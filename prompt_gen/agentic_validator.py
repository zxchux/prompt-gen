"""Evaluate prompts in a four-turn conversation: proposer, challenger, defender, judge."""

import json
import logging

from prompt_gen.config import PromptExtractionConfig
from prompt_gen.llm_client import OpenRouterClient
from prompt_gen.models import (
    CrawledPage,
    ExtractedPrompt,
    QualityRating,
    ValidationResult,
)

logger = logging.getLogger(__name__)

DEBATE_SYSTEM_PROMPT = """\
You are participating in a rigorous multi-agent debate to evaluate whether \
a search prompt/query is valid, accurate, faithful, and high-quality for a \
specific website.

You will be asked to take on different roles during this debate:
- PROPOSER: Argue FOR the prompt's validity with evidence
- CHALLENGER: Argue AGAINST the prompt, find every weakness
- DEFENDER: Counter the challenges with specific evidence
- JUDGE: Weigh all arguments and deliver a final verdict

Be thorough, specific, and cite evidence from the source content. Do not \
be superficial - dig deep into potential issues. When challenging, be \
genuinely adversarial. When defending, address each challenge directly. \
When judging, be fair but rigorous.

Always respond with valid JSON as specified in each turn."""

TURN_1_PROPOSER = """ROLE: PROPOSER
You are arguing FOR the validity of this search prompt.

Search Prompt: "{prompt_text}"
Extraction Reasoning: "{reasoning}"
Search Intent: {search_intent}
Topic Cluster: {topic_cluster}

Source Page Content:
{page_content}

Make the strongest possible case that this prompt is:
1. ACCURATE: It genuinely matches what the source page covers
2. FAITHFUL: Real users would actually search for this exact query
3. HIGH-QUALITY: It's a valuable prompt worth targeting for SEO

Provide specific evidence from the page content. Be thorough.

Respond with JSON:
{{
  "accuracy_argument": "detailed argument with evidence for accuracy",
  "faithfulness_argument": "detailed argument for why real users search this",
  "quality_argument": "detailed argument for strategic value",
  "key_evidence": ["list of specific evidence points from the content"],
  "confidence": 0.0-1.0
}}"""

TURN_2_CHALLENGER = """ROLE: CHALLENGER (Devil's Advocate)
You must now CHALLENGE the Proposer's arguments. Your job is to find \
every possible weakness, flaw, and reason this prompt might be INVALID.

The Proposer argued:
{proposer_arguments}

Now challenge each point. Consider:

ACCURACY CHALLENGES:
- Is the prompt too broad/narrow for the actual page content?
- Does the page REALLY cover this topic, or is it a stretch?
- Would a user searching this actually find the page useful?
- Are there factual mismatches between the prompt and content?

FAITHFULNESS CHALLENGES:
- Would a real person actually type this exact query?
- Is the phrasing natural or artificially constructed?
- Is this a manufactured query that no one actually searches?
- Does the search intent claim actually match the query?

QUALITY CHALLENGES:
- Is this too competitive to realistically rank for?
- Is the search volume likely too low to matter?
- Is this a vanity query with no real business value?
- Are there much better alternative queries to target?
- Is this just a generic query that any website could target?

Be genuinely adversarial. Find REAL problems, not superficial ones.

Respond with JSON:
{{
  "accuracy_challenges": ["list of specific accuracy concerns"],
  "faithfulness_challenges": ["list of specific faithfulness concerns"],
  "quality_challenges": ["list of specific quality concerns"],
  "strongest_objection": "the single most damaging argument against this prompt",
  "recommended_verdict": "reject" | "needs_improvement" | "acceptable",
  "severity": "critical" | "moderate" | "minor"
}}"""

TURN_3_DEFENDER = """ROLE: DEFENDER
The Challenger has raised these objections against the prompt "{prompt_text}".
You must now DEFEND the prompt by addressing each challenge directly.

Challenger's Objections:
{challenger_arguments}

Original Proposer's Case:
{proposer_arguments}

For each challenge:
1. Acknowledge if the concern has merit
2. Provide counter-evidence or counter-arguments
3. Explain why the prompt should still be considered valid \
(or concede if the challenge is too strong)

Be honest - if a challenge is valid and cannot be countered, admit it.

Respond with JSON:
{{
  "accuracy_defense": "response to accuracy challenges with evidence",
  "faithfulness_defense": "response to faithfulness challenges",
  "quality_defense": "response to quality challenges",
  "concessions": ["list of challenges you concede are valid"],
  "strongest_counter": "your strongest counter-argument",
  "revised_confidence": 0.0-1.0
}}"""

TURN_4_JUDGE = """ROLE: JUDGE
You have observed the full debate about the search prompt: "{prompt_text}"

PROPOSER'S CASE:
{proposer_arguments}

CHALLENGER'S OBJECTIONS:
{challenger_arguments}

DEFENDER'S RESPONSE:
{defender_arguments}

Now deliver your final verdict. Weigh all arguments carefully:

1. Score ACCURACY (0.0-1.0): How well does this prompt match the source content?
2. Score FAITHFULNESS (0.0-1.0): How realistic is this as an actual search query?
3. Score QUALITY (0.0-1.0): How valuable is this prompt for the website?
4. Calculate OVERALL score (weighted: accuracy 30%, faithfulness 30%, quality 40%)
5. Determine if the prompt PASSES or FAILS (threshold: 0.7)

Consider:
- Which side presented stronger evidence?
- Were the challenges adequately addressed?
- What concessions were made and how significant are they?
- On balance, is this a prompt worth pursuing?

Respond with JSON:
{{
  "accuracy_score": 0.0-1.0,
  "accuracy_reasoning": "final assessment of accuracy",
  "is_accurate": true/false,
  "faithfulness_score": 0.0-1.0,
  "faithfulness_reasoning": "final assessment of faithfulness",
  "is_faithful": true/false,
  "quality_score": 0.0-1.0,
  "quality_rating": "high" | "medium" | "low" | "rejected",
  "quality_reasoning": "final assessment of quality",
  "overall_score": 0.0-1.0,
  "verdict": "pass" | "fail",
  "verdict_reasoning": "comprehensive summary of why this prompt passes or fails",
  "key_strengths": ["strengths identified during debate"],
  "key_weaknesses": ["weaknesses that survived the debate"],
  "improvement_suggestions": ["how the prompt could be improved if applicable"]
}}"""


class AgenticValidator:
    """Evaluate each prompt with four roles sharing one conversation history."""

    def __init__(self, config: PromptExtractionConfig, llm_client: OpenRouterClient):
        self.config = config
        self.llm = llm_client

    def _get_page_content_for_prompt(
        self, prompt: ExtractedPrompt, pages: list[CrawledPage]
    ) -> str:
        """Get relevant page content for a prompt's source URLs."""
        relevant_pages = []
        for page in pages:
            if page.url in prompt.source_urls:
                relevant_pages.append(page)

        if not relevant_pages:
            relevant_pages = pages[:3]

        content_parts = []
        for page in relevant_pages[:3]:
            parts = [f"URL: {page.url}"]
            if page.title:
                parts.append(f"Title: {page.title}")
            if page.meta_description:
                parts.append(f"Meta: {page.meta_description}")
            if page.h1_tags:
                parts.append(f"H1: {', '.join(page.h1_tags)}")
            if page.h2_tags:
                parts.append(f"H2: {', '.join(page.h2_tags[:5])}")
            body_preview = page.body_text[:1500] if page.body_text else ""
            if body_preview:
                parts.append(f"Content: {body_preview}")
            content_parts.append("\n".join(parts))

        return "\n\n---\n\n".join(content_parts)

    def _truncate_for_context(self, text: str, max_chars: int = 800) -> str:
        """Truncate text to fit in context while keeping it meaningful."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def validate_prompt(
        self, prompt: ExtractedPrompt, pages: list[CrawledPage]
    ) -> ValidationResult:
        """Run the four debate turns and return the judge's scores and a debate summary."""
        chain_of_thought = []
        page_content = self._get_page_content_for_prompt(prompt, pages)

        messages = [{"role": "system", "content": DEBATE_SYSTEM_PROMPT}]

        logger.debug(f"Debate Turn 1 (Proposer): {prompt.prompt_text[:50]}")
        turn1_prompt = TURN_1_PROPOSER.format(
            prompt_text=prompt.prompt_text,
            reasoning=prompt.extraction_reasoning or "N/A",
            search_intent=prompt.search_intent,
            topic_cluster=prompt.topic_cluster,
            page_content=page_content,
        )
        messages.append({"role": "user", "content": turn1_prompt})

        try:
            proposer_response = self.llm.chat(
                messages=messages,
                temperature=0.4,
                max_tokens=1024,
            )
            messages.append({"role": "assistant", "content": proposer_response})

            try:
                proposer_data = json.loads(proposer_response)
            except json.JSONDecodeError:
                start = proposer_response.find("{")
                end = proposer_response.rfind("}") + 1
                if start >= 0 and end > start:
                    proposer_data = json.loads(proposer_response[start:end])
                else:
                    proposer_data = {"confidence": 0.5}

            chain_of_thought.append(
                f"PROPOSER (confidence: {proposer_data.get('confidence', 'N/A')}): "
                f"Argued for accuracy, faithfulness, and quality with "
                f"{len(proposer_data.get('key_evidence', []))} evidence points"
            )
        except Exception as e:
            logger.error(f"Proposer turn failed: {e}")
            proposer_response = '{"confidence": 0.5, "accuracy_argument": "Analysis failed"}'
            messages.append({"role": "assistant", "content": proposer_response})
            chain_of_thought.append(f"PROPOSER: FAILED - {e}")

        logger.debug(f"Debate Turn 2 (Challenger): {prompt.prompt_text[:50]}")
        turn2_prompt = TURN_2_CHALLENGER.format(
            proposer_arguments=self._truncate_for_context(proposer_response),
        )
        messages.append({"role": "user", "content": turn2_prompt})

        try:
            challenger_response = self.llm.chat(
                messages=messages,
                temperature=0.5,  # Slightly higher temp for more creative challenges
                max_tokens=1024,
            )
            messages.append({"role": "assistant", "content": challenger_response})

            try:
                challenger_data = json.loads(challenger_response)
            except json.JSONDecodeError:
                start = challenger_response.find("{")
                end = challenger_response.rfind("}") + 1
                if start >= 0 and end > start:
                    challenger_data = json.loads(challenger_response[start:end])
                else:
                    challenger_data = {"severity": "minor", "recommended_verdict": "acceptable"}

            chain_of_thought.append(
                f"CHALLENGER (severity: {challenger_data.get('severity', 'N/A')}): "
                f"Raised {len(challenger_data.get('accuracy_challenges', []))} accuracy, "
                f"{len(challenger_data.get('faithfulness_challenges', []))} faithfulness, "
                f"{len(challenger_data.get('quality_challenges', []))} quality challenges. "
                f"Strongest: {challenger_data.get('strongest_objection', 'N/A')[:100]}"
            )
        except Exception as e:
            logger.error(f"Challenger turn failed: {e}")
            challenger_response = '{"severity": "minor", "recommended_verdict": "acceptable"}'
            messages.append({"role": "assistant", "content": challenger_response})
            chain_of_thought.append(f"CHALLENGER: FAILED - {e}")

        logger.debug(f"Debate Turn 3 (Defender): {prompt.prompt_text[:50]}")
        turn3_prompt = TURN_3_DEFENDER.format(
            prompt_text=prompt.prompt_text,
            challenger_arguments=self._truncate_for_context(challenger_response),
            proposer_arguments=self._truncate_for_context(proposer_response, 500),
        )
        messages.append({"role": "user", "content": turn3_prompt})

        try:
            defender_response = self.llm.chat(
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            messages.append({"role": "assistant", "content": defender_response})

            try:
                defender_data = json.loads(defender_response)
            except json.JSONDecodeError:
                start = defender_response.find("{")
                end = defender_response.rfind("}") + 1
                if start >= 0 and end > start:
                    defender_data = json.loads(defender_response[start:end])
                else:
                    defender_data = {"revised_confidence": 0.5, "concessions": []}

            concessions = defender_data.get("concessions", [])
            chain_of_thought.append(
                f"DEFENDER (revised confidence: {defender_data.get('revised_confidence', 'N/A')}): "
                f"Conceded {len(concessions)} points. "
                f"Counter: {defender_data.get('strongest_counter', 'N/A')[:100]}"
            )
        except Exception as e:
            logger.error(f"Defender turn failed: {e}")
            defender_response = '{"revised_confidence": 0.5, "concessions": []}'
            messages.append({"role": "assistant", "content": defender_response})
            chain_of_thought.append(f"DEFENDER: FAILED - {e}")

        logger.debug(f"Debate Turn 4 (Judge): {prompt.prompt_text[:50]}")
        turn4_prompt = TURN_4_JUDGE.format(
            prompt_text=prompt.prompt_text,
            proposer_arguments=self._truncate_for_context(proposer_response, 600),
            challenger_arguments=self._truncate_for_context(challenger_response, 600),
            defender_arguments=self._truncate_for_context(defender_response, 600),
        )
        messages.append({"role": "user", "content": turn4_prompt})

        try:
            judge_response = self.llm.chat(
                messages=messages,
                temperature=0.2,  # Low temp for consistent judging
                max_tokens=1024,
            )

            try:
                judge_data = json.loads(judge_response)
            except json.JSONDecodeError:
                start = judge_response.find("{")
                end = judge_response.rfind("}") + 1
                if start >= 0 and end > start:
                    judge_data = json.loads(judge_response[start:end])
                else:
                    judge_data = {}

            accuracy_score = float(judge_data.get("accuracy_score", 0.5))
            is_accurate = judge_data.get("is_accurate", accuracy_score >= 0.6)
            accuracy_reasoning = judge_data.get("accuracy_reasoning", "")

            faithfulness_score = float(judge_data.get("faithfulness_score", 0.5))
            is_faithful = judge_data.get("is_faithful", faithfulness_score >= 0.6)
            faithfulness_reasoning = judge_data.get("faithfulness_reasoning", "")

            quality_score = float(judge_data.get("quality_score", 0.5))
            quality_rating_str = judge_data.get("quality_rating", "medium")
            quality_reasoning = judge_data.get("quality_reasoning", "")

            overall_score = float(judge_data.get("overall_score", 0.5))
            verdict = judge_data.get("verdict", "fail")

            chain_of_thought.append(
                f"JUDGE (verdict: {verdict}, score: {overall_score:.2f}): "
                f"{judge_data.get('verdict_reasoning', 'N/A')[:150]}"
            )

        except Exception as e:
            logger.error(f"Judge turn failed: {e}")
            accuracy_score = 0.5
            is_accurate = True
            accuracy_reasoning = "Judge analysis failed"
            faithfulness_score = 0.5
            is_faithful = True
            faithfulness_reasoning = "Judge analysis failed"
            quality_score = 0.5
            quality_rating_str = "medium"
            quality_reasoning = "Judge analysis failed"
            overall_score = 0.5
            chain_of_thought.append(f"JUDGE: FAILED - {e}")

        quality_rating_map = {
            "high": QualityRating.HIGH,
            "medium": QualityRating.MEDIUM,
            "low": QualityRating.LOW,
            "rejected": QualityRating.REJECTED,
        }
        quality_rating = quality_rating_map.get(
            quality_rating_str.lower(), QualityRating.MEDIUM
        )

        return ValidationResult(
            prompt_text=prompt.prompt_text,
            is_accurate=is_accurate,
            accuracy_score=accuracy_score,
            accuracy_reasoning=accuracy_reasoning,
            is_faithful=is_faithful,
            faithfulness_score=faithfulness_score,
            faithfulness_reasoning=faithfulness_reasoning,
            quality_rating=quality_rating,
            quality_score=quality_score,
            quality_reasoning=quality_reasoning,
            overall_score=overall_score,
            agent_chain_of_thought=chain_of_thought,
        )

    def validate_prompts(
        self,
        prompts: list[ExtractedPrompt],
        pages: list[CrawledPage],
        progress_callback=None,
    ) -> list[ValidationResult]:
        """Validate prompts in order, calling progress_callback(current, total) after each."""
        results = []
        total = len(prompts)

        logger.info(f"Starting adversarial agentic validation of {total} prompts")
        logger.info("Each prompt goes through: Proposer → Challenger → Defender → Judge")

        for i, prompt in enumerate(prompts):
            logger.info(
                f"Debating prompt [{i + 1}/{total}]: {prompt.prompt_text[:60]}..."
            )

            result = self.validate_prompt(prompt, pages)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

            passed = result.overall_score >= self.config.min_quality_score
            verdict = "PASS" if passed else "FAIL"
            logger.info(
                f"  → Verdict: {verdict} "
                f"(score: {result.overall_score:.2f}) | "
                f"Accuracy: {result.accuracy_score:.2f} | "
                f"Faithful: {result.faithfulness_score:.2f} | "
                f"Quality: {result.quality_rating.value}"
            )
            for step in result.agent_chain_of_thought:
                logger.debug(f"    {step}")

        passed = sum(
            1 for r in results if r.overall_score >= self.config.min_quality_score
        )
        avg_score = sum(r.overall_score for r in results) / len(results) if results else 0
        logger.info(
            f"Validation complete: {passed}/{total} prompts passed "
            f"(threshold: {self.config.min_quality_score}, avg score: {avg_score:.2f})"
        )

        return results

    def filter_validated(
        self,
        prompts: list[ExtractedPrompt],
        results: list[ValidationResult],
    ) -> list[tuple[ExtractedPrompt, ValidationResult]]:
        """Return passing (prompt, result) pairs, sorted by overall score."""
        passed = []
        for prompt, result in zip(prompts, results):
            if (
                result.overall_score >= self.config.min_quality_score
                and result.quality_rating != QualityRating.REJECTED
            ):
                passed.append((prompt, result))

        passed.sort(key=lambda x: x[1].overall_score, reverse=True)
        return passed
