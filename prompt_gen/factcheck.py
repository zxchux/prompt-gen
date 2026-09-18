"""Extract claims, make them self-contained, verify them, and combine the verdicts."""

import json
import logging

from prompt_gen.config import FactcheckConfig
from prompt_gen.llm_client import OpenRouterClient
from prompt_gen.models import (
    CrawledPage,
    ExtractedPrompt,
    FactcheckClaim,
    FactcheckResult,
)

logger = logging.getLogger(__name__)

DECOMPOSE_SYSTEM = """\
You are a factcheck specialist following a rigorous claim-based factcheck methodology. \
Your task is to DECOMPOSE a search prompt and its context into individual, \
atomic claims that can be independently verified.

The factcheck method requires:
1. Break the prompt into its constituent factual claims
2. Each claim should be a single, atomic assertion
3. Include implicit claims (assumptions embedded in the prompt)
4. Include claims about the relationship between the prompt and the source content
5. Identify any claims about search intent, topic relevance, or content coverage

Be thorough - extract ALL verifiable claims, even subtle ones.

Respond with valid JSON:
{
  "claims": [
    {
      "claim_text": "The atomic claim statement",
      "claim_type": "explicit" | "implicit" | "relational",
      "source_context": "The part of the content this claim relates to",
      "is_verifiable": true/false,
      "verification_difficulty": "easy" | "medium" | "hard"
    }
  ],
  "decomposition_reasoning": "Explanation of how claims were extracted"
}"""

DECONTEXTUALIZE_SYSTEM = """\
You are a factcheck specialist. Your task is to DECONTEXTUALIZE claims - \
make each claim fully self-contained so it can be verified independently \
without needing the original context.

For each claim:
1. Replace pronouns with specific references
2. Add necessary context that was implicit
3. Make the claim unambiguous
4. Ensure it can stand alone as a verifiable statement

Respond with valid JSON:
{
  "decontextualized_claims": [
    {
      "original_claim": "the original claim",
      "decontextualized_claim": "the self-contained version",
      "added_context": "what context was added"
    }
  ]
}"""

VERIFY_SYSTEM = """\
You are a factcheck specialist performing the VERIFICATION step of the \
claim-based factcheck methodology. For each claim, determine whether it is:

- VERIFIED: The claim is supported by the provided evidence/content
- REFUTED: The claim contradicts the provided evidence/content
- UNVERIFIABLE: There is insufficient evidence to verify or refute the claim

For each claim, provide:
1. A verification verdict
2. The specific evidence supporting your verdict
3. Your confidence level (0.0-1.0)
4. Detailed reasoning

Be rigorous - only mark claims as VERIFIED if there is clear supporting evidence.

Respond with valid JSON:
{
  "verifications": [
    {
      "claim_text": "the claim being verified",
      "verdict": "verified" | "refuted" | "unverifiable",
      "evidence": ["specific evidence points"],
      "confidence": 0.0-1.0,
      "reasoning": "detailed explanation"
    }
  ]
}"""

AGGREGATE_SYSTEM = """\
You are a factcheck specialist performing the AGGREGATION step of the \
claim-based factcheck methodology. Given the individual claim verification \
results, produce an overall factcheck verdict for the search prompt.

Consider:
1. What proportion of claims were verified vs refuted vs unverifiable?
2. How critical are the refuted claims? (A single critical refutation \
can invalidate the whole prompt)
3. What is the overall confidence level?
4. Is the prompt fundamentally sound despite minor issues?

Scoring guidelines:
- 0.9-1.0: All claims verified with high confidence
- 0.7-0.89: Most claims verified, minor issues only
- 0.5-0.69: Mixed results, some concerns
- 0.3-0.49: Significant issues, multiple refuted claims
- 0.0-0.29: Fundamentally flawed, critical claims refuted

Respond with valid JSON:
{
  "overall_score": 0.0-1.0,
  "is_factually_sound": true/false,
  "confidence_level": 0.0-1.0,
  "summary": "Overall factcheck summary",
  "critical_issues": ["list of critical issues if any"],
  "minor_issues": ["list of minor issues if any"],
  "strengths": ["factual strengths of the prompt"]
}"""


class Factchecker:
    """Check prompt claims against source content through four model-assisted steps."""

    def __init__(self, config: FactcheckConfig, llm_client: OpenRouterClient):
        self.config = config
        self.llm = llm_client

    def _get_source_content(
        self, prompt: ExtractedPrompt, pages: list[CrawledPage]
    ) -> str:
        """Get source content for factchecking a prompt."""
        relevant_pages = [p for p in pages if p.url in prompt.source_urls]
        if not relevant_pages:
            relevant_pages = pages[:3]

        content_parts = []
        for page in relevant_pages[:3]:
            parts = []
            if page.title:
                parts.append(f"Title: {page.title}")
            if page.meta_description:
                parts.append(f"Description: {page.meta_description}")
            if page.h1_tags:
                parts.append(f"H1: {', '.join(page.h1_tags)}")
            if page.h2_tags:
                parts.append(f"H2: {', '.join(page.h2_tags[:8])}")
            body = page.body_text[:2000] if page.body_text else ""
            if body:
                parts.append(f"Content: {body}")
            content_parts.append("\n".join(parts))

        return "\n\n---\n\n".join(content_parts)

    def _step_decompose(
        self, prompt: ExtractedPrompt, source_content: str
    ) -> list[FactcheckClaim]:
        """Step 1: Decompose the prompt into atomic claims."""
        user_prompt = (
            f"Decompose this search prompt and its relationship to the source content "
            f"into individual verifiable claims.\n\n"
            f"Search Prompt: \"{prompt.prompt_text}\"\n"
            f"Search Intent: {prompt.search_intent}\n"
            f"Topic Cluster: {prompt.topic_cluster}\n"
            f"Extraction Reasoning: {prompt.extraction_reasoning}\n\n"
            f"Source Content:\n{source_content}"
        )

        try:
            result = self.llm.analyze_json(
                system_prompt=DECOMPOSE_SYSTEM,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=2048,
            )

            claims = []
            for item in result.get("claims", [])[:self.config.max_claims_per_prompt]:
                claim = FactcheckClaim(
                    claim_text=item.get("claim_text", ""),
                    source_context=item.get("source_context", ""),
                    is_verifiable=item.get("is_verifiable", True),
                )
                if claim.claim_text:
                    claims.append(claim)

            logger.debug(
                f"Decomposed into {len(claims)} claims: {prompt.prompt_text[:50]}"
            )
            return claims

        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            # Create a single claim from the prompt itself
            return [
                FactcheckClaim(
                    claim_text=(
                        f"The search prompt '{prompt.prompt_text}' "
                        f"is relevant to the source content"
                    ),
                    source_context="",
                    is_verifiable=True,
                )
            ]

    def _step_decontextualize(
        self, claims: list[FactcheckClaim], prompt: ExtractedPrompt
    ) -> list[FactcheckClaim]:
        """Step 2: Decontextualize claims to be self-contained."""
        if not claims:
            return claims

        claims_data = [
            {"claim_text": c.claim_text, "source_context": c.source_context}
            for c in claims
        ]

        user_prompt = (
            f"Decontextualize these claims extracted from the search prompt "
            f"\"{prompt.prompt_text}\" so each can be verified independently.\n\n"
            f"Claims:\n{json.dumps(claims_data, indent=2)}"
        )

        try:
            result = self.llm.analyze_json(
                system_prompt=DECONTEXTUALIZE_SYSTEM,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=2048,
            )

            decontextualized = result.get("decontextualized_claims", [])
            for i, decon in enumerate(decontextualized):
                if i < len(claims):
                    new_text = decon.get("decontextualized_claim", "")
                    if new_text:
                        claims[i].claim_text = new_text

            logger.debug(f"Decontextualized {len(claims)} claims")
            return claims

        except Exception as e:
            logger.error(f"Decontextualization failed: {e}")
            return claims  # Return original claims if this step fails

    def _step_verify(
        self,
        claims: list[FactcheckClaim],
        source_content: str,
        prompt: ExtractedPrompt,
    ) -> list[FactcheckClaim]:
        """Step 3: Verify each claim against source content."""
        if not claims:
            return claims

        # Run verification in rounds for higher confidence
        for round_num in range(self.config.verification_rounds):
            claims_data = [
                {
                    "claim_text": c.claim_text,
                    "is_verifiable": c.is_verifiable,
                    "previous_verdict": c.verification_status
                    if round_num > 0
                    else "pending",
                }
                for c in claims
            ]

            user_prompt = (
                f"Verify these claims against the source content. "
                f"This is verification round {round_num + 1}/{self.config.verification_rounds}.\n\n"
                f"Search Prompt: \"{prompt.prompt_text}\"\n\n"
                f"Claims to verify:\n{json.dumps(claims_data, indent=2)}\n\n"
                f"Source Content:\n{source_content}"
            )

            try:
                result = self.llm.analyze_json(
                    system_prompt=VERIFY_SYSTEM,
                    user_prompt=user_prompt,
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=2048,
                )

                verifications = result.get("verifications", [])
                for i, verification in enumerate(verifications):
                    if i < len(claims):
                        claims[i].verification_status = verification.get(
                            "verdict", "unverifiable"
                        )
                        claims[i].confidence = float(
                            verification.get("confidence", 0.5)
                        )
                        claims[i].evidence = verification.get("evidence", [])
                        claims[i].reasoning = verification.get("reasoning", "")

                logger.debug(
                    f"Verification round {round_num + 1} complete for "
                    f"{len(claims)} claims"
                )

            except Exception as e:
                logger.error(f"Verification round {round_num + 1} failed: {e}")
                # Mark unverified claims
                for claim in claims:
                    if claim.verification_status == "pending":
                        claim.verification_status = "unverifiable"
                        claim.confidence = 0.3
                break  # Stop further rounds on failure

        return claims

    def _step_aggregate(
        self, claims: list[FactcheckClaim], prompt: ExtractedPrompt
    ) -> tuple[float, bool, float, str]:
        """Step 4: Aggregate claim verdicts into overall score."""
        if not claims:
            return 0.5, True, 0.5, "No claims to verify"

        verification_summary = []
        for claim in claims:
            verification_summary.append(
                {
                    "claim": claim.claim_text[:100],
                    "verdict": claim.verification_status,
                    "confidence": claim.confidence,
                }
            )

        user_prompt = (
            f"Aggregate these claim verification results into an overall "
            f"factcheck verdict for the search prompt.\n\n"
            f"Search Prompt: \"{prompt.prompt_text}\"\n\n"
            f"Claim Verifications:\n{json.dumps(verification_summary, indent=2)}"
        )

        try:
            result = self.llm.analyze_json(
                system_prompt=AGGREGATE_SYSTEM,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=1024,
            )

            overall_score = float(result.get("overall_score", 0.5))
            is_sound = result.get("is_factually_sound", False)
            confidence = float(result.get("confidence_level", 0.5))
            summary = result.get("summary", "")

            return overall_score, is_sound, confidence, summary

        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            # Calculate manually
            verified = sum(
                1 for c in claims if c.verification_status == "verified"
            )
            refuted = sum(
                1 for c in claims if c.verification_status == "refuted"
            )
            total = len(claims)

            if total == 0:
                return 0.5, True, 0.5, "No claims to aggregate"

            score = verified / total
            is_sound = refuted == 0 and score >= 0.5
            confidence = sum(c.confidence for c in claims) / total

            return (
                score,
                is_sound,
                confidence,
                f"Manual aggregation: {verified}/{total} verified, {refuted} refuted",
            )

    def factcheck_prompt(
        self, prompt: ExtractedPrompt, pages: list[CrawledPage]
    ) -> FactcheckResult:
        """Check a prompt against its source pages and return claim verdicts and scores."""
        source_content = self._get_source_content(prompt, pages)

        logger.debug(f"Factcheck Step 1 - Decompose: {prompt.prompt_text[:50]}")
        claims = self._step_decompose(prompt, source_content)

        logger.debug(f"Factcheck Step 2 - Decontextualize: {len(claims)} claims")
        claims = self._step_decontextualize(claims, prompt)

        logger.debug(f"Factcheck Step 3 - Verify: {len(claims)} claims")
        claims = self._step_verify(claims, source_content, prompt)

        logger.debug("Factcheck Step 4 - Aggregate")
        overall_score, is_sound, confidence, summary = self._step_aggregate(
            claims, prompt
        )

        return FactcheckResult(
            prompt_text=prompt.prompt_text,
            claims=claims,
            overall_factual_score=overall_score,
            decomposition_reasoning="",
            verification_summary=summary,
            is_factually_sound=is_sound,
            confidence_level=confidence,
        )

    def factcheck_prompts(
        self,
        prompts: list[ExtractedPrompt],
        pages: list[CrawledPage],
        progress_callback=None,
    ) -> list[FactcheckResult]:
        """Check prompts in order, calling progress_callback(current, total) after each."""
        results = []
        total = len(prompts)

        logger.info(f"Starting factcheck for {total} prompts")

        for i, prompt in enumerate(prompts):
            logger.info(
                f"Factchecking [{i + 1}/{total}]: {prompt.prompt_text[:60]}..."
            )

            result = self.factcheck_prompt(prompt, pages)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

            verified_count = sum(
                1 for c in result.claims if c.verification_status == "verified"
            )
            logger.info(
                f"  -> Score: {result.overall_factual_score:.2f} | "
                f"Sound: {result.is_factually_sound} | "
                f"Claims: {verified_count}/{len(result.claims)} verified"
            )

        passed = sum(
            1
            for r in results
            if r.overall_factual_score >= self.config.confidence_threshold
        )
        logger.info(
            f"Factcheck complete: {passed}/{total} prompts passed "
            f"(threshold: {self.config.confidence_threshold})"
        )

        return results


# Alias for backward-compatible imports
Factchecker = Factchecker
