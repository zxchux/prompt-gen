"""Match prompts to Search Console queries and Analytics pages, then score opportunities."""

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Optional
from urllib.parse import urlparse

from prompt_gen.config import Settings
from prompt_gen.llm_client import OpenRouterClient
from prompt_gen.models import (
    CorrelatedPrompt,
    ExtractedPrompt,
    FactcheckResult,
    GAData,
    GSCData,
    ValidationResult,
)

logger = logging.getLogger(__name__)

CORRELATION_SYSTEM_PROMPT = """You are a search analytics expert. Analyze the correlation between a search prompt and the available Google Search Console / Google Analytics data.

Evaluate:
1. DATA MATCH: How closely does the prompt match existing search queries in GSC?
2. TRAFFIC POTENTIAL: Based on current metrics, what's the traffic potential?
3. OPTIMIZATION OPPORTUNITY: Is there a gap between current performance and potential?
4. IMPACT PREDICTION: What impact would monitoring/optimizing for this prompt have?

Consider:
- Prompts matching high-impression, low-CTR queries = high optimization opportunity
- Prompts matching high-position queries (>10) = ranking improvement opportunity
- Prompts with no GSC match but high relevance = content gap opportunity
- Pages with high bounce rate for related queries = content quality opportunity

Respond with valid JSON:
{
  "data_correlation_score": 0.0-1.0,
  "impact_score": 0.0-1.0,
  "opportunity_type": "optimization" | "content_gap" | "ranking_improvement" | "new_opportunity",
  "recommendation": "specific actionable recommendation",
  "reasoning": "detailed analysis",
  "estimated_traffic_impact": "high" | "medium" | "low",
  "priority": "critical" | "high" | "medium" | "low"
}"""


class CorrelationEngine:
    """Match search and traffic data to prompts and rank the resulting opportunities."""

    def __init__(self, settings: Settings, llm_client: OpenRouterClient):
        self.settings = settings
        self.llm = llm_client

    def _normalize_query(self, query: str) -> str:
        """Normalize a query for comparison."""
        return re.sub(r"\s+", " ", query.lower().strip())

    def _fuzzy_match_score(self, text1: str, text2: str) -> float:
        """Calculate fuzzy match score between two strings."""
        norm1 = self._normalize_query(text1)
        norm2 = self._normalize_query(text2)

        if norm1 == norm2:
            return 1.0

        seq_score = SequenceMatcher(None, norm1, norm2).ratio()

        words1 = set(norm1.split())
        words2 = set(norm2.split())
        if words1 and words2:
            overlap = len(words1 & words2)
            word_score = overlap / max(len(words1), len(words2))
        else:
            word_score = 0.0

        containment_score = 0.0
        if norm1 in norm2 or norm2 in norm1:
            containment_score = 0.8

        return max(seq_score, word_score, containment_score)

    def _find_matching_gsc_data(
        self, prompt: ExtractedPrompt, gsc_data: list[GSCData], threshold: float = 0.5
    ) -> list[tuple[GSCData, float]]:
        """Return up to ten (query, match_score) pairs, highest score first."""
        matches = []
        prompt_normalized = self._normalize_query(prompt.prompt_text)

        for gsc in gsc_data:
            score = self._fuzzy_match_score(prompt.prompt_text, gsc.query)
            if score >= threshold:
                matches.append((gsc, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:10]  # Top 10 matches

    def _find_matching_ga_data(
        self,
        prompt: ExtractedPrompt,
        ga_data: dict[str, GAData],
        source_urls: list[str],
    ) -> Optional[GAData]:
        """Find Analytics data using source URL paths, trying exact matches first."""
        for url in source_urls:
            parsed = urlparse(url)
            page_path = parsed.path

            # Try exact match
            if page_path in ga_data:
                return ga_data[page_path]

            # Try with/without trailing slash
            alt_path = page_path.rstrip("/") if page_path.endswith("/") else page_path + "/"
            if alt_path in ga_data:
                return ga_data[alt_path]

            # Try partial match
            for ga_path, data in ga_data.items():
                if page_path in ga_path or ga_path in page_path:
                    return data

        return None

    def _calculate_data_correlation_score(
        self,
        gsc_matches: list[tuple[GSCData, float]],
        ga_match: Optional[GAData],
    ) -> float:
        """Calculate how well the prompt correlates with existing data."""
        score = 0.0

        # GSC correlation (0-0.6)
        if gsc_matches:
            best_match_score = gsc_matches[0][1]
            best_gsc = gsc_matches[0][0]

            score += best_match_score * 0.3

            # Data richness (has clicks/impressions)
            if best_gsc.impressions > 0:
                score += 0.15
            if best_gsc.clicks > 0:
                score += 0.15

        # GA correlation (0-0.4)
        if ga_match:
            score += 0.2
            if ga_match.organic_sessions > 0:
                score += 0.1
            if ga_match.sessions > 10:
                score += 0.1

        return min(score, 1.0)

    def _calculate_impact_score(
        self,
        prompt: ExtractedPrompt,
        validation: Optional[ValidationResult],
        factcheck: Optional[FactcheckResult],
        gsc_matches: list[tuple[GSCData, float]],
        ga_match: Optional[GAData],
    ) -> float:
        """Score relevance, validation, search opportunity, and traffic potential.

        Normalize by the weights available when validation or factcheck results are missing.
        """
        score = 0.0
        weights_used = 0.0

        # Validation quality (weight: 0.2)
        if validation:
            score += validation.overall_score * 0.2
            weights_used += 0.2

        # Factcheck confidence (weight: 0.15)
        if factcheck:
            score += factcheck.overall_factual_score * 0.15
            weights_used += 0.15

        # Prompt relevance (weight: 0.15)
        score += prompt.relevance_score * 0.15
        weights_used += 0.15

        # Search performance opportunity (weight: 0.3)
        if gsc_matches:
            best_gsc = gsc_matches[0][0]

            # High impressions + low CTR = big opportunity
            if best_gsc.impressions > 100 and best_gsc.ctr < 0.05:
                score += 0.3
            elif best_gsc.impressions > 50 and best_gsc.ctr < 0.1:
                score += 0.2
            elif best_gsc.impressions > 10:
                score += 0.1

            # Position opportunity (not on page 1)
            if best_gsc.position > 10:
                score += 0.1  # Room to improve to page 1
            elif best_gsc.position > 3:
                score += 0.05  # Room to improve to top 3

            weights_used += 0.3
        else:
            # No GSC data = potential new opportunity
            if prompt.estimated_volume in ("high", "medium"):
                score += 0.15
            weights_used += 0.3

        # Traffic potential (weight: 0.2)
        if ga_match:
            if ga_match.organic_sessions > 100:
                score += 0.2
            elif ga_match.organic_sessions > 50:
                score += 0.15
            elif ga_match.organic_sessions > 10:
                score += 0.1
            else:
                score += 0.05
            weights_used += 0.2
        else:
            # Estimate based on volume
            volume_scores = {"high": 0.15, "medium": 0.1, "low": 0.05}
            score += volume_scores.get(prompt.estimated_volume, 0.05)
            weights_used += 0.2

        # Normalize if not all weights were used
        if weights_used > 0:
            score = score / weights_used * 1.0

        return min(score, 1.0)

    def _llm_correlation_analysis(
        self,
        prompt: ExtractedPrompt,
        validation: Optional[ValidationResult],
        factcheck: Optional[FactcheckResult],
        gsc_matches: list[tuple[GSCData, float]],
        ga_match: Optional[GAData],
    ) -> dict[str, Any]:
        """Use LLM to perform deeper correlation analysis."""
        context_parts = [
            f"Search Prompt: \"{prompt.prompt_text}\"",
            f"Search Intent: {prompt.search_intent}",
            f"Topic: {prompt.topic_cluster}",
            f"Estimated Volume: {prompt.estimated_volume}",
            f"Difficulty: {prompt.difficulty}",
        ]

        if validation:
            context_parts.append(
                f"Validation Score: {validation.overall_score:.2f} "
                f"(Accuracy: {validation.accuracy_score:.2f}, "
                f"Faithfulness: {validation.faithfulness_score:.2f}, "
                f"Quality: {validation.quality_score:.2f})"
            )

        if factcheck:
            context_parts.append(
                f"Factcheck Score: {factcheck.overall_factual_score:.2f} "
                f"(Sound: {factcheck.is_factually_sound})"
            )

        if gsc_matches:
            gsc_context = []
            for gsc, match_score in gsc_matches[:5]:
                gsc_context.append(
                    f"  - \"{gsc.query}\" (match: {match_score:.2f}) - "
                    f"Clicks: {gsc.clicks}, Impressions: {gsc.impressions}, "
                    f"CTR: {gsc.ctr:.3f}, Position: {gsc.position:.1f}"
                )
            context_parts.append(
                f"GSC Matching Queries:\n" + "\n".join(gsc_context)
            )
        else:
            context_parts.append("GSC Data: No matching queries found")

        if ga_match:
            context_parts.append(
                f"GA Data: Sessions: {ga_match.sessions}, "
                f"Organic: {ga_match.organic_sessions}, "
                f"Pageviews: {ga_match.pageviews}, "
                f"Bounce Rate: {ga_match.bounce_rate:.2f}"
            )
        else:
            context_parts.append("GA Data: No matching page data found")

        user_prompt = (
            "Analyze the correlation between this search prompt and the available "
            "search/analytics data. Determine the impact potential.\n\n"
            + "\n".join(context_parts)
        )

        try:
            return self.llm.analyze_json(
                system_prompt=CORRELATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception as e:
            logger.error(f"LLM correlation analysis failed: {e}")
            return {}

    def correlate_prompt(
        self,
        prompt: ExtractedPrompt,
        validation: Optional[ValidationResult],
        factcheck: Optional[FactcheckResult],
        gsc_data: list[GSCData],
        ga_data: dict[str, GAData],
        use_llm: bool = True,
    ) -> CorrelatedPrompt:
        """
        Correlate a single prompt with all available data.

        Args:
            prompt: The extracted prompt.
            validation: Validation result (if available).
            factcheck: Factcheck result (if available).
            gsc_data: All GSC query data.
            ga_data: All GA page data (keyed by page path).
            use_llm: Whether to use LLM for deeper analysis.

        Returns:
            CorrelatedPrompt with all scores and recommendations.
        """
        gsc_matches = self._find_matching_gsc_data(prompt, gsc_data)
        ga_match = self._find_matching_ga_data(prompt, ga_data, prompt.source_urls)

        data_correlation = self._calculate_data_correlation_score(gsc_matches, ga_match)
        impact = self._calculate_impact_score(
            prompt, validation, factcheck, gsc_matches, ga_match
        )

        correlated = CorrelatedPrompt(
            prompt=prompt,
            validation=validation,
            factcheck=factcheck,
            gsc_data=gsc_matches[0][0] if gsc_matches else None,
            ga_data=ga_match,
            data_correlation_score=data_correlation,
            impact_score=impact,
        )

        # LLM-based deeper analysis for top candidates
        if use_llm and (data_correlation > 0.3 or impact > 0.5):
            llm_result = self._llm_correlation_analysis(
                prompt, validation, factcheck, gsc_matches, ga_match
            )
            if llm_result:
                # Blend model scores with the metric-based scores.
                if "data_correlation_score" in llm_result:
                    correlated.data_correlation_score = (
                        data_correlation * 0.4
                        + float(llm_result["data_correlation_score"]) * 0.6
                    )
                if "impact_score" in llm_result:
                    correlated.impact_score = (
                        impact * 0.4 + float(llm_result["impact_score"]) * 0.6
                    )
                correlated.recommendation = llm_result.get("recommendation", "")
                correlated.correlation_reasoning = llm_result.get("reasoning", "")
        else:
            if gsc_matches and gsc_matches[0][0].ctr < 0.05:
                correlated.recommendation = (
                    f"Optimize meta title/description for '{prompt.prompt_text}' - "
                    f"high impressions but low CTR"
                )
            elif not gsc_matches and prompt.estimated_volume in ("high", "medium"):
                correlated.recommendation = (
                    f"Create/optimize content targeting '{prompt.prompt_text}' - "
                    f"potential content gap"
                )
            else:
                correlated.recommendation = (
                    f"Monitor performance for '{prompt.prompt_text}'"
                )

        return correlated

    def correlate_all(
        self,
        prompts: list[ExtractedPrompt],
        validations: list[Optional[ValidationResult]],
        factchecks: list[Optional[FactcheckResult]],
        gsc_data: list[GSCData],
        ga_data: dict[str, GAData],
        use_llm: bool = True,
        progress_callback=None,
    ) -> list[CorrelatedPrompt]:
        """
        Correlate all prompts with search and analytics data.

        Args:
            prompts: List of extracted prompts.
            validations: Corresponding validation results.
            factchecks: Corresponding factcheck results.
            gsc_data: All GSC query data.
            ga_data: All GA page data.
            use_llm: Whether to use LLM for deeper analysis.
            progress_callback: Optional callback(current, total).

        Returns:
            List of CorrelatedPrompt objects, ranked by impact.
        """
        total = len(prompts)
        logger.info(f"Starting correlation analysis for {total} prompts")

        correlated_prompts = []

        for i, prompt in enumerate(prompts):
            validation = validations[i] if i < len(validations) else None
            factcheck = factchecks[i] if i < len(factchecks) else None

            logger.info(
                f"Correlating [{i + 1}/{total}]: {prompt.prompt_text[:60]}..."
            )

            correlated = self.correlate_prompt(
                prompt=prompt,
                validation=validation,
                factcheck=factcheck,
                gsc_data=gsc_data,
                ga_data=ga_data,
                use_llm=use_llm,
            )
            correlated_prompts.append(correlated)

            if progress_callback:
                progress_callback(i + 1, total)

        correlated_prompts.sort(key=lambda x: x.impact_score, reverse=True)

        for rank, cp in enumerate(correlated_prompts, 1):
            cp.priority_rank = rank

        logger.info("Top 10 correlated prompts by impact:")
        for cp in correlated_prompts[:10]:
            logger.info(
                f"  #{cp.priority_rank}: {cp.prompt.prompt_text[:50]} "
                f"(impact: {cp.impact_score:.2f}, "
                f"correlation: {cp.data_correlation_score:.2f})"
            )

        return correlated_prompts

    def filter_high_impact(
        self,
        correlated: list[CorrelatedPrompt],
        min_impact: float = 0.5,
        min_correlation: float = 0.3,
    ) -> list[CorrelatedPrompt]:
        """Keep prompts meeting either the impact or data correlation threshold."""
        filtered = [
            cp
            for cp in correlated
            if cp.impact_score >= min_impact
            or cp.data_correlation_score >= min_correlation
        ]

        logger.info(
            f"Filtered to {len(filtered)} high-impact prompts "
            f"(from {len(correlated)} total)"
        )

        return filtered
