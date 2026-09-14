"""Write analysis results as JSON, CSV, and text reports."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from prompt_gen.models import CorrelatedPrompt, CrawledPage

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates output reports from the analysis pipeline."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_timestamp(self) -> str:
        """Generate a timestamp string for filenames."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_json_report(
        self,
        correlated_prompts: list[CorrelatedPrompt],
        pages: list[CrawledPage],
        target_url: str,
        llm_stats: dict[str, Any],
    ) -> str:
        """Write the full JSON report and return its path."""
        timestamp = self._generate_timestamp()

        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "target_url": target_url,
                "total_pages_crawled": len(pages),
                "total_prompts_analyzed": len(correlated_prompts),
                "llm_usage": llm_stats,
            },
            "summary": {
                "total_prompts": len(correlated_prompts),
                "high_impact_prompts": sum(
                    1 for cp in correlated_prompts if cp.impact_score >= 0.7
                ),
                "medium_impact_prompts": sum(
                    1
                    for cp in correlated_prompts
                    if 0.4 <= cp.impact_score < 0.7
                ),
                "low_impact_prompts": sum(
                    1 for cp in correlated_prompts if cp.impact_score < 0.4
                ),
                "with_gsc_data": sum(
                    1 for cp in correlated_prompts if cp.gsc_data is not None
                ),
                "with_ga_data": sum(
                    1 for cp in correlated_prompts if cp.ga_data is not None
                ),
                "avg_impact_score": (
                    sum(cp.impact_score for cp in correlated_prompts)
                    / len(correlated_prompts)
                    if correlated_prompts
                    else 0
                ),
                "avg_correlation_score": (
                    sum(cp.data_correlation_score for cp in correlated_prompts)
                    / len(correlated_prompts)
                    if correlated_prompts
                    else 0
                ),
            },
            "top_prompts": [
                cp.to_dict() for cp in correlated_prompts[:50]
            ],
            "all_prompts": [cp.to_dict() for cp in correlated_prompts],
            "topic_clusters": self._group_by_topic(correlated_prompts),
            "intent_distribution": self._intent_distribution(correlated_prompts),
            "recommendations": self._generate_recommendations(correlated_prompts),
        }

        filepath = self.output_dir / f"prompt_report_{timestamp}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"JSON report saved to {filepath}")
        return str(filepath)

    def generate_summary_report(
        self,
        correlated_prompts: list[CorrelatedPrompt],
        pages: list[CrawledPage],
        target_url: str,
    ) -> str:
        """Write a text summary and return its path."""
        timestamp = self._generate_timestamp()
        lines = []

        lines.append("=" * 80)
        lines.append("PROMPTGEN - SEARCH PROMPT DISCOVERY REPORT")
        lines.append("=" * 80)
        lines.append(f"Target: {target_url}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Pages Crawled: {len(pages)}")
        lines.append(f"Prompts Analyzed: {len(correlated_prompts)}")
        lines.append("")

        lines.append("-" * 80)
        lines.append("SUMMARY")
        lines.append("-" * 80)
        high = sum(1 for cp in correlated_prompts if cp.impact_score >= 0.7)
        medium = sum(
            1 for cp in correlated_prompts if 0.4 <= cp.impact_score < 0.7
        )
        low = sum(1 for cp in correlated_prompts if cp.impact_score < 0.4)
        lines.append(f"High Impact Prompts:   {high}")
        lines.append(f"Medium Impact Prompts: {medium}")
        lines.append(f"Low Impact Prompts:    {low}")
        lines.append(
            f"With GSC Data:         "
            f"{sum(1 for cp in correlated_prompts if cp.gsc_data)}"
        )
        lines.append(
            f"With GA Data:          "
            f"{sum(1 for cp in correlated_prompts if cp.ga_data)}"
        )
        lines.append("")

        lines.append("-" * 80)
        lines.append("TOP 50 HIGH-IMPACT PROMPTS")
        lines.append("-" * 80)
        lines.append(
            f"{'#':<4} {'Prompt':<50} {'Impact':<8} {'Corr.':<8} {'Intent':<15}"
        )
        lines.append("-" * 85)

        for cp in correlated_prompts[:50]:
            prompt_text = cp.prompt.prompt_text[:48]
            lines.append(
                f"{cp.priority_rank:<4} {prompt_text:<50} "
                f"{cp.impact_score:<8.2f} {cp.data_correlation_score:<8.2f} "
                f"{cp.prompt.search_intent:<15}"
            )

        lines.append("")

        lines.append("-" * 80)
        lines.append("TOPIC CLUSTERS")
        lines.append("-" * 80)
        clusters = self._group_by_topic(correlated_prompts)
        for cluster_name, cluster_data in sorted(
            clusters.items(), key=lambda x: x[1]["count"], reverse=True
        )[:20]:
            lines.append(
                f"  {cluster_name}: {cluster_data['count']} prompts "
                f"(avg impact: {cluster_data['avg_impact']:.2f})"
            )

        lines.append("")

        lines.append("-" * 80)
        lines.append("TOP RECOMMENDATIONS")
        lines.append("-" * 80)
        for i, cp in enumerate(correlated_prompts[:20], 1):
            if cp.recommendation:
                lines.append(f"  {i}. {cp.recommendation}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        report_text = "\n".join(lines)

        filepath = self.output_dir / f"prompt_summary_{timestamp}.txt"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info(f"Summary report saved to {filepath}")
        return str(filepath)

    def generate_csv_report(
        self, correlated_prompts: list[CorrelatedPrompt]
    ) -> str:
        """Write prompt scores and metrics to CSV and return its path."""
        import csv

        timestamp = self._generate_timestamp()
        filepath = self.output_dir / f"prompts_{timestamp}.csv"

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "Rank",
                    "Prompt",
                    "Impact Score",
                    "Correlation Score",
                    "Search Intent",
                    "Topic Cluster",
                    "Est. Volume",
                    "Difficulty",
                    "Validation Score",
                    "Factcheck Score",
                    "GSC Clicks",
                    "GSC Impressions",
                    "GSC CTR",
                    "GSC Position",
                    "GA Sessions",
                    "GA Organic Sessions",
                    "GA Bounce Rate",
                    "Recommendation",
                    "Source URLs",
                ]
            )

            for cp in correlated_prompts:
                writer.writerow(
                    [
                        cp.priority_rank,
                        cp.prompt.prompt_text,
                        f"{cp.impact_score:.3f}",
                        f"{cp.data_correlation_score:.3f}",
                        cp.prompt.search_intent,
                        cp.prompt.topic_cluster,
                        cp.prompt.estimated_volume,
                        cp.prompt.difficulty,
                        f"{cp.validation.overall_score:.3f}"
                        if cp.validation
                        else "",
                        f"{cp.factcheck.overall_factual_score:.3f}"
                        if cp.factcheck
                        else "",
                        cp.gsc_data.clicks if cp.gsc_data else "",
                        cp.gsc_data.impressions if cp.gsc_data else "",
                        f"{cp.gsc_data.ctr:.4f}" if cp.gsc_data else "",
                        f"{cp.gsc_data.position:.1f}" if cp.gsc_data else "",
                        cp.ga_data.sessions if cp.ga_data else "",
                        cp.ga_data.organic_sessions if cp.ga_data else "",
                        f"{cp.ga_data.bounce_rate:.2f}" if cp.ga_data else "",
                        cp.recommendation,
                        "; ".join(cp.prompt.source_urls[:3]),
                    ]
                )

        logger.info(f"CSV report saved to {filepath}")
        return str(filepath)

    def _group_by_topic(
        self, correlated_prompts: list[CorrelatedPrompt]
    ) -> dict[str, Any]:
        """Group prompts by topic cluster."""
        clusters: dict[str, list[CorrelatedPrompt]] = {}
        for cp in correlated_prompts:
            topic = cp.prompt.topic_cluster or "Uncategorized"
            if topic not in clusters:
                clusters[topic] = []
            clusters[topic].append(cp)

        result = {}
        for topic, prompts in clusters.items():
            result[topic] = {
                "count": len(prompts),
                "avg_impact": sum(p.impact_score for p in prompts) / len(prompts),
                "avg_correlation": sum(p.data_correlation_score for p in prompts)
                / len(prompts),
                "top_prompts": [
                    p.prompt.prompt_text
                    for p in sorted(
                        prompts, key=lambda x: x.impact_score, reverse=True
                    )[:5]
                ],
            }

        return result

    def _intent_distribution(
        self, correlated_prompts: list[CorrelatedPrompt]
    ) -> dict[str, int]:
        """Calculate search intent distribution."""
        distribution: dict[str, int] = {}
        for cp in correlated_prompts:
            intent = cp.prompt.search_intent or "unknown"
            distribution[intent] = distribution.get(intent, 0) + 1
        return distribution

    def _generate_recommendations(
        self, correlated_prompts: list[CorrelatedPrompt]
    ) -> list[dict[str, Any]]:
        """Generate prioritized recommendations."""
        recommendations = []

        for cp in correlated_prompts[:30]:
            if cp.recommendation:
                recommendations.append(
                    {
                        "priority": cp.priority_rank,
                        "prompt": cp.prompt.prompt_text,
                        "impact_score": cp.impact_score,
                        "recommendation": cp.recommendation,
                        "reasoning": cp.correlation_reasoning[:200]
                        if cp.correlation_reasoning
                        else "",
                    }
                )

        return recommendations
