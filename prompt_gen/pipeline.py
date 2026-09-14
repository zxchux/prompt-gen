"""Run crawling, extraction, validation, factchecking, correlation, and reporting."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from prompt_gen.agentic_validator import AgenticValidator
from prompt_gen.config import Settings
from prompt_gen.correlation_engine import CorrelationEngine
from prompt_gen.crawler import WebCrawler
from prompt_gen.csv_importer import CSVImporter
from prompt_gen.factcheck import Factchecker
from prompt_gen.google_analytics import AnalyticsClient
from prompt_gen.google_search_console import SearchConsoleClient
from prompt_gen.llm_client import OpenRouterClient
from prompt_gen.models import (
    CorrelatedPrompt,
    CrawledPage,
    ExtractedPrompt,
    FactcheckResult,
    PromptStatus,
    ValidationResult,
)
from prompt_gen.prompt_extractor import PromptExtractor
from prompt_gen.reporter import ReportGenerator

logger = logging.getLogger(__name__)


class Pipeline:
    """Run analysis stages and save intermediate results to JSON cache files."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = OpenRouterClient(settings.openrouter)
        self.crawler = WebCrawler(settings.crawler)
        self.extractor = PromptExtractor(settings.prompt_extraction, self.llm)
        self.validator = AgenticValidator(settings.prompt_extraction, self.llm)
        self.factchecker = Factchecker(settings.factcheck, self.llm)
        self.gsc_client = SearchConsoleClient(settings.google)
        self.ga_client = AnalyticsClient(settings.google)
        self.correlation = CorrelationEngine(settings, self.llm)
        self.reporter = ReportGenerator(settings.output_dir)

        self.pages: list[CrawledPage] = []
        self.prompts: list[ExtractedPrompt] = []
        self.validations: list[ValidationResult] = []
        self.factchecks: list[FactcheckResult] = []
        self.correlated: list[CorrelatedPrompt] = []

        self._cache_dir = Path(settings.cache.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _save_cache(self, stage: str, data: Any) -> None:
        """Save intermediate results to cache."""
        if not self.settings.cache.enable_cache:
            return

        cache_file = self._cache_dir / f"{stage}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Cached {stage} results to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache {stage}: {e}")

    def _load_cache(self, stage: str) -> Optional[Any]:
        """Load cached intermediate results."""
        if not self.settings.cache.enable_cache:
            return None

        cache_file = self._cache_dir / f"{stage}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded cached {stage} results from {cache_file}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache for {stage}: {e}")

        return None

    def stage_crawl(self, target_url: str, progress_callback=None) -> list[CrawledPage]:
        """Crawl the target URL and store the pages.

        The callback receives (pages_crawled, total_found).
        """
        logger.info(f"=== STAGE 1: CRAWLING {target_url} ===")

        self.pages = self.crawler.crawl(target_url, progress_callback)

        self._save_cache(
            "crawl", [p.to_dict() for p in self.pages]
        )

        logger.info(f"Crawl complete: {len(self.pages)} pages")
        return self.pages

    def stage_extract(self, progress_callback=None) -> list[ExtractedPrompt]:
        """Extract and store prompts from the crawled pages."""
        if not self.pages:
            raise ValueError("No pages available. Run crawl stage first.")

        logger.info(f"=== STAGE 2: EXTRACTING PROMPTS FROM {len(self.pages)} PAGES ===")

        self.prompts = self.extractor.extract_prompts(self.pages, progress_callback)

        self._save_cache(
            "extract", [p.to_dict() for p in self.prompts]
        )

        logger.info(f"Extraction complete: {len(self.prompts)} prompts")
        return self.prompts

    def stage_validate(self, progress_callback=None) -> list[ValidationResult]:
        """Validate extracted prompts and update their statuses."""
        if not self.prompts:
            raise ValueError("No prompts available. Run extract stage first.")

        logger.info(
            f"=== STAGE 3: VALIDATING {len(self.prompts)} PROMPTS ==="
        )

        self.validations = self.validator.validate_prompts(
            self.prompts, self.pages, progress_callback
        )

        # Filter to validated prompts
        validated_pairs = self.validator.filter_validated(
            self.prompts, self.validations
        )

        validated_texts = {p.prompt_text for p, _ in validated_pairs}
        for prompt in self.prompts:
            if prompt.prompt_text in validated_texts:
                prompt.status = PromptStatus.VALIDATED
            else:
                prompt.status = PromptStatus.REJECTED

        self._save_cache(
            "validate",
            [v.to_dict() for v in self.validations],
        )

        passed = sum(
            1
            for v in self.validations
            if v.overall_score >= self.settings.prompt_extraction.min_quality_score
        )
        logger.info(f"Validation complete: {passed}/{len(self.validations)} passed")
        return self.validations

    def stage_factcheck(self, progress_callback=None) -> list[FactcheckResult]:
        """Factcheck prompts whose status is VALIDATED."""
        # Only factcheck validated prompts
        validated_prompts = [
            p for p in self.prompts if p.status == PromptStatus.VALIDATED
        ]

        if not validated_prompts:
            logger.warning("No validated prompts to factcheck")
            self.factchecks = []
            return self.factchecks

        logger.info(
            f"=== STAGE 4: FACTCHECKING {len(validated_prompts)} VALIDATED PROMPTS ==="
        )

        self.factchecks = self.factchecker.factcheck_prompts(
            validated_prompts, self.pages, progress_callback
        )

        for prompt, fc in zip(validated_prompts, self.factchecks):
            if fc.is_factually_sound:
                prompt.status = PromptStatus.FACTCHECKED

        self._save_cache(
            "factcheck",
            [f.to_dict() for f in self.factchecks],
        )

        passed = sum(1 for f in self.factchecks if f.is_factually_sound)
        logger.info(
            f"Factcheck complete: {passed}/{len(self.factchecks)} passed"
        )
        return self.factchecks

    def stage_correlate(
        self,
        target_url: str,
        skip_google_apis: bool = False,
        gsc_csv_path: Optional[str] = None,
        ga_csv_path: Optional[str] = None,
        progress_callback=None,
    ) -> list[CorrelatedPrompt]:
        """Match prompts to analytics data and rank them by impact.

        CSV input takes precedence for each service. Otherwise, fetch from its API
        unless skip_google_apis is set. Continue without data when a source is unavailable.
        """
        # Use factchecked prompts, or validated if no factcheck was run
        active_prompts = [
            p
            for p in self.prompts
            if p.status in (PromptStatus.FACTCHECKED, PromptStatus.VALIDATED)
        ]

        if not active_prompts:
            logger.warning("No validated/factchecked prompts to correlate")
            active_prompts = self.prompts  # Fall back to all prompts

        logger.info(
            f"=== STAGE 5: CORRELATING {len(active_prompts)} PROMPTS WITH DATA ==="
        )

        # Fetch Google data from CSV or API
        gsc_data = []
        ga_data = {}
        csv_importer = CSVImporter()

        # Priority 1: CSV imports
        if gsc_csv_path:
            gsc_data = csv_importer.import_gsc_queries(gsc_csv_path)
            logger.info(f"Loaded {len(gsc_data)} GSC queries from CSV: {gsc_csv_path}")

        if ga_csv_path:
            ga_data = csv_importer.import_ga_pages(ga_csv_path)
            logger.info(f"Loaded GA data for {len(ga_data)} pages from CSV: {ga_csv_path}")

        # Priority 2: Google APIs (only if no CSV provided and not skipped)
        if not gsc_csv_path and not skip_google_apis:
            if self.gsc_client.authenticate():
                from urllib.parse import urlparse

                domain = urlparse(target_url).netloc
                site_url = self.gsc_client.get_site_url(domain)
                if site_url:
                    gsc_data = self.gsc_client.fetch_query_data(site_url)
                    logger.info(f"Fetched {len(gsc_data)} GSC queries via API")
            else:
                logger.warning(
                    "GSC authentication failed - proceeding without GSC data"
                )

        if not ga_csv_path and not skip_google_apis:
            if self.ga_client.authenticate():
                ga_data = self.ga_client.fetch_landing_page_data()
                logger.info(f"Fetched GA data for {len(ga_data)} pages via API")
            else:
                logger.warning(
                    "GA authentication failed - proceeding without GA data"
                )

        if skip_google_apis and not gsc_csv_path and not ga_csv_path:
            logger.info("No data sources provided - correlating without GSC/GA data")

        # Build validation/factcheck lookups
        validation_map = {v.prompt_text: v for v in self.validations}
        factcheck_map = {f.prompt_text: f for f in self.factchecks}

        validations = [
            validation_map.get(p.prompt_text) for p in active_prompts
        ]
        factchecks = [
            factcheck_map.get(p.prompt_text) for p in active_prompts
        ]

        self.correlated = self.correlation.correlate_all(
            prompts=active_prompts,
            validations=validations,
            factchecks=factchecks,
            gsc_data=gsc_data,
            ga_data=ga_data,
            use_llm=True,
            progress_callback=progress_callback,
        )

        self._save_cache(
            "correlate",
            [c.to_dict() for c in self.correlated],
        )

        logger.info(f"Correlation complete: {len(self.correlated)} prompts ranked")
        return self.correlated

    def stage_report(
        self,
        target_url: str,
    ) -> dict[str, str]:
        """Write JSON, text, and CSV reports and return their paths keyed by format."""
        if not self.correlated:
            raise ValueError("No correlated data. Run correlate stage first.")

        logger.info("=== STAGE 6: GENERATING REPORTS ===")

        reports = {}

        json_path = self.reporter.generate_json_report(
            correlated_prompts=self.correlated,
            pages=self.pages,
            target_url=target_url,
            llm_stats=self.llm.get_usage_stats(),
        )
        reports["json"] = json_path

        summary_path = self.reporter.generate_summary_report(
            correlated_prompts=self.correlated,
            pages=self.pages,
            target_url=target_url,
        )
        reports["summary"] = summary_path

        csv_path = self.reporter.generate_csv_report(self.correlated)
        reports["csv"] = csv_path

        logger.info(f"Reports generated: {reports}")
        return reports

    def run(
        self,
        target_url: str,
        skip_google_apis: bool = False,
        skip_validation: bool = False,
        skip_factcheck: bool = False,
        gsc_csv_path: Optional[str] = None,
        ga_csv_path: Optional[str] = None,
        progress_callback=None,
    ) -> dict[str, str]:
        """Run the pipeline and return report paths keyed by format.

        Validation, factchecking, and Google API calls can be skipped independently.
        CSV inputs take precedence over Google APIs. The progress_callback parameter
        is retained for compatibility but is not called; use stage callbacks for progress.
        """
        logger.info(f"Starting PromptGen pipeline for: {target_url}")
        logger.info(f"Options: skip_google={skip_google_apis}, "
                     f"skip_validation={skip_validation}, "
                     f"skip_factcheck={skip_factcheck}")
        if gsc_csv_path:
            logger.info(f"GSC data source: CSV file ({gsc_csv_path})")
        if ga_csv_path:
            logger.info(f"GA data source: CSV file ({ga_csv_path})")

        self.stage_crawl(target_url)

        if not self.pages:
            raise RuntimeError(f"No pages crawled from {target_url}")

        self.stage_extract()

        if not self.prompts:
            raise RuntimeError("No prompts extracted from crawled pages")

        if not skip_validation:
            self.stage_validate()
        else:
            logger.info("Skipping validation stage")
            for p in self.prompts:
                p.status = PromptStatus.VALIDATED

        if not skip_factcheck:
            self.stage_factcheck()
        else:
            logger.info("Skipping factcheck stage")
            for p in self.prompts:
                if p.status == PromptStatus.VALIDATED:
                    p.status = PromptStatus.FACTCHECKED

        self.stage_correlate(
            target_url,
            skip_google_apis=skip_google_apis,
            gsc_csv_path=gsc_csv_path,
            ga_csv_path=ga_csv_path,
        )

        reports = self.stage_report(target_url)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Pages crawled: {len(self.pages)}")
        logger.info(f"Prompts extracted: {len(self.prompts)}")
        if self.validations:
            min_score = self.settings.prompt_extraction.min_quality_score
            valid_count = sum(
                1 for v in self.validations
                if v.overall_score >= min_score
            )
            logger.info(f"Prompts validated: {valid_count}")
        else:
            logger.info("Prompts validated: skipped")
        if self.factchecks:
            fc_count = sum(
                1 for f in self.factchecks if f.is_factually_sound
            )
            logger.info(f"Prompts factchecked: {fc_count}")
        else:
            logger.info("Prompts factchecked: skipped")
        logger.info(f"Prompts correlated: {len(self.correlated)}")
        logger.info(f"LLM usage: {self.llm.get_usage_stats()}")
        logger.info(f"Reports: {reports}")
        logger.info("=" * 60)

        return reports
