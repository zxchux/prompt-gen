"""
Competitor Analysis Module for PromptGen.

Auto-discovers competitors based on the target website's content,
crawls competitor sites, and extracts the prompts they rank for.
These competitor prompts are merged into the main prompt list to
identify opportunities the target site is missing.

Discovery methods:
1. LLM analysis of the target site's content to identify competitors
2. Extraction of competitor domains mentioned/linked on the target site
3. Optional manual competitor list via CLI

For each competitor:
1. Crawl their site (limited pages)
2. Extract prompts they likely rank for
3. Tag prompts as "competitor_opportunity" for prioritization
"""

import logging
from typing import Optional
from urllib.parse import urlparse

from prompt_gen.config import CrawlerConfig, PromptExtractionConfig
from prompt_gen.crawler import WebCrawler
from prompt_gen.llm_client import OpenRouterClient
from prompt_gen.models import CrawledPage, ExtractedPrompt

logger = logging.getLogger(__name__)

DISCOVER_COMPETITORS_PROMPT = """You are an expert competitive intelligence analyst. Based on the website content provided, identify the top competitors for this website.

Website URL: {target_url}
Website Domain: {target_domain}

Website Content Summary:
{site_summary}

Identify 5-10 direct competitors. For each competitor, provide:
1. Their domain name (e.g., competitor.com)
2. Why they are a competitor (what overlapping market/audience)
3. How strong a competitor they are (direct, indirect, or aspirational)

Rules:
- Only include REAL websites that actually exist
- Focus on direct competitors in the same niche/market
- Include both well-known and emerging competitors
- Do NOT include generic platforms (google.com, youtube.com, wikipedia.org, etc.)
- Do NOT include social media platforms
- Do NOT include the target website itself

Respond with valid JSON:
{{
  "target_analysis": {{
    "industry": "the industry/niche",
    "primary_offering": "what the site offers",
    "target_audience": "who the site serves"
  }},
  "competitors": [
    {{
      "domain": "competitor.com",
      "name": "Competitor Name",
      "reason": "why they compete",
      "strength": "direct" | "indirect" | "aspirational",
      "overlap_areas": ["area1", "area2"]
    }}
  ]
}}"""

EXTRACT_COMPETITOR_PROMPTS_PROMPT = """You are an expert SEO analyst. Analyze this competitor website's content and identify the search prompts/queries they are likely ranking for.

Competitor: {competitor_domain}
Competitor Content:
{competitor_content}

Target Website: {target_url} (the site we're analyzing FOR)
Target Industry: {industry}

Identify search queries that:
1. This competitor page would rank for
2. Are ALSO relevant to the target website ({target_url})
3. Represent opportunities the target site could compete for

For each prompt, indicate:
- Whether the target site currently covers this topic (likely/unlikely/unknown)
- The competitive gap (how much opportunity there is)

Respond with valid JSON:
{{
  "prompts": [
    {{
      "prompt_text": "the search query",
      "relevance_score": 0.0-1.0,
      "search_intent": "informational" | "navigational" | "transactional" | "commercial",
      "topic_cluster": "topic category",
      "estimated_volume": "high" | "medium" | "low",
      "difficulty": "easy" | "medium" | "hard",
      "target_coverage": "likely" | "unlikely" | "unknown",
      "competitive_gap": "high" | "medium" | "low",
      "reasoning": "why this is an opportunity"
    }}
  ]
}}"""


class CompetitorAnalyzer:
    """
    Discovers competitors and extracts their ranking prompts.

    Workflow:
    1. Analyze target site content to identify competitors
    2. Crawl each competitor (limited scope)
    3. Extract prompts competitors rank for
    4. Filter to prompts relevant to the target site
    5. Tag as competitor opportunities
    """

    def __init__(
        self,
        crawler_config: CrawlerConfig,
        extraction_config: PromptExtractionConfig,
        llm_client: OpenRouterClient,
    ):
        self.crawler_config = crawler_config
        self.extraction_config = extraction_config
        self.llm = llm_client

    def _build_site_summary(self, pages: list[CrawledPage]) -> str:
        """Build a summary of the target site from crawled pages."""
        summaries = []
        for page in pages[:20]:
            parts = []
            if page.title:
                parts.append(f"Title: {page.title}")
            if page.meta_description:
                parts.append(f"Description: {page.meta_description}")
            if page.h1_tags:
                parts.append(f"H1: {', '.join(page.h1_tags[:3])}")
            body_preview = page.body_text[:200] if page.body_text else ""
            if body_preview:
                parts.append(f"Content: {body_preview}")
            if parts:
                summaries.append(f"Page: {page.url}\n" + "\n".join(parts))

        return "\n\n".join(summaries[:15])

    def _extract_linked_competitors(
        self, pages: list[CrawledPage], target_domain: str
    ) -> list[str]:
        """Extract potential competitor domains from external links."""
        domain_counts: dict[str, int] = {}
        target_base = target_domain.lower().removeprefix("www.")

        skip_domains = {
            "google.com", "youtube.com", "facebook.com", "twitter.com",
            "instagram.com", "linkedin.com", "github.com", "wikipedia.org",
            "amazon.com", "apple.com", "microsoft.com", "w3.org",
            "schema.org", "fonts.googleapis.com", "cdn.jsdelivr.net",
            "cloudflare.com", "googleapis.com", "gstatic.com",
            "pinterest.com", "reddit.com", "tiktok.com", "x.com",
        }

        for page in pages:
            for link in page.external_links:
                try:
                    parsed = urlparse(link)
                    domain = parsed.netloc.lower().removeprefix("www.")
                    if (
                        domain
                        and domain != target_base
                        and domain not in skip_domains
                        and not any(skip in domain for skip in skip_domains)
                    ):
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
                except Exception:
                    pass

        # Sort by frequency and return top domains
        sorted_domains = sorted(
            domain_counts.items(), key=lambda x: x[1], reverse=True
        )
        return [d for d, _ in sorted_domains[:10]]

    def discover_competitors(
        self,
        target_url: str,
        pages: list[CrawledPage],
        manual_competitors: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Discover competitors for the target website.

        Uses three methods:
        1. Manual list (if provided)
        2. LLM analysis of site content
        3. External link analysis

        Args:
            target_url: The target website URL.
            pages: Crawled pages from the target site.
            manual_competitors: Optional list of competitor domains.

        Returns:
            List of competitor dicts with domain, name, reason, strength.
        """
        parsed = urlparse(target_url)
        target_domain = parsed.netloc

        all_competitors: dict[str, dict] = {}

        # Method 1: Manual competitors
        if manual_competitors:
            for domain in manual_competitors:
                domain = domain.strip().lower().removeprefix("https://").removeprefix("http://").removeprefix("www.").rstrip("/")
                all_competitors[domain] = {
                    "domain": domain,
                    "name": domain,
                    "reason": "Manually specified",
                    "strength": "direct",
                    "overlap_areas": [],
                    "source": "manual",
                }
            logger.info(f"Added {len(manual_competitors)} manual competitors")

        # Method 2: LLM discovery
        logger.info("Discovering competitors via LLM analysis...")
        site_summary = self._build_site_summary(pages)

        try:
            result = self.llm.analyze_json(
                system_prompt="You are a competitive intelligence analyst. Respond only with valid JSON.",
                user_prompt=DISCOVER_COMPETITORS_PROMPT.format(
                    target_url=target_url,
                    target_domain=target_domain,
                    site_summary=site_summary,
                ),
                temperature=0.4,
                max_tokens=2048,
            )

            industry = result.get("target_analysis", {}).get("industry", "")
            llm_competitors = result.get("competitors", [])

            for comp in llm_competitors:
                domain = comp.get("domain", "").lower().removeprefix("www.").strip()
                if domain and domain not in all_competitors:
                    all_competitors[domain] = {
                        "domain": domain,
                        "name": comp.get("name", domain),
                        "reason": comp.get("reason", ""),
                        "strength": comp.get("strength", "indirect"),
                        "overlap_areas": comp.get("overlap_areas", []),
                        "source": "llm_discovery",
                        "industry": industry,
                    }

            logger.info(
                f"LLM discovered {len(llm_competitors)} competitors "
                f"(industry: {industry})"
            )

        except Exception as e:
            logger.error(f"LLM competitor discovery failed: {e}")
            industry = ""

        # Method 3: External link analysis
        linked_domains = self._extract_linked_competitors(pages, target_domain)
        for domain in linked_domains:
            if domain not in all_competitors:
                all_competitors[domain] = {
                    "domain": domain,
                    "name": domain,
                    "reason": "Found in external links from target site",
                    "strength": "indirect",
                    "overlap_areas": [],
                    "source": "link_analysis",
                }

        if linked_domains:
            logger.info(
                f"Link analysis found {len(linked_domains)} potential competitor domains"
            )

        competitors = list(all_competitors.values())

        # Sort: manual first, then direct, then indirect
        strength_order = {"direct": 0, "indirect": 1, "aspirational": 2}
        source_order = {"manual": 0, "llm_discovery": 1, "link_analysis": 2}
        competitors.sort(
            key=lambda c: (
                source_order.get(c.get("source", ""), 3),
                strength_order.get(c.get("strength", ""), 3),
            )
        )

        logger.info(f"Total competitors identified: {len(competitors)}")
        for comp in competitors:
            logger.info(
                f"  • {comp['domain']} ({comp['strength']}) - {comp['reason'][:60]}"
            )

        return competitors

    def crawl_competitor(
        self,
        competitor_domain: str,
        max_pages: int = 20,
    ) -> list[CrawledPage]:
        """
        Crawl a competitor website with limited scope.

        Args:
            competitor_domain: The competitor's domain.
            max_pages: Maximum pages to crawl per competitor.

        Returns:
            List of crawled pages from the competitor.
        """
        # Create a separate crawler config for competitors (lighter crawl)
        comp_config = CrawlerConfig(
            max_pages=max_pages,
            crawl_delay=self.crawler_config.crawl_delay,
            max_concurrent_requests=self.crawler_config.max_concurrent_requests,
            user_agent=self.crawler_config.user_agent,
            request_timeout=self.crawler_config.request_timeout,
            respect_robots_txt=True,
            max_depth=3,  # Shallower crawl for competitors
        )

        crawler = WebCrawler(comp_config)
        competitor_url = f"https://{competitor_domain}"

        logger.info(f"Crawling competitor: {competitor_domain} (max {max_pages} pages)")

        try:
            pages = crawler.crawl(competitor_url)
            logger.info(
                f"Crawled {len(pages)} pages from {competitor_domain}"
            )
            return pages
        except Exception as e:
            logger.error(f"Failed to crawl competitor {competitor_domain}: {e}")
            return []

    def extract_competitor_prompts(
        self,
        competitor: dict,
        competitor_pages: list[CrawledPage],
        target_url: str,
        industry: str = "",
    ) -> list[ExtractedPrompt]:
        """
        Extract prompts a competitor ranks for that are relevant to the target.

        Args:
            competitor: Competitor info dict.
            competitor_pages: Crawled pages from the competitor.
            target_url: The target website URL.
            industry: The industry/niche.

        Returns:
            List of ExtractedPrompt objects tagged as competitor opportunities.
        """
        if not competitor_pages:
            return []

        competitor_domain = competitor["domain"]
        all_prompts: list[ExtractedPrompt] = []

        # Process competitor pages in batches
        batch_size = self.extraction_config.batch_size
        for i in range(0, len(competitor_pages), batch_size):
            batch = competitor_pages[i : i + batch_size]

            # Build content for the batch
            content_parts = []
            for page in batch:
                parts = [f"URL: {page.url}"]
                if page.title:
                    parts.append(f"Title: {page.title}")
                if page.meta_description:
                    parts.append(f"Meta: {page.meta_description}")
                if page.h1_tags:
                    parts.append(f"H1: {', '.join(page.h1_tags[:3])}")
                if page.h2_tags:
                    parts.append(f"H2: {', '.join(page.h2_tags[:5])}")
                body = page.body_text[:1500] if page.body_text else ""
                if body:
                    parts.append(f"Content: {body}")
                content_parts.append("\n".join(parts))

            competitor_content = "\n\n---\n\n".join(content_parts)

            try:
                result = self.llm.analyze_json(
                    system_prompt="You are an expert SEO analyst specializing in competitive analysis. Respond only with valid JSON.",
                    user_prompt=EXTRACT_COMPETITOR_PROMPTS_PROMPT.format(
                        competitor_domain=competitor_domain,
                        competitor_content=competitor_content,
                        target_url=target_url,
                        industry=industry or "general",
                    ),
                    temperature=0.4,
                    max_tokens=4096,
                )

                for item in result.get("prompts", []):
                    prompt = ExtractedPrompt(
                        prompt_text=item.get("prompt_text", "").strip(),
                        source_urls=[p.url for p in batch],
                        relevance_score=float(item.get("relevance_score", 0.5)),
                        search_intent=item.get("search_intent", "informational"),
                        topic_cluster=item.get("topic_cluster", ""),
                        estimated_volume=item.get("estimated_volume", "medium"),
                        difficulty=item.get("difficulty", "medium"),
                        extraction_reasoning=(
                            f"Competitor opportunity from {competitor_domain}: "
                            f"{item.get('reasoning', '')} "
                            f"[gap: {item.get('competitive_gap', 'unknown')}, "
                            f"target coverage: {item.get('target_coverage', 'unknown')}]"
                        ),
                    )
                    if prompt.prompt_text:
                        all_prompts.append(prompt)

            except Exception as e:
                logger.error(
                    f"Error extracting prompts from {competitor_domain} batch: {e}"
                )

        logger.info(
            f"Extracted {len(all_prompts)} prompts from competitor {competitor_domain}"
        )
        return all_prompts

    def analyze_competitors(
        self,
        target_url: str,
        target_pages: list[CrawledPage],
        manual_competitors: Optional[list[str]] = None,
        max_competitors: int = 5,
        max_pages_per_competitor: int = 15,
        progress_callback=None,
    ) -> tuple[list[dict], list[ExtractedPrompt]]:
        """
        Run the full competitor analysis pipeline.

        1. Discover competitors
        2. Crawl each competitor
        3. Extract their ranking prompts
        4. Return competitor info and prompts

        Args:
            target_url: The target website URL.
            target_pages: Crawled pages from the target site.
            manual_competitors: Optional list of competitor domains.
            max_competitors: Maximum number of competitors to analyze.
            max_pages_per_competitor: Max pages to crawl per competitor.
            progress_callback: Optional callback(current, total, competitor_name).

        Returns:
            Tuple of (competitor_info_list, all_competitor_prompts).
        """
        logger.info("=== COMPETITOR ANALYSIS ===")

        # Step 1: Discover competitors
        competitors = self.discover_competitors(
            target_url, target_pages, manual_competitors
        )

        # Limit to max_competitors
        competitors = competitors[:max_competitors]
        if not competitors:
            logger.warning("No competitors identified")
            return [], []

        logger.info(f"Analyzing top {len(competitors)} competitors")

        # Get industry from first competitor's data
        industry = ""
        for comp in competitors:
            if comp.get("industry"):
                industry = comp["industry"]
                break

        # Step 2 & 3: Crawl and extract for each competitor
        all_competitor_prompts: list[ExtractedPrompt] = []

        for i, competitor in enumerate(competitors):
            domain = competitor["domain"]
            logger.info(
                f"Analyzing competitor [{i + 1}/{len(competitors)}]: {domain}"
            )

            if progress_callback:
                progress_callback(i + 1, len(competitors), domain)

            # Crawl competitor
            comp_pages = self.crawl_competitor(
                domain, max_pages=max_pages_per_competitor
            )

            if not comp_pages:
                logger.warning(f"No pages crawled from {domain}, skipping")
                competitor["pages_crawled"] = 0
                competitor["prompts_found"] = 0
                continue

            competitor["pages_crawled"] = len(comp_pages)

            # Extract prompts
            comp_prompts = self.extract_competitor_prompts(
                competitor, comp_pages, target_url, industry
            )

            competitor["prompts_found"] = len(comp_prompts)
            all_competitor_prompts.extend(comp_prompts)

            logger.info(
                f"  → {len(comp_pages)} pages crawled, "
                f"{len(comp_prompts)} prompts extracted from {domain}"
            )

        # Deduplicate competitor prompts
        seen: dict[str, ExtractedPrompt] = {}
        for prompt in all_competitor_prompts:
            key = prompt.prompt_text.lower().strip()
            if key in seen:
                existing = seen[key]
                existing.source_urls = list(
                    set(existing.source_urls + prompt.source_urls)
                )
                existing.relevance_score = max(
                    existing.relevance_score, prompt.relevance_score
                )
            else:
                seen[key] = prompt

        deduped_prompts = list(seen.values())

        logger.info(
            f"Competitor analysis complete: "
            f"{len(competitors)} competitors analyzed, "
            f"{len(deduped_prompts)} unique prompts extracted "
            f"(from {len(all_competitor_prompts)} total)"
        )

        return competitors, deduped_prompts
