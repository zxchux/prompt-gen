"""Suggest search queries from page content, merge duplicates, and rank by relevance."""

import json
import logging
from typing import Any

from prompt_gen.config import PromptExtractionConfig
from prompt_gen.llm_client import OpenRouterClient
from prompt_gen.models import CrawledPage, ExtractedPrompt

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_EXTRACT = """You are an expert SEO analyst and search intent specialist. Your task is to analyze website page content and identify the search queries/prompts that this page would rank for in search engines.

For each page, identify search queries that:
1. Match the page's topic, content, and intent
2. Represent real user search behavior (how people actually search)
3. Cover different search intents: informational, navigational, transactional, commercial
4. Include both head terms (short, high-volume) and long-tail queries
5. Consider question-based queries (who, what, when, where, why, how)
6. Include related/semantic variations

For each prompt, provide:
- prompt_text: The actual search query
- relevance_score: 0.0-1.0 how relevant this query is to the page content
- search_intent: informational, navigational, transactional, or commercial
- topic_cluster: The broad topic category
- estimated_volume: high, medium, or low
- difficulty: easy, medium, or hard (how competitive this query is)
- reasoning: Brief explanation of why this page would rank for this query

You MUST respond with valid JSON in this exact format:
{
  "prompts": [
    {
      "prompt_text": "example search query",
      "relevance_score": 0.85,
      "search_intent": "informational",
      "topic_cluster": "topic name",
      "estimated_volume": "medium",
      "difficulty": "medium",
      "reasoning": "This page covers..."
    }
  ]
}"""

SYSTEM_PROMPT_CONSOLIDATE = """You are an expert SEO analyst. You have been given a large list of search prompts/queries extracted from multiple pages of a website. Your task is to:

1. DEDUPLICATE: Remove near-duplicate queries (keep the best version)
2. RANK: Score each unique query by its overall value considering:
   - Relevance to the website's core topics
   - Search volume potential
   - Content match quality
   - Strategic importance for the website
3. CATEGORIZE: Group queries into meaningful topic clusters
4. SELECT: Pick the top queries that represent the best opportunities

For each prompt, provide a consolidated score (0.0-1.0) and updated metadata.

You MUST respond with valid JSON in this exact format:
{
  "prompts": [
    {
      "prompt_text": "search query",
      "relevance_score": 0.9,
      "search_intent": "informational",
      "topic_cluster": "topic",
      "estimated_volume": "high",
      "difficulty": "medium",
      "content_match_score": 0.85,
      "source_urls": ["url1", "url2"],
      "reasoning": "explanation"
    }
  ]
}"""

SYSTEM_PROMPT_SITE_ANALYSIS = """You are an expert SEO analyst. Analyze the overall structure and content of this website based on the page summaries provided. Identify:

1. The website's primary topics and themes
2. The target audience
3. The types of content (blog posts, product pages, documentation, etc.)
4. Key topic clusters the site covers

Then generate additional search prompts that the website SHOULD rank for based on its overall content strategy, even if specific pages weren't perfectly optimized for them.

You MUST respond with valid JSON in this exact format:
{
  "site_analysis": {
    "primary_topics": ["topic1", "topic2"],
    "target_audience": "description",
    "content_types": ["type1", "type2"],
    "topic_clusters": ["cluster1", "cluster2"]
  },
  "additional_prompts": [
    {
      "prompt_text": "search query",
      "relevance_score": 0.8,
      "search_intent": "informational",
      "topic_cluster": "topic",
      "estimated_volume": "medium",
      "difficulty": "medium",
      "reasoning": "explanation"
    }
  ]
}"""


class PromptExtractor:
    """Extracts and ranks search prompts from crawled website content."""

    def __init__(self, config: PromptExtractionConfig, llm_client: OpenRouterClient):
        self.config = config
        self.llm = llm_client

    def _prepare_page_content(self, page: CrawledPage) -> str:
        """Prepare page content for LLM analysis."""
        content_parts = []

        content_parts.append(f"URL: {page.url}")
        if page.title:
            content_parts.append(f"Title: {page.title}")
        if page.meta_description:
            content_parts.append(f"Meta Description: {page.meta_description}")
        if page.h1_tags:
            content_parts.append(f"H1 Tags: {', '.join(page.h1_tags)}")
        if page.h2_tags:
            content_parts.append(f"H2 Tags: {', '.join(page.h2_tags[:10])}")
        if page.h3_tags:
            content_parts.append(f"H3 Tags: {', '.join(page.h3_tags[:10])}")

        # Truncate body text to fit in context window
        body_preview = page.body_text[:3000] if page.body_text else ""
        if body_preview:
            content_parts.append(f"Content Preview:\n{body_preview}")

        content_parts.append(f"Word Count: {page.word_count}")

        return "\n\n".join(content_parts)

    def _extract_from_page_batch(
        self, pages: list[CrawledPage]
    ) -> list[ExtractedPrompt]:
        """Extract prompts from a batch of pages."""
        batch_content = []
        for i, page in enumerate(pages, 1):
            page_content = self._prepare_page_content(page)
            batch_content.append(f"=== PAGE {i} ===\n{page_content}")

        combined_content = "\n\n".join(batch_content)

        user_prompt = (
            f"Analyze these {len(pages)} web pages and identify ALL search queries/prompts "
            f"that these pages would rank for. Generate at least 10-15 prompts per page. "
            f"Focus on real user search behavior.\n\n{combined_content}"
        )

        try:
            result = self.llm.analyze_json(
                system_prompt=SYSTEM_PROMPT_EXTRACT,
                user_prompt=user_prompt,
                max_tokens=4096,
                temperature=0.4,
            )

            prompts = []
            for item in result.get("prompts", []):
                prompt = ExtractedPrompt(
                    prompt_text=item.get("prompt_text", "").strip(),
                    source_urls=[p.url for p in pages],
                    relevance_score=float(item.get("relevance_score", 0.5)),
                    search_intent=item.get("search_intent", "informational"),
                    topic_cluster=item.get("topic_cluster", ""),
                    estimated_volume=item.get("estimated_volume", "medium"),
                    difficulty=item.get("difficulty", "medium"),
                    extraction_reasoning=item.get("reasoning", ""),
                )
                if prompt.prompt_text:
                    prompts.append(prompt)

            logger.info(
                f"Extracted {len(prompts)} prompts from batch of {len(pages)} pages"
            )
            return prompts

        except Exception as e:
            logger.error(f"Error extracting prompts from batch: {e}")
            return []

    def _analyze_site_structure(
        self, pages: list[CrawledPage]
    ) -> list[ExtractedPrompt]:
        """Analyze overall site structure to find additional prompts."""
        page_summaries = []
        for page in pages[:50]:  # Use top 50 pages for site analysis
            summary = f"- {page.title or page.url}"
            if page.meta_description:
                summary += f": {page.meta_description[:100]}"
            page_summaries.append(summary)

        site_summary = "\n".join(page_summaries)

        user_prompt = (
            f"Here is a summary of {len(pages)} pages from a website. "
            f"Analyze the site's content strategy and generate additional search prompts "
            f"that this website should rank for.\n\n"
            f"Page Summaries:\n{site_summary}"
        )

        try:
            result = self.llm.analyze_json(
                system_prompt=SYSTEM_PROMPT_SITE_ANALYSIS,
                user_prompt=user_prompt,
                max_tokens=4096,
                temperature=0.5,
            )

            prompts = []
            for item in result.get("additional_prompts", []):
                prompt = ExtractedPrompt(
                    prompt_text=item.get("prompt_text", "").strip(),
                    source_urls=[],
                    relevance_score=float(item.get("relevance_score", 0.5)),
                    search_intent=item.get("search_intent", "informational"),
                    topic_cluster=item.get("topic_cluster", ""),
                    estimated_volume=item.get("estimated_volume", "medium"),
                    difficulty=item.get("difficulty", "medium"),
                    extraction_reasoning=item.get("reasoning", ""),
                )
                if prompt.prompt_text:
                    prompts.append(prompt)

            logger.info(
                f"Site analysis generated {len(prompts)} additional prompts"
            )
            return prompts

        except Exception as e:
            logger.error(f"Error in site analysis: {e}")
            return []

    def _consolidate_and_rank(
        self, all_prompts: list[ExtractedPrompt]
    ) -> list[ExtractedPrompt]:
        """Consolidate, deduplicate, and rank all extracted prompts."""
        if not all_prompts:
            return []

        # First pass: simple deduplication by normalized text
        seen_texts: dict[str, ExtractedPrompt] = {}
        for prompt in all_prompts:
            normalized = prompt.prompt_text.lower().strip()
            if normalized in seen_texts:
                # Merge source URLs and keep higher score
                existing = seen_texts[normalized]
                existing.source_urls = list(
                    set(existing.source_urls + prompt.source_urls)
                )
                existing.relevance_score = max(
                    existing.relevance_score, prompt.relevance_score
                )
            else:
                seen_texts[normalized] = prompt

        unique_prompts = list(seen_texts.values())
        logger.info(
            f"After dedup: {len(unique_prompts)} unique prompts "
            f"(from {len(all_prompts)} total)"
        )

        # If we have too many, use LLM to consolidate in batches
        if len(unique_prompts) > self.config.top_prompts_count * 2:
            consolidated = self._llm_consolidate(unique_prompts)
        else:
            consolidated = unique_prompts

        # Sort by relevance score and return top N
        consolidated.sort(key=lambda p: p.relevance_score, reverse=True)
        top_prompts = consolidated[: self.config.top_prompts_count]

        logger.info(
            f"Final selection: {len(top_prompts)} top prompts "
            f"(from {len(consolidated)} consolidated)"
        )

        return top_prompts

    def _llm_consolidate(
        self, prompts: list[ExtractedPrompt]
    ) -> list[ExtractedPrompt]:
        """Use LLM to consolidate and rank a large list of prompts."""
        chunk_size = 100
        all_consolidated = []

        for i in range(0, len(prompts), chunk_size):
            chunk = prompts[i : i + chunk_size]

            prompt_list = []
            for p in chunk:
                prompt_list.append(
                    {
                        "prompt_text": p.prompt_text,
                        "relevance_score": p.relevance_score,
                        "search_intent": p.search_intent,
                        "topic_cluster": p.topic_cluster,
                        "estimated_volume": p.estimated_volume,
                        "source_urls": p.source_urls[:3],
                    }
                )

            user_prompt = (
                f"Consolidate and rank these {len(chunk)} search prompts. "
                f"Remove duplicates and near-duplicates, keeping the best version. "
                f"Return the top {min(len(chunk), self.config.top_prompts_count)} "
                f"most valuable prompts.\n\n"
                f"Prompts:\n{json.dumps(prompt_list, indent=2)}"
            )

            try:
                result = self.llm.analyze_json(
                    system_prompt=SYSTEM_PROMPT_CONSOLIDATE,
                    user_prompt=user_prompt,
                    max_tokens=4096,
                    temperature=0.3,
                )

                for item in result.get("prompts", []):
                    prompt = ExtractedPrompt(
                        prompt_text=item.get("prompt_text", "").strip(),
                        source_urls=item.get("source_urls", []),
                        relevance_score=float(item.get("relevance_score", 0.5)),
                        search_intent=item.get("search_intent", "informational"),
                        topic_cluster=item.get("topic_cluster", ""),
                        estimated_volume=item.get("estimated_volume", "medium"),
                        difficulty=item.get("difficulty", "medium"),
                        content_match_score=float(
                            item.get("content_match_score", 0.5)
                        ),
                        extraction_reasoning=item.get("reasoning", ""),
                    )
                    if prompt.prompt_text:
                        all_consolidated.append(prompt)

            except Exception as e:
                logger.error(f"Error consolidating chunk: {e}")
                # Fall back to keeping the chunk as-is
                all_consolidated.extend(chunk)

        return all_consolidated

    def extract_prompts(
        self, pages: list[CrawledPage], progress_callback=None
    ) -> list[ExtractedPrompt]:
        """Extract queries in page batches, add site-level suggestions, and rank the results.

        The callback receives (batch_num, total_batches) after each page batch.
        """
        if not pages:
            logger.warning("No pages to extract prompts from")
            return []

        logger.info(
            f"Starting prompt extraction from {len(pages)} pages "
            f"(batch size: {self.config.batch_size})"
        )

        all_prompts: list[ExtractedPrompt] = []

        total_batches = (len(pages) + self.config.batch_size - 1) // self.config.batch_size
        for batch_num in range(total_batches):
            start_idx = batch_num * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(pages))
            batch = pages[start_idx:end_idx]

            logger.info(
                f"Processing batch {batch_num + 1}/{total_batches} "
                f"({len(batch)} pages)"
            )

            batch_prompts = self._extract_from_page_batch(batch)
            all_prompts.extend(batch_prompts)

            if progress_callback:
                progress_callback(batch_num + 1, total_batches)

        # Site-level analysis for additional prompts
        logger.info("Running site-level analysis for additional prompts...")
        site_prompts = self._analyze_site_structure(pages)
        all_prompts.extend(site_prompts)

        logger.info(f"Total raw prompts extracted: {len(all_prompts)}")

        top_prompts = self._consolidate_and_rank(all_prompts)

        return top_prompts
