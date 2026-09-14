"""Crawl website pages and sitemaps, then extract text, metadata, and links."""

import logging
import time
from collections import deque
from typing import Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
import urllib3
from bs4 import BeautifulSoup

from prompt_gen.config import CrawlerConfig
from prompt_gen.models import CrawledPage

# Suppress InsecureRequestWarning when SSL verification is disabled
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class WebCrawler:
    """Crawls a website and extracts structured content from pages."""

    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.session = requests.Session()
        self.session.verify = False  # Handle sites with SSL issues
        self.session.headers.update(
            {
                "User-Agent": config.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )
        self.visited_urls: set[str] = set()
        self.robots_parser: Optional[RobotFileParser] = None
        self._domain: str = ""

    def _normalize_url(self, url: str) -> str:
        """Normalize a URL by removing fragments and trailing slashes."""
        parsed = urlparse(url)
        normalized = parsed._replace(fragment="")
        result = normalized.geturl().rstrip("/")
        return result

    def _strip_www(self, domain: str) -> str:
        """Strip www. prefix from a domain for comparison."""
        return domain.lower().removeprefix("www.")

    def _is_same_domain(self, url: str) -> bool:
        """Check if a URL belongs to the same domain as the target (handles www/non-www)."""
        parsed = urlparse(url)
        return self._strip_www(parsed.netloc) == self._strip_www(self._domain)

    def _is_valid_page_url(self, url: str) -> bool:
        """Check if a URL is a valid page to crawl (not a file, image, etc.)."""
        skip_extensions = {
            ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
            ".mp4", ".mp3", ".avi", ".mov", ".zip", ".tar", ".gz",
            ".css", ".js", ".xml", ".json", ".ico", ".woff", ".woff2",
            ".ttf", ".eot", ".map",
        }
        parsed = urlparse(url)
        path_lower = parsed.path.lower()

        for ext in skip_extensions:
            if path_lower.endswith(ext):
                return False

        # Skip common non-content paths
        skip_paths = {
            "/wp-admin", "/wp-login", "/admin", "/login", "/logout",
            "/cart", "/checkout", "/account", "/api/", "/feed",
        }
        for skip in skip_paths:
            if skip in path_lower:
                return False

        return True

    def _load_robots_txt(self, base_url: str) -> None:
        """Load and parse robots.txt for the target domain."""
        if not self.config.respect_robots_txt:
            return

        robots_url = urljoin(base_url, "/robots.txt")
        self.robots_parser = RobotFileParser()
        self.robots_parser.set_url(robots_url)

        try:
            # Use our session (with SSL verify=False) instead of stdlib urllib
            response = self.session.get(robots_url, timeout=10)
            if response.status_code == 200:
                self.robots_parser.parse(response.text.splitlines())
                logger.info(f"Loaded robots.txt from {robots_url}")
            else:
                logger.warning(
                    f"robots.txt returned status {response.status_code}, allowing all"
                )
                self.robots_parser = None
        except Exception as e:
            logger.warning(f"Could not load robots.txt: {e}")
            self.robots_parser = None

    def _can_fetch(self, url: str) -> bool:
        """Check if we're allowed to fetch a URL per robots.txt."""
        if not self.robots_parser:
            return True
        return self.robots_parser.can_fetch(self.config.user_agent, url)

    def _extract_page_content(self, url: str, html: str) -> CrawledPage:
        """Extract structured content from an HTML page."""
        # First try trafilatura for better content extraction
        body_text = ""
        try:
            import trafilatura
            extracted = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                favor_recall=True,
            )
            if extracted:
                body_text = extracted
        except Exception as e:
            logger.debug(f"Trafilatura extraction failed for {url}: {e}")

        soup = BeautifulSoup(html, "lxml")

        # Extract title (before decomposing elements)
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag:
            meta_desc = meta_tag.get("content", "")

        # Extract other meta tags for SPA support
        og_title = ""
        og_tag = soup.find("meta", attrs={"property": "og:title"})
        if og_tag:
            og_title = og_tag.get("content", "")

        og_desc = ""
        og_desc_tag = soup.find("meta", attrs={"property": "og:description"})
        if og_desc_tag:
            og_desc = og_desc_tag.get("content", "")

        # Extract JSON-LD structured data (common in SPAs)
        json_ld_text = ""
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                import json
                ld_data = json.loads(script.string or "")
                if isinstance(ld_data, dict):
                    for key in ("name", "description", "headline", "articleBody", "text"):
                        if key in ld_data and isinstance(ld_data[key], str):
                            json_ld_text += " " + ld_data[key]
                elif isinstance(ld_data, list):
                    for item in ld_data:
                        if isinstance(item, dict):
                            for key in ("name", "description", "headline"):
                                if key in item and isinstance(item[key], str):
                                    json_ld_text += " " + item[key]
            except Exception:
                pass

        # Extract links before decomposing
        internal_links = []
        external_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            absolute_url = urljoin(url, href)

            if self._is_same_domain(absolute_url):
                internal_links.append(self._normalize_url(absolute_url))
            elif href.startswith("http"):
                external_links.append(absolute_url)

        # Also extract links from sitemap-style references in the HTML
        # (some SPAs embed route info in script tags)
        for script in soup.find_all("script"):
            if script.string:
                import re
                # Find URL-like paths in script content
                paths = re.findall(r'["\']/([\w-]+(?:/[\w-]+)*)["\']', script.string)
                for path in paths:
                    skip_prefixes = ("static", "assets", "_next", "chunk")
                    if len(path) > 1 and not path.startswith(skip_prefixes):
                        candidate = urljoin(url, f"/{path}")
                        if (
                            self._is_same_domain(candidate)
                            and self._is_valid_page_url(candidate)
                        ):
                            internal_links.append(self._normalize_url(candidate))

        # Now decompose non-content elements for BS4 text extraction
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        h1_tags = [h.get_text(strip=True) for h in soup.find_all("h1")]
        h2_tags = [h.get_text(strip=True) for h in soup.find_all("h2")]
        h3_tags = [h.get_text(strip=True) for h in soup.find_all("h3")]

        # If trafilatura didn't get text, try BS4
        if not body_text:
            body = soup.find("body")
            if body:
                main_content = body.find("main") or body.find("article") or body
                body_text = main_content.get_text(separator=" ", strip=True)
                body_text = " ".join(body_text.split())

        # For SPAs with no body text, compose from metadata
        if not body_text or len(body_text.split()) < 10:
            meta_parts = []
            if title:
                meta_parts.append(title)
            if og_title and og_title != title:
                meta_parts.append(og_title)
            if meta_desc:
                meta_parts.append(meta_desc)
            if og_desc and og_desc != meta_desc:
                meta_parts.append(og_desc)
            if json_ld_text.strip():
                meta_parts.append(json_ld_text.strip())
            if meta_parts:
                body_text = ". ".join(meta_parts) + (". " + body_text if body_text else "")

        word_count = len(body_text.split()) if body_text else 0

        return CrawledPage(
            url=url,
            title=title or og_title,
            meta_description=meta_desc or og_desc,
            h1_tags=h1_tags,
            h2_tags=h2_tags,
            h3_tags=h3_tags,
            body_text=body_text,
            internal_links=list(set(internal_links)),
            external_links=list(set(external_links[:50])),
            word_count=word_count,
        )

    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch a single page and return its HTML content."""
        try:
            response = self.session.get(
                url,
                timeout=self.config.request_timeout,
                allow_redirects=True,
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                logger.debug(f"Skipping non-HTML content: {url} ({content_type})")
                return None

            return response.text

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching {url}")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP error fetching {url}: {e}")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error fetching {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")

        return None

    def crawl(self, start_url: str, progress_callback=None) -> list[CrawledPage]:
        """
        Crawl a website starting from the given URL.

        Uses BFS to discover and crawl pages up to the configured maximum,
        respecting robots.txt and rate limits.

        Args:
            start_url: The starting URL to crawl.
            progress_callback: Optional callback(pages_crawled, total_found)
                for progress reporting.

        Returns:
            List of CrawledPage objects with extracted content.
        """
        # Normalize and validate start URL
        parsed = urlparse(start_url)
        if not parsed.scheme:
            start_url = f"https://{start_url}"
            parsed = urlparse(start_url)

        self._domain = parsed.netloc
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        logger.info(f"Starting crawl of {base_url} (max {self.config.max_pages} pages)")

        self._load_robots_txt(base_url)

        # Try sitemap discovery first to seed the queue
        sitemap_urls = self.get_sitemap_urls(base_url)

        queue: deque[tuple[str, int]] = deque()  # (url, depth)
        queue.append((self._normalize_url(start_url), 0))

        for surl in sitemap_urls:
            if self._is_same_domain(surl):
                queue.append((self._normalize_url(surl), 1))

        if sitemap_urls:
            logger.info(f"Seeded queue with {len(sitemap_urls)} sitemap URLs")

        self.visited_urls = set()
        pages: list[CrawledPage] = []
        is_first_page = True

        while queue and len(pages) < self.config.max_pages:
            url, depth = queue.popleft()

            if url in self.visited_urls:
                continue
            if depth > self.config.max_depth:
                continue

            self.visited_urls.add(url)

            if not self._can_fetch(url):
                logger.debug(f"Blocked by robots.txt: {url}")
                continue

            if not self._is_valid_page_url(url):
                continue

            time.sleep(self.config.crawl_delay)

            html = self._fetch_page(url)
            if html is None:
                continue

            page = self._extract_page_content(url, html)
            page.depth = depth
            page.status_code = 200

            # Accept a title-only homepage; sparse SPAs may expose only metadata.
            min_words = 5 if is_first_page else 10
            if page.word_count >= min_words or (page.title and is_first_page):
                pages.append(page)
                logger.info(
                    f"Crawled [{len(pages)}/{self.config.max_pages}]: {url} "
                    f"({page.word_count} words)"
                )

                if progress_callback:
                    progress_callback(len(pages), len(queue) + len(pages))
            else:
                logger.debug(
                    f"Skipped low-content page: {url} ({page.word_count} words)"
                )

            is_first_page = False

            for link in page.internal_links:
                if link not in self.visited_urls and self._is_same_domain(link):
                    queue.append((link, depth + 1))

        logger.info(
            f"Crawl complete: {len(pages)} pages crawled, "
            f"{len(self.visited_urls)} URLs visited"
        )

        return pages

    def get_sitemap_urls(self, base_url: str) -> list[str]:
        """Find URLs in common sitemap locations and their immediate child sitemaps."""
        sitemap_urls = [
            urljoin(base_url, "/sitemap.xml"),
            urljoin(base_url, "/sitemap_index.xml"),
            urljoin(base_url, "/sitemap/sitemap.xml"),
        ]

        discovered_urls = []

        for sitemap_url in sitemap_urls:
            try:
                response = self.session.get(sitemap_url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "lxml-xml")

                    sitemaps = soup.find_all("sitemap")
                    if sitemaps:
                        for sm in sitemaps:
                            loc = sm.find("loc")
                            if loc:
                                # Fetch each child sitemap once.
                                try:
                                    sub_response = self.session.get(
                                        loc.text.strip(), timeout=10
                                    )
                                    sub_soup = BeautifulSoup(
                                        sub_response.text, "lxml-xml"
                                    )
                                    for url_tag in sub_soup.find_all("url"):
                                        loc_tag = url_tag.find("loc")
                                        if loc_tag:
                                            discovered_urls.append(loc_tag.text.strip())
                                except Exception:
                                    pass

                    for url_tag in soup.find_all("url"):
                        loc = url_tag.find("loc")
                        if loc:
                            discovered_urls.append(loc.text.strip())

                    if discovered_urls:
                        logger.info(
                            f"Found {len(discovered_urls)} URLs in sitemap: {sitemap_url}"
                        )
                        break

            except Exception as e:
                logger.debug(f"Could not fetch sitemap {sitemap_url}: {e}")

        return discovered_urls
