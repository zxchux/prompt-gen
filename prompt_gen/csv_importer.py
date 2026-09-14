"""Read Search Console query exports and GA4 page and organic traffic CSVs.

Column aliases cover common export headers; Google credentials are not required.
"""

import csv
import logging
from pathlib import Path
from typing import Optional

from prompt_gen.models import GAData, GSCData

logger = logging.getLogger(__name__)


class CSVImporter:
    """Imports GSC and GA data from CSV files."""

    # Known column name mappings for GSC exports
    GSC_QUERY_COLUMNS = {
        "query": ["top queries", "queries", "query", "search query", "keyword", "top query"],
        "clicks": ["clicks", "click"],
        "impressions": ["impressions", "impression"],
        "ctr": ["ctr", "click-through rate", "click through rate"],
        "position": ["position", "average position", "avg position", "avg. position"],
    }

    GSC_PAGE_COLUMNS = {
        "page": ["top pages", "pages", "page", "url", "landing page"],
        "clicks": ["clicks", "click"],
        "impressions": ["impressions", "impression"],
        "ctr": ["ctr", "click-through rate", "click through rate"],
        "position": ["position", "average position", "avg position", "avg. position"],
    }

    # Known column name mappings for GA exports
    GA_COLUMNS = {
        "page_path": [
            "page path", "page path + query string", "page", "pages",
            "page path and screen class", "landing page", "landing page + query string",
            "page title and screen class", "page title",
        ],
        "sessions": ["sessions", "session"],
        "users": ["users", "total users", "active users"],
        "pageviews": [
            "views", "pageviews", "page views", "screen page views",
            "screenPageViews",
        ],
        "avg_session_duration": [
            "avg. session duration", "average session duration",
            "avg engagement time", "average engagement time per session",
            "avg. engagement time",
        ],
        "bounce_rate": ["bounce rate", "bounceRate", "bounce rate (%)"],
        "conversions": ["conversions", "key events", "goal completions"],
    }

    def _normalize_header(self, header: str) -> str:
        """Normalize a CSV header for matching."""
        return header.strip().lower().replace("_", " ").replace("-", " ")

    def _find_column_index(
        self, headers: list[str], column_names: list[str]
    ) -> Optional[int]:
        """Find the index of a column by trying multiple name variants."""
        normalized_headers = [self._normalize_header(h) for h in headers]
        for name in column_names:
            normalized_name = self._normalize_header(name)
            for i, header in enumerate(normalized_headers):
                if normalized_name == header or normalized_name in header:
                    return i
        return None

    def _parse_number(self, value: str) -> float:
        """Parse a number from a CSV value, handling commas and percentages."""
        if not value or value.strip() in ("", "-", "N/A", "n/a"):
            return 0.0
        # Remove commas, percentage signs, and whitespace
        cleaned = value.strip().replace(",", "").replace("%", "").replace("$", "")
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def _detect_delimiter(self, filepath: str) -> str:
        """Detect the CSV delimiter (comma, tab, or semicolon)."""
        with open(filepath, "r", encoding="utf-8-sig") as f:
            first_line = f.readline()
            tab_count = first_line.count("\t")
            comma_count = first_line.count(",")
            semicolon_count = first_line.count(";")

            if tab_count > comma_count and tab_count > semicolon_count:
                return "\t"
            elif semicolon_count > comma_count:
                return ";"
            return ","

    def _skip_metadata_rows(self, filepath: str, delimiter: str) -> int:
        """
        Detect how many metadata rows to skip before the actual header.
        GSC and GA exports sometimes have metadata rows at the top.
        """
        with open(filepath, "r", encoding="utf-8-sig") as f:
            for i, line in enumerate(f):
                stripped = line.strip()
                if not stripped:
                    continue
                # Check if this looks like a header row (has multiple columns)
                parts = stripped.split(delimiter)
                if len(parts) >= 2:
                    # Check if any part looks like a known column name
                    for part in parts:
                        normalized = self._normalize_header(part)
                        all_known = []
                        for names in self.GSC_QUERY_COLUMNS.values():
                            all_known.extend(names)
                        for names in self.GA_COLUMNS.values():
                            all_known.extend(names)
                        for known in all_known:
                            if known in normalized or normalized in known:
                                return i
                # Skip export metadata such as "# ..." or "Date range: ...".
                if stripped.startswith("#") or ":" in parts[0]:
                    continue
                return i
        return 0

    def import_gsc_queries(self, filepath: str) -> list[GSCData]:
        """Read a Search Console Performance export from the Queries tab."""
        filepath = str(filepath)
        if not Path(filepath).exists():
            logger.error(f"GSC CSV file not found: {filepath}")
            return []

        delimiter = self._detect_delimiter(filepath)
        skip_rows = self._skip_metadata_rows(filepath, delimiter)

        logger.info(f"Importing GSC query data from {filepath}")

        data = []
        try:
            with open(filepath, "r", encoding="utf-8-sig") as f:
                for _ in range(skip_rows):
                    next(f)

                reader = csv.reader(f, delimiter=delimiter)
                headers = next(reader)

                query_idx = self._find_column_index(
                    headers, self.GSC_QUERY_COLUMNS["query"]
                )
                clicks_idx = self._find_column_index(
                    headers, self.GSC_QUERY_COLUMNS["clicks"]
                )
                impressions_idx = self._find_column_index(
                    headers, self.GSC_QUERY_COLUMNS["impressions"]
                )
                ctr_idx = self._find_column_index(
                    headers, self.GSC_QUERY_COLUMNS["ctr"]
                )
                position_idx = self._find_column_index(
                    headers, self.GSC_QUERY_COLUMNS["position"]
                )

                if query_idx is None:
                    logger.error(
                        f"Could not find query column in CSV. "
                        f"Headers found: {headers}"
                    )
                    return []

                for row in reader:
                    if not row or len(row) <= query_idx:
                        continue

                    query = row[query_idx].strip()
                    if not query:
                        continue

                    clicks = int(self._parse_number(row[clicks_idx])) if clicks_idx is not None and clicks_idx < len(row) else 0
                    impressions = int(self._parse_number(row[impressions_idx])) if impressions_idx is not None and impressions_idx < len(row) else 0

                    ctr_val = self._parse_number(row[ctr_idx]) if ctr_idx is not None and ctr_idx < len(row) else 0.0
                    # GSC exports CTR as percentage (e.g., "3.5%") or decimal (0.035)
                    if ctr_val > 1:
                        ctr_val = ctr_val / 100.0

                    position = self._parse_number(row[position_idx]) if position_idx is not None and position_idx < len(row) else 0.0

                    gsc_data = GSCData(
                        query=query,
                        clicks=clicks,
                        impressions=impressions,
                        ctr=ctr_val,
                        position=position,
                        pages=[],
                        date_range="imported from CSV",
                    )
                    data.append(gsc_data)

            logger.info(f"Imported {len(data)} GSC queries from CSV")
            return data

        except Exception as e:
            logger.error(f"Error importing GSC CSV: {e}")
            return []

    def import_ga_pages(self, filepath: str) -> dict[str, GAData]:
        """Read a GA4 Pages and screens export, keyed by page path."""
        filepath = str(filepath)
        if not Path(filepath).exists():
            logger.error(f"GA CSV file not found: {filepath}")
            return {}

        delimiter = self._detect_delimiter(filepath)
        skip_rows = self._skip_metadata_rows(filepath, delimiter)

        logger.info(f"Importing GA page data from {filepath}")

        page_map: dict[str, GAData] = {}
        try:
            with open(filepath, "r", encoding="utf-8-sig") as f:
                for _ in range(skip_rows):
                    next(f)

                reader = csv.reader(f, delimiter=delimiter)
                headers = next(reader)

                page_idx = self._find_column_index(
                    headers, self.GA_COLUMNS["page_path"]
                )
                sessions_idx = self._find_column_index(
                    headers, self.GA_COLUMNS["sessions"]
                )
                users_idx = self._find_column_index(
                    headers, self.GA_COLUMNS["users"]
                )
                pageviews_idx = self._find_column_index(
                    headers, self.GA_COLUMNS["pageviews"]
                )
                duration_idx = self._find_column_index(
                    headers, self.GA_COLUMNS["avg_session_duration"]
                )
                bounce_idx = self._find_column_index(
                    headers, self.GA_COLUMNS["bounce_rate"]
                )
                conversions_idx = self._find_column_index(
                    headers, self.GA_COLUMNS["conversions"]
                )

                if page_idx is None:
                    logger.error(
                        f"Could not find page path column in CSV. "
                        f"Headers found: {headers}"
                    )
                    return {}

                for row in reader:
                    if not row or len(row) <= page_idx:
                        continue

                    page_path = row[page_idx].strip()
                    if not page_path:
                        continue

                    sessions = int(self._parse_number(row[sessions_idx])) if sessions_idx is not None and sessions_idx < len(row) else 0
                    users = int(self._parse_number(row[users_idx])) if users_idx is not None and users_idx < len(row) else 0
                    pageviews = int(self._parse_number(row[pageviews_idx])) if pageviews_idx is not None and pageviews_idx < len(row) else 0
                    duration = self._parse_number(row[duration_idx]) if duration_idx is not None and duration_idx < len(row) else 0.0
                    bounce = self._parse_number(row[bounce_idx]) if bounce_idx is not None and bounce_idx < len(row) else 0.0
                    conversions = int(self._parse_number(row[conversions_idx])) if conversions_idx is not None and conversions_idx < len(row) else 0

                    if bounce > 1:
                        bounce = bounce / 100.0

                    ga_data = GAData(
                        page_path=page_path,
                        sessions=sessions,
                        users=users,
                        pageviews=pageviews,
                        avg_session_duration=duration,
                        bounce_rate=bounce,
                        conversions=conversions,
                        organic_sessions=0,  # Will be merged if organic CSV provided
                    )
                    page_map[page_path] = ga_data

            logger.info(f"Imported GA data for {len(page_map)} pages from CSV")
            return page_map

        except Exception as e:
            logger.error(f"Error importing GA CSV: {e}")
            return {}

    def import_ga_organic(
        self, filepath: str, existing_data: Optional[dict[str, GAData]] = None
    ) -> dict[str, GAData]:
        """Merge organic session counts from CSV into existing page records.

        Returns a new dictionary; matching records in existing_data are updated in place.
        """
        filepath = str(filepath)
        if not Path(filepath).exists():
            logger.error(f"GA organic CSV file not found: {filepath}")
            return existing_data or {}

        delimiter = self._detect_delimiter(filepath)
        skip_rows = self._skip_metadata_rows(filepath, delimiter)

        page_map = dict(existing_data) if existing_data else {}

        try:
            with open(filepath, "r", encoding="utf-8-sig") as f:
                for _ in range(skip_rows):
                    next(f)

                reader = csv.reader(f, delimiter=delimiter)
                headers = next(reader)

                page_idx = self._find_column_index(
                    headers, self.GA_COLUMNS["page_path"]
                )
                sessions_idx = self._find_column_index(
                    headers, self.GA_COLUMNS["sessions"]
                )

                if page_idx is None:
                    logger.error(f"Could not find page column. Headers: {headers}")
                    return page_map

                for row in reader:
                    if not row or len(row) <= page_idx:
                        continue

                    page_path = row[page_idx].strip()
                    if not page_path:
                        continue

                    organic_sessions = int(self._parse_number(row[sessions_idx])) if sessions_idx is not None and sessions_idx < len(row) else 0

                    if page_path in page_map:
                        page_map[page_path].organic_sessions = organic_sessions
                    else:
                        page_map[page_path] = GAData(
                            page_path=page_path,
                            organic_sessions=organic_sessions,
                        )

            logger.info(f"Imported organic data, total pages: {len(page_map)}")
            return page_map

        except Exception as e:
            logger.error(f"Error importing GA organic CSV: {e}")
            return page_map
