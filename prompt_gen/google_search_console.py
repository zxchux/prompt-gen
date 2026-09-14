"""Fetch query and page performance data from Google Search Console."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from prompt_gen.config import GoogleConfig
from prompt_gen.models import GSCData

logger = logging.getLogger(__name__)


class SearchConsoleClient:
    """Client for Google Search Console API."""

    def __init__(self, config: GoogleConfig):
        self.config = config
        self.service = None
        self._authenticated = False

    def authenticate(self) -> bool:
        """
        Authenticate with Google Search Console API using OAuth2.

        Requires credentials JSON file from Google Cloud Console.
        On first run, opens a browser for OAuth consent flow.
        Subsequent runs use the saved token.

        Returns:
            True if authentication was successful.
        """
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build

            scopes = ["https://www.googleapis.com/auth/webmasters.readonly"]

            creds = None
            token_path = Path(self.config.gsc_token_file)
            creds_path = Path(self.config.gsc_credentials_file)

            if token_path.exists():
                creds = Credentials.from_authorized_user_file(
                    str(token_path), scopes
                )

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not creds_path.exists():
                        logger.error(
                            f"GSC credentials file not found: {creds_path}. "
                            f"Download OAuth2 credentials from Google Cloud Console "
                            f"and save to {creds_path}"
                        )
                        return False

                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(creds_path), scopes
                    )
                    creds = flow.run_local_server(port=0)

                token_path.parent.mkdir(parents=True, exist_ok=True)
                with open(token_path, "w") as token_file:
                    token_file.write(creds.to_json())

            self.service = build("searchconsole", "v1", credentials=creds)
            self._authenticated = True
            logger.info("Successfully authenticated with Google Search Console")
            return True

        except ImportError:
            logger.error(
                "Google API libraries not installed. "
                "Run: pip install google-api-python-client google-auth-oauthlib"
            )
            return False
        except Exception as e:
            logger.error(f"GSC authentication failed: {e}")
            return False

    def get_site_url(self, target_domain: str) -> Optional[str]:
        """Return the configured GSC site, or find one containing target_domain."""
        if self.config.gsc_site_url:
            return self.config.gsc_site_url

        if not self._authenticated or not self.service:
            logger.warning("Not authenticated with GSC")
            return None

        try:
            site_list = self.service.sites().list().execute()
            sites = site_list.get("siteEntry", [])

            for site in sites:
                site_url = site.get("siteUrl", "")
                if target_domain in site_url:
                    logger.info(f"Found GSC site: {site_url}")
                    return site_url

            logger.warning(
                f"No GSC site found for domain: {target_domain}. "
                f"Available sites: {[s.get('siteUrl') for s in sites]}"
            )
            return None

        except Exception as e:
            logger.error(f"Error listing GSC sites: {e}")
            return None

    def fetch_query_data(
        self,
        site_url: str,
        days: Optional[int] = None,
        row_limit: int = 5000,
    ) -> list[GSCData]:
        """
        Fetch search query performance data from GSC.

        Args:
            site_url: The GSC site URL.
            days: Number of days to look back (default from config).
            row_limit: Maximum rows to fetch.

        Returns:
            List of GSCData objects with query metrics.
        """
        if not self._authenticated or not self.service:
            logger.warning("Not authenticated with GSC")
            return []

        lookback = days or self.config.data_lookback_days
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")

        date_range = f"{start_date} to {end_date}"
        logger.info(f"Fetching GSC data for {site_url} ({date_range})")

        all_data: list[GSCData] = []
        start_row = 0

        while True:
            try:
                request_body = {
                    "startDate": start_date,
                    "endDate": end_date,
                    "dimensions": ["query", "page"],
                    "rowLimit": min(row_limit - start_row, 25000),
                    "startRow": start_row,
                    "dataState": "final",
                }

                response = (
                    self.service.searchanalytics()
                    .query(siteUrl=site_url, body=request_body)
                    .execute()
                )

                rows = response.get("rows", [])
                if not rows:
                    break

                query_map: dict[str, GSCData] = {}
                for row in rows:
                    keys = row.get("keys", [])
                    if len(keys) < 2:
                        continue

                    query = keys[0]
                    page = keys[1]

                    if query not in query_map:
                        query_map[query] = GSCData(
                            query=query,
                            clicks=0,
                            impressions=0,
                            ctr=0.0,
                            position=0.0,
                            pages=[],
                            date_range=date_range,
                        )

                    data = query_map[query]
                    data.clicks += int(row.get("clicks", 0))
                    data.impressions += int(row.get("impressions", 0))
                    if page not in data.pages:
                        data.pages.append(page)

                for data in query_map.values():
                    if data.impressions > 0:
                        data.ctr = data.clicks / data.impressions

                all_data.extend(query_map.values())

                start_row += len(rows)
                if len(rows) < 25000 or start_row >= row_limit:
                    break

            except Exception as e:
                logger.error(f"Error fetching GSC data (row {start_row}): {e}")
                break

        final_map: dict[str, GSCData] = {}
        for data in all_data:
            if data.query in final_map:
                existing = final_map[data.query]
                existing.clicks += data.clicks
                existing.impressions += data.impressions
                existing.pages = list(set(existing.pages + data.pages))
                if existing.impressions > 0:
                    existing.ctr = existing.clicks / existing.impressions
            else:
                final_map[data.query] = data

        result = list(final_map.values())
        logger.info(f"Fetched GSC data for {len(result)} unique queries")
        return result

    def fetch_page_data(
        self,
        site_url: str,
        days: Optional[int] = None,
    ) -> dict[str, dict]:
        """Fetch page performance metrics, keyed by page URL."""
        if not self._authenticated or not self.service:
            return {}

        lookback = days or self.config.data_lookback_days
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")

        try:
            request_body = {
                "startDate": start_date,
                "endDate": end_date,
                "dimensions": ["page"],
                "rowLimit": 5000,
                "dataState": "final",
            }

            response = (
                self.service.searchanalytics()
                .query(siteUrl=site_url, body=request_body)
                .execute()
            )

            page_data = {}
            for row in response.get("rows", []):
                page_url = row.get("keys", [""])[0]
                page_data[page_url] = {
                    "clicks": row.get("clicks", 0),
                    "impressions": row.get("impressions", 0),
                    "ctr": row.get("ctr", 0.0),
                    "position": row.get("position", 0.0),
                }

            logger.info(f"Fetched GSC page data for {len(page_data)} pages")
            return page_data

        except Exception as e:
            logger.error(f"Error fetching GSC page data: {e}")
            return {}
