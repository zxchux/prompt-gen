"""Fetch page and organic traffic metrics from the Google Analytics 4 Data API."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from prompt_gen.config import GoogleConfig
from prompt_gen.models import GAData

logger = logging.getLogger(__name__)


class AnalyticsClient:
    """Client for Google Analytics 4 Data API."""

    def __init__(self, config: GoogleConfig):
        self.config = config
        self.client = None
        self._authenticated = False

    def authenticate(self) -> bool:
        """Load GA4 service account credentials. Return True on success."""
        try:
            from google.analytics.data_v1beta import BetaAnalyticsDataClient
            from google.oauth2 import service_account

            creds_path = Path(self.config.ga_credentials_file)

            if not creds_path.exists():
                logger.error(
                    f"GA credentials file not found: {creds_path}. "
                    f"Download service account credentials from Google Cloud Console "
                    f"and save to {creds_path}"
                )
                return False

            credentials = service_account.Credentials.from_service_account_file(
                str(creds_path),
                scopes=["https://www.googleapis.com/auth/analytics.readonly"],
            )

            self.client = BetaAnalyticsDataClient(credentials=credentials)
            self._authenticated = True
            logger.info("Successfully authenticated with Google Analytics")
            return True

        except ImportError:
            logger.error(
                "Google Analytics library not installed. "
                "Run: pip install google-analytics-data"
            )
            return False
        except Exception as e:
            logger.error(f"GA authentication failed: {e}")
            return False

    def fetch_page_data(
        self,
        days: Optional[int] = None,
        limit: int = 5000,
    ) -> list[GAData]:
        """
        Fetch page-level analytics data from GA4.

        Args:
            days: Number of days to look back (default from config).
            limit: Maximum number of rows to fetch.

        Returns:
            List of GAData objects with page metrics.
        """
        if not self._authenticated or not self.client:
            logger.warning("Not authenticated with GA")
            return []

        if not self.config.ga_property_id:
            logger.error("GA property ID not configured")
            return []

        try:
            from google.analytics.data_v1beta.types import (
                DateRange,
                Dimension,
                Metric,
                RunReportRequest,
            )

            lookback = days or self.config.data_lookback_days
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=lookback)).strftime(
                "%Y-%m-%d"
            )

            property_id = self.config.ga_property_id
            if not property_id.startswith("properties/"):
                property_id = f"properties/{property_id}"

            request = RunReportRequest(
                property=property_id,
                date_ranges=[
                    DateRange(start_date=start_date, end_date=end_date)
                ],
                dimensions=[
                    Dimension(name="pagePath"),
                ],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="totalUsers"),
                    Metric(name="screenPageViews"),
                    Metric(name="averageSessionDuration"),
                    Metric(name="bounceRate"),
                    Metric(name="conversions"),
                ],
                limit=limit,
                order_bys=[
                    {
                        "metric": {"metric_name": "sessions"},
                        "desc": True,
                    }
                ],
            )

            response = self.client.run_report(request)

            page_data = []
            for row in response.rows:
                page_path = row.dimension_values[0].value
                metrics = row.metric_values

                ga_data = GAData(
                    page_path=page_path,
                    sessions=int(metrics[0].value) if metrics[0].value else 0,
                    users=int(metrics[1].value) if metrics[1].value else 0,
                    pageviews=int(metrics[2].value) if metrics[2].value else 0,
                    avg_session_duration=float(metrics[3].value)
                    if metrics[3].value
                    else 0.0,
                    bounce_rate=float(metrics[4].value)
                    if metrics[4].value
                    else 0.0,
                    conversions=int(float(metrics[5].value))
                    if metrics[5].value
                    else 0,
                )
                page_data.append(ga_data)

            logger.info(f"Fetched GA data for {len(page_data)} pages")
            return page_data

        except Exception as e:
            logger.error(f"Error fetching GA page data: {e}")
            return []

    def fetch_organic_data(
        self,
        days: Optional[int] = None,
        limit: int = 5000,
    ) -> list[GAData]:
        """
        Fetch organic traffic data from GA4.

        Filters for organic search sessions to correlate with search prompts.

        Args:
            days: Number of days to look back.
            limit: Maximum rows to fetch.

        Returns:
            List of GAData objects with organic traffic metrics.
        """
        if not self._authenticated or not self.client:
            logger.warning("Not authenticated with GA")
            return []

        if not self.config.ga_property_id:
            logger.error("GA property ID not configured")
            return []

        try:
            from google.analytics.data_v1beta.types import (
                DateRange,
                Dimension,
                Filter,
                FilterExpression,
                Metric,
                RunReportRequest,
            )

            lookback = days or self.config.data_lookback_days
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=lookback)).strftime(
                "%Y-%m-%d"
            )

            property_id = self.config.ga_property_id
            if not property_id.startswith("properties/"):
                property_id = f"properties/{property_id}"

            request = RunReportRequest(
                property=property_id,
                date_ranges=[
                    DateRange(start_date=start_date, end_date=end_date)
                ],
                dimensions=[
                    Dimension(name="pagePath"),
                    Dimension(name="sessionDefaultChannelGroup"),
                ],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="totalUsers"),
                    Metric(name="screenPageViews"),
                    Metric(name="bounceRate"),
                ],
                dimension_filter=FilterExpression(
                    filter=Filter(
                        field_name="sessionDefaultChannelGroup",
                        string_filter=Filter.StringFilter(
                            value="Organic Search",
                            match_type=Filter.StringFilter.MatchType.EXACT,
                        ),
                    )
                ),
                limit=limit,
                order_bys=[
                    {
                        "metric": {"metric_name": "sessions"},
                        "desc": True,
                    }
                ],
            )

            response = self.client.run_report(request)

            organic_data = []
            for row in response.rows:
                page_path = row.dimension_values[0].value
                metrics = row.metric_values

                ga_data = GAData(
                    page_path=page_path,
                    organic_sessions=int(metrics[0].value)
                    if metrics[0].value
                    else 0,
                    users=int(metrics[1].value) if metrics[1].value else 0,
                    pageviews=int(metrics[2].value) if metrics[2].value else 0,
                    bounce_rate=float(metrics[3].value)
                    if metrics[3].value
                    else 0.0,
                )
                organic_data.append(ga_data)

            logger.info(
                f"Fetched organic GA data for {len(organic_data)} pages"
            )
            return organic_data

        except Exception as e:
            logger.error(f"Error fetching organic GA data: {e}")
            return []

    def fetch_landing_page_data(
        self,
        days: Optional[int] = None,
        limit: int = 5000,
    ) -> dict[str, GAData]:
        """Merge total and organic traffic metrics into a dictionary keyed by page path."""
        all_pages = self.fetch_page_data(days=days, limit=limit)
        organic_pages = self.fetch_organic_data(days=days, limit=limit)

        page_map: dict[str, GAData] = {}
        for page in all_pages:
            page_map[page.page_path] = page

        for organic in organic_pages:
            if organic.page_path in page_map:
                page_map[organic.page_path].organic_sessions = (
                    organic.organic_sessions
                )
            else:
                page_map[organic.page_path] = organic

        logger.info(
            f"Combined GA data: {len(page_map)} pages "
            f"({len(all_pages)} total, {len(organic_pages)} organic)"
        )
        return page_map
