"""Minimal SuccessFactors OData V2 client.

Uses only the standard library so the same code runs locally and inside
AWS Lambda without bundling third-party packages.
"""

import base64
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

SF_DATE_PREFIX = "/Date("
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ODataError(Exception):
    """Raised when the OData API returns a non-retryable error."""


def parse_sf_date(value):
    """Convert SuccessFactors '/Date(1691200000000)/' strings to ISO-8601 UTC.

    Values that are not SF date strings are returned unchanged.
    """
    if isinstance(value, str) and value.startswith(SF_DATE_PREFIX):
        inner = value[len(SF_DATE_PREFIX):value.index(")")]
        millis = int(inner.split("+")[0].split("-")[0]) if inner[0] != "-" else int(inner)
        dt = datetime.fromtimestamp(millis / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return value


def normalize_record(record):
    """Strip OData metadata and convert SF date strings to ISO-8601."""
    return {
        k: parse_sf_date(v)
        for k, v in record.items()
        if k != "__metadata"
    }


class ODataClient:
    def __init__(self, base_url, username, password,
                 timeout=30, max_retries=5, page_size=100):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.page_size = page_size
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        self._headers = {
            "Authorization": f"Basic {token}",
            "Accept": "application/json",
        }

    def _request(self, url):
        last_error = None
        for attempt in range(self.max_retries):
            if attempt:
                delay = 2 ** attempt
                logger.warning("retrying in %ss (attempt %s): %s", delay, attempt + 1, url)
                time.sleep(delay)
            try:
                req = urllib.request.Request(url, headers=self._headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                if exc.code in RETRYABLE_STATUS:
                    last_error = exc
                    continue
                raise ODataError(f"HTTP {exc.code} for {url}: {exc.read().decode()[:500]}") from exc
            except urllib.error.URLError as exc:
                last_error = exc
                continue
        raise ODataError(f"request failed after {self.max_retries} attempts: {url}") from last_error

    def fetch_entity(self, entity, select=None, modified_after=None):
        """Yield normalized records for an entity set, following pagination.

        When ``modified_after`` (ISO-8601) is given, only records changed
        strictly after that instant are returned (delta extraction).
        """
        params = {"$format": "json", "$top": str(self.page_size)}
        if select:
            params["$select"] = select
        if modified_after:
            params["$filter"] = (
                f"lastModifiedDateTime gt datetimeoffset'{modified_after}'"
            )
        url = f"{self.base_url}/{entity}?{urllib.parse.urlencode(params)}"
        pages = 0
        while url:
            body = self._request(url)["d"]
            results = body["results"] if isinstance(body, dict) else body
            pages += 1
            for record in results:
                yield normalize_record(record)
            url = body.get("__next") if isinstance(body, dict) else None
        logger.info("fetched %s in %s page(s)", entity, pages)

    def fetch_metadata(self):
        req = urllib.request.Request(
            f"{self.base_url}/$metadata",
            headers={**self._headers, "Accept": "application/xml"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode()
