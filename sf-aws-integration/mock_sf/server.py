"""Mock SuccessFactors OData V2 server for local development and tests.

Faithfully mimics the behaviors the extractor must handle:
  - Basic authentication (401 without valid credentials)
  - JSON envelope: {"d": {"results": [...], "__next": "..."}}
  - Pagination via $top / $skiptoken with a __next continuation link
  - $select projection
  - $filter with "lastModifiedDateTime gt datetimeoffset'...'" (delta)
  - SF legacy date serialization: "/Date(1691200000000)/"

Usage:
    python mock_sf/server.py [--port 8000]
    (credentials default to demo@DEMO / demo; override with MOCK_USER/MOCK_PASS)
"""

import argparse
import base64
import json
import os
import re
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from data import DATASET

MOCK_USER = os.environ.get("MOCK_USER", "demo@DEMO")
MOCK_PASS = os.environ.get("MOCK_PASS", "demo")
MAX_PAGE_SIZE = 1000
FILTER_RE = re.compile(r"lastModifiedDateTime\s+gt\s+datetimeoffset'([^']+)'")

METADATA_XML = """<?xml version="1.0" encoding="utf-8"?>
<edmx:Edmx xmlns:edmx="http://schemas.microsoft.com/ado/2007/06/edmx" Version="1.0">
  <edmx:DataServices>
    <Schema xmlns="http://schemas.microsoft.com/ado/2008/09/edm" Namespace="SFOData">
      {entities}
    </Schema>
  </edmx:DataServices>
</edmx:Edmx>
"""


def _iso_to_millis(value):
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _serialize(record):
    """Render internal records with epoch-milli date fields as SF JSON."""
    out = {}
    for key, value in record.items():
        if key.endswith("_ms"):
            out[key[:-3]] = f"/Date({value})/"
        else:
            out[key] = value
    out["__metadata"] = {"type": "SFOData.mock"}
    return out


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep test output quiet
        pass

    def _send(self, status, body, content_type="application/json"):
        payload = body if isinstance(body, bytes) else body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _authorized(self):
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(header[6:]).decode()
        except Exception:
            return False
        return decoded == f"{MOCK_USER}:{MOCK_PASS}"

    def do_GET(self):
        if not self._authorized():
            self._send(401, json.dumps({"error": "unauthorized"}))
            return

        split = urllib.parse.urlsplit(self.path)
        parts = [p for p in split.path.split("/") if p]
        if len(parts) < 3 or parts[0] != "odata" or parts[1] != "v2":
            self._send(404, json.dumps({"error": "not found"}))
            return

        target = urllib.parse.unquote(parts[2])
        if target == "$metadata":
            entities = "\n".join(
                f'<EntityType Name="{name}"/>' for name in DATASET
            )
            self._send(200, METADATA_XML.format(entities=entities), "application/xml")
            return

        if target not in DATASET:
            self._send(404, json.dumps({"error": f"unknown entity {target}"}))
            return

        query = urllib.parse.parse_qs(split.query)
        records = sorted(DATASET[target], key=lambda r: str(r[next(iter(r))]))

        filter_expr = query.get("$filter", [None])[0]
        if filter_expr:
            match = FILTER_RE.search(filter_expr)
            if not match:
                self._send(400, json.dumps({"error": "unsupported $filter"}))
                return
            threshold = _iso_to_millis(match.group(1))
            records = [r for r in records if r["lastModifiedDateTime_ms"] > threshold]

        top = min(int(query.get("$top", ["100"])[0]), MAX_PAGE_SIZE)
        skip = int(query.get("$skiptoken", ["0"])[0])
        page = records[skip:skip + top]

        select = query.get("$select", [None])[0]
        results = []
        for record in page:
            serialized = _serialize(record)
            if select:
                wanted = {f.strip() for f in select.split(",")}
                results.append(
                    {k: v for k, v in serialized.items()
                     if k in wanted or k == "__metadata"}
                )
            else:
                results.append(serialized)

        body = {"d": {"results": results}}
        if skip + top < len(records):
            next_query = {
                k: v[0] for k, v in query.items() if k != "$skiptoken"
            }
            next_query["$skiptoken"] = str(skip + top)
            host = self.headers.get("Host", "localhost")
            body["d"]["__next"] = (
                f"http://{host}{split.path}?{urllib.parse.urlencode(next_query)}"
            )
        self._send(200, json.dumps(body))


def make_server(port=0):
    return ThreadingHTTPServer(("127.0.0.1", port), MockHandler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = make_server(args.port)
    print(f"mock SuccessFactors OData API on http://127.0.0.1:{server.server_address[1]}/odata/v2")
    print(f"credentials: {MOCK_USER} / {MOCK_PASS}")
    server.serve_forever()
