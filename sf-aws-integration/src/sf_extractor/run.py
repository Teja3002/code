"""CLI for running the extraction locally (Phase 1 / development).

Usage:
    export SF_USERNAME="USER@COMPANYID" SF_PASSWORD="..."
    python -m sf_extractor.run --base-url https://apisalesdemo2.successfactors.eu/odata/v2 \
        --output ./data --state ./state.json
"""

import argparse
import json
import logging
import os
import sys

from .extractor import run_extraction
from .odata_client import ODataClient
from .state import LocalFileState
from .storage import LocalStorage


def main():
    parser = argparse.ArgumentParser(description="SuccessFactors OData extractor")
    parser.add_argument("--base-url", required=True, help="OData V2 base URL")
    parser.add_argument("--output", default="./data", help="local data lake root")
    parser.add_argument("--state", default="./state.json", help="watermark state file")
    parser.add_argument("--entities", help="comma-separated subset of entities")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--full", action="store_true", help="ignore watermarks (full extract)")
    args = parser.parse_args()

    username = os.environ.get("SF_USERNAME")
    password = os.environ.get("SF_PASSWORD")
    if not username or not password:
        sys.exit("set SF_USERNAME and SF_PASSWORD environment variables")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    client = ODataClient(args.base_url, username, password, page_size=args.page_size)
    summary = run_extraction(
        client,
        LocalStorage(args.output),
        LocalFileState(args.state),
        entities=args.entities.split(",") if args.entities else None,
        full=args.full,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
