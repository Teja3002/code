"""AWS Lambda entry point for the scheduled extraction.

Environment variables:
    SF_BASE_URL   OData V2 base URL
    SECRET_ARN    Secrets Manager secret with {"username": ..., "password": ...}
    BUCKET        S3 data-lake bucket
    STATE_TABLE   DynamoDB sync-state table
    ENTITIES      optional comma-separated subset
"""

import json
import logging
import os

import boto3

from sf_extractor.extractor import run_extraction
from sf_extractor.odata_client import ODataClient
from sf_extractor.state import DynamoDBState
from sf_extractor.storage import S3Storage

logging.getLogger().setLevel(logging.INFO)


def _credentials():
    secret_arn = os.environ["SECRET_ARN"]
    response = boto3.client("secretsmanager").get_secret_value(SecretId=secret_arn)
    secret = json.loads(response["SecretString"])
    return secret["username"], secret["password"]


def handler(event, context):
    username, password = _credentials()
    client = ODataClient(os.environ["SF_BASE_URL"], username, password)
    entities_env = os.environ.get("ENTITIES")
    summary = run_extraction(
        client,
        S3Storage(os.environ["BUCKET"]),
        DynamoDBState(os.environ["STATE_TABLE"]),
        entities=entities_env.split(",") if entities_env else None,
        full=bool(event.get("full")) if isinstance(event, dict) else False,
    )
    logging.info("run summary: %s", json.dumps(summary))
    return summary
