"""Natural-language Q&A over the curated HR data using Amazon Bedrock.

Invoke with: {"question": "How many people work in Engineering in Berlin?"}

Environment variables:
    BUCKET     S3 data-lake bucket (reads curated/<entity>.csv)
    MODEL_ID   Bedrock model id (default: Claude 3 Haiku)
    ENTITIES   optional comma-separated subset to load as context
"""

import json
import logging
import os

import boto3

logging.getLogger().setLevel(logging.INFO)

DEFAULT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
DEFAULT_ENTITIES = ["EmpJob", "FODepartment", "PerPersonal"]
MAX_CHARS_PER_ENTITY = 15000

# Responsible-AI guardrail: this demo dataset holds no compensation or
# government-ID data, and questions seeking it are refused outright.
BLOCKED_TERMS = [
    "salary", "compensation", "pay", "wage", "ssn", "social security",
    "bank", "passport", "medical", "disability",
]

SYSTEM_PROMPT = (
    "You are an HR data analyst assistant. Answer questions using ONLY the "
    "CSV data provided in the user message. If the data cannot answer the "
    "question, say so explicitly instead of guessing. Never reveal or infer "
    "sensitive personal attributes. Keep answers concise and, when helpful, "
    "cite the rows or values the answer is based on."
)


def _load_context(bucket, entities):
    s3 = boto3.client("s3")
    sections = []
    for entity in entities:
        try:
            obj = s3.get_object(Bucket=bucket, Key=f"curated/{entity}.csv")
        except s3.exceptions.NoSuchKey:
            continue
        text = obj["Body"].read().decode()[:MAX_CHARS_PER_ENTITY]
        sections.append(f"### {entity}\n{text}")
    return "\n\n".join(sections)


def handler(event, context):
    question = (event or {}).get("question", "").strip()
    if not question:
        return {"error": "provide a 'question' field"}

    lowered = question.lower()
    blocked = [term for term in BLOCKED_TERMS if term in lowered]
    if blocked:
        return {
            "refused": True,
            "reason": f"question touches restricted attributes: {', '.join(blocked)}",
        }

    entities = os.environ.get("ENTITIES", ",".join(DEFAULT_ENTITIES)).split(",")
    data = _load_context(os.environ["BUCKET"], [e.strip() for e in entities])
    if not data:
        return {"error": "no curated data found; run the extraction first"}

    bedrock = boto3.client("bedrock-runtime")
    response = bedrock.converse(
        modelId=os.environ.get("MODEL_ID", DEFAULT_MODEL),
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{
            "role": "user",
            "content": [{"text": f"HR data:\n\n{data}\n\nQuestion: {question}"}],
        }],
        inferenceConfig={"maxTokens": 500, "temperature": 0},
    )
    answer = response["output"]["message"]["content"][0]["text"]
    usage = response.get("usage", {})
    logging.info("tokens in=%s out=%s", usage.get("inputTokens"), usage.get("outputTokens"))
    return {"question": question, "answer": answer, "usage": usage}
