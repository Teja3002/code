"""Storage backends for the raw zone and curated layer.

Raw zone:      raw/<entity>/dt=YYYY-MM-DD/<run-timestamp>.json  (append-only)
Curated layer: curated/<entity>.csv  (latest state per business key, upserted
               on every run so delta extractions keep it current)
"""

import csv
import io
import json
import os
from datetime import datetime, timezone


def _run_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _to_csv(rows):
    fieldnames = sorted({k for row in rows for k in row})
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _from_csv(text):
    return [dict(row) for row in csv.DictReader(io.StringIO(text))]


def merge_curated(existing, new_records, key):
    """Upsert new records into the curated snapshot by business key."""
    merged = {row[key]: row for row in existing if key in row}
    for record in new_records:
        merged[str(record[key])] = {k: str(v) if v is not None else "" for k, v in record.items()}
    return list(merged.values())


class LocalStorage:
    def __init__(self, root):
        self.root = root

    def _path(self, relative):
        path = os.path.join(self.root, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def write_raw(self, entity, records):
        ts = _run_timestamp()
        relative = f"raw/{entity}/dt={ts[:10]}/{ts}.json"
        with open(self._path(relative), "w") as f:
            json.dump(records, f, indent=2)
        return relative

    def read_curated(self, entity):
        path = os.path.join(self.root, f"curated/{entity}.csv")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return _from_csv(f.read())

    def write_curated(self, entity, rows):
        with open(self._path(f"curated/{entity}.csv"), "w") as f:
            f.write(_to_csv(rows))


class S3Storage:
    def __init__(self, bucket, boto3_session=None):
        import boto3
        session = boto3_session or boto3
        self.bucket = bucket
        self.client = session.client("s3")

    def write_raw(self, entity, records):
        ts = _run_timestamp()
        key = f"raw/{entity}/dt={ts[:10]}/{ts}.json"
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(records).encode(),
            ContentType="application/json",
        )
        return key

    def read_curated(self, entity):
        try:
            obj = self.client.get_object(
                Bucket=self.bucket, Key=f"curated/{entity}.csv"
            )
            return _from_csv(obj["Body"].read().decode())
        except self.client.exceptions.NoSuchKey:
            return []

    def write_curated(self, entity, rows):
        self.client.put_object(
            Bucket=self.bucket,
            Key=f"curated/{entity}.csv",
            Body=_to_csv(rows).encode(),
            ContentType="text/csv",
        )
