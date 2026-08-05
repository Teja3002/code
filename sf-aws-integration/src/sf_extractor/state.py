"""Sync-state (watermark) stores.

The watermark per entity is the highest lastModifiedDateTime seen so far,
in ISO-8601. The next run filters on it so only changed records are pulled.
"""

import json
import os


class LocalFileState:
    def __init__(self, path):
        self.path = path

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return {}

    def get(self, entity):
        return self._load().get(entity)

    def set(self, entity, watermark):
        data = self._load()
        data[entity] = watermark
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)


class DynamoDBState:
    def __init__(self, table_name, boto3_session=None):
        import boto3
        session = boto3_session or boto3
        self.table = session.resource("dynamodb").Table(table_name)

    def get(self, entity):
        item = self.table.get_item(Key={"entity_name": entity}).get("Item")
        return item["watermark"] if item else None

    def set(self, entity, watermark):
        self.table.put_item(Item={"entity_name": entity, "watermark": watermark})
