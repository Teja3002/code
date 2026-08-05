"""Orchestrates one extraction run across all configured entities."""

import json
import logging

from .entities import ENTITIES
from .storage import merge_curated

logger = logging.getLogger(__name__)


def run_extraction(client, storage, state, entities=None, full=False):
    """Extract each entity incrementally, land raw JSON, upsert curated CSV.

    Returns a per-entity summary dict (records extracted, new watermark).
    """
    summary = {}
    names = entities or list(ENTITIES)
    for name in names:
        config = ENTITIES[name]
        watermark = None if full else state.get(name)
        records = list(
            client.fetch_entity(name, select=config["select"], modified_after=watermark)
        )
        entry = {"records": len(records), "previous_watermark": watermark}
        if records:
            raw_key = storage.write_raw(name, records)
            curated = merge_curated(
                storage.read_curated(name), records, config["key"]
            )
            storage.write_curated(name, curated)
            new_watermark = max(
                r["lastModifiedDateTime"] for r in records if "lastModifiedDateTime" in r
            )
            state.set(name, new_watermark)
            entry.update(
                raw_key=raw_key,
                curated_rows=len(curated),
                new_watermark=new_watermark,
            )
        summary[name] = entry
        logger.info("extracted %s: %s", name, json.dumps(entry))
    return summary
