import json
import os
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "mock_sf"))

from data import DATASET, EMPLOYEE_COUNT  # noqa: E402
from server import make_server  # noqa: E402

from sf_extractor.extractor import run_extraction  # noqa: E402
from sf_extractor.odata_client import ODataClient, ODataError, parse_sf_date  # noqa: E402
from sf_extractor.state import LocalFileState  # noqa: E402
from sf_extractor.storage import LocalStorage, merge_curated  # noqa: E402


@pytest.fixture(scope="module")
def base_url():
    server = make_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/odata/v2"
    server.shutdown()


@pytest.fixture
def client(base_url):
    return ODataClient(base_url, "demo@DEMO", "demo", page_size=100)


def test_parse_sf_date():
    assert parse_sf_date("/Date(1750000000000)/") == "2025-06-15T15:06:40Z"
    assert parse_sf_date("/Date(1750000000000+0000)/") == "2025-06-15T15:06:40Z"
    assert parse_sf_date("plain string") == "plain string"
    assert parse_sf_date(42) == 42


def test_auth_required(base_url):
    bad = ODataClient(base_url, "wrong", "creds")
    with pytest.raises(ODataError, match="401"):
        list(bad.fetch_entity("PerPerson"))


def test_pagination_returns_all_records(base_url):
    paged = ODataClient(base_url, "demo@DEMO", "demo", page_size=17)
    records = list(paged.fetch_entity("EmpJob"))
    assert len(records) == EMPLOYEE_COUNT
    assert len({r["userId"] for r in records}) == EMPLOYEE_COUNT


def test_select_projects_fields(client):
    record = next(iter(client.fetch_entity("EmpJob", select="userId,jobTitle")))
    assert set(record) == {"userId", "jobTitle"}


def test_records_are_normalized(client):
    record = next(iter(client.fetch_entity("EmpJob")))
    assert "__metadata" not in record
    assert record["lastModifiedDateTime"].endswith("Z")


def test_delta_filter(client):
    everything = list(client.fetch_entity("EmpJob"))
    watermarks = sorted(r["lastModifiedDateTime"] for r in everything)
    midpoint = watermarks[len(watermarks) // 2]
    delta = list(client.fetch_entity("EmpJob", modified_after=midpoint))
    expected = [r for r in everything if r["lastModifiedDateTime"] > midpoint]
    assert len(delta) == len(expected)
    assert 0 < len(delta) < len(everything)


def test_metadata_lists_entities(client):
    xml = client.fetch_metadata()
    for entity in DATASET:
        assert entity in xml


def test_merge_curated_upserts():
    existing = [{"userId": "U1", "jobTitle": "Engineer"}]
    new = [
        {"userId": "U1", "jobTitle": "Senior Engineer"},
        {"userId": "U2", "jobTitle": "Analyst"},
    ]
    merged = {row["userId"]: row for row in merge_curated(existing, new, "userId")}
    assert merged["U1"]["jobTitle"] == "Senior Engineer"
    assert merged["U2"]["jobTitle"] == "Analyst"


def test_end_to_end_full_then_delta(client, tmp_path):
    storage = LocalStorage(str(tmp_path / "lake"))
    state = LocalFileState(str(tmp_path / "state.json"))

    first = run_extraction(client, storage, state)
    assert first["EmpJob"]["records"] == EMPLOYEE_COUNT
    assert first["EmpJob"]["curated_rows"] == EMPLOYEE_COUNT
    assert (tmp_path / "lake" / "curated" / "EmpJob.csv").exists()

    with open(tmp_path / "state.json") as f:
        watermarks = json.load(f)
    assert set(watermarks) == set(first)

    second = run_extraction(client, storage, state)
    assert all(entry["records"] == 0 for entry in second.values())

    curated = storage.read_curated("EmpJob")
    assert len(curated) == EMPLOYEE_COUNT
