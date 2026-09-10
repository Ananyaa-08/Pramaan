"""Tests for LocalFilesConnector (design §4.1 — local CSV/JSONL/Parquet)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from auditor.connectors.base import Connector
from auditor.connectors.errors import (
    EmptySourceError,
    MalformedRecordError,
    RecordNotFoundError,
    SourceNotFoundError,
)
from auditor.connectors.local import LocalFilesConnector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    """CSV with a missing value and a type-mismatched numeric field as text."""
    path = tmp_path / "sample.csv"
    path.write_text(
        "id,label,score\n"
        "r1,hello,1.5\n"
        "r2,,2.0\n"  # missing label
        "r3,sign,not_a_number\n"  # type mismatch in score
        "r4,world,4.0\n"
        "r5,ok,5.0\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def jsonl_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample.jsonl"
    rows = [
        {"id": "r1", "label": "hello", "score": 1.5},
        {"id": "r2", "label": None, "score": 2.0},
        {"id": "r3", "label": "sign", "score": "not_a_number"},
        {"id": "r4", "label": "world", "score": 4.0},
        {"id": "r5", "label": "ok", "score": 5.0},
    ]
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def parquet_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample.parquet"
    table = pa.table(
        {
            "id": ["r1", "r2", "r3", "r4", "r5"],
            "label": ["hello", None, "sign", "world", "ok"],
            "score": [1.5, 2.0, None, 4.0, 5.0],
        }
    )
    pq.write_table(table, path)
    return path


@pytest.fixture
def malformed_jsonl_path(tmp_path: Path) -> Path:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "r1"}\nNOT_JSON\n{"id": "r3"}\n', encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Interface + capabilities
# ---------------------------------------------------------------------------


def test_local_files_connector_is_connector(csv_path: Path) -> None:
    conn = LocalFilesConnector(csv_path)
    assert isinstance(conn, Connector)


@pytest.mark.parametrize(
    ("fixture_name", "expected_stream", "expected_format"),
    [
        ("csv_path", True, "csv"),
        ("jsonl_path", True, "jsonl"),
        ("parquet_path", True, "parquet"),
    ],
)
def test_capabilities_reflect_format_reality(
    fixture_name: str,
    expected_stream: bool,
    expected_format: str,
    request: pytest.FixtureRequest,
) -> None:
    path: Path = request.getfixturevalue(fixture_name)
    conn = LocalFilesConnector(path)
    caps = conn.get_capabilities()

    assert caps.can_stream is expected_stream
    assert caps.can_random_access is True
    assert caps.has_index_metadata is True
    assert caps.media_is_referenced_not_present is False
    assert caps.requires_auth is False
    assert caps.estimated_record_count == 5
    assert conn.get_source_metadata()["format"] == expected_format


def test_estimated_record_count_matches_parquet_metadata(
    parquet_path: Path,
) -> None:
    conn = LocalFilesConnector(parquet_path)
    pf = pq.ParquetFile(parquet_path)
    assert conn.get_capabilities().estimated_record_count == pf.metadata.num_rows


# ---------------------------------------------------------------------------
# Laziness per format (production-counter pattern)
# ---------------------------------------------------------------------------


def _assert_lazy_list_records(
    conn: LocalFilesConnector, monkeypatch: pytest.MonkeyPatch
) -> None:
    produced = {"count": 0}
    original = conn._iter_raw_records

    def counting() -> Iterator[dict[str, Any]]:
        for record in original():
            produced["count"] += 1
            yield record

    monkeypatch.setattr(conn, "_iter_raw_records", counting)

    assert produced["count"] == 0
    iterator = conn.list_records()
    assert isinstance(iterator, Iterator)
    assert not isinstance(iterator, list)
    assert produced["count"] == 0

    first = next(iterator)
    assert produced["count"] == 1
    assert "id" in first

    second = next(iterator)
    assert produced["count"] == 2
    assert second["id"] != first["id"]


def test_list_records_lazy_csv(
    csv_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_lazy_list_records(LocalFilesConnector(csv_path), monkeypatch)


def test_list_records_lazy_jsonl(
    jsonl_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_lazy_list_records(LocalFilesConnector(jsonl_path), monkeypatch)


def test_list_records_lazy_parquet(
    parquet_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_lazy_list_records(LocalFilesConnector(parquet_path), monkeypatch)


def test_list_records_respects_limit(csv_path: Path) -> None:
    conn = LocalFilesConnector(csv_path)
    rows = list(conn.list_records(limit=2))
    assert len(rows) == 2
    assert rows[0]["id"] == "r1"


# ---------------------------------------------------------------------------
# get_record / schema / metadata
# ---------------------------------------------------------------------------


def test_get_record_by_id(csv_path: Path) -> None:
    conn = LocalFilesConnector(csv_path, id_column="id")
    row = conn.get_record("r4")
    assert row["label"] == "world"


def test_get_record_missing_raises_typed_error(csv_path: Path) -> None:
    conn = LocalFilesConnector(csv_path, id_column="id")
    with pytest.raises(RecordNotFoundError):
        conn.get_record("missing-id")


def test_get_schema_csv_notes_partial_inference(csv_path: Path) -> None:
    schema = LocalFilesConnector(csv_path).get_schema()
    assert "id" in schema or "columns" in schema
    # Honest note that CSV/JSONL schema may be sample-inferred.
    inference = schema.get("_inference") or schema.get("inference")
    assert inference in {"sample_head", "partial", "full"}


def test_get_schema_parquet_is_complete(parquet_path: Path) -> None:
    schema = LocalFilesConnector(parquet_path).get_schema()
    inference = schema.get("_inference") or schema.get("inference")
    assert inference == "full"
    columns = schema.get("columns", schema)
    assert "id" in columns
    assert "score" in columns


def test_revision_changes_when_file_content_changes(tmp_path: Path) -> None:
    path = tmp_path / "rev.csv"
    path.write_text("id,label\nr1,a\n", encoding="utf-8")
    conn = LocalFilesConnector(path)
    rev1 = conn.get_source_metadata()["revision"]

    path.write_text("id,label\nr1,a\nr2,b\n", encoding="utf-8")
    # New connector instance (or same path) must observe new content hash.
    conn2 = LocalFilesConnector(path)
    rev2 = conn2.get_source_metadata()["revision"]

    assert rev1 != rev2


def test_source_metadata_has_absolute_uri_and_provider(csv_path: Path) -> None:
    meta = LocalFilesConnector(csv_path).get_source_metadata()
    assert Path(meta["source_uri"]).is_absolute()
    assert meta["provider"] == "local_files"
    assert "revision" in meta


# ---------------------------------------------------------------------------
# Typed failures — never silent / fabricated
# ---------------------------------------------------------------------------


def test_missing_file_raises_on_construction(tmp_path: Path) -> None:
    with pytest.raises(SourceNotFoundError):
        LocalFilesConnector(tmp_path / "does_not_exist.csv")


def test_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(EmptySourceError):
        LocalFilesConnector(path)


def test_malformed_jsonl_raises_typed_error(
    malformed_jsonl_path: Path,
) -> None:
    conn = LocalFilesConnector(malformed_jsonl_path)
    with pytest.raises(MalformedRecordError):
        list(conn.list_records())


def test_explicit_format_override(tmp_path: Path) -> None:
    path = tmp_path / "data.dat"
    path.write_text("id,label\nr1,a\nr2,b\n", encoding="utf-8")
    conn = LocalFilesConnector(path, format="csv")
    assert conn.get_capabilities().estimated_record_count == 2
    assert list(conn.list_records())[0]["id"] == "r1"
