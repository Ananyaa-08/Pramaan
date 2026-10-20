"""Tests for LocalFilesConnector (design §4.1 — local CSV/JSONL/Parquet)."""

from __future__ import annotations

import hashlib
import json
import os
import re
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


def test_list_records_lazy_csv(csv_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_5_5_local_metadata_revision_matches_get_revision(csv_path: Path) -> None:
    connector = LocalFilesConnector(csv_path)
    assert connector.get_source_metadata()["revision"] == connector.get_revision()


def test_5_5_local_metadata_delegates_revision(
    csv_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connector = LocalFilesConnector(csv_path)
    sentinel = "sha256:delegated-local-revision"
    monkeypatch.setattr(connector, "get_revision", lambda: sentinel)
    assert connector.get_source_metadata()["revision"] == sentinel


def test_5_5_local_revision_has_exact_content_hash_format(tmp_path: Path) -> None:
    content = b"id,label\nr1,alpha\nr2,beta\n"
    path = tmp_path / "content.csv"
    path.write_bytes(content)
    expected = "sha256:" + hashlib.sha256(content).hexdigest()
    revision = LocalFilesConnector(path).get_revision()
    assert revision == expected
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", revision) is not None


def test_5_5_local_revision_ignores_mtime_changes(csv_path: Path) -> None:
    first = LocalFilesConnector(csv_path)
    original_revision = first.get_source_metadata()["revision"]
    original_bytes = csv_path.read_bytes()
    original_stat = os.stat(csv_path)
    os.utime(
        csv_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 10_000_000_000),
    )
    assert os.stat(csv_path).st_mtime_ns != original_stat.st_mtime_ns
    assert csv_path.read_bytes() == original_bytes
    second = LocalFilesConnector(csv_path)
    assert second.get_source_metadata()["revision"] == original_revision


def test_5_5_local_metadata_exposes_mtime_ns(csv_path: Path) -> None:
    connector = LocalFilesConnector(csv_path)
    metadata = connector.get_source_metadata()
    stat = os.stat(csv_path)
    assert "mtime_ns" in metadata
    assert type(metadata["mtime_ns"]) is int
    assert metadata["mtime_ns"] == stat.st_mtime_ns


def test_5_5_local_revision_is_computed_once(
    csv_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = "sha256:" + hashlib.sha256(csv_path.read_bytes()).hexdigest()
    original_sha256 = hashlib.sha256
    hash_calls = 0

    def counting_sha256(*args: Any, **kwargs: Any) -> Any:
        nonlocal hash_calls
        hash_calls += 1
        return original_sha256(*args, **kwargs)

    monkeypatch.setattr("auditor.connectors.local.hashlib.sha256", counting_sha256)
    connector = LocalFilesConnector(csv_path)
    for _ in range(3):
        assert connector.get_revision() == expected
        assert connector.get_source_metadata()["revision"] == expected
    assert hash_calls == 1


def test_5_5_local_revision_is_pinned_for_instance_lifetime(tmp_path: Path) -> None:
    path = tmp_path / "pinned.csv"
    content_a = b"id,label\nr1,alpha\n"
    content_b = b"id,label\nr1,beta\nr2,gamma\n"
    path.write_bytes(content_a)
    first = LocalFilesConnector(path)
    original_revision = first.get_source_metadata()["revision"]
    path.write_bytes(content_b)
    assert first.get_revision() == original_revision
    assert first.get_source_metadata()["revision"] == original_revision
    second = LocalFilesConnector(path)
    expected_b = "sha256:" + hashlib.sha256(content_b).hexdigest()
    assert second.get_revision() == expected_b
    assert second.get_source_metadata()["revision"] == expected_b
    assert second.get_revision() != original_revision


def _write_schema_records(
    tmp_path: Path, suffix: str, records: list[dict[str, Any]]
) -> Path:
    path = tmp_path / f"schema{suffix}"
    if suffix == ".json":
        content = json.dumps(records)
    else:
        content = "\n".join(json.dumps(record) for record in records) + "\n"
    path.write_text(content, encoding="utf-8")
    return path


def _assert_two_row_records_and_schema(
    path: Path, records: list[dict[str, Any]], columns: dict[str, str]
) -> None:
    connector = LocalFilesConnector(path)
    assert list(connector.list_records()) == records
    assert connector.get_schema() == {
        "columns": columns,
        "_inference": "sample_head",
        "_sample_rows": 2,
    }


def _numeric_string_records() -> list[dict[str, Any]]:
    return [
        {"integer": "007", "decimal": "1.5", "exponent": "1e3"},
        {"integer": "12", "decimal": "2.75", "exponent": "2E-2"},
    ]


def _native_scalar_records() -> list[dict[str, Any]]:
    return [
        {"integer": 7, "floating": 1.5, "boolean": True, "text": "alpha"},
        {"integer": 12, "floating": 2.75, "boolean": False, "text": "beta"},
    ]


def _nullable_records() -> list[dict[str, Any]]:
    return [
        {"text": "ordinary", "integer": 7, "boolean": False, "empty": None},
        {"text": None, "integer": None, "boolean": None, "empty": None},
    ]


def _mixed_and_nested_records() -> list[dict[str, Any]]:
    return [
        {"mixed": "ordinary", "bool_number": True, "array": [1], "object": {"x": 1}},
        {"mixed": 2, "bool_number": 1, "array": ["two"], "object": {"x": "two"}},
    ]


def test_6_3_json_numeric_looking_strings_remain_strings(tmp_path: Path) -> None:
    records = _numeric_string_records()
    path = _write_schema_records(tmp_path, ".json", records)
    _assert_two_row_records_and_schema(
        path, records, {"integer": "str", "decimal": "str", "exponent": "str"}
    )


def test_6_3_jsonl_numeric_looking_strings_remain_strings(tmp_path: Path) -> None:
    records = _numeric_string_records()
    path = _write_schema_records(tmp_path, ".jsonl", records)
    _assert_two_row_records_and_schema(
        path, records, {"integer": "str", "decimal": "str", "exponent": "str"}
    )


def test_6_3_ndjson_numeric_looking_strings_remain_strings(tmp_path: Path) -> None:
    records = _numeric_string_records()
    path = _write_schema_records(tmp_path, ".ndjson", records)
    _assert_two_row_records_and_schema(
        path, records, {"integer": "str", "decimal": "str", "exponent": "str"}
    )


def test_6_3_json_native_scalar_types_are_preserved(tmp_path: Path) -> None:
    records = _native_scalar_records()
    path = _write_schema_records(tmp_path, ".json", records)
    _assert_two_row_records_and_schema(
        path,
        records,
        {"integer": "int", "floating": "float", "boolean": "bool", "text": "str"},
    )


def test_6_3_jsonl_native_scalar_types_are_preserved(tmp_path: Path) -> None:
    records = _native_scalar_records()
    path = _write_schema_records(tmp_path, ".jsonl", records)
    _assert_two_row_records_and_schema(
        path,
        records,
        {"integer": "int", "floating": "float", "boolean": "bool", "text": "str"},
    )


def test_6_3_json_integer_float_mixture_is_float(tmp_path: Path) -> None:
    records = [{"number": 1}, {"number": 1.5}]
    path = _write_schema_records(tmp_path, ".json", records)
    _assert_two_row_records_and_schema(path, records, {"number": "float"})


def test_6_3_jsonl_integer_float_mixture_is_float(tmp_path: Path) -> None:
    records = [{"number": 1}, {"number": 1.5}]
    path = _write_schema_records(tmp_path, ".jsonl", records)
    _assert_two_row_records_and_schema(path, records, {"number": "float"})


def test_6_3_json_nulls_do_not_create_mixed_types(tmp_path: Path) -> None:
    records = _nullable_records()
    path = _write_schema_records(tmp_path, ".json", records)
    _assert_two_row_records_and_schema(
        path,
        records,
        {"text": "str", "integer": "int", "boolean": "bool", "empty": "null"},
    )


def test_6_3_jsonl_nulls_do_not_create_mixed_types(tmp_path: Path) -> None:
    records = _nullable_records()
    path = _write_schema_records(tmp_path, ".jsonl", records)
    _assert_two_row_records_and_schema(
        path,
        records,
        {"text": "str", "integer": "int", "boolean": "bool", "empty": "null"},
    )


def test_6_3_csv_numeric_string_inference_is_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "schema.csv"
    path.write_text(
        "integer,floating,text,boolean,empty\n"
        "007,1.5,alpha,true,\n"
        "12,2.75,beta,false,\n",
        encoding="utf-8",
    )
    records = [
        {
            "integer": "007",
            "floating": "1.5",
            "text": "alpha",
            "boolean": "true",
            "empty": None,
        },
        {
            "integer": "12",
            "floating": "2.75",
            "text": "beta",
            "boolean": "false",
            "empty": None,
        },
    ]
    _assert_two_row_records_and_schema(
        path,
        records,
        {
            "integer": "int",
            "floating": "float",
            "text": "str",
            "boolean": "str",
            "empty": "null",
        },
    )


def test_6_3_json_mixed_and_nested_classification_is_unchanged(tmp_path: Path) -> None:
    records = _mixed_and_nested_records()
    path = _write_schema_records(tmp_path, ".json", records)
    _assert_two_row_records_and_schema(
        path,
        records,
        {"mixed": "mixed", "bool_number": "mixed", "array": "mixed", "object": "mixed"},
    )


def test_6_3_jsonl_mixed_and_nested_classification_is_unchanged(tmp_path: Path) -> None:
    records = _mixed_and_nested_records()
    path = _write_schema_records(tmp_path, ".jsonl", records)
    _assert_two_row_records_and_schema(
        path,
        records,
        {"mixed": "mixed", "bool_number": "mixed", "array": "mixed", "object": "mixed"},
    )


def _assert_real_schema_sample_consumption(
    connector: LocalFilesConnector,
    monkeypatch: pytest.MonkeyPatch,
    read_boundary: str,
) -> None:
    original_records = connector._iter_raw_records
    original_read = getattr(connector, read_boundary)
    counts = {"iterators": 0, "yielded": 0, "reads": 0}

    def guarded_iterator(actual: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for record in actual:
            counts["yielded"] += 1
            assert counts["yielded"] <= 50, "schema consumed a 51st sample record"
            yield record

    def counted_records() -> Iterator[dict[str, Any]]:
        counts["iterators"] += 1
        assert counts["iterators"] == 1, "schema requested a second raw-record iterator"
        return guarded_iterator(original_records())

    def counted_read() -> Any:
        counts["reads"] += 1
        return original_read()

    monkeypatch.setattr(connector, "_iter_raw_records", counted_records)
    monkeypatch.setattr(connector, read_boundary, counted_read)
    schema = connector.get_schema()
    assert counts == {"iterators": 1, "yielded": 50, "reads": 1}
    assert schema == {
        "columns": {"number": "int"},
        "_inference": "sample_head",
        "_sample_rows": 50,
    }


def test_6_3_json_schema_sample_consumption_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [{"number": number} for number in range(50)] + [{"number": "conflict"}]
    path = _write_schema_records(tmp_path, ".json", records)
    connector = LocalFilesConnector(path)
    _assert_real_schema_sample_consumption(connector, monkeypatch, "_read_text")


def test_6_3_jsonl_schema_sample_consumption_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [{"number": number} for number in range(50)] + [{"number": "conflict"}]
    path = _write_schema_records(tmp_path, ".jsonl", records)
    connector = LocalFilesConnector(path)
    _assert_real_schema_sample_consumption(connector, monkeypatch, "_open_text")
