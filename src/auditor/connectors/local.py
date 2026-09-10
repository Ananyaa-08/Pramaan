"""Local filesystem connector — CSV / JSON / JSONL / Parquet (design §4.1)."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq

from auditor.connectors.base import CapabilityDescriptor, Connector
from auditor.connectors.errors import (
    EmptySourceError,
    MalformedRecordError,
    MalformedSourceError,
    RecordNotFoundError,
    SourceNotFoundError,
    UnsupportedFormatError,
)
from auditor.connectors.retry import with_retry

FormatName = Literal["csv", "json", "jsonl", "parquet"]

_EXTENSION_MAP: dict[str, FormatName] = {
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
    ".pq": "parquet",
}

_SCHEMA_SAMPLE_ROWS = 50


class LocalFilesConnector(Connector):
    """
    Connector for local tabular files.

    Reference implementation for later HF/GitHub adapters: capabilities are
    honest, list_records is a true generator, failures are typed.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        format: FormatName | str | None = None,
        id_column: str | None = "id",
    ) -> None:
        self._path = Path(path).expanduser().resolve()
        self._id_column = id_column
        self._format = self._resolve_format(format)
        self._validate_source_exists()
        self._estimated_count = self._compute_record_count()
        if self._estimated_count == 0:
            raise EmptySourceError(f"source is empty: {self._path}")

    # ------------------------------------------------------------------
    # Connector API
    # ------------------------------------------------------------------

    def get_capabilities(self) -> CapabilityDescriptor:
        # JSON arrays require a full stdlib parse — not streamed.
        # CSV / JSONL / Parquet stream row-by-row (or batch-then-yield).
        return CapabilityDescriptor(
            can_stream=self._format in {"csv", "jsonl", "parquet"},
            can_random_access=True,
            has_index_metadata=True,
            media_is_referenced_not_present=False,
            requires_auth=False,
            estimated_record_count=self._estimated_count,
        )

    def get_schema(self) -> dict[str, Any]:
        if self._format == "parquet":
            arrow_schema = self._parquet_file().schema_arrow
            schema = {field.name: str(field.type) for field in arrow_schema}
            return {"columns": schema, "_inference": "full"}

        sample: list[dict[str, Any]] = []
        for i, row in enumerate(self._iter_raw_records()):
            sample.append(row)
            if i + 1 >= _SCHEMA_SAMPLE_ROWS:
                break
        columns = _infer_types_from_sample(sample)
        return {
            "columns": columns,
            "_inference": "sample_head",
            "_sample_rows": len(sample),
        }

    def list_records(
        self, limit: int | None = None
    ) -> Iterator[dict[str, Any]]:
        count = 0
        for record in self._iter_raw_records():
            if limit is not None and count >= limit:
                return
            yield record
            count += 1

    def get_record(self, record_id: str) -> dict[str, Any]:
        # Row-index access: "0", "1", ...
        if self._id_column is None or record_id.isdigit():
            target = int(record_id)
            for idx, record in enumerate(self._iter_raw_records()):
                if idx == target:
                    return record
            raise RecordNotFoundError(record_id)

        for record in self._iter_raw_records():
            value = record.get(self._id_column)
            if value is not None and str(value) == record_id:
                return record
        raise RecordNotFoundError(record_id)

    def get_source_metadata(self) -> dict[str, Any]:
        stat = self._path.stat()
        content_hash = self._content_hash()
        return {
            "source_uri": str(self._path),
            "revision": f"mtime:{stat.st_mtime_ns}:sha256:{content_hash}",
            "provider": "local_files",
            "format": self._format,
        }

    # ------------------------------------------------------------------
    # Internal: format dispatch (lazy iterators)
    # ------------------------------------------------------------------

    def _iter_raw_records(self) -> Iterator[dict[str, Any]]:
        if self._format == "csv":
            yield from self._iter_csv()
        elif self._format == "jsonl":
            yield from self._iter_jsonl()
        elif self._format == "json":
            yield from self._iter_json()
        elif self._format == "parquet":
            yield from self._iter_parquet()
        else:  # pragma: no cover
            raise UnsupportedFormatError(self._format)

    def _iter_csv(self) -> Iterator[dict[str, Any]]:
        with self._open_text() as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise EmptySourceError(f"CSV has no header: {self._path}")
            for row in reader:
                yield {
                    key: _normalize_csv_value(value) for key, value in row.items()
                }

    def _iter_jsonl(self) -> Iterator[dict[str, Any]]:
        with self._open_text() as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise MalformedRecordError(
                        f"invalid JSONL at {self._path}:{line_no}: {exc}"
                    ) from exc
                if not isinstance(obj, dict):
                    raise MalformedRecordError(
                        f"JSONL record at {self._path}:{line_no} is not an object"
                    )
                yield obj

    def _iter_json(self) -> Iterator[dict[str, Any]]:
        # Stdlib cannot stream a JSON array without a full parse — capabilities
        # report can_stream=False for this format.
        try:
            payload = json.loads(self._read_text())
        except json.JSONDecodeError as exc:
            raise MalformedSourceError(
                f"invalid JSON file {self._path}: {exc}"
            ) from exc
        if not isinstance(payload, list):
            raise MalformedSourceError(
                f"JSON root must be an array of objects: {self._path}"
            )
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                raise MalformedRecordError(
                    f"JSON record at index {idx} is not an object"
                )
            yield item

    def _iter_parquet(self) -> Iterator[dict[str, Any]]:
        # Row-group / batch streaming via pyarrow — never loads the full table.
        pf = self._parquet_file()
        for batch in pf.iter_batches(batch_size=64):
            yield from batch.to_pylist()

    # ------------------------------------------------------------------
    # Internal: counts, opens, hashing
    # ------------------------------------------------------------------

    def _resolve_format(self, format_override: str | None) -> FormatName:
        if format_override is not None:
            normalized = format_override.lower().strip()
            if normalized not in {"csv", "json", "jsonl", "parquet"}:
                raise UnsupportedFormatError(f"unsupported format: {format_override}")
            return normalized  # type: ignore[return-value]
        ext = self._path.suffix.lower()
        if ext not in _EXTENSION_MAP:
            raise UnsupportedFormatError(
                f"cannot infer format from extension {ext!r} for {self._path}"
            )
        return _EXTENSION_MAP[ext]

    def _validate_source_exists(self) -> None:
        if not self._path.exists() or not self._path.is_file():
            raise SourceNotFoundError(f"source not found: {self._path}")

    def _compute_record_count(self) -> int:
        if self._format == "parquet":
            return int(self._parquet_file().metadata.num_rows)
        if self._format == "csv":
            with self._open_text() as handle:
                # Header + data rows; empty file already caught by size/header.
                lines = sum(1 for _ in handle)
            return max(0, lines - 1)
        if self._format == "jsonl":
            with self._open_text() as handle:
                return sum(1 for line in handle if line.strip())
        # json array
        try:
            payload = json.loads(self._read_text())
        except json.JSONDecodeError as exc:
            raise MalformedSourceError(
                f"invalid JSON file {self._path}: {exc}"
            ) from exc
        if not isinstance(payload, list):
            raise MalformedSourceError(
                f"JSON root must be an array of objects: {self._path}"
            )
        return len(payload)

    @with_retry
    def _open_text(self):
        """Open the source text file (retry wrapper proves the shared pattern)."""
        return self._path.open("r", encoding="utf-8", newline="")

    @with_retry
    def _read_text(self) -> str:
        return self._path.read_text(encoding="utf-8")

    @with_retry
    def _parquet_file(self) -> pq.ParquetFile:
        return pq.ParquetFile(self._path)

    @with_retry
    def _content_hash(self) -> str:
        digest = hashlib.sha256()
        with self._path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def _normalize_csv_value(value: str | None) -> Any:
    if value is None or value == "":
        return None
    return value


def _infer_types_from_sample(sample: list[dict[str, Any]]) -> dict[str, str]:
    if not sample:
        return {}
    columns: dict[str, str] = {}
    keys: set[str] = set()
    for row in sample:
        keys.update(row.keys())
    for key in sorted(keys):
        columns[key] = _infer_column_type([row.get(key) for row in sample])
    return columns


def _infer_column_type(values: list[Any]) -> str:
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "null"
    if all(isinstance(v, bool) for v in non_null):
        return "bool"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in non_null):
        return "int"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
        return "float"
    # CSV values arrive as strings — try numeric parse honesty.
    if all(isinstance(v, str) for v in non_null):
        if _all_int_strings(non_null):
            return "int"
        if _all_float_strings(non_null):
            return "float"
        return "str"
    return "mixed"


def _all_int_strings(values: list[Any]) -> bool:
    for value in values:
        try:
            int(value)
        except (TypeError, ValueError):
            return False
    return True


def _all_float_strings(values: list[Any]) -> bool:
    for value in values:
        try:
            float(value)
        except (TypeError, ValueError):
            return False
    return True
