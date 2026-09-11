"""GitHub REST API connector for repository-hosted datasets."""

from __future__ import annotations

import base64
import os
import re
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

import httpx

from auditor.connectors.base import CapabilityDescriptor, Connector
from auditor.connectors.errors import (
    ConnectorError,
    MalformedSourceError,
    RecordNotFoundError,
    SourceNotFoundError,
)
from auditor.connectors.retry import with_retry

_API_ROOT = "https://api.github.com"
_DATA_EXTENSIONS = {
    ".arrow",
    ".avro",
    ".csv",
    ".db",
    ".json",
    ".jsonl",
    ".ndjson",
    ".parquet",
    ".pq",
    ".sqlite",
    ".tsv",
}
_MEDIA_EXTENSIONS = {
    ".aac",
    ".avi",
    ".bmp",
    ".flac",
    ".flv",
    ".gif",
    ".jpeg",
    ".jpg",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ogg",
    ".png",
    ".tif",
    ".tiff",
    ".wav",
    ".webm",
    ".webp",
}
_DOCUMENT_EXTENSIONS = {".md", ".markdown", ".rst", ".txt"}
_LFS_POINTER_MAX_BYTES = 1_024
_LFS_POINTER_CHECK_LIMIT = 20
_DOC_SCAN_MAX_BYTES = 256 * 1_024
_DOC_SCAN_LIMIT = 5
_LFS_HEADER = "version https://git-lfs.github.com/spec/v1"
_URL_REFERENCE_RE = re.compile(
    r"https?://[^\s<>()\[\]{}]+(?:"
    + "|".join(
        re.escape(extension)
        for extension in sorted(_DATA_EXTENSIONS | _MEDIA_EXTENSIONS)
    )
    + r")(?:\?[^\s<>()\[\]{}]*)?(?:#[^\s<>()\[\]{}]*)?",
    re.IGNORECASE,
)


class GitHubConnector(Connector):
    """
    Treat data/media files in a GitHub repository as dataset records.

    A record is one tracked blob whose extension identifies data or media.
    Documentation and code are excluded from ``list_records``, but any tracked
    file (including a README or dataset card) can be read with ``get_record``.
    GitHub's recursive tree response is materialized by the REST API, so
    ``list_records`` is a lazy Python generator but ``can_stream`` is honestly
    false. File contents are never opened to infer the record schema.

    Missing-media detection is structural first. A non-empty tree containing
    only documentation blobs and no data/media blob is reported as referenced
    but not present. If auxiliary code is also present, at most the first five
    documentation files of at most 256 KiB are inspected; the only accepted
    textual signal is an HTTP(S) URL whose path ends in one of the explicit
    data/media extensions in this module (query strings and fragments may
    follow). No natural-language keyword heuristic is used.

    Git LFS is a secondary per-file confirmation. Only the first 20
    media-looking blobs whose tree-reported size is at most 1 KiB are fetched.
    A file is considered an LFS pointer only when its decoded content begins
    with ``version https://git-lfs.github.com/spec/v1``. Larger media blobs are
    considered present using tree metadata and are never fetched during
    capability discovery. A confirmed pointer is excluded from records and
    makes ``media_is_referenced_not_present`` true.
    """

    def __init__(
        self,
        repository: str,
        *,
        ref: str | None = None,
        token: str | None = None,
        client: httpx.Client | Any | None = None,
    ) -> None:
        parts = repository.strip().strip("/").split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError("repository must have the form 'owner/name'")
        self._repository = "/".join(parts)
        self._token = token or os.getenv("GITHUB_TOKEN")
        self._client = client or httpx.Client(timeout=30.0)
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token is not None:
            self._headers["Authorization"] = f"Bearer {self._token}"

        repo_info = self._request_json(f"/repos/{self._repository}")
        self._requires_auth = bool(repo_info.get("private", False))
        self._source_uri = repo_info.get(
            "html_url", f"https://github.com/{self._repository}"
        )
        self._requested_ref = ref or repo_info.get("default_branch")
        if not isinstance(self._requested_ref, str) or not self._requested_ref:
            raise MalformedSourceError(
                f"GitHub did not report a default branch for {self._repository}"
            )

        commit = self._request_json(
            f"/repos/{self._repository}/commits/{quote(self._requested_ref, safe='')}"
        )
        revision = commit.get("sha")
        if not isinstance(revision, str) or not revision:
            raise MalformedSourceError(
                f"GitHub did not return a commit SHA for {self._repository}"
            )
        self._revision = revision

        tree_payload = self._request_json(
            f"/repos/{self._repository}/git/trees/{self._revision}",
            params={"recursive": "1"},
        )
        tree = tree_payload.get("tree")
        if not isinstance(tree, list):
            raise MalformedSourceError(
                f"GitHub returned a malformed tree for {self._repository}"
            )
        self._tree_truncated = bool(tree_payload.get("truncated", False))
        self._blobs = [entry for entry in tree if entry.get("type") == "blob"]
        self._records, lfs_reference = self._classify_records()
        self._media_is_referenced_not_present = (
            lfs_reference or self._detect_structural_or_document_reference()
        )

    def get_capabilities(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            can_stream=False,
            can_random_access=True,
            has_index_metadata=not self._tree_truncated,
            media_is_referenced_not_present=(self._media_is_referenced_not_present),
            requires_auth=self._requires_auth,
            estimated_record_count=(
                None if self._tree_truncated else len(self._records)
            ),
        )

    def get_schema(self) -> dict[str, Any]:
        return {
            "columns": {
                "path": "str",
                "sha": "str",
                "size": "int",
                "kind": "str",
                "download_url": "str | null",
                "revision": "str",
            },
            "_inference": "full",
            "record_model": "repository_file",
        }

    def list_records(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None")
        iterator = iter(self._iter_repository_records())
        yielded = 0
        while limit is None or yielded < limit:
            try:
                record = next(iterator)
            except StopIteration:
                return
            yield record
            yielded += 1

    def get_record(self, record_id: str) -> dict[str, Any]:
        blob = next(
            (entry for entry in self._blobs if entry.get("path") == record_id),
            None,
        )
        if blob is None:
            raise RecordNotFoundError(record_id)
        try:
            payload = self._get_file_payload(record_id)
        except SourceNotFoundError as exc:
            raise RecordNotFoundError(record_id) from exc
        content = _decode_content(payload)
        return {
            "path": record_id,
            "sha": payload.get("sha", blob.get("sha")),
            "size": payload.get("size", blob.get("size", 0)),
            "kind": _path_kind(record_id),
            "download_url": payload.get("download_url"),
            "revision": self._revision,
            "content": content,
        }

    def get_source_metadata(self) -> dict[str, Any]:
        return {
            "source_uri": self._source_uri,
            "revision": self._revision,
            "provider": "github",
            "requested_ref": self._requested_ref,
            "authenticated": self._token is not None,
        }

    def _iter_repository_records(self) -> Iterator[dict[str, Any]]:
        yield from self._records

    def _classify_records(self) -> tuple[list[dict[str, Any]], bool]:
        records: list[dict[str, Any]] = []
        lfs_reference = False
        lfs_checks = 0
        for blob in self._blobs:
            path = blob.get("path")
            if not isinstance(path, str):
                continue
            kind = _path_kind(path)
            if kind not in {"data", "media"}:
                continue

            is_lfs_pointer = False
            size = blob.get("size")
            if (
                kind == "media"
                and isinstance(size, int)
                and size <= _LFS_POINTER_MAX_BYTES
                and lfs_checks < _LFS_POINTER_CHECK_LIMIT
            ):
                lfs_checks += 1
                is_lfs_pointer = self._is_lfs_pointer(path)
            if is_lfs_pointer:
                lfs_reference = True
                continue

            records.append(self._file_record(blob, kind))
        return records, lfs_reference

    def _detect_structural_or_document_reference(self) -> bool:
        if self._records:
            return False
        if not self._blobs:
            return False

        doc_blobs = [
            blob
            for blob in self._blobs
            if isinstance(blob.get("path"), str)
            and _path_kind(blob["path"]) == "documentation"
        ]
        if doc_blobs and len(doc_blobs) == len(self._blobs):
            return True

        checked = 0
        for blob in doc_blobs:
            size = blob.get("size")
            if not isinstance(size, int) or size > _DOC_SCAN_MAX_BYTES:
                continue
            if checked >= _DOC_SCAN_LIMIT:
                break
            checked += 1
            path = blob["path"]
            try:
                content = _decode_content(self._get_file_payload(path))
            except (ConnectorError, UnicodeDecodeError):
                continue
            if isinstance(content, str) and _URL_REFERENCE_RE.search(content):
                return True
        return False

    def _is_lfs_pointer(self, path: str) -> bool:
        try:
            content = _decode_content(self._get_file_payload(path))
        except (ConnectorError, UnicodeDecodeError):
            return False
        return isinstance(content, str) and content.startswith(_LFS_HEADER)

    def _file_record(self, blob: dict[str, Any], kind: str) -> dict[str, Any]:
        path = str(blob["path"])
        return {
            "path": path,
            "sha": str(blob.get("sha", "")),
            "size": int(blob.get("size", 0)),
            "kind": kind,
            "download_url": (
                f"https://raw.githubusercontent.com/{self._repository}/"
                f"{self._revision}/{quote(path, safe='/')}"
            ),
            "revision": self._revision,
        }

    def _get_file_payload(self, path: str) -> dict[str, Any]:
        return self._request_json(
            f"/repos/{self._repository}/contents/{quote(path, safe='/')}",
            params={"ref": self._revision},
        )

    @with_retry
    def _request_json(
        self, path: str, *, params: dict[str, str] | None = None
    ) -> dict[str, Any]:
        response = self._client.get(
            f"{_API_ROOT}{path}", headers=self._headers, params=params
        )
        if response.status_code == 404:
            raise SourceNotFoundError(
                f"GitHub resource not found: {self._repository} ({path})"
            )
        if response.status_code >= 400:
            raise ConnectorError(
                f"GitHub API returned HTTP {response.status_code} for {path}: "
                f"{response.text}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise MalformedSourceError(
                f"GitHub returned a non-object response for {path}"
            )
        return payload


def _path_kind(path: str) -> str:
    lower_path = path.lower()
    dot = lower_path.rfind(".")
    extension = lower_path[dot:] if dot >= 0 else ""
    if extension in _DATA_EXTENSIONS:
        return "data"
    if extension in _MEDIA_EXTENSIONS:
        return "media"
    if extension in _DOCUMENT_EXTENSIONS:
        return "documentation"
    return "other"


def _decode_content(payload: dict[str, Any]) -> str | bytes:
    content = payload.get("content")
    if not isinstance(content, str):
        raise MalformedSourceError("GitHub content response has no encoded content")
    if payload.get("encoding") != "base64":
        raise MalformedSourceError("GitHub content response is not base64 encoded")
    try:
        raw = base64.b64decode(content, validate=False)
    except (ValueError, TypeError) as exc:
        raise MalformedSourceError("GitHub returned invalid base64 content") from exc
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw
