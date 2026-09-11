"""Streaming Hugging Face Hub connector."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping
from typing import Any

import datasets
from huggingface_hub import HfApi
from huggingface_hub.errors import (
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from auditor.connectors.base import CapabilityDescriptor, Connector
from auditor.connectors.errors import (
    ConnectorError,
    MalformedSourceError,
    SourceNotFoundError,
)
from auditor.connectors.retry import with_retry

# Exposed at module scope so tests can replace the network boundary.
load_dataset = datasets.load_dataset

_MISSING_WORDS = r"(?:not found|does not exist|doesn't exist)"
_MISSING_CONFIG_OR_SPLIT = re.compile(
    rf"(?:\b(?:builder\s*)?config(?:uration)?\b.*\b{_MISSING_WORDS}\b"
    rf"|\bsplit\b.*\b{_MISSING_WORDS}\b"
    r"|\bunknown\s+(?:config(?:uration)?|split)\b)",
    re.IGNORECASE,
)


class HFConnector(Connector):
    """
    Stream one Hugging Face dataset configuration and split.

    The requested revision (which may be a branch) is resolved through the
    Hub API first. Both metadata and streaming reads then use that immutable
    dataset commit SHA. Records are rows yielded by ``datasets`` streaming
    mode; random access is deliberately unsupported because satisfying it
    would require a separate non-streaming index or a scan.

    ``has_index_metadata`` is true only when a declared feature schema exists
    and none of its (possibly nested) features is a real ``datasets.Video``,
    ``datasets.Audio``, or ``datasets.Image`` instance. Those media feature
    types expose their shape in the schema, but properties such as duration,
    frame rate, and dimensions can require opening media bytes, so the
    connector does not claim that metadata is cheaply indexed.
    """

    def __init__(
        self,
        dataset_id: str,
        *,
        config_name: str | None = None,
        split: str = "train",
        revision: str | None = None,
        token: str | None = None,
    ) -> None:
        self._dataset_id = dataset_id
        self._config_name = config_name
        self._split = split
        self._requested_revision = revision
        self._token = token or os.getenv("HF_TOKEN")
        self._api = HfApi(token=self._token)

        info = self._fetch_dataset_info()
        resolved_revision = getattr(info, "sha", None)
        if not isinstance(resolved_revision, str) or not resolved_revision:
            raise MalformedSourceError(
                f"Hugging Face did not return a commit SHA for {dataset_id}"
            )
        self._revision = resolved_revision
        self._requires_auth = bool(
            getattr(info, "private", False) or getattr(info, "gated", False)
        )
        self._dataset = self._load_streaming_dataset()

    def get_capabilities(self) -> CapabilityDescriptor:
        features = getattr(self._dataset, "features", None)
        return CapabilityDescriptor(
            can_stream=True,
            can_random_access=False,
            has_index_metadata=bool(features) and not _contains_media_feature(features),
            media_is_referenced_not_present=False,
            requires_auth=self._requires_auth,
            estimated_record_count=self._estimated_record_count(),
        )

    def get_schema(self) -> dict[str, Any]:
        features = getattr(self._dataset, "features", None)
        if not isinstance(features, Mapping):
            return {"columns": {}, "_inference": "unavailable"}
        return {
            "columns": {
                str(name): _feature_type(feature) for name, feature in features.items()
            },
            "_inference": "full",
        }

    def list_records(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None")

        iterator = iter(self._dataset)
        yielded = 0
        while limit is None or yielded < limit:
            try:
                record = self._next_stream_record(iterator)
            except StopIteration:
                return
            if not isinstance(record, dict):
                raise MalformedSourceError(
                    f"Hugging Face row is not a mapping: {type(record).__name__}"
                )
            yield record
            yielded += 1

    def get_record(self, record_id: str) -> dict[str, Any]:
        raise NotImplementedError(
            "HF streaming datasets do not support random access "
            "(can_random_access=False)"
        )

    def get_source_metadata(self) -> dict[str, Any]:
        return {
            "source_uri": (f"https://huggingface.co/datasets/{self._dataset_id}"),
            "revision": self._revision,
            "provider": "huggingface",
            "config_name": self._config_name,
            "split": self._split,
            "authenticated": self._token is not None,
        }

    @with_retry
    def _fetch_dataset_info(self) -> Any:
        try:
            return self._api.dataset_info(
                self._dataset_id, revision=self._requested_revision
            )
        except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
            raise SourceNotFoundError(
                f"Hugging Face dataset or revision not found: "
                f"{self._dataset_id}@{self._requested_revision or 'default'}"
            ) from exc
        except GatedRepoError as exc:
            raise ConnectorError(
                f"Hugging Face dataset requires authentication: {self._dataset_id}"
            ) from exc

    @with_retry
    def _load_streaming_dataset(self) -> Any:
        try:
            return load_dataset(
                self._dataset_id,
                name=self._config_name,
                split=self._split,
                revision=self._revision,
                streaming=True,
                token=self._token,
            )
        except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
            raise SourceNotFoundError(
                f"Hugging Face dataset or revision not found: {self._dataset_id}"
            ) from exc
        except GatedRepoError as exc:
            raise ConnectorError(
                f"Hugging Face dataset requires authentication: {self._dataset_id}"
            ) from exc
        except ValueError as exc:
            if _MISSING_CONFIG_OR_SPLIT.search(str(exc)):
                raise SourceNotFoundError(
                    f"Hugging Face config or split not found for "
                    f"{self._dataset_id}: config={self._config_name!r}, "
                    f"split={self._split!r}"
                ) from exc
            raise

    @with_retry
    def _next_stream_record(self, iterator: Iterator[dict[str, Any]]) -> dict[str, Any]:
        return next(iterator)

    def _estimated_record_count(self) -> int | None:
        dataset_info = getattr(self._dataset, "info", None)
        splits = getattr(dataset_info, "splits", None)
        if not isinstance(splits, Mapping):
            return None
        split_info = splits.get(self._split)
        count = getattr(split_info, "num_examples", None)
        return count if isinstance(count, int) and count >= 0 else None


def _contains_media_feature(feature: Any) -> bool:
    if isinstance(feature, (datasets.Video, datasets.Audio, datasets.Image)):
        return True
    if isinstance(feature, Mapping):
        return any(_contains_media_feature(value) for value in feature.values())

    # Sequence/List/LargeList expose their nested declaration as ``feature``.
    nested = getattr(feature, "feature", None)
    if nested is not None and nested is not feature:
        return _contains_media_feature(nested)
    return False


def _feature_type(feature: Any) -> str:
    dtype = getattr(feature, "dtype", None)
    if isinstance(dtype, str):
        return dtype
    return str(feature)
