"""Deterministic Dataset Profiler (Phase 2)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from uuid import UUID

from auditor.connectors.base import Connector
from auditor.evidence.models import EvidenceStatus, Fact
from auditor.evidence.writers import TrustedWriter

_PROFILER_VERSION = "0.1.0"

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_CONTENT_TOKENS = frozenset({"text", "image", "audio", "video"})
_CONTENT_ORDER = ("text", "image", "audio", "video")
_EXACT_SCALAR_TOKENS = frozenset(
    {
        "int",
        "integer",
        "uint",
        "float",
        "double",
        "decimal",
        "bool",
        "boolean",
        "str",
        "string",
        "null",
        "none",
        "date",
        "timestamp",
        "datetime",
    }
)
_SIZED_SCALAR_RE = re.compile(
    r"^(?:int|uint|float|decimal|date)\d+$",
    re.IGNORECASE,
)

_SCHEMA_INCOMPLETE_EXPLANATION = (
    "Connector schema is incomplete or unavailable for modality classification."
)
_MEDIA_REFERENCED_EXPLANATION = (
    "Dataset content is referenced but not present, and the schema does not "
    "declare a content modality."
)
_MODALITY_AMBIGUOUS_EXPLANATION = (
    "Connector schema types do not establish a supported modality."
)


def _source_provenance(source_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_uri": source_metadata["source_uri"],
        "revision": source_metadata["revision"],
        "file_path": None,
        "field": None,
        "record_id": None,
    }


def _method_provenance() -> dict[str, Any]:
    return {
        "analyzer_name": "dataset_profiler",
        "analyzer_version": _PROFILER_VERSION,
        "model_name": None,
        "model_version": None,
        "parameters": {"operation": "metadata_profiling"},
        "sampling_strategy": None,
        "sampling_seed": None,
    }


def _exact_count_method_provenance(scan_max_records: int) -> dict[str, Any]:
    method = _method_provenance()
    method["parameters"] = {
        "operation": "exact_record_count",
        "scan_max_records": scan_max_records,
    }
    return method


def _modality_method_provenance() -> dict[str, Any]:
    method = _method_provenance()
    method["parameters"] = {"operation": "modality_classification"}
    return method


def _is_scalar_token(token: str) -> bool:
    lowered = token.lower()
    return lowered in _EXACT_SCALAR_TOKENS or bool(_SIZED_SCALAR_RE.fullmatch(lowered))


def _collect_type_descriptions(schema: Mapping[str, Any]) -> list[str]:
    columns = schema.get("columns")
    if isinstance(columns, Mapping):
        roots: list[Any] = list(columns.values())
    else:
        roots = [
            value
            for key, value in schema.items()
            if not str(key).startswith("_") and key != "record_model"
        ]

    descriptions: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
        elif isinstance(node, str):
            descriptions.append(node)

    for root in roots:
        walk(root)
    return descriptions


def _classify_modality(
    schema: Mapping[str, Any],
    *,
    media_is_referenced_not_present: bool,
) -> tuple[EvidenceStatus, str, dict[str, Any]]:
    schema_inference = schema.get("_inference")
    base_payload: dict[str, Any] = {
        "metric": "profiler.modality",
        "schema_inference": schema_inference,
    }

    if schema.get("record_model") == "repository_file":
        return (
            EvidenceStatus.OBSERVED,
            "Connector schema declares repository-file records",
            {
                **base_payload,
                "modalities": ["repository_file"],
                "basis": "schema_record_model",
            },
        )

    descriptions = _collect_type_descriptions(schema)
    if schema.get("_inference") == "unavailable" or not descriptions:
        return (
            EvidenceStatus.UNKNOWN,
            "Dataset modality could not be determined",
            {
                **base_payload,
                "modalities": ["unknown"],
                "basis": "none",
                "reason": "schema_incomplete",
                "explanation": _SCHEMA_INCOMPLETE_EXPLANATION,
            },
        )

    content_tokens: set[str] = set()
    all_scalar_only = True
    for description in descriptions:
        tokens = _TOKEN_RE.findall(description)
        description_content = {
            token.lower() for token in tokens if token.lower() in _CONTENT_TOKENS
        }
        if description_content:
            content_tokens |= description_content
            all_scalar_only = False
            continue
        if not tokens or not all(_is_scalar_token(token) for token in tokens):
            all_scalar_only = False

    if content_tokens:
        ordered = [token for token in _CONTENT_ORDER if token in content_tokens]
        modalities = ordered if len(ordered) == 1 else ["multimodal", *ordered]
        return (
            EvidenceStatus.OBSERVED,
            "Connector schema declares dataset modality",
            {
                **base_payload,
                "modalities": modalities,
                "basis": "schema_types",
            },
        )

    if media_is_referenced_not_present:
        return (
            EvidenceStatus.UNKNOWN,
            "Dataset modality could not be determined",
            {
                **base_payload,
                "modalities": ["unknown"],
                "basis": "none",
                "reason": "media_referenced_not_present",
                "explanation": _MEDIA_REFERENCED_EXPLANATION,
            },
        )

    if all_scalar_only:
        return (
            EvidenceStatus.COMPUTED,
            "Dataset modality classified as tabular from connector schema types",
            {
                **base_payload,
                "modalities": ["tabular"],
                "basis": "schema_types",
            },
        )

    return (
        EvidenceStatus.UNKNOWN,
        "Dataset modality could not be determined",
        {
            **base_payload,
            "modalities": ["unknown"],
            "basis": "none",
            "reason": "modality_ambiguous",
            "explanation": _MODALITY_AMBIGUOUS_EXPLANATION,
        },
    )


class DatasetProfiler:
    """Metadata-first deterministic profiler over the frozen Connector interface."""

    def profile(
        self,
        connector: Connector,
        writer: TrustedWriter,
        run_id: UUID,
        *,
        scan_max_records: int = 0,
    ) -> list[Fact]:
        if scan_max_records < 0:
            raise ValueError("scan_max_records must be non-negative")

        source_metadata = deepcopy(connector.get_source_metadata())
        capability_descriptor = connector.get_capabilities()
        capabilities = capability_descriptor.model_dump(mode="json")
        estimated_record_count = capability_descriptor.estimated_record_count
        schema = deepcopy(connector.get_schema())

        if estimated_record_count is not None:
            count_fact = Fact(
                status=EvidenceStatus.OBSERVED,
                claim="Connector-reported estimated record count",
                value={
                    "metric": "profiler.record_count.indexed",
                    "value": estimated_record_count,
                },
                source=_source_provenance(source_metadata),
                method=_method_provenance(),
                run_id=run_id,
                parent_fact_ids=[],
                confidence=None,
            )
        else:
            count_fact = Fact(
                status=EvidenceStatus.UNKNOWN,
                claim="Connector did not report an estimated record count",
                value={
                    "metric": "profiler.record_count.indexed",
                    "value": None,
                    "reason": "metadata_unavailable",
                    "explanation": (
                        "Connector did not provide an estimated record count."
                    ),
                },
                source=_source_provenance(source_metadata),
                method=_method_provenance(),
                run_id=run_id,
                parent_fact_ids=[],
                confidence=None,
            )

        scan_authorized = (
            scan_max_records > 0
            and capability_descriptor.has_index_metadata
            and not capability_descriptor.media_is_referenced_not_present
            and estimated_record_count is not None
            and estimated_record_count <= scan_max_records
        )

        if scan_authorized:
            exact_record_count = sum(1 for _ in connector.list_records())
            exact_fact = Fact(
                status=EvidenceStatus.COMPUTED,
                claim="Exact record count computed from an authorized record scan",
                value={
                    "metric": "profiler.record_count.exact",
                    "value": exact_record_count,
                },
                source=_source_provenance(source_metadata),
                method=_exact_count_method_provenance(scan_max_records),
                run_id=run_id,
                parent_fact_ids=[],
                confidence=None,
            )
        else:
            exact_fact = Fact(
                status=EvidenceStatus.UNKNOWN,
                claim="Exact record count was not computed",
                value={
                    "metric": "profiler.record_count.exact",
                    "value": None,
                    "reason": "scan_not_authorized",
                    "explanation": (
                        "Exact record count was not computed because "
                        "a full record scan was not authorized."
                    ),
                },
                source=_source_provenance(source_metadata),
                method=_exact_count_method_provenance(scan_max_records),
                run_id=run_id,
                parent_fact_ids=[],
                confidence=None,
            )

        source_fact = Fact(
            status=EvidenceStatus.OBSERVED,
            claim="Connector source metadata as reported by the connector",
            value={
                "metric": "profiler.source_metadata",
                "value": deepcopy(source_metadata),
            },
            source=_source_provenance(source_metadata),
            method=_method_provenance(),
            run_id=run_id,
            parent_fact_ids=[],
            confidence=None,
        )
        capabilities_fact = Fact(
            status=EvidenceStatus.OBSERVED,
            claim="Connector-reported capabilities as published by the connector",
            value={
                "metric": "profiler.capabilities",
                "value": deepcopy(capabilities),
            },
            source=_source_provenance(source_metadata),
            method=_method_provenance(),
            run_id=run_id,
            parent_fact_ids=[],
            confidence=None,
        )
        schema_fact = Fact(
            status=EvidenceStatus.OBSERVED,
            claim="Connector-reported schema as returned by the connector",
            value={
                "metric": "profiler.schema",
                "value": deepcopy(schema),
            },
            source=_source_provenance(source_metadata),
            method=_method_provenance(),
            run_id=run_id,
            parent_fact_ids=[],
            confidence=None,
        )

        modality_status, modality_claim, modality_value = _classify_modality(
            schema,
            media_is_referenced_not_present=(
                capability_descriptor.media_is_referenced_not_present
            ),
        )
        if modality_value.get("reason") == "media_referenced_not_present":
            modality_parents = [capabilities_fact.id, schema_fact.id]
        else:
            modality_parents = [schema_fact.id]

        modality_fact = Fact(
            status=modality_status,
            claim=modality_claim,
            value=modality_value,
            source=_source_provenance(source_metadata),
            method=_modality_method_provenance(),
            run_id=run_id,
            parent_fact_ids=modality_parents,
            confidence=None,
        )

        facts = [
            source_fact,
            capabilities_fact,
            schema_fact,
            count_fact,
            exact_fact,
            modality_fact,
        ]

        for fact in facts:
            writer.write(fact)
        return facts
