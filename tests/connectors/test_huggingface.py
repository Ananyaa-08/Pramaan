"""Contract tests for the streaming Hugging Face dataset connector."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import datasets
import pytest

from auditor.connectors.base import Connector
from auditor.connectors.errors import SourceNotFoundError
from auditor.connectors.huggingface import HFConnector

PINNED_SHA = "0123456789abcdef0123456789abcdef01234567"
ROWS = [
    {"id": "r1", "text": "hello", "score": 1},
    {"id": "r2", "text": "world", "score": 2},
    {"id": "r3", "text": "sign", "score": 3},
]


class Value:
    """Small stand-in for datasets.Value used by schema tests."""

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype

    def __str__(self) -> str:
        return self.dtype


class GuardedRows(Iterator[dict[str, Any]]):
    """Fail if a limited consumer pulls one record too many."""

    def __init__(self, rows: list[dict[str, Any]], max_pulls: int) -> None:
        self._rows = rows
        self._max_pulls = max_pulls
        self.pulls = 0

    def __iter__(self) -> GuardedRows:
        return self

    def __next__(self) -> dict[str, Any]:
        self.pulls += 1
        if self.pulls > self._max_pulls:
            raise AssertionError("the streaming source was consumed past the limit")
        return self._rows[self.pulls - 1]


class FakeStreamingDataset:
    def __init__(
        self,
        rows: Iterator[dict[str, Any]],
        *,
        features: dict[str, Any] | None = None,
        count: int | None = 3,
    ) -> None:
        self._rows = rows
        self.features = features or {
            "id": Value("string"),
            "text": Value("string"),
            "score": Value("int64"),
        }
        split_info = SimpleNamespace(num_examples=count)
        self.info = SimpleNamespace(splits={"train": split_info})

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self._rows


class FakeHfApi:
    def __init__(
        self,
        *,
        private: bool = False,
        gated: bool | str = False,
        failures_before_success: int = 0,
    ) -> None:
        self.private = private
        self.gated = gated
        self.failures_before_success = failures_before_success
        self.calls = 0

    def dataset_info(self, repo_id: str, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise ConnectionError("temporary Hub outage")
        return SimpleNamespace(
            id=repo_id,
            sha=PINNED_SHA,
            private=self.private,
            gated=self.gated,
        )


def _install_hf_fakes(
    monkeypatch: pytest.MonkeyPatch,
    dataset: FakeStreamingDataset,
    *,
    api: FakeHfApi | None = None,
) -> tuple[FakeHfApi, list[dict[str, Any]]]:
    fake_api = api or FakeHfApi()
    load_calls: list[dict[str, Any]] = []

    def api_factory(*args: Any, **kwargs: Any) -> FakeHfApi:
        return fake_api

    def fake_load_dataset(*args: Any, **kwargs: Any) -> FakeStreamingDataset:
        load_calls.append({"args": args, **kwargs})
        return dataset

    monkeypatch.setattr("auditor.connectors.huggingface.HfApi", api_factory)
    monkeypatch.setattr(
        "auditor.connectors.huggingface.load_dataset", fake_load_dataset
    )
    return fake_api, load_calls


def test_hf_connector_is_connector_and_loads_streaming_at_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeStreamingDataset(iter(ROWS))
    _, load_calls = _install_hf_fakes(monkeypatch, source)

    connector = HFConnector(
        "acme/example", config_name="default", split="train", revision="main"
    )

    assert isinstance(connector, Connector)
    assert len(load_calls) == 1
    assert load_calls[0]["args"][0] == "acme/example"
    assert load_calls[0]["name"] == "default"
    assert load_calls[0]["split"] == "train"
    assert load_calls[0]["streaming"] is True
    assert load_calls[0]["revision"] == PINNED_SHA


def test_list_records_is_lazy_and_limit_does_not_overconsume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guarded = GuardedRows(ROWS, max_pulls=2)
    source = FakeStreamingDataset(guarded)
    _install_hf_fakes(monkeypatch, source)
    connector = HFConnector("acme/example", split="train")

    assert guarded.pulls == 0
    records = connector.list_records(limit=2)
    assert isinstance(records, Iterator)
    assert not isinstance(records, list)
    assert guarded.pulls == 0

    assert next(records)["id"] == "r1"
    assert guarded.pulls == 1
    assert next(records)["id"] == "r2"
    assert guarded.pulls == 2
    with pytest.raises(StopIteration):
        next(records)
    assert guarded.pulls == 2


def test_media_features_do_not_claim_cheap_index_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeStreamingDataset(
        iter(ROWS),
        features={"id": Value("string"), "video": datasets.Video()},
    )
    _install_hf_fakes(monkeypatch, source)

    caps = HFConnector("acme/video-dataset", split="train").get_capabilities()

    assert caps.can_stream is True
    assert caps.can_random_access is False
    assert caps.has_index_metadata is False
    assert caps.media_is_referenced_not_present is False
    assert caps.estimated_record_count == 3


def test_scalar_features_can_use_hub_index_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeStreamingDataset(iter(ROWS))
    _install_hf_fakes(monkeypatch, source)

    caps = HFConnector("acme/tabular", split="train").get_capabilities()

    assert caps.has_index_metadata is True
    assert caps.estimated_record_count == 3


def test_schema_comes_from_features_without_consuming_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guarded = GuardedRows(ROWS, max_pulls=0)
    source = FakeStreamingDataset(guarded)
    _install_hf_fakes(monkeypatch, source)
    connector = HFConnector("acme/example", split="train")

    schema = connector.get_schema()

    assert schema["columns"] == {
        "id": "string",
        "text": "string",
        "score": "int64",
    }
    assert schema["_inference"] == "full"
    assert guarded.pulls == 0


def test_source_metadata_uses_resolved_sha_never_requested_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeStreamingDataset(iter(ROWS))
    _install_hf_fakes(monkeypatch, source)

    meta = HFConnector(
        "acme/example", config_name="default", split="train", revision="main"
    ).get_source_metadata()

    assert meta["source_uri"] == "https://huggingface.co/datasets/acme/example"
    assert meta["provider"] == "huggingface"
    assert meta["revision"] == PINNED_SHA
    assert meta["revision"] not in {"main", "master", "latest"}
    assert meta["config_name"] == "default"
    assert meta["split"] == "train"


def test_public_dataset_reports_anonymous_access_honestly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    source = FakeStreamingDataset(iter(ROWS))
    _install_hf_fakes(monkeypatch, source, api=FakeHfApi(private=False))

    connector = HFConnector("acme/public", split="train")

    assert connector.get_capabilities().requires_auth is False
    assert connector.get_source_metadata()["authenticated"] is False


def test_private_dataset_distinguishes_required_and_used_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    source = FakeStreamingDataset(iter(ROWS))
    _install_hf_fakes(monkeypatch, source, api=FakeHfApi(private=True))

    connector = HFConnector("acme/private", split="train")

    assert connector.get_capabilities().requires_auth is True
    assert connector.get_source_metadata()["authenticated"] is True


def test_get_record_is_explicitly_unsupported_for_streaming_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeStreamingDataset(iter(ROWS))
    _install_hf_fakes(monkeypatch, source)
    connector = HFConnector("acme/example", split="train")

    with pytest.raises(NotImplementedError):
        connector.get_record("r1")


def test_missing_config_raises_source_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "auditor.connectors.huggingface.HfApi", lambda **kwargs: FakeHfApi()
    )

    def missing_config(*args: Any, **kwargs: Any) -> None:
        raise ValueError("BuilderConfig 'does-not-exist' not found")

    monkeypatch.setattr("auditor.connectors.huggingface.load_dataset", missing_config)

    with pytest.raises(SourceNotFoundError):
        HFConnector("acme/example", config_name="does-not-exist", split="train")


def test_unrelated_value_error_is_not_mislabeled_source_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "auditor.connectors.huggingface.HfApi", lambda **kwargs: FakeHfApi()
    )

    def unrelated_value_error(*args: Any, **kwargs: Any) -> None:
        raise ValueError("invalid trust_remote_code setting")

    monkeypatch.setattr(
        "auditor.connectors.huggingface.load_dataset", unrelated_value_error
    )

    with pytest.raises(ValueError, match="trust_remote_code"):
        HFConnector("acme/example", config_name="default", split="train")


def test_hub_metadata_call_retries_transient_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeHfApi(failures_before_success=1)
    source = FakeStreamingDataset(iter(ROWS))
    _install_hf_fakes(monkeypatch, source, api=api)

    connector = HFConnector("acme/example", split="train")

    assert connector.get_source_metadata()["revision"] == PINNED_SHA
    assert api.calls == 2
