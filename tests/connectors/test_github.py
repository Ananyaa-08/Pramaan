"""Contract tests for the GitHub-hosted dataset connector."""

from __future__ import annotations

import base64
import inspect
from collections.abc import Iterator
from typing import Any

import pytest

from auditor.connectors.base import Connector
from auditor.connectors.errors import RecordNotFoundError
from auditor.connectors.github import GitHubConnector

PINNED_SHA = "abcdef0123456789abcdef0123456789abcdef01"


def _content_payload(path: str, content: str, sha: str = "blob-sha") -> dict[str, Any]:
    return {
        "type": "file",
        "path": path,
        "name": path.rsplit("/", 1)[-1],
        "sha": sha,
        "size": len(content.encode()),
        "encoding": "base64",
        "content": base64.b64encode(content.encode()).decode(),
        "download_url": f"https://raw.githubusercontent.com/acme/dataset/{sha}/{path}",
    }


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> Any:
        return self._payload


class FakeGitHubClient:
    def __init__(
        self,
        *,
        tree: list[dict[str, Any]],
        files: dict[str, str] | None = None,
        private: bool = False,
        failures_before_success: int = 0,
    ) -> None:
        self.tree = tree
        self.files = files or {}
        self.private = private
        self.failures_before_success = failures_before_success
        self.repo_calls = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if url.endswith("/repos/acme/dataset"):
            self.repo_calls += 1
            if self.repo_calls <= self.failures_before_success:
                raise ConnectionError("temporary GitHub outage")
            return FakeResponse(
                {
                    "full_name": "acme/dataset",
                    "private": self.private,
                    "default_branch": "main",
                    "html_url": "https://github.com/acme/dataset",
                }
            )
        if url.endswith("/repos/acme/dataset/commits/main"):
            return FakeResponse({"sha": PINNED_SHA})
        if url.endswith(f"/repos/acme/dataset/git/trees/{PINNED_SHA}"):
            return FakeResponse(
                {"sha": PINNED_SHA, "truncated": False, "tree": self.tree}
            )
        marker = "/repos/acme/dataset/contents/"
        if marker in url:
            path = url.split(marker, maxsplit=1)[1]
            if path in self.files:
                return FakeResponse(_content_payload(path, self.files[path]))
            return FakeResponse({"message": "Not Found"}, status_code=404)
        raise AssertionError(f"unexpected GitHub request: {url}")


def _blob(path: str, *, size: int = 10, sha: str | None = None) -> dict[str, Any]:
    return {
        "path": path,
        "mode": "100644",
        "type": "blob",
        "sha": sha or f"sha-{path}",
        "size": size,
        "url": f"https://api.github.com/repos/acme/dataset/git/blobs/{sha or path}",
    }


PRESENT_TREE = [
    _blob("README.md", size=100),
    _blob("data/items.csv", size=500),
    _blob("media/clip.mp4", size=10_000),
    _blob("src/download.py", size=200),
]


class GuardedRecords(Iterator[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]], max_pulls: int) -> None:
        self.records = records
        self.max_pulls = max_pulls
        self.pulls = 0

    def __iter__(self) -> GuardedRecords:
        return self

    def __next__(self) -> dict[str, Any]:
        self.pulls += 1
        if self.pulls > self.max_pulls:
            raise AssertionError("repository records were consumed past the limit")
        return self.records[self.pulls - 1]


def test_github_connector_is_connector_and_documents_file_record_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    connector = GitHubConnector(
        "acme/dataset", ref="main", client=FakeGitHubClient(tree=PRESENT_TREE)
    )

    doc = inspect.getdoc(GitHubConnector) or ""
    assert isinstance(connector, Connector)
    assert "file" in doc.lower()
    assert "record" in doc.lower()
    assert "documentation" in doc.lower()


def test_source_metadata_pins_exact_commit_not_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    connector = GitHubConnector(
        "acme/dataset", ref="main", client=FakeGitHubClient(tree=PRESENT_TREE)
    )

    meta = connector.get_source_metadata()

    assert meta["source_uri"] == "https://github.com/acme/dataset"
    assert meta["provider"] == "github"
    assert meta["revision"] == PINNED_SHA
    assert meta["revision"] not in {"main", "master", "latest"}
    assert meta["requested_ref"] == "main"


def test_list_records_yields_only_present_data_or_media_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    connector = GitHubConnector(
        "acme/dataset", ref="main", client=FakeGitHubClient(tree=PRESENT_TREE)
    )

    records = list(connector.list_records())

    assert [record["path"] for record in records] == [
        "data/items.csv",
        "media/clip.mp4",
    ]
    assert [record["kind"] for record in records] == ["data", "media"]
    assert all(record["revision"] == PINNED_SHA for record in records)


def test_list_records_is_lazy_and_limit_does_not_overconsume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    connector = GitHubConnector(
        "acme/dataset", ref="main", client=FakeGitHubClient(tree=PRESENT_TREE)
    )
    guarded = GuardedRecords(
        [
            {"path": "data/one.csv"},
            {"path": "data/two.jsonl"},
            {"path": "data/three.parquet"},
        ],
        max_pulls=2,
    )
    monkeypatch.setattr(connector, "_iter_repository_records", lambda: guarded)

    assert guarded.pulls == 0
    records = connector.list_records(limit=2)
    assert isinstance(records, Iterator)
    assert not isinstance(records, list)
    assert guarded.pulls == 0

    assert next(records)["path"] == "data/one.csv"
    assert guarded.pulls == 1
    assert next(records)["path"] == "data/two.jsonl"
    assert guarded.pulls == 2
    with pytest.raises(StopIteration):
        next(records)
    assert guarded.pulls == 2


def test_schema_describes_repository_file_records_without_opening_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    client = FakeGitHubClient(tree=PRESENT_TREE)
    connector = GitHubConnector("acme/dataset", ref="main", client=client)

    schema = connector.get_schema()

    assert schema["_inference"] == "full"
    assert schema["record_model"] == "repository_file"
    assert schema["columns"] == {
        "path": "str",
        "sha": "str",
        "size": "int",
        "kind": "str",
        "download_url": "str | null",
        "revision": "str",
    }
    assert not any("/contents/data/" in url for url, _ in client.calls)


def test_readme_can_be_read_by_path_at_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    readme = "# Dataset\nThe media files are included under media/.\n"
    client = FakeGitHubClient(tree=PRESENT_TREE, files={"README.md": readme})
    connector = GitHubConnector("acme/dataset", ref="main", client=client)

    record = connector.get_record("README.md")

    assert record["path"] == "README.md"
    assert record["kind"] == "documentation"
    assert record["content"] == readme
    read_call = next(url for url, _ in client.calls if "/contents/README.md" in url)
    assert read_call.endswith("/contents/README.md")
    read_params = next(
        kwargs.get("params", {})
        for url, kwargs in client.calls
        if "/contents/README.md" in url
    )
    assert read_params["ref"] == PINNED_SHA


def test_docs_with_external_media_are_reported_as_referenced_not_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    readme = (
        "# Video Dataset\n"
        "Videos are hosted externally: https://cdn.example.org/videos/clip.mp4\n"
    )
    client = FakeGitHubClient(
        tree=[
            _blob("README.md", size=len(readme)),
            _blob("scripts/download.py", size=200),
        ],
        files={"README.md": readme},
    )
    connector = GitHubConnector("acme/dataset", ref="main", client=client)

    caps = connector.get_capabilities()

    assert caps.media_is_referenced_not_present is True
    assert caps.estimated_record_count == 0
    assert list(connector.list_records()) == []


def test_document_url_without_known_extension_is_not_a_media_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    readme = "Download from https://example.org/datasets/latest\n"
    client = FakeGitHubClient(
        tree=[
            _blob("README.md", size=len(readme)),
            _blob("scripts/download.py", size=200),
        ],
        files={"README.md": readme},
    )
    connector = GitHubConnector("acme/dataset", ref="main", client=client)

    assert connector.get_capabilities().media_is_referenced_not_present is False


def test_git_lfs_pointer_is_reference_not_present_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:1234567890abcdef\n"
        "size 99999999\n"
    )
    client = FakeGitHubClient(
        tree=[_blob("videos/clip.mp4", size=len(pointer))],
        files={"videos/clip.mp4": pointer},
    )
    connector = GitHubConnector("acme/dataset", ref="main", client=client)

    caps = connector.get_capabilities()

    assert caps.media_is_referenced_not_present is True
    assert caps.estimated_record_count == 0
    assert list(connector.list_records()) == []


def test_present_files_report_random_access_and_index_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    client = FakeGitHubClient(tree=PRESENT_TREE)
    connector = GitHubConnector("acme/dataset", ref="main", client=client)

    caps = connector.get_capabilities()

    assert caps.can_stream is False
    assert caps.can_random_access is True
    assert caps.has_index_metadata is True
    assert caps.media_is_referenced_not_present is False
    assert caps.estimated_record_count == 2
    assert not any("/contents/media/clip.mp4" in url for url, _ in client.calls)


def test_missing_repository_path_raises_typed_record_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    connector = GitHubConnector(
        "acme/dataset", ref="main", client=FakeGitHubClient(tree=PRESENT_TREE)
    )

    with pytest.raises(RecordNotFoundError):
        connector.get_record("data/missing.csv")


def test_public_repo_reports_no_auth_required_even_when_token_improves_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token")
    connector = GitHubConnector(
        "acme/dataset",
        ref="main",
        client=FakeGitHubClient(tree=PRESENT_TREE, private=False),
    )

    assert connector.get_capabilities().requires_auth is False
    assert connector.get_source_metadata()["authenticated"] is True


def test_private_repo_reports_auth_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token")
    connector = GitHubConnector(
        "acme/dataset",
        ref="main",
        client=FakeGitHubClient(tree=PRESENT_TREE, private=True),
    )

    assert connector.get_capabilities().requires_auth is True
    assert connector.get_source_metadata()["authenticated"] is True


def test_github_api_calls_retry_transient_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    client = FakeGitHubClient(tree=PRESENT_TREE, failures_before_success=1)

    connector = GitHubConnector("acme/dataset", ref="main", client=client)

    assert connector.get_source_metadata()["revision"] == PINNED_SHA
    assert client.repo_calls == 2


def test_docs_only_tree_sets_referenced_not_present_without_url_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    readme = "This dataset's media is not included in this repository.\n"
    client = FakeGitHubClient(
        tree=[_blob("README.md", size=len(readme))],
        files={"README.md": readme},
    )
    connector = GitHubConnector("acme/dataset", ref="main", client=client)

    caps = connector.get_capabilities()

    assert caps.media_is_referenced_not_present is True
