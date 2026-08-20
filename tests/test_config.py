"""Tests for pydantic Settings scaffolding."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from auditor.config import Settings


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear settings-related env vars so tests control inputs."""
    for key in (
        "HF_TOKEN",
        "GITHUB_TOKEN",
        "LLM_API_KEY",
        "LOG_LEVEL",
        "DEFAULT_BUDGET",
    ):
        monkeypatch.delenv(key, raising=False)


def test_settings_loads_with_defaults(clean_env: None) -> None:
    settings = Settings(_env_file=None)

    assert settings.log_level == "INFO"
    assert settings.default_budget == 100.0
    assert settings.hf_token is None
    assert settings.github_token is None
    assert settings.llm_api_key is None


def test_settings_loads_from_environment(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-secret-value")
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret-value")
    monkeypatch.setenv("LLM_API_KEY", "llm-secret-value")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DEFAULT_BUDGET", "42.5")

    settings = Settings(_env_file=None)

    assert isinstance(settings.hf_token, SecretStr)
    assert settings.hf_token.get_secret_value() == "hf-secret-value"
    assert settings.github_token.get_secret_value() == "gh-secret-value"
    assert settings.llm_api_key.get_secret_value() == "llm-secret-value"
    assert settings.log_level == "DEBUG"
    assert settings.default_budget == 42.5


def test_secrets_do_not_leak_in_str_or_repr(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "super-secret-token-should-not-appear"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setenv("GITHUB_TOKEN", secret)
    monkeypatch.setenv("LLM_API_KEY", secret)

    settings = Settings(_env_file=None)

    for rendered in (str(settings), repr(settings)):
        assert secret not in rendered
        assert "**********" in rendered or "SecretStr" in rendered


def test_settings_loads_from_dotenv_file(
    clean_env: None, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LOG_LEVEL=WARNING\nDEFAULT_BUDGET=7\nHF_TOKEN=dotenv-hf-token\n",
        encoding="utf-8",
    )
    for key in ("LOG_LEVEL", "DEFAULT_BUDGET", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=env_file)

    assert settings.log_level == "WARNING"
    assert settings.default_budget == 7.0
    assert settings.hf_token is not None
    assert settings.hf_token.get_secret_value() == "dotenv-hf-token"
    assert "dotenv-hf-token" not in str(settings)
