"""Application settings loaded from environment variables and .env."""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Secrets are SecretStr and never printed in cleartext."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Field names map to UPPER_SNAKE env vars (hf_token -> HF_TOKEN).
    )

    hf_token: SecretStr | None = None
    github_token: SecretStr | None = None
    llm_api_key: SecretStr | None = None
    log_level: str = "INFO"
    default_budget: float = Field(default=100.0)


def get_settings() -> Settings:
    """Load settings from the environment / .env file."""
    return Settings()
