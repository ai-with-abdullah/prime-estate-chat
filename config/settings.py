"""Environment-driven configuration for the composition root.

Every deployment-specific value lives here and nowhere else: business logic
never reads ``os.environ`` directly, so the same code runs in production (real
Groq key, real calendar/sheet ids) and in tests (no environment at all —
settings are only constructed at the composition root, which tests replace
wholesale with in-memory fakes).

Pydantic ``BaseSettings`` gives typed, validated config with ``.env`` support
for free; a missing required value (the API key) fails loudly at startup, not
at the first LLM call twenty turns into a conversation.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, sourced from environment variables / ``.env``.

    Variable names are prefixed ``PRIME_`` (e.g. ``PRIME_GROQ_API_KEY``) so
    they cannot collide with other software on the host.
    """

    model_config = SettingsConfigDict(
        env_prefix="PRIME_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = Field(..., description="Groq API key; the only required secret.")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model id used for both routing and agent turns.",
    )
    session_ttl_seconds: float = Field(
        default=1800.0,
        gt=0,
        description="Sticky-session TTL. 30 minutes, matching the original workflow.",
    )
    calendar_id: str = Field(
        default="",
        description="Google Calendar id for the production calendar tool. Unused by the in-memory tool.",
    )
    sheet_id: str = Field(
        default="",
        description="Google Sheet id for the production lead datastore. Unused by the in-memory tool.",
    )
    log_level: str = Field(
        default="INFO",
        description="Root log level for the prime_estate logger namespace.",
    )
