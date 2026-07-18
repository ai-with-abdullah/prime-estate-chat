"""LLM client abstraction.

The system talks to exactly one LLM capability — "given a system prompt and a
conversation, return the next assistant message" — so that is the entire
interface. Wrapping it in a Protocol keeps the router and agents ignorant of the
provider: Groq in production (chosen originally for its low latency, which
matters on a WhatsApp turn), but swappable for any chat model without touching
call sites, and trivially replaceable with a scripted fake in tests.

Retry with bounded exponential backoff lives here rather than in every caller,
because transient 429/5xx from the inference API is an infrastructure concern,
not something an agent's business logic should have to reason about.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from prime_estate.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChatMessage:
    """A single turn in a conversation. ``role`` is one of system/user/assistant."""

    role: str
    content: str


@runtime_checkable
class ChatModel(Protocol):
    """Minimal chat-completion contract the rest of the system depends on."""

    def complete(self, *, system: str, messages: list[ChatMessage]) -> str:
        """Return the assistant's reply given a system prompt and history."""
        ...


class GroqChatModel:
    """Groq-backed :class:`ChatModel` with bounded retry.

    The SDK client is injected rather than constructed internally so that
    credentials, timeouts, and base URLs are configured once at the composition
    root and this class stays a thin, testable adapter.
    """

    def __init__(
        self,
        *,
        client: object,
        model: str,
        temperature: float = 0.3,
        max_retries: int = 3,
        base_delay: float = 0.5,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._base_delay = base_delay

    def complete(self, *, system: str, messages: list[ChatMessage]) -> str:
        """Call Groq, retrying transient failures with exponential backoff.

        Temperature defaults low (0.3): every use in this system — intent
        classification, slot-filling, structured signal emission — wants
        consistency over creativity. A high temperature here would surface as
        flickering intents and malformed DONE payloads.
        """
        payload = [{"role": "system", "content": system}]
        payload += [{"role": m.role, "content": m.content} for m in messages]

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(  # type: ignore[attr-defined]
                    model=self._model,
                    temperature=self._temperature,
                    messages=payload,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 — adapter deliberately broad
                last_error = exc
                delay = self._base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "groq.complete failed (attempt %d/%d): %s; retrying in %.2fs",
                    attempt, self._max_retries, exc, delay,
                )
                time.sleep(delay)

        raise LLMUnavailableError(
            f"Groq call failed after {self._max_retries} attempts"
        ) from last_error


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM cannot be reached after all retries are exhausted."""
