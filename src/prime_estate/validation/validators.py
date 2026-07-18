"""Field-level validation, ported from the original ``Validate Input`` node.

The regexes are intentionally identical to the JavaScript originals so that the
Python system accepts and rejects exactly the same inputs the production n8n bot
did. A booking is only as trustworthy as the contact details behind it — an
agent that captures ``+971...`` typo'd as ``00971`` produces a lead nobody can
call back, so validation runs as a hard gate before any persistence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Compiled once at import; these mirror the original node's patterns exactly.
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
_PHONE_RE = re.compile(r"^\+[0-9]{7,15}$")          # E.164-ish: + then 7–15 digits
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")        # YYYY-MM-DD
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")  # HH:MM, 24h


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a candidate lead's contactable fields."""

    ok: bool
    errors: list[str]

    def as_client_prompt(self) -> str | None:
        """Render a human, WhatsApp-friendly correction request.

        Returns ``None`` when validation passed. The phrasing mirrors the
        original bot so the ported system speaks with the same voice.
        """
        if self.ok:
            return None
        joined = " and ".join(self.errors)
        return (
            f"I noticed an issue with your {joined}. "
            "Could you please double-check and provide them again?"
        )


def validate_email(value: str) -> bool:
    """True if *value* is a syntactically plausible email address."""
    return bool(_EMAIL_RE.match(value or ""))


def validate_phone(value: str) -> bool:
    """True if *value* is a ``+<country><number>`` phone with 7–15 digits."""
    return bool(_PHONE_RE.match(value or ""))


def validate_date(value: str) -> bool:
    """True if *value* is formatted ``YYYY-MM-DD`` (format only, not calendar)."""
    return bool(_DATE_RE.match(value or ""))


def validate_time(value: str) -> bool:
    """True if *value* is a 24-hour ``HH:MM`` timestamp."""
    return bool(_TIME_RE.match(value or ""))


def validate_lead_fields(
    *, email: str, phone: str, meeting_date: str, meeting_time: str
) -> ValidationResult:
    """Validate the four gating fields before a lead is allowed to persist.

    Error messages match the original node so downstream copy is unchanged.
    Collecting all failures in one pass (rather than failing fast) lets the
    agent ask the user to fix everything at once instead of round-tripping per
    field.
    """
    errors: list[str] = []
    if not validate_email(email):
        errors.append("email appears invalid")
    if not validate_phone(phone):
        errors.append(
            "phone number format is invalid (must include country code, e.g. +971...)"
        )
    if not validate_date(meeting_date):
        errors.append("meeting date format is invalid")
    if not validate_time(meeting_time):
        errors.append("meeting time format is invalid")
    return ValidationResult(ok=not errors, errors=errors)
