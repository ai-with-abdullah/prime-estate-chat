"""Core domain models for the Prime Estate agent system.

Every value that crosses a module boundary is one of these typed models. This is
deliberate: in the original n8n implementation, data moved between nodes as
untyped JSON blobs, which made it impossible to reason about what a given agent
could rely on. Porting to explicit Pydantic models turns those implicit
contracts into enforced ones — an agent that emits a malformed ``Lead`` fails at
construction time, not three nodes downstream when a Google Sheets write
silently drops a column.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """The finite set of conversation intents the router can dispatch to.

    Mirrors the ``[[INTENT:x]]`` tag scheme from the original router prompt. Kept
    as a closed enum rather than free-form strings so that an unrecognised intent
    is a programming error, not a silent routing failure.
    """

    QUALIFY = "qualify"      # buyer — property purchase interest
    SELLER = "seller"        # listing a property for sale / valuation
    INVESTOR = "investor"    # ROI, rental yield, buy-to-let, portfolio
    RENT = "rent"            # letting or renting a property
    RESCHEDULE = "reschedule"
    CANCEL = "cancel"
    FOLLOWUP = "followup"    # status check on an existing lead
    GENERAL = "general"      # greetings, FAQ, anything uncategorised (was "fals")

    @classmethod
    def from_tag(cls, raw: str) -> Intent:
        """Parse a raw router output such as ``[[INTENT:seller]]`` into an Intent.

        Falls back to :attr:`GENERAL` on anything unparseable. The router is an
        LLM and will occasionally wrap the tag in prose despite instructions;
        we extract defensively rather than trust the format.
        """
        token = raw.strip().lower()
        if "[[intent:" in token:
            token = token.split("[[intent:", 1)[1].split("]]", 1)[0].strip()
        # legacy alias from the original workflow
        if token == "fals":
            return cls.GENERAL
        try:
            return cls(token)
        except ValueError:
            return cls.GENERAL


class LeadStage(str, Enum):
    """Lifecycle stage of a lead record in the datastore."""

    NEW = "New"
    BOOKED = "Booked"
    RESCHEDULED = "Rescheduled"
    CANCELLED = "Cancelled"


class LeadScore(str, Enum):
    """Silent qualification score. Never surfaced to the client."""

    HOT = "HOT"
    WARM = "Warm"
    COLD = "Cold"


class Lead(BaseModel):
    """A captured lead ready to persist and book.

    This is the ``SELLER_DONE:{...}`` / ``QUALIFY_DONE:{...}`` payload from the
    original agents, promoted to a validated model. Fields common to every
    vertical live here; vertical-specific attributes go in ``extra``.
    """

    full_name: str
    email: str
    phone: str
    intent: Intent
    meeting_date: str = Field(..., description="YYYY-MM-DD, Mon–Sat, never a day name")
    meeting_time: str = Field(..., description="HH:MM, 09:00–18:00")
    timezone: str = "PKT"
    stage: LeadStage = LeadStage.NEW
    score: LeadScore | None = None
    calendar_event_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def dedup_key(self) -> tuple[str, str]:
        """Return normalised (email, phone) used for duplicate detection.

        Normalisation mirrors the original JS exactly: lowercase-trim the email,
        strip whitespace and a single leading ``+`` from the phone. A match on
        *either* component is treated as a duplicate (see
        :class:`~prime_estate.tools.datastore.LeadDatastore`).
        """
        email = self.email.lower().strip()
        phone = self.phone.replace(" ", "").lstrip("+").strip()
        return email, phone


class SlotCheck(BaseModel):
    """Result of checking a lead against the datastore before booking."""

    is_duplicate: bool
    existing_row: int | None = None
    slot_taken: bool


class Session(BaseModel):
    """Per-user conversation state, keyed by WhatsApp sender number.

    Holds the sticky intent so that a user mid-flow (e.g. answering the seller
    agent's questions) is not re-classified on every turn. TTL and override
    behaviour are enforced by :class:`~prime_estate.core.session.SessionStore`,
    not here — this model is a plain data carrier.
    """

    session_id: str
    intent: Intent | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    collected: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        """Refresh the last-activity timestamp (extends the TTL window)."""
        self.updated_at = time.time()


class InboundMessage(BaseModel):
    """A normalised inbound message, decoupled from the WhatsApp payload shape."""

    session_id: str
    text: str
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentReply(BaseModel):
    """What an agent hands back to the orchestrator after processing a turn.

    ``lead`` is populated only on the final turn, when the agent has collected
    and confirmed every field and would have emitted its DONE signal in the
    original design. ``is_final`` drives the orchestrator's decision to run the
    booking pipeline.
    """

    text: str
    is_final: bool = False
    lead: Lead | None = None
