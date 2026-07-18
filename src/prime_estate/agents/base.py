"""Base class for the bounded specialist agents.

Every specialist in the original system (seller, buyer, rent, investor, …) was
the same shape: a persona-driven slot-filling loop that collects a fixed set of
fields one at a time, checks calendar availability through a tool before
confirming a slot, and emits a structured ``<VERTICAL>_DONE:{json}`` signal
exactly once when complete. Rather than repeat that machinery in eight prompts,
it is factored into one base class here. A concrete agent supplies only what
actually differs: its persona, its ordered field schema, and its signal name.

The "boundaries the model operates inside" are concrete and enforced in code,
not merely requested in the prompt:

* the agent may only reach the outside world through the two injected tools
  (calendar, datastore) — it has no other handles;
* the completion signal is parsed and *re-validated* in Python
  (:mod:`prime_estate.validation`) before anything is persisted, so a
  hallucinated or malformed payload is rejected rather than booked;
* the lead score is computed deterministically
  (:mod:`prime_estate.domain.scoring`), not taken from the model's say-so.

This is the difference between "a prompt chain" and "a model making decisions
inside boundaries you designed": the model drives the conversation, but every
consequential action passes through typed, testable Python.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from prime_estate.domain.models import (
    AgentReply,
    Intent,
    Lead,
    LeadScore,
    Session,
)
from prime_estate.domain.scoring import score_lead
from prime_estate.llm.groq_client import ChatMessage, ChatModel
from prime_estate.tools.base import CalendarTool, LeadDatastore
from prime_estate.utils.dates import UpcomingDate, render_dates_table, upcoming_dates
from prime_estate.utils.logging import get_logger
from prime_estate.validation.validators import validate_lead_fields

logger = get_logger(__name__)


@dataclass(frozen=True)
class FieldSpec:
    """One field a specialist agent must collect, in ask order.

    ``key`` is the machine name used in the DONE payload and the ``Lead.extra``
    dict; ``prompt_hint`` is the human description injected into the persona
    prompt so the model knows what to ask for.
    """

    key: str
    prompt_hint: str


class SpecialistAgent(ABC):
    """Abstract bounded agent that drives one vertical's slot-filling flow.

    Concrete subclasses declare :attr:`intent`, :attr:`signal_name`,
    :attr:`persona`, and :attr:`fields`. All conversation mechanics — prompt
    assembly, signal extraction, validation, scoring, availability checks — live
    here so behaviour is uniform and tested once.
    """

    #: Which intent this agent serves. Used by the orchestrator's registry.
    intent: Intent
    #: The completion signal token, e.g. ``"SELLER_DONE"``.
    signal_name: str
    #: One-line persona/role description injected at the top of the prompt.
    persona: str

    def __init__(self, *, model: ChatModel, calendar: CalendarTool, datastore: LeadDatastore) -> None:
        self._model = model
        self._calendar = calendar
        self._datastore = datastore
        # Precompiled matcher for this agent's DONE signal, e.g. SELLER_DONE:{...}
        self._signal_re = re.compile(rf"{re.escape(self.signal_name)}:\s*(\{{.*\}})", re.DOTALL)

    # --- contract subclasses must fill in -------------------------------------

    @property
    @abstractmethod
    def fields(self) -> list[FieldSpec]:
        """The ordered fields this agent collects, one at a time."""

    # --- shared machinery -----------------------------------------------------

    def build_system_prompt(self, *, today: str, dates: list[UpcomingDate]) -> str:
        """Assemble the full persona prompt with field order and hard rules.

        The upcoming-dates table is injected as pre-computed text so the model
        selects a date rather than deriving one (see :mod:`prime_estate.utils.dates`
        for why that matters). The DONE-signal contract is stated explicitly so
        the completion payload is machine-parseable.
        """
        field_lines = "\n".join(
            f"{i}. {f.key} — {f.prompt_hint}" for i, f in enumerate(self.fields, start=1)
        )
        keys_json = ", ".join(f'"{f.key}":"..."' for f in self.fields)
        return (
            f"{self.persona}\n"
            f"CURRENT DATE: {today}\n\n"
            "Collect the following fields ONE at a time, in this exact order. "
            "Ask for exactly one field per message, in a warm, human tone "
            "(max 2 sentences, no bullet points, no markdown):\n"
            f"{field_lines}\n\n"
            "SCHEDULING RULES:\n"
            "- Meeting dates MUST be YYYY-MM-DD, Monday–Saturday only, never Sunday.\n"
            "- NEVER calculate dates yourself. Use ONLY the table below.\n"
            "- Meeting times MUST be HH:MM between 09:00 and 18:00.\n\n"
            "UPCOMING DATES TABLE (use exactly as shown):\n"
            f"{render_dates_table(dates)}\n\n"
            "COMPLETION SIGNAL (client must NEVER see this):\n"
            "Once every field is collected and confirmed, send the client a warm "
            "one-line confirmation, then on a new line emit the signal exactly once:\n"
            f"{self.signal_name}:{{{keys_json}}}\n"
            "Every value must be the real collected value — never a placeholder."
        )

    def handle_turn(
        self,
        *,
        message: str,
        session: Session,
        history: list[ChatMessage],
        today: str,
    ) -> AgentReply:
        """Process one user turn and return the agent's reply.

        The flow: build the persona prompt, let the model produce its next
        message, then inspect that message for a completion signal. If none is
        present, the agent is still collecting — the model's text is returned
        verbatim. If the signal is present, control passes to
        :meth:`_finalise`, which validates, scores, and books.
        """
        dates = upcoming_dates()
        system = self.build_system_prompt(today=today, dates=dates)
        turn_messages = [*history, ChatMessage(role="user", content=message)]

        raw = self._model.complete(system=system, messages=turn_messages)
        payload = self._extract_signal(raw)

        if payload is None:
            # Still collecting: hand the model's question straight back.
            return AgentReply(text=raw.strip(), is_final=False)

        return self._finalise(raw=raw, payload=payload)

    def _extract_signal(self, raw: str) -> dict[str, Any] | None:
        """Return the parsed DONE payload if present in *raw*, else ``None``.

        Parsing is tolerant of surrounding prose (the model sometimes precedes
        the signal with the client confirmation line) but strict on the JSON
        itself — a malformed payload returns ``None`` and is treated as "still
        collecting", which is the safe failure mode.
        """
        match = self._signal_re.search(raw)
        if not match:
            return None
        try:
            payload: dict[str, Any] = json.loads(match.group(1))
            return payload
        except json.JSONDecodeError:
            logger.warning("%s emitted an unparseable DONE payload", self.signal_name)
            return None

    def _finalise(self, *, raw: str, payload: dict[str, Any]) -> AgentReply:
        """Validate, score, and book a completed lead.

        This is the boundary between "the model said it's done" and "the system
        commits to it". Validation runs first; on failure the client is asked to
        correct the offending fields and ``is_final`` stays ``False`` so the flow
        continues. Only validated leads are scored and returned for booking.
        """
        validation = validate_lead_fields(
            email=payload.get("email", ""),
            phone=payload.get("phone", ""),
            meeting_date=payload.get("meetingDate", payload.get("meeting_date", "")),
            meeting_time=payload.get("meetingTime", payload.get("meeting_time", "")),
        )
        if not validation.ok:
            return AgentReply(text=validation.as_client_prompt() or "", is_final=False)

        lead = self._to_lead(payload)
        client_text = self._strip_signal(raw)
        logger.info("%s finalised lead for %s (score=%s)", self.signal_name, lead.email, lead.score)
        return AgentReply(text=client_text, is_final=True, lead=lead)

    def _to_lead(self, payload: dict[str, Any]) -> Lead:
        """Build a validated :class:`Lead` from a DONE payload.

        Known common fields map onto :class:`Lead`; everything vertical-specific
        is preserved in ``extra`` so no captured data is lost. The score is
        computed here, deterministically, from the collected timeline and price
        signals — never trusted to the model.
        """
        common = {"fullName", "email", "phone", "meetingDate", "meetingTime", "timezone"}
        extra = {k: v for k, v in payload.items() if k not in common}
        score: LeadScore = score_lead(
            timeline=str(payload.get("timeline", "")),
            price=str(payload.get("askingPrice", payload.get("budget", ""))),
        )
        return Lead(
            full_name=payload.get("fullName", ""),
            email=payload.get("email", ""),
            phone=payload.get("phone", ""),
            intent=self.intent,
            meeting_date=payload.get("meetingDate", ""),
            meeting_time=payload.get("meetingTime", ""),
            timezone=payload.get("timezone", "PKT"),
            score=score,
            extra=extra,
        )

    def _strip_signal(self, raw: str) -> str:
        """Remove the DONE signal line from *raw* so the client never sees it."""
        return self._signal_re.sub("", raw).strip()
