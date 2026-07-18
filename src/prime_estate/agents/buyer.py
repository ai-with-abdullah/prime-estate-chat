"""Buyer vertical — property purchase qualification.

The buyer flow is two-staged in the original design: qualify first (is this a
real buyer, with a real budget and timeline?), collect contact details second.
The stage split is a funnel decision — asking for a phone number before the
client has said what they want loses leads, so contact capture is deliberately
deferred until qualification is complete. The base class collects fields in
declared order, which preserves exactly that staging; the persona text makes
the two-stage intent explicit to the model so it frames the transition
naturally ("great, let me grab a few details so our consultant can call you").
"""

from __future__ import annotations

import re

from prime_estate.agents.base import FieldSpec, SpecialistAgent
from prime_estate.domain.models import Intent
from prime_estate.llm.groq_client import ChatModel
from prime_estate.tools.base import CalendarTool, LeadDatastore


class BuyerAgent(SpecialistAgent):
    """Slot-filling specialist for property buyers.

    Deviation note (signal token): in the original workflow the buyer's
    completion token is literally ``DONE`` — not ``BUYER_DONE`` — and that is
    preserved here for fidelity. A bare ``DONE:`` is fragile, though: it is a
    substring of every other agent's token (``SELLER_DONE``, ``RENT_DONE``, ...)
    and could in principle match mid-word in model prose. Rather than rename
    the token and diverge from the production contract, the matcher is
    recompiled with a negative lookbehind so ``DONE:`` only matches when not
    preceded by an identifier character. Same wire format, safer parse.
    """

    intent = Intent.QUALIFY
    signal_name = "DONE"
    persona = (
        "You are Ahmed Raza from Prime Estate, a real human property consultant. "
        "You are warm, professional, and concise. NEVER reveal that you are an AI "
        "or an assistant under any circumstances — you are a person. "
        "Work in two stages: STAGE 1 qualifies the buyer (their intent, property "
        "type, location, budget, timeline). Only once stage 1 is complete, move "
        "to STAGE 2 and collect contact details plus a meeting slot."
    )

    def __init__(self, *, model: ChatModel, calendar: CalendarTool, datastore: LeadDatastore) -> None:
        super().__init__(model=model, calendar=calendar, datastore=datastore)
        # See class docstring: keep the production `DONE` token, but refuse to
        # match it as the tail of another agent's token or a longer identifier.
        self._signal_re = re.compile(r"(?<![A-Za-z_])DONE:\s*(\{.*\})", re.DOTALL)

    @property
    def fields(self) -> list[FieldSpec]:
        """Stage 1 (qualify) then stage 2 (contact + meeting), in ask order."""
        return [
            # STAGE 1 — qualification
            FieldSpec(
                "intent", "what the client is looking to do (buy to live, buy to resell, first home, upgrade)"
            ),
            FieldSpec("propertyType", "type of property they want (house, flat, plot, commercial)"),
            FieldSpec("location", "preferred city and area"),
            FieldSpec("budget", "their purchase budget"),
            FieldSpec("timeline", "how soon they want to buy (e.g. ASAP, within 3 months, flexible)"),
            # STAGE 2 — contact + meeting
            FieldSpec("fullName", "client's full name"),
            FieldSpec("phone", "phone number with country code, e.g. +92..."),
            FieldSpec("email", "email address"),
            FieldSpec("meetingDate", "meeting date chosen from the upcoming dates table (YYYY-MM-DD)"),
            FieldSpec("meetingTime", "meeting time between 09:00 and 18:00 (HH:MM)"),
        ]
