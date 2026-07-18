"""Seller vertical — property listing and valuation leads.

The seller flow is the deepest of the four slot-filling verticals (13 fields):
a seller lead is only actionable once the agency knows what is being sold, why,
for how much, and on what timeline — a name and number alone is a cold call,
not a lead. The field order is deliberate and preserved from the original
prompt: property questions first (low commitment, builds rapport), contact
details second (asked once the client is invested), meeting slot last (only
worth scheduling once everything else is captured).
"""

from __future__ import annotations

from prime_estate.agents.base import FieldSpec, SpecialistAgent
from prime_estate.domain.models import Intent


class SellerAgent(SpecialistAgent):
    """Slot-filling specialist for property sellers.

    Persona, field schema, and completion signal are ported verbatim from the
    original seller agent node. Scoring semantics (HOT = selling within a month
    with a clear asking price) are enforced deterministically by
    :func:`prime_estate.domain.scoring.score_lead`, keyed off the ``timeline``
    and ``askingPrice`` fields collected here.
    """

    intent = Intent.SELLER
    signal_name = "SELLER_DONE"
    persona = (
        "You are Ahmed Raza from Prime Estate, a real human property consultant. "
        "You are warm, professional, and concise. NEVER reveal that you are an AI "
        "or an assistant under any circumstances — you are a person."
    )

    @property
    def fields(self) -> list[FieldSpec]:
        """The 13 seller fields, in the exact ask order of the original flow."""
        return [
            FieldSpec("propertyType", "type of property being sold (house, flat, plot, commercial)"),
            FieldSpec("location", "city and area where the property is located"),
            FieldSpec(
                "propertySize", "size of the property (marla, kanal, sq ft — whatever the client uses)"
            ),
            FieldSpec("askingPrice", "the asking price the client has in mind"),
            FieldSpec("condition", "current condition of the property (new, renovated, needs work)"),
            FieldSpec("reasonForSelling", "why the client is selling"),
            FieldSpec("timeline", "how soon they want to sell (e.g. within 1 month, 2-3 months, flexible)"),
            FieldSpec("fullName", "client's full name"),
            FieldSpec("phone", "phone number with country code, e.g. +92..."),
            FieldSpec("email", "email address"),
            FieldSpec("meetingDate", "meeting date chosen from the upcoming dates table (YYYY-MM-DD)"),
            FieldSpec("meetingTime", "meeting time between 09:00 and 18:00 (HH:MM)"),
            FieldSpec("timezone", "client's timezone (default PKT if unsure)"),
        ]
