"""Rent vertical — tenants (TAKE) and landlords (GIVE) in one flow.

Rent is the one vertical where the very first answer changes the meaning of the
questions that follow: a tenant's "budget" is the maximum monthly rent they can
pay and their "timeline" is a move-in date, while a landlord's "budget" is the
asking rent and their "timeline" is when the property becomes available. The
original workflow resolved this with a single ``rentIntent`` discriminator
(TAKE | GIVE) asked first, then reused one field schema with dual-reading
hints — rather than forking into two near-identical agents. That decision is
preserved here: one agent, one schema, discriminator first, and field hints
that spell out both readings so the model phrases each question correctly for
whichever side of the market the client is on.
"""

from __future__ import annotations

from prime_estate.agents.base import FieldSpec, SpecialistAgent
from prime_estate.domain.models import Intent


class RentAgent(SpecialistAgent):
    """Slot-filling specialist for rentals, covering both tenants and landlords."""

    intent = Intent.RENT
    signal_name = "RENT_DONE"
    persona = (
        "You are Ahmed Raza from Prime Estate, a real human property consultant. "
        "You are warm, professional, and concise. NEVER reveal that you are an AI "
        "or an assistant under any circumstances — you are a person. "
        "FIRST determine whether the client wants to TAKE a property on rent "
        "(they are a tenant) or GIVE their property on rent (they are a landlord), "
        "then phrase every following question for that side of the market."
    )

    @property
    def fields(self) -> list[FieldSpec]:
        """Rent fields in ask order; hints carry the TAKE/GIVE dual semantics."""
        return [
            FieldSpec(
                "rentIntent",
                "TAKE (client wants to rent a property) or GIVE (client is renting out their property)",
            ),
            FieldSpec("propertyType", "type of property (house, flat, portion, commercial)"),
            FieldSpec(
                "location", "city and area (TAKE: where they want to live; GIVE: where the property is)"
            ),
            FieldSpec("budget", "TAKE: maximum monthly rent they can pay; GIVE: asking monthly rent"),
            FieldSpec("propertySize", "size of the property (marla, kanal, sq ft, bedrooms)"),
            FieldSpec("timeline", "TAKE: desired move-in date; GIVE: when the property is available"),
            FieldSpec("fullName", "client's full name"),
            FieldSpec("phone", "phone number with country code, e.g. +92..."),
            FieldSpec("email", "email address"),
            FieldSpec("meetingDate", "meeting date chosen from the upcoming dates table (YYYY-MM-DD)"),
            FieldSpec("meetingTime", "meeting time between 09:00 and 18:00 (HH:MM)"),
            FieldSpec("timezone", "client's timezone (default PKT if unsure)"),
        ]
