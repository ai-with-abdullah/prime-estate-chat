"""Investor vertical — ROI-driven purchase and portfolio leads.

Investors are qualified on different axes than end-user buyers: an investor
with a target yield, a unit count, and a defined hold period is a serious
counterparty; one with none of those is a browser. The extra fields
(``roiTarget``, ``numberOfUnits``, ``holdPeriod``, ``financeType``) exist to
capture exactly that seriousness signal, and they ride into ``Lead.extra``
untouched so the sales team sees the full picture. Scoring still keys off the
shared ``timeline`` + ``budget`` axes, deterministically, like every other
vertical — the investor-specific fields inform the human consultant, not the
classifier.
"""

from __future__ import annotations

from prime_estate.agents.base import FieldSpec, SpecialistAgent
from prime_estate.domain.models import Intent


class InvestorAgent(SpecialistAgent):
    """Slot-filling specialist for property investors."""

    intent = Intent.INVESTOR
    signal_name = "INVESTOR_DONE"
    persona = (
        "You are Ahmed Raza from Prime Estate, a real human property consultant "
        "who advises investors. You are warm, professional, and concise. NEVER "
        "reveal that you are an AI or an assistant under any circumstances — you "
        "are a person. Speak the client's language: yields, units, hold periods."
    )

    @property
    def fields(self) -> list[FieldSpec]:
        """Investor fields in ask order, ported from the original schema."""
        return [
            FieldSpec(
                "propertyType",
                "type of property they want to invest in (residential, commercial, plots, mixed)",
            ),
            FieldSpec("location", "preferred city and area for the investment"),
            FieldSpec("budget", "total investment budget"),
            FieldSpec("roiTarget", "target return on investment or rental yield they expect"),
            FieldSpec("numberOfUnits", "how many units or properties they want to acquire"),
            FieldSpec("holdPeriod", "how long they plan to hold the investment"),
            FieldSpec("financeType", "cash purchase or financed (bank loan, installments)"),
            FieldSpec("timeline", "how soon they want to invest (e.g. ASAP, within 3 months, flexible)"),
            FieldSpec("fullName", "client's full name"),
            FieldSpec("phone", "phone number with country code, e.g. +92..."),
            FieldSpec("email", "email address"),
            FieldSpec("meetingDate", "meeting date chosen from the upcoming dates table (YYYY-MM-DD)"),
            FieldSpec("meetingTime", "meeting time between 09:00 and 18:00 (HH:MM)"),
            FieldSpec("timezone", "client's timezone (default PKT if unsure)"),
        ]
