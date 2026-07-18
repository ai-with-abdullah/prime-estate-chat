"""Cancellation flow — verify identity, then release a booking.

Cancellation is the most destructive action in the system (it frees a calendar
slot and dead-ends a lead), so its boundary is the strictest: the record must
be located via BOTH email and phone (``find_by_contact``), and the actual
mutation in :meth:`CancelAgent._apply` operates on the record the datastore
returned during lookup — never on values echoed back by the model. The model's
tag block triggers the action; Python decides what the action touches.
"""

from __future__ import annotations

from typing import Any

from prime_estate.agents.lookup import LookupAgent
from prime_estate.domain.models import AgentReply, Intent, LeadStage, Session
from prime_estate.utils.logging import get_logger

logger = get_logger(__name__)

# Lead intent -> the CANCEL_TYPE vocabulary of the original tag scheme. Derived
# in Python from the verified record rather than trusted from the model's tag.
_INTENT_TO_TYPE: dict[Intent, str] = {
    Intent.QUALIFY: "buyer",
    Intent.SELLER: "seller",
    Intent.RENT: "renter",
    Intent.INVESTOR: "investor",
}


class CancelAgent(LookupAgent):
    """Lookup agent that cancels an existing booking after identity checks."""

    intent = Intent.CANCEL
    persona = (
        "You are Ahmed Raza from Prime Estate, a real human property consultant. "
        "You are warm, professional, and concise. NEVER reveal that you are an AI. "
        "A client wants to cancel their booked meeting."
    )
    final_tags = ("CANCEL_EMAIL", "CANCEL_TYPE", "CANCEL_REASON", "CANCEL_NAME")

    def _flow_rules(self) -> str:
        """Cancellation flow, ported from the original cancel agent."""
        return (
            "FLOW RULES:\n"
            "1. Ask for the client's email AND phone number together, in one "
            "message. Both are required to locate the booking.\n"
            '2. When you have both, emit LOOKUP:{"email":"...","phone":"..."}.\n'
            "3. If the tool result says no record was found, apologise, say you "
            "could not find a booking under those details, and stop — do NOT "
            "emit any tags.\n"
            "4. If a record was found, confirm the client's name, meeting date "
            "and time back to them, then ask the reason for cancelling.\n"
            "5. Only after the reason is given, confirm the cancellation and "
            "emit the tags. CANCEL_TYPE is one of: buyer, seller, renter, "
            "investor.\n"
        )

    def _run_lookup(self, *, payload: dict[str, Any], session: Session) -> str:
        """Locate the booking by email + phone; stash the verified identity."""
        email = str(payload.get("email", ""))
        phone = str(payload.get("phone", ""))
        found = self._datastore.find_by_contact(email=email, phone=phone)
        if found is None:
            return "NO RECORD FOUND for that email and phone combination."
        row, lead = found
        # Stash the verified contact pair, not the row's contents: _apply
        # re-reads the datastore at action time so a record updated between
        # turns is never acted on from a stale copy.
        session.collected["verified_email"] = email
        session.collected["verified_phone"] = phone
        return (
            f"RECORD FOUND (row {row}): name={lead.full_name}, "
            f"type={_INTENT_TO_TYPE.get(lead.intent, 'buyer')}, "
            f"meeting={lead.meeting_date} {lead.meeting_time}, stage={lead.stage.value}"
        )

    def _apply(self, *, tags: dict[str, str], session: Session, raw: str) -> AgentReply:
        """Cancel the verified booking: stage -> Cancelled, calendar event freed.

        The record is re-fetched using the identity verified during lookup. If
        that identity is missing (the model skipped the lookup step and jumped
        straight to tags), the action is refused — tags alone are not proof a
        record exists.
        """
        email = str(session.collected.get("verified_email", ""))
        phone = str(session.collected.get("verified_phone", ""))
        found = self._datastore.find_by_contact(email=email, phone=phone) if email and phone else None
        if found is None:
            logger.warning("cancel tags emitted without a verified lookup; refusing")
            return AgentReply(
                text=(
                    "I could not verify a booking under those details. Could you "
                    "share the email and phone number you booked with?"
                ),
                is_final=False,
            )

        row, lead = found
        cancelled = lead.model_copy(update={"stage": LeadStage.CANCELLED})
        self._datastore.update(row=row, lead=cancelled)
        if lead.calendar_event_id:
            self._calendar.delete_event(event_id=lead.calendar_event_id)
        logger.info(
            "cancelled booking row=%d email=%s reason=%r",
            row, lead.email, tags.get("CANCEL_REASON", ""),
        )
        return AgentReply(text=self._strip_machine_lines(raw), is_final=True)
