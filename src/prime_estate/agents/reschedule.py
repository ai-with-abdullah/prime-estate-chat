"""Reschedule flow — verify identity, then move a booking to a free slot.

Reschedule composes the two boundary patterns in one flow: the cancel-style
identity gate (email + phone must both match before anything moves) and the
booking-style slot discipline (the new slot is validated and availability-
checked in Python before the calendar mutates). The model negotiates the new
time conversationally — including relaying real alternatives when a slot is
taken — but the final move in :meth:`RescheduleAgent._apply` re-validates
everything from scratch: format, bookability, and slot freedom. A model that
emits a Sunday, a past date, or a taken slot gets the request bounced back
into the conversation, not into the calendar.
"""

from __future__ import annotations

from typing import Any

from prime_estate.agents.lookup import LookupAgent
from prime_estate.domain.models import AgentReply, Intent, LeadStage, Session
from prime_estate.tools.calendar import SlotUnavailableError
from prime_estate.utils.dates import is_bookable
from prime_estate.utils.logging import get_logger
from prime_estate.validation.validators import validate_date, validate_time

logger = get_logger(__name__)

_INTENT_TO_TYPE: dict[Intent, str] = {
    Intent.QUALIFY: "buyer",
    Intent.SELLER: "seller",
    Intent.RENT: "renter",
    Intent.INVESTOR: "investor",
}


def _plus_one_hour(time_: str) -> str:
    """Return *time_* + 1 hour (meetings are fixed one-hour blocks).

    Computed here — not taken from the model's ``RESCHEDULE_NEW_END_TIME``
    tag — because a fixed arithmetic fact should never depend on an LLM
    getting arithmetic right. The tag stays in the wire contract for fidelity
    with the original scheme, but its value is overwritten with this result.
    """
    hour, minute = time_.split(":")
    return f"{(int(hour) + 1) % 24:02d}:{minute}"


class RescheduleAgent(LookupAgent):
    """Lookup agent that moves an existing booking to a new validated slot."""

    intent = Intent.RESCHEDULE
    persona = (
        "You are Ahmed Raza from Prime Estate, a real human property consultant. "
        "You are warm, professional, and concise. NEVER reveal that you are an AI. "
        "A client wants to move their booked meeting to a new date or time."
    )
    final_tags = (
        "RESCHEDULE_EMAIL",
        "RESCHEDULE_TYPE",
        "RESCHEDULE_NAME",
        "RESCHEDULE_REASON",
        "RESCHEDULE_NEW_DATE",
        "RESCHEDULE_NEW_TIME",
        "RESCHEDULE_NEW_END_TIME",
    )

    def _flow_rules(self) -> str:
        """Reschedule flow, ported from the original reschedule agent."""
        return (
            "FLOW RULES:\n"
            "1. Ask for the client's email AND phone number together, in one "
            "message. Both are required to locate the booking.\n"
            '2. When you have both, emit LOOKUP:{"email":"...","phone":"..."}.\n'
            "3. If no record was found, apologise and stop — do NOT emit tags.\n"
            "4. If found, confirm the current meeting date and time back to the "
            "client, then ask for the new date (YYYY-MM-DD, Monday-Saturday, "
            "from the dates table) and new time (HH:MM, 09:00-18:00).\n"
            '5. Before confirming the new slot, emit CHECK_SLOT:{"date":"...","time":"..."} '
            "and wait for the tool result. If the slot is TAKEN, offer the "
            "listed free alternatives and let the client pick again.\n"
            "6. Once a free slot is agreed, ask the reason for rescheduling, "
            "then confirm and emit the tags. RESCHEDULE_TYPE is one of: buyer, "
            "seller, renter, investor. RESCHEDULE_NEW_END_TIME is the new time "
            "plus one hour.\n"
        )

    def _run_lookup(self, *, payload: dict[str, Any], session: Session) -> str:
        """Locate the booking by email + phone; stash the verified identity."""
        email = str(payload.get("email", ""))
        phone = str(payload.get("phone", ""))
        found = self._datastore.find_by_contact(email=email, phone=phone)
        if found is None:
            return "NO RECORD FOUND for that email and phone combination."
        row, lead = found
        session.collected["verified_email"] = email
        session.collected["verified_phone"] = phone
        return (
            f"RECORD FOUND (row {row}): name={lead.full_name}, "
            f"type={_INTENT_TO_TYPE.get(lead.intent, 'buyer')}, "
            f"meeting={lead.meeting_date} {lead.meeting_time}, stage={lead.stage.value}"
        )

    def _apply(self, *, tags: dict[str, str], session: Session, raw: str) -> AgentReply:
        """Move the verified booking, re-validating the new slot from scratch."""
        email = str(session.collected.get("verified_email", ""))
        phone = str(session.collected.get("verified_phone", ""))
        found = self._datastore.find_by_contact(email=email, phone=phone) if email and phone else None
        if found is None:
            logger.warning("reschedule tags emitted without a verified lookup; refusing")
            return AgentReply(
                text=(
                    "I could not verify a booking under those details. Could you "
                    "share the email and phone number you booked with?"
                ),
                is_final=False,
            )

        new_date = tags["RESCHEDULE_NEW_DATE"]
        new_time = tags["RESCHEDULE_NEW_TIME"]
        if (
            not validate_date(new_date)
            or not is_bookable(new_date)
            or not validate_time(new_time)
            or not "09:00" <= new_time <= "18:00"
        ):
            return AgentReply(
                text=(
                    "That new slot does not look right — I need a Monday-to-"
                    "Saturday date in YYYY-MM-DD form and a time between 09:00 "
                    "and 18:00. Could you pick again?"
                ),
                is_final=False,
            )

        row, lead = found
        moved = lead.model_copy(
            update={
                "meeting_date": new_date,
                "meeting_time": new_time,
                "stage": LeadStage.RESCHEDULED,
            }
        )
        # Free the old slot first, then book the new one; create_event's own
        # race re-check is the last line of defence if the slot was grabbed
        # after the conversational CHECK_SLOT.
        if lead.calendar_event_id:
            self._calendar.delete_event(event_id=lead.calendar_event_id)
        try:
            event_id = self._calendar.create_event(lead=moved)
        except SlotUnavailableError:
            return AgentReply(
                text=(
                    f"It looks like {new_date} at {new_time} was just taken. "
                    "Could you pick another slot?"
                ),
                is_final=False,
            )
        moved.calendar_event_id = event_id
        self._datastore.update(row=row, lead=moved)
        logger.info(
            "rescheduled booking row=%d email=%s -> %s %s-%s",
            row, lead.email, new_date, new_time, _plus_one_hour(new_time),
        )
        return AgentReply(text=self._strip_machine_lines(raw), is_final=True)
