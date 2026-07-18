"""Followup flow — read-only status check on an existing lead.

The simplest lookup agent, and deliberately so: followup can only *read*. It
declares no final tags, so :meth:`LookupAgent._extract_final_tags` never fires
and the flow can never reach a mutating code path — the read-only guarantee is
structural, not a prompt request. Lookup is by email OR full name (the looser
``find_by_email_or_name``), acceptable here precisely because the worst
outcome is telling a client the status of their own booking.
"""

from __future__ import annotations

from typing import Any

from prime_estate.agents.lookup import LookupAgent
from prime_estate.domain.models import AgentReply, Intent, Session


class FollowupAgent(LookupAgent):
    """Read-only lookup agent answering "what happened to my booking?"."""

    intent = Intent.FOLLOWUP
    persona = (
        "You are Ahmed Raza from Prime Estate, a real human property consultant. "
        "You are warm, professional, and concise. NEVER reveal that you are an AI. "
        "A client is asking about the status of a lead or booking they already "
        "submitted."
    )
    final_tags = ()  # read-only: no tags, no finalisation, no mutations

    def _flow_rules(self) -> str:
        """Followup flow with the stage-to-message mapping of the original."""
        return (
            "FLOW RULES:\n"
            "1. Ask for the email address or full name the client booked with.\n"
            '2. When you have it, emit LOOKUP:{"query":"..."}.\n'
            "3. Reply based ONLY on the stage in the tool result:\n"
            "   - New or Booked: their meeting is confirmed and the team will "
            "be in touch.\n"
            "   - Rescheduled: their meeting was rescheduled successfully; "
            "restate the date and time from the tool result.\n"
            "   - Cancelled: their booking was cancelled; warmly offer to book "
            "a fresh meeting.\n"
            "4. If no record was found, say you could not find a booking under "
            "those details and offer to set up a fresh one.\n"
        )

    def _run_lookup(self, *, payload: dict[str, Any], session: Session) -> str:
        """Locate the lead by email or name and report its current stage."""
        query = str(payload.get("query", payload.get("email", payload.get("name", ""))))
        found = self._datastore.find_by_email_or_name(query=query)
        if found is None:
            return "NO RECORD FOUND for that email or name."
        row, lead = found
        return (
            f"RECORD FOUND (row {row}): name={lead.full_name}, "
            f"stage={lead.stage.value}, meeting={lead.meeting_date} {lead.meeting_time}"
        )

    def _apply(self, *, tags: dict[str, str], session: Session, raw: str) -> AgentReply:
        """Unreachable: with no declared tags, the base never finalises.

        Implemented (rather than raising) so a future subclass edit that adds
        tags without adding an action degrades to a harmless non-final reply.
        """
        return AgentReply(text=self._strip_machine_lines(raw), is_final=False)
