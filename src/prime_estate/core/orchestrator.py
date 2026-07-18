"""The orchestrator — one inbound message in, one reply out.

This is the spine the whole architecture hangs off, and the clearest statement
of the design thesis: the model drives the conversation, but every
consequential action passes through typed, testable Python boundaries.

Trace a message through and count where the LLM is allowed to decide:

1. **Session resolution** (pure Python) — TTL, override keywords, sticky
   intent. Decides *whether* the model gets to classify at all.
2. **Routing** (model decides, inside a closed enum) — only when there is no
   sticky session. The worst possible model output degrades to GENERAL.
3. **The agent turn** (model decides, inside its field schema / tool loop) —
   conversation wording, which field to ask next, when to signal completion.
4. **The booking pipeline** (pure Python) — validation already ran inside the
   agent; here the lead faces duplicate detection, slot-conflict detection,
   and the calendar's own race re-check. The model cannot bypass any of it:
   a booking happens because ``datastore.check`` and ``calendar.create_event``
   said yes, never because the model said DONE.

Everything is injected through the constructor. The orchestrator owns no I/O,
no credentials, and no LLM specifics — which is why the integration tests run
the entire loop with in-memory tools and a scripted model.
"""

from __future__ import annotations

from prime_estate.core.intents import ConversationAgent
from prime_estate.core.router import IntentRouter
from prime_estate.core.session import SessionStore
from prime_estate.domain.models import AgentReply, InboundMessage, Intent, LeadStage
from prime_estate.llm.groq_client import ChatMessage
from prime_estate.tools.base import CalendarTool, LeadDatastore
from prime_estate.tools.calendar import SlotUnavailableError
from prime_estate.utils.dates import upcoming_dates
from prime_estate.utils.logging import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """Routes inbound messages to agents and guards the booking pipeline.

    Conversation history lives here (keyed by session id) rather than in
    :class:`SessionStore`: the session store answers the *policy* question
    "which intent is this user stuck to?", while history is *transport* state
    the agents need to stay coherent. Conflating them would give the TTL
    policy object a second job. Both are cleared together when a flow ends.
    """

    def __init__(
        self,
        *,
        router: IntentRouter,
        registry: dict[Intent, ConversationAgent],
        sessions: SessionStore,
        calendar: CalendarTool,
        datastore: LeadDatastore,
    ) -> None:
        self._router = router
        self._registry = registry
        self._sessions = sessions
        self._calendar = calendar
        self._datastore = datastore
        self._history: dict[str, list[ChatMessage]] = {}

    def handle(self, message: InboundMessage) -> AgentReply:
        """Process one inbound message end to end and return the reply."""
        now = message.received_at.timestamp()
        today = message.received_at.date().isoformat()
        history = self._history.setdefault(message.session_id, [])

        # Boundary 1-2: sticky session short-circuits the router entirely;
        # otherwise the model classifies into the closed enum.
        intent = self._sessions.resolve(session_id=message.session_id, text=message.text, now=now)
        if intent is None:
            intent = self._router.classify(text=message.text, history=history)
        session = self._sessions.remember(session_id=message.session_id, intent=intent, now=now)

        # Boundary 3: the agent turn.
        agent = self._registry[intent]
        reply = agent.handle_turn(
            message=message.text, session=session, history=history, today=today
        )

        # Boundary 4: the booking pipeline, for agents that produced a lead.
        if reply.is_final and reply.lead is not None:
            reply = self._book(reply=reply, session_id=message.session_id)

        history.append(ChatMessage(role="user", content=message.text))
        history.append(ChatMessage(role="assistant", content=reply.text))

        if reply.is_final:
            # Flow complete (booked, cancelled, or dead-ended): drop both the
            # sticky intent and the transcript so the next message starts clean.
            self._sessions.clear(session_id=message.session_id)
            self._history.pop(message.session_id, None)

        return reply

    def _book(self, *, reply: AgentReply, session_id: str) -> AgentReply:
        """Run duplicate/slot guards and, if clean, persist + book the lead.

        Outcomes:

        * duplicate -> the flow ends without a save; the client is pointed at
          reschedule/cancel (the original system's behaviour: one live booking
          per contact).
        * slot taken (either at check time or lost to a race inside
          ``create_event``) -> the flow STAYS OPEN (``is_final=False`` keeps
          the session sticky) and the client is offered real alternatives.
        * clean -> event created, stage set to Booked, lead persisted, session
          cleared by the caller.
        """
        lead = reply.lead
        assert lead is not None  # guarded by the caller

        check = self._datastore.check(lead=lead)
        if check.is_duplicate:
            logger.info("duplicate lead for %s (row %s); not saving", lead.email, check.existing_row)
            return AgentReply(
                text=(
                    "It looks like you already have a booking with us under these "
                    "contact details. If you would like to change or cancel it, "
                    "just say 'reschedule' or 'cancel' and I will sort it out."
                ),
                is_final=True,
            )

        if check.slot_taken:
            return self._offer_alternatives(lead_date=lead.meeting_date, lead_time=lead.meeting_time)

        try:
            event_id = self._calendar.create_event(lead=lead)
        except SlotUnavailableError:
            # Lost the race between check and create; same client experience
            # as an ordinary slot conflict.
            return self._offer_alternatives(lead_date=lead.meeting_date, lead_time=lead.meeting_time)

        lead.calendar_event_id = event_id
        lead.stage = LeadStage.BOOKED
        row = self._datastore.save(lead=lead)
        logger.info(
            "booked %s %s %s (row %d, event %s, score %s)",
            lead.email, lead.meeting_date, lead.meeting_time, row, event_id, lead.score,
        )
        return reply

    def _offer_alternatives(self, *, lead_date: str, lead_time: str) -> AgentReply:
        """Build a non-final reply offering up to three genuinely free slots.

        Alternatives are computed against the Protocol surface
        (``is_slot_free`` over the deterministic upcoming-dates table) rather
        than ``InMemoryCalendar.suggest_alternatives``, which is a convenience
        of one implementation and deliberately not part of the
        :class:`CalendarTool` contract. The orchestrator depends only on the
        contract.
        """
        alternatives = [
            f"{d.day_name} {d.iso_date} at {lead_time}"
            for d in upcoming_dates()
            if d.iso_date != lead_date and self._calendar.is_slot_free(iso_date=d.iso_date, time=lead_time)
        ][:3]
        if alternatives:
            offer = "; ".join(alternatives)
            text = (
                f"Unfortunately {lead_date} at {lead_time} has just been taken. "
                f"I do have these slots free: {offer}. Would any of those work?"
            )
        else:
            text = (
                f"Unfortunately {lead_date} at {lead_time} has just been taken, "
                "and the next two weeks are looking full at that hour. Could you "
                "suggest a different time of day?"
            )
        return AgentReply(text=text, is_final=False)
