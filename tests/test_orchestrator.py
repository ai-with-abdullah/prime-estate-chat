"""End-to-end orchestrator tests with in-memory tools and a scripted model.

These are the integration proof of the architecture: one composition root,
zero network, and the full path — session, router, agent, booking pipeline —
exercised exactly as production wires it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from prime_estate.core.intents import build_registry
from prime_estate.core.orchestrator import Orchestrator
from prime_estate.core.router import IntentRouter
from prime_estate.core.session import SessionStore
from prime_estate.domain.models import InboundMessage, LeadStage
from prime_estate.tools.calendar import InMemoryCalendar
from prime_estate.tools.datastore import InMemoryLeadDatastore
from tests.conftest import ScriptedChatModel, make_lead

_T0 = datetime(2030, 1, 2, 10, 0, tzinfo=UTC)

_SELLER_PAYLOAD = {
    "propertyType": "house",
    "location": "DHA Lahore",
    "propertySize": "5 marla",
    "askingPrice": "45000000",
    "condition": "renovated",
    "reasonForSelling": "relocating",
    "timeline": "within 1 month",
    "fullName": "Ayesha Khan",
    "phone": "+923001234567",
    "email": "ayesha@example.com",
    "meetingDate": "2030-01-07",
    "meetingTime": "11:00",
    "timezone": "PKT",
}


def _build(
    responses: list[str],
) -> tuple[Orchestrator, ScriptedChatModel, InMemoryLeadDatastore, InMemoryCalendar]:
    model = ScriptedChatModel(responses)
    calendar = InMemoryCalendar()
    datastore = InMemoryLeadDatastore()
    orchestrator = Orchestrator(
        router=IntentRouter(model=model),
        registry=build_registry(model=model, calendar=calendar, datastore=datastore),
        sessions=SessionStore(),
        calendar=calendar,
        datastore=datastore,
    )
    return orchestrator, model, datastore, calendar


def _msg(text: str, *, session_id: str = "wa:+92300", minutes: int = 0) -> InboundMessage:
    return InboundMessage(
        session_id=session_id, text=text, received_at=_T0 + timedelta(minutes=minutes)
    )


class TestRoutingAndStickiness:
    def test_fresh_message_routes_then_stays_sticky(self) -> None:
        orchestrator, model, _, _ = _build(
            [
                "[[INTENT:seller]]",
                "What type of property are you selling?",
                "Which city and area is it in?",
            ]
        )
        r1 = orchestrator.handle(_msg("Hi, I want to sell my flat"))
        assert not r1.is_final
        assert len(model.calls) == 2  # router + seller agent

        r2 = orchestrator.handle(_msg("It is a 5 marla house", minutes=1))
        assert not r2.is_final
        # Sticky session: exactly one more model call (the agent), no router.
        assert len(model.calls) == 3
        assert "SELLER_DONE" in str(model.calls[2]["system"])

    def test_garbled_router_output_lands_in_general(self) -> None:
        orchestrator, model, _, _ = _build(
            [
                "erm, hard to say really",
                "Hello! How can I help you with your property needs today?",
            ]
        )
        reply = orchestrator.handle(_msg("???"))
        assert not reply.is_final
        assert "help" in reply.text


class TestBookingPipeline:
    def test_final_turn_books_persists_and_clears_session(self) -> None:
        orchestrator, model, datastore, calendar = _build(
            [
                "[[INTENT:seller]]",
                "You are all set!\nSELLER_DONE:" + json.dumps(_SELLER_PAYLOAD),
                "[[INTENT:general]]",
                "Hello again! How can I help?",
            ]
        )
        reply = orchestrator.handle(_msg("sell my house, details attached"))
        assert reply.is_final
        assert "SELLER_DONE" not in reply.text

        stored = datastore.all_leads()
        assert len(stored) == 1
        assert stored[0].stage is LeadStage.BOOKED
        assert stored[0].calendar_event_id is not None
        assert not calendar.is_slot_free(iso_date="2030-01-07", time="11:00")

        # Session was cleared: the next message is re-routed from scratch.
        orchestrator.handle(_msg("hello", minutes=2))
        assert "[[INTENT" not in str(model.calls[-1]["system"])
        assert len(model.calls) == 4  # router consulted again

    def test_taken_slot_offers_alternatives_and_keeps_flow_open(self) -> None:
        other = dict(_SELLER_PAYLOAD, email="other@example.com", phone="+929998887766")
        orchestrator, _, datastore, _ = _build(
            [
                "[[INTENT:seller]]",
                "Booked!\nSELLER_DONE:" + json.dumps(_SELLER_PAYLOAD),
                "[[INTENT:seller]]",
                "Booked!\nSELLER_DONE:" + json.dumps(other),
            ]
        )
        orchestrator.handle(_msg("sell", session_id="wa:first"))
        reply = orchestrator.handle(_msg("sell", session_id="wa:second"))

        # Same slot, different person: no booking, real alternatives offered,
        # and the flow stays open so the client can pick one.
        assert not reply.is_final
        assert "taken" in reply.text
        assert len(datastore.all_leads()) == 1

    def test_duplicate_contact_is_not_saved_twice(self) -> None:
        same_person_new_slot = dict(_SELLER_PAYLOAD, meetingDate="2030-01-09", meetingTime="15:00")
        orchestrator, _, datastore, _ = _build(
            [
                "[[INTENT:seller]]",
                "Booked!\nSELLER_DONE:" + json.dumps(_SELLER_PAYLOAD),
                "[[INTENT:seller]]",
                "Booked!\nSELLER_DONE:" + json.dumps(same_person_new_slot),
            ]
        )
        orchestrator.handle(_msg("sell", session_id="wa:first"))
        reply = orchestrator.handle(_msg("sell again", session_id="wa:third"))

        assert reply.is_final
        assert "already have a booking" in reply.text
        assert len(datastore.all_leads()) == 1

    def test_race_lost_inside_create_event_offers_alternatives(self) -> None:
        # Occupy the calendar slot directly (as if another system booked it)
        # while the datastore knows nothing — so check() passes and the race
        # re-check inside create_event is what fires.
        orchestrator, _, datastore, calendar = _build(
            [
                "[[INTENT:seller]]",
                "Booked!\nSELLER_DONE:" + json.dumps(_SELLER_PAYLOAD),
            ]
        )
        calendar.create_event(lead=make_lead(email="external@example.com", phone="+920000000009"))
        reply = orchestrator.handle(_msg("sell"))
        assert not reply.is_final
        assert "taken" in reply.text
        assert len(datastore.all_leads()) == 0
