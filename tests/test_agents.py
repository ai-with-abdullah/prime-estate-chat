"""Agent-level tests: slot-filling finalisation and the lookup tool loop.

Every test drives a real agent with a :class:`ScriptedChatModel` — the same
code paths production runs, minus the network. The assertions target the
boundaries: a DONE signal only finalises when its payload survives Python
validation, and a lookup agent only mutates records the datastore verified.
"""

from __future__ import annotations

import json

from prime_estate.agents.buyer import BuyerAgent
from prime_estate.agents.cancel import CancelAgent
from prime_estate.agents.followup import FollowupAgent
from prime_estate.agents.reschedule import RescheduleAgent
from prime_estate.agents.seller import SellerAgent
from prime_estate.domain.models import Intent, LeadScore, LeadStage, Session
from prime_estate.tools.calendar import InMemoryCalendar
from prime_estate.tools.datastore import InMemoryLeadDatastore
from tests.conftest import ScriptedChatModel, make_lead

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


def _session() -> Session:
    return Session(session_id="u1")


class TestSellerSlotFilling:
    def test_mid_flow_turn_is_not_final(
        self, calendar: InMemoryCalendar, datastore: InMemoryLeadDatastore
    ) -> None:
        model = ScriptedChatModel(["What type of property are you selling?"])
        agent = SellerAgent(model=model, calendar=calendar, datastore=datastore)
        reply = agent.handle_turn(
            message="I want to sell", session=_session(), history=[], today="2030-01-02"
        )
        assert not reply.is_final
        assert reply.lead is None
        assert "property" in reply.text

    def test_done_signal_finalises_with_validated_scored_lead(
        self, calendar: InMemoryCalendar, datastore: InMemoryLeadDatastore
    ) -> None:
        raw = "Perfect, you are all set!\nSELLER_DONE:" + json.dumps(_SELLER_PAYLOAD)
        model = ScriptedChatModel([raw])
        agent = SellerAgent(model=model, calendar=calendar, datastore=datastore)
        reply = agent.handle_turn(
            message="yes, confirm", session=_session(), history=[], today="2030-01-02"
        )
        assert reply.is_final
        lead = reply.lead
        assert lead is not None
        assert lead.intent is Intent.SELLER
        assert lead.email == "ayesha@example.com"
        assert lead.meeting_date == "2030-01-07"
        # Deterministic scoring: within 1 month + concrete price -> HOT.
        assert lead.score is LeadScore.HOT
        # Vertical-specific fields survive in extra.
        assert lead.extra["propertyType"] == "house"
        # The client never sees the machine signal.
        assert "SELLER_DONE" not in reply.text

    def test_malformed_payload_does_not_finalise(
        self, calendar: InMemoryCalendar, datastore: InMemoryLeadDatastore
    ) -> None:
        model = ScriptedChatModel(['SELLER_DONE:{"fullName": "broken json'])
        agent = SellerAgent(model=model, calendar=calendar, datastore=datastore)
        reply = agent.handle_turn(
            message="confirm", session=_session(), history=[], today="2030-01-02"
        )
        assert not reply.is_final
        assert reply.lead is None

    def test_invalid_email_bounces_back_to_the_client(
        self, calendar: InMemoryCalendar, datastore: InMemoryLeadDatastore
    ) -> None:
        payload = dict(_SELLER_PAYLOAD, email="not-an-email")
        model = ScriptedChatModel(["Done!\nSELLER_DONE:" + json.dumps(payload)])
        agent = SellerAgent(model=model, calendar=calendar, datastore=datastore)
        reply = agent.handle_turn(
            message="confirm", session=_session(), history=[], today="2030-01-02"
        )
        assert not reply.is_final
        assert reply.lead is None
        assert "email" in reply.text


class TestBuyerSignalToken:
    def test_bare_done_finalises(
        self, calendar: InMemoryCalendar, datastore: InMemoryLeadDatastore
    ) -> None:
        payload = {
            "intent": "first home",
            "propertyType": "flat",
            "location": "Gulberg",
            "budget": "15000000",
            "timeline": "ASAP",
            "fullName": "Bilal Ahmed",
            "phone": "+923009998877",
            "email": "bilal@example.com",
            "meetingDate": "2030-01-08",
            "meetingTime": "10:00",
        }
        model = ScriptedChatModel(["You are booked!\nDONE:" + json.dumps(payload)])
        agent = BuyerAgent(model=model, calendar=calendar, datastore=datastore)
        reply = agent.handle_turn(
            message="confirm", session=_session(), history=[], today="2030-01-02"
        )
        assert reply.is_final
        assert reply.lead is not None
        assert reply.lead.intent is Intent.QUALIFY
        assert reply.lead.score is LeadScore.HOT

    def test_lookbehind_ignores_other_agents_tokens(
        self, calendar: InMemoryCalendar, datastore: InMemoryLeadDatastore
    ) -> None:
        # A stray SELLER_DONE must not satisfy the buyer's bare-DONE matcher.
        model = ScriptedChatModel(["SELLER_DONE:" + json.dumps(_SELLER_PAYLOAD)])
        agent = BuyerAgent(model=model, calendar=calendar, datastore=datastore)
        reply = agent.handle_turn(
            message="confirm", session=_session(), history=[], today="2030-01-02"
        )
        assert not reply.is_final
        assert reply.lead is None


class TestCancelFlow:
    def _booked_datastore(
        self, calendar: InMemoryCalendar, datastore: InMemoryLeadDatastore
    ) -> None:
        lead = make_lead(
            email="ayesha@example.com",
            phone="+923001234567",
            full_name="Ayesha Khan",
            stage=LeadStage.BOOKED,
        )
        lead.calendar_event_id = calendar.create_event(lead=lead)
        datastore.save(lead=lead)

    def test_full_cancel_flow(
        self, calendar: InMemoryCalendar, datastore: InMemoryLeadDatastore
    ) -> None:
        self._booked_datastore(calendar, datastore)
        model = ScriptedChatModel(
            [
                "Of course. Could you share the email and phone you booked with?",
                'LOOKUP:{"email":"ayesha@example.com","phone":"+923001234567"}',
                "Found it, Ayesha — 2030-01-07 at 11:00. May I ask why you are cancelling?",
                "All done, your meeting is cancelled. Hope to see you again!\n"
                "CANCEL_EMAIL: ayesha@example.com\n"
                "CANCEL_TYPE: seller\n"
                "CANCEL_REASON: schedule conflict\n"
                "CANCEL_NAME: Ayesha Khan",
            ]
        )
        agent = CancelAgent(model=model, calendar=calendar, datastore=datastore)
        session = _session()

        r1 = agent.handle_turn(
            message="cancel my meeting", session=session, history=[], today="2030-01-02"
        )
        assert not r1.is_final

        r2 = agent.handle_turn(
            message="ayesha@example.com, +923001234567",
            session=session,
            history=[],
            today="2030-01-02",
        )
        assert not r2.is_final
        # The tool loop fed the datastore's answer back to the model.
        fed = model.calls[2]["messages"]
        assert isinstance(fed, list)
        assert any("[TOOL RESULT] RECORD FOUND" in m.content for m in fed)

        r3 = agent.handle_turn(
            message="schedule conflict", session=session, history=[], today="2030-01-02"
        )
        assert r3.is_final
        assert "CANCEL_EMAIL" not in r3.text
        stored = datastore.all_leads()[0]
        assert stored.stage is LeadStage.CANCELLED
        # The calendar slot was genuinely freed.
        assert calendar.is_slot_free(iso_date="2030-01-07", time="11:00")

    def test_tags_without_verified_lookup_are_refused(
        self, calendar: InMemoryCalendar, datastore: InMemoryLeadDatastore
    ) -> None:
        self._booked_datastore(calendar, datastore)
        model = ScriptedChatModel(
            [
                "Cancelled!\n"
                "CANCEL_EMAIL: ayesha@example.com\n"
                "CANCEL_TYPE: seller\n"
                "CANCEL_REASON: none\n"
                "CANCEL_NAME: Ayesha Khan"
            ]
        )
        agent = CancelAgent(model=model, calendar=calendar, datastore=datastore)
        reply = agent.handle_turn(
            message="cancel it", session=_session(), history=[], today="2030-01-02"
        )
        # The model skipped the lookup: tags alone must not mutate anything.
        assert not reply.is_final
        assert datastore.all_leads()[0].stage is LeadStage.BOOKED

    def test_unknown_contact_reports_no_record(
        self, calendar: InMemoryCalendar, datastore: InMemoryLeadDatastore
    ) -> None:
        model = ScriptedChatModel(
            [
                'LOOKUP:{"email":"ghost@example.com","phone":"+920000000000"}',
                "I am sorry, I could not find a booking under those details.",
            ]
        )
        agent = CancelAgent(model=model, calendar=calendar, datastore=datastore)
        reply = agent.handle_turn(
            message="ghost@example.com, +920000000000",
            session=_session(),
            history=[],
            today="2030-01-02",
        )
        assert not reply.is_final
        fed = model.calls[1]["messages"]
        assert isinstance(fed, list)
        assert any("NO RECORD FOUND" in m.content for m in fed)


class TestRescheduleFlow:
    def test_full_reschedule_flow(
        self, calendar: InMemoryCalendar, datastore: InMemoryLeadDatastore
    ) -> None:
        lead = make_lead(
            email="ayesha@example.com",
            phone="+923001234567",
            full_name="Ayesha Khan",
            stage=LeadStage.BOOKED,
        )
        lead.calendar_event_id = calendar.create_event(lead=lead)
        datastore.save(lead=lead)

        model = ScriptedChatModel(
            [
                'LOOKUP:{"email":"ayesha@example.com","phone":"+923001234567"}',
                "You are booked for 2030-01-07 at 11:00. What new date and time suit you?",
                'CHECK_SLOT:{"date":"2030-01-08","time":"12:00"}',
                "2030-01-08 at 12:00 is free! May I ask the reason for the change?",
                "Done — moved to 2030-01-08 at 12:00. See you then!\n"
                "RESCHEDULE_EMAIL: ayesha@example.com\n"
                "RESCHEDULE_TYPE: seller\n"
                "RESCHEDULE_NAME: Ayesha Khan\n"
                "RESCHEDULE_REASON: travel\n"
                "RESCHEDULE_NEW_DATE: 2030-01-08\n"
                "RESCHEDULE_NEW_TIME: 12:00\n"
                "RESCHEDULE_NEW_END_TIME: 13:00",
            ]
        )
        agent = RescheduleAgent(model=model, calendar=calendar, datastore=datastore)
        session = _session()

        r1 = agent.handle_turn(
            message="ayesha@example.com, +923001234567",
            session=session,
            history=[],
            today="2030-01-02",
        )
        assert not r1.is_final

        r2 = agent.handle_turn(
            message="2030-01-08 at 12:00", session=session, history=[], today="2030-01-02"
        )
        assert not r2.is_final

        r3 = agent.handle_turn(
            message="travelling that day", session=session, history=[], today="2030-01-02"
        )
        assert r3.is_final
        stored = datastore.all_leads()[0]
        assert stored.stage is LeadStage.RESCHEDULED
        assert stored.meeting_date == "2030-01-08"
        assert stored.meeting_time == "12:00"
        # Old slot freed, new slot held.
        assert calendar.is_slot_free(iso_date="2030-01-07", time="11:00")
        assert not calendar.is_slot_free(iso_date="2030-01-08", time="12:00")

    def test_invalid_new_slot_is_bounced(
        self, calendar: InMemoryCalendar, datastore: InMemoryLeadDatastore
    ) -> None:
        lead = make_lead(email="ayesha@example.com", phone="+923001234567")
        datastore.save(lead=lead)
        model = ScriptedChatModel(
            [
                'LOOKUP:{"email":"ayesha@example.com","phone":"+923001234567"}',
                "Found it. New date and time?",
                # 2030-01-06 is a Sunday: format-valid but not bookable.
                "Moved!\n"
                "RESCHEDULE_EMAIL: ayesha@example.com\n"
                "RESCHEDULE_TYPE: seller\n"
                "RESCHEDULE_NAME: Ayesha Khan\n"
                "RESCHEDULE_REASON: travel\n"
                "RESCHEDULE_NEW_DATE: 2030-01-06\n"
                "RESCHEDULE_NEW_TIME: 12:00\n"
                "RESCHEDULE_NEW_END_TIME: 13:00",
            ]
        )
        agent = RescheduleAgent(model=model, calendar=calendar, datastore=datastore)
        session = _session()
        agent.handle_turn(
            message="ayesha@example.com, +923001234567",
            session=session,
            history=[],
            today="2030-01-02",
        )
        reply = agent.handle_turn(
            message="sunday at noon", session=session, history=[], today="2030-01-02"
        )
        assert not reply.is_final
        assert datastore.all_leads()[0].meeting_date == "2030-01-07"  # unchanged


class TestFollowupFlow:
    def test_reports_stage_and_never_finalises(
        self, calendar: InMemoryCalendar, datastore: InMemoryLeadDatastore
    ) -> None:
        datastore.save(
            lead=make_lead(email="ayesha@example.com", full_name="Ayesha Khan", stage=LeadStage.BOOKED)
        )
        model = ScriptedChatModel(
            [
                'LOOKUP:{"query":"ayesha@example.com"}',
                "Your meeting is confirmed and our team will be in touch shortly!",
            ]
        )
        agent = FollowupAgent(model=model, calendar=calendar, datastore=datastore)
        reply = agent.handle_turn(
            message="ayesha@example.com", session=_session(), history=[], today="2030-01-02"
        )
        assert not reply.is_final
        assert "confirmed" in reply.text
        fed = model.calls[1]["messages"]
        assert isinstance(fed, list)
        assert any("stage=Booked" in m.content for m in fed)
        # Read-only guarantee: nothing changed.
        assert datastore.all_leads()[0].stage is LeadStage.BOOKED
