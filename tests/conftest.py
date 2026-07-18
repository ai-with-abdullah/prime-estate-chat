"""Shared test fixtures.

The single most important object here is :class:`ScriptedChatModel`: a fake
satisfying the ``ChatModel`` Protocol that replays a fixed list of responses.
Because every LLM touchpoint in the system goes through that Protocol, the
entire agent loop — routing, slot-filling, tool rounds, booking — runs under
test with zero network and byte-for-byte reproducibility. That is the payoff
of the dependency-injection rule the codebase follows everywhere.
"""

from __future__ import annotations

import pytest

from prime_estate.domain.models import Intent, Lead, LeadStage
from prime_estate.llm.groq_client import ChatMessage
from prime_estate.tools.calendar import InMemoryCalendar
from prime_estate.tools.datastore import InMemoryLeadDatastore


class ScriptedChatModel:
    """A ``ChatModel`` fake that replays canned responses in order.

    Also records every call (system prompt + messages) so tests can assert on
    what the model was shown — e.g. that the router was NOT consulted on a
    sticky turn, or that a tool result was fed back into the conversation.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete(self, *, system: str, messages: list[ChatMessage]) -> str:
        self.calls.append({"system": system, "messages": list(messages)})
        if not self._responses:
            raise AssertionError("ScriptedChatModel ran out of scripted responses")
        return self._responses.pop(0)


def make_lead(
    *,
    email: str = "client@example.com",
    phone: str = "+923001234567",
    full_name: str = "Test Client",
    intent: Intent = Intent.SELLER,
    meeting_date: str = "2030-01-07",
    meeting_time: str = "11:00",
    stage: LeadStage = LeadStage.NEW,
) -> Lead:
    """Build a valid lead with overridable fields for datastore/calendar tests."""
    return Lead(
        full_name=full_name,
        email=email,
        phone=phone,
        intent=intent,
        meeting_date=meeting_date,
        meeting_time=meeting_time,
        stage=stage,
    )


@pytest.fixture()
def calendar() -> InMemoryCalendar:
    return InMemoryCalendar()


@pytest.fixture()
def datastore() -> InMemoryLeadDatastore:
    return InMemoryLeadDatastore()
