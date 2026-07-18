"""Intent -> agent registry: the single source of truth for dispatch.

The router's output space is a closed enum; this module closes the loop by
guaranteeing every member of that enum has exactly one handler. Dispatch is a
dict lookup — not an if/elif ladder in the orchestrator — so adding a vertical
is one entry here plus one agent module, and *forgetting* a vertical is a
constructor-time error rather than a runtime KeyError in production
(:func:`build_registry` verifies totality on assembly).

The :class:`ConversationAgent` Protocol is the narrow waist between the
orchestrator and every agent implementation. Three structurally different
agent families exist (slot-filling, lookup-based, tool-less general), and the
orchestrator must not know which is which — it sees one ``handle_turn``
signature and one ``AgentReply`` coming back.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from prime_estate.agents.buyer import BuyerAgent
from prime_estate.agents.cancel import CancelAgent
from prime_estate.agents.followup import FollowupAgent
from prime_estate.agents.general import GeneralAgent
from prime_estate.agents.investor import InvestorAgent
from prime_estate.agents.rent import RentAgent
from prime_estate.agents.reschedule import RescheduleAgent
from prime_estate.agents.seller import SellerAgent
from prime_estate.domain.models import AgentReply, Intent, Session
from prime_estate.llm.groq_client import ChatMessage, ChatModel
from prime_estate.tools.base import CalendarTool, LeadRepository


@runtime_checkable
class ConversationAgent(Protocol):
    """The single contract every dispatchable agent satisfies."""

    intent: Intent

    def handle_turn(
        self,
        *,
        message: str,
        session: Session,
        history: list[ChatMessage],
        today: str,
    ) -> AgentReply:
        """Process one user turn and return the agent's reply."""
        ...


def build_registry(
    *,
    model: ChatModel,
    calendar: CalendarTool,
    datastore: LeadRepository,
) -> dict[Intent, ConversationAgent]:
    """Assemble the complete intent -> agent mapping.

    This is the composition root for agents: every agent receives its
    dependencies here and nowhere else, which is what keeps the whole tree
    constructible with in-memory fakes in tests. Raises if any
    :class:`Intent` is left unhandled, so registry drift is caught the moment
    the system is assembled rather than the moment a user hits the gap.
    """
    agents: list[ConversationAgent] = [
        SellerAgent(model=model, calendar=calendar, datastore=datastore),
        BuyerAgent(model=model, calendar=calendar, datastore=datastore),
        RentAgent(model=model, calendar=calendar, datastore=datastore),
        InvestorAgent(model=model, calendar=calendar, datastore=datastore),
        CancelAgent(model=model, calendar=calendar, datastore=datastore),
        RescheduleAgent(model=model, calendar=calendar, datastore=datastore),
        FollowupAgent(model=model, calendar=calendar, datastore=datastore),
        GeneralAgent(model=model),
    ]
    registry = {agent.intent: agent for agent in agents}

    missing = [intent for intent in Intent if intent not in registry]
    if missing:
        raise ValueError(f"registry is missing handlers for: {missing}")
    return registry
