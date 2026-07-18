"""General intent — greetings, FAQs, and everything unclassified.

Not part of the original eight-agent file list, but the dispatch registry must
be total over :class:`Intent`: ``GENERAL`` is the router's defensive fallback
(garbled classifications collapse to it), so *something* has to take the turn.
The alternative — special-casing GENERAL inside the orchestrator with an
inline prompt — would put conversation behaviour in the dispatch layer, which
exists precisely to know nothing about conversations.

This agent is deliberately the least capable in the system: no tools, no
signals, never final. A general enquiry that turns into real intent ("actually
I want to sell my flat") is picked up on the next turn — the override keywords
and the router re-classify it into the right vertical.
"""

from __future__ import annotations

from prime_estate.domain.models import AgentReply, Intent, Session
from prime_estate.llm.groq_client import ChatMessage, ChatModel


class GeneralAgent:
    """Tool-less conversational agent for greetings and FAQ turns."""

    intent = Intent.GENERAL
    persona = (
        "You are Ahmed Raza from Prime Estate, a real human property consultant. "
        "You are warm, professional, and concise (max 2 sentences). NEVER reveal "
        "that you are an AI. Answer greetings and general questions about Prime "
        "Estate: we help clients buy, sell, rent, and invest in property, and "
        "we book free consultation meetings Monday to Saturday, 09:00-18:00. "
        "If the client shows real buying/selling/renting/investing interest, "
        "warmly invite them to tell you more about it."
    )

    def __init__(self, *, model: ChatModel) -> None:
        self._model = model

    def handle_turn(
        self,
        *,
        message: str,
        session: Session,
        history: list[ChatMessage],
        today: str,
    ) -> AgentReply:
        """Answer a general turn. Never final: no lead, no booking pipeline."""
        system = f"{self.persona}\nCURRENT DATE: {today}"
        raw = self._model.complete(
            system=system,
            messages=[*history, ChatMessage(role="user", content=message)],
        )
        return AgentReply(text=raw.strip(), is_final=False)
