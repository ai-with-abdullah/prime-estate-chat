"""Interactive terminal demo of the full agent loop, wired to a real LLM.

Runs the exact composition root the tests exercise — orchestrator, router,
session store, registry, in-memory calendar and datastore — but with a live
Groq model instead of a scripted fake. Nothing else changes, which is the
point: the demo *is* the production wiring minus the WhatsApp transport.

Usage:
    pip install -e .
    export PRIME_GROQ_API_KEY=gsk_...   # free key from console.groq.com
    python demo_chat.py

Commands inside the chat: ``leads`` prints the datastore, ``reset`` starts a
fresh session (a new "phone number"), ``quit`` exits.
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import UTC, datetime

from prime_estate.core.intents import build_registry
from prime_estate.core.orchestrator import Orchestrator
from prime_estate.core.router import IntentRouter
from prime_estate.core.session import SessionStore
from prime_estate.domain.models import InboundMessage
from prime_estate.llm.groq_client import ChatModel, GroqChatModel
from prime_estate.tools.calendar import InMemoryCalendar
from prime_estate.tools.datastore import InMemoryLeadDatastore


def build_app(model: ChatModel) -> tuple[Orchestrator, InMemoryLeadDatastore, InMemoryCalendar]:
    """Assemble the full system around any ChatModel (real or scripted)."""
    calendar = InMemoryCalendar()
    datastore = InMemoryLeadDatastore()
    orchestrator = Orchestrator(
        router=IntentRouter(model=model),
        registry=build_registry(model=model, calendar=calendar, datastore=datastore),
        sessions=SessionStore(),
        calendar=calendar,
        datastore=datastore,
    )
    return orchestrator, datastore, calendar


def _print_leads(datastore: InMemoryLeadDatastore) -> None:
    leads = datastore.all_leads()
    if not leads:
        print("  (no leads saved yet)")
    for lead in leads:
        score = lead.score.value if lead.score else "-"
        print(
            f"  {lead.stage.value:12} {lead.intent.value:10} {lead.full_name:20} "
            f"{lead.email:28} {lead.meeting_date} {lead.meeting_time}  score={score}"
        )


def main() -> None:
    api_key = os.environ.get("PRIME_GROQ_API_KEY", "")
    if not api_key:
        sys.exit("Set PRIME_GROQ_API_KEY first (free key at console.groq.com).")
    from groq import Groq  # imported here so the module loads without the SDK

    model = GroqChatModel(
        client=Groq(api_key=api_key),
        model=os.environ.get("PRIME_GROQ_MODEL", "llama-3.3-70b-versatile"),
    )
    orchestrator, datastore, _ = build_app(model)
    session_id = f"demo:{uuid.uuid4().hex[:8]}"
    print("Prime Estate demo. Talk to the agent. Commands: leads / reset / quit")

    while True:
        try:
            text = input("you>   ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        lowered = text.lower()
        if lowered in {"quit", "exit"}:
            break
        if lowered == "leads":
            _print_leads(datastore)
            continue
        if lowered == "reset":
            session_id = f"demo:{uuid.uuid4().hex[:8]}"
            print("  [new session started]")
            continue

        reply = orchestrator.handle(
            InboundMessage(session_id=session_id, text=text, received_at=datetime.now(UTC))
        )
        print(f"agent> {reply.text}")
        if reply.is_final:
            print("  [flow complete — session cleared]")


if __name__ == "__main__":
    main()
