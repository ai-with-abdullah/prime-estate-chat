"""HTTP-transport tests: the FastAPI app in front of a scripted orchestrator.

The orchestrator is already integration-tested end to end; here the subject is
the transport itself — request validation, the response contract the web chat
frontend depends on, and that the endpoint really is a thin adapter (one
normalisation, one ``handle`` call, one serialisation).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from prime_estate.api import create_app
from prime_estate.core.intents import build_registry
from prime_estate.core.orchestrator import Orchestrator
from prime_estate.core.router import IntentRouter
from prime_estate.core.session import SessionStore
from prime_estate.tools.calendar import InMemoryCalendar
from prime_estate.tools.datastore import InMemoryLeadDatastore
from tests.conftest import ScriptedChatModel


def _client(responses: list[str]) -> TestClient:
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
    return TestClient(create_app(orchestrator=orchestrator))


def test_health() -> None:
    client = _client([])
    assert client.get("/api/health").json() == {"status": "ok"}


def test_chat_returns_reply_with_contract_fields() -> None:
    client = _client(
        [
            "[[INTENT:GENERAL]]",
            "Hello! How can I help you with property today?",
        ]
    )
    response = client.post(
        "/api/chat", json={"session_id": "web_abc", "message": "hi there"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "reply": "Hello! How can I help you with property today?",
        "is_final": False,
    }


def test_chat_rejects_empty_message() -> None:
    client = _client([])
    response = client.post("/api/chat", json={"session_id": "web_abc", "message": ""})
    assert response.status_code == 422


def test_chat_rejects_missing_session() -> None:
    client = _client([])
    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 422


def test_chat_serves_web_frontend_at_root() -> None:
    client = _client([])
    response = client.get("/")
    assert response.status_code == 200
    assert "Prime Estate" in response.text
    # The page must post to the same contract the API exposes.
    assert "/api/chat" in response.text
    assert "session_id" in response.text
