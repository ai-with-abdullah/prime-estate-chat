"""HTTP transport: a FastAPI app exposing the agent as a web chat backend.

This module is the *composition root* — the only place where configuration,
credentials, and concrete tool implementations meet. The rule that everything
below this layer is injected pays off here: swapping the whole persistence and
scheduling stack (in-memory vs Google Sheets + Calendar) is a pair of ``if``
statements over :class:`Settings`, and zero business-logic modules change.

Run from the repository root (so ``config/`` and ``web/`` resolve):

    uvicorn prime_estate.api:app --reload

Tool selection:

* ``PRIME_GOOGLE_SERVICE_ACCOUNT_FILE`` + ``PRIME_SHEET_ID``  -> Google Sheets
  datastore, else in-memory.
* ``PRIME_GOOGLE_SERVICE_ACCOUNT_FILE`` + ``PRIME_CALENDAR_ID`` -> Google
  Calendar tool, else in-memory.
* ``PRIME_GMAIL_USER`` + ``PRIME_GMAIL_APP_PASSWORD`` -> Gmail confirmations,
  else no email.

The transport stays deliberately thin: one POST endpoint that normalises a
browser message into the same :class:`InboundMessage` the WhatsApp webhook
would produce, calls ``Orchestrator.handle``, and returns the reply. The
orchestrator neither knows nor cares that the channel changed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from prime_estate.core.intents import build_registry
from prime_estate.core.orchestrator import Orchestrator
from prime_estate.core.router import IntentRouter
from prime_estate.core.session import SessionStore
from prime_estate.domain.models import InboundMessage
from prime_estate.llm.groq_client import GroqChatModel
from prime_estate.tools.base import CalendarTool, LeadRepository
from prime_estate.tools.calendar import InMemoryCalendar
from prime_estate.tools.datastore import InMemoryLeadDatastore
from prime_estate.tools.email import GmailSmtpNotifier, LeadNotifier
from prime_estate.utils.logging import get_logger

logger = get_logger(__name__)

_WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class ChatRequest(BaseModel):
    """One inbound chat message from the browser."""

    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    """The agent's reply for one turn."""

    reply: str
    is_final: bool


def build_orchestrator_from_settings() -> Orchestrator:
    """Wire the production orchestrator according to environment settings."""
    from config.settings import Settings
    from groq import Groq

    # Required fields (the Groq key) come from the environment / .env at
    # runtime; mypy cannot see that, hence the targeted ignore.
    settings = Settings()  # type: ignore[call-arg]
    model = GroqChatModel(client=Groq(api_key=settings.groq_api_key), model=settings.groq_model)

    datastore: LeadRepository
    if settings.google_service_account_file and settings.sheet_id:
        from prime_estate.tools.google_sheets import GoogleSheetsLeadDatastore, open_worksheet

        datastore = GoogleSheetsLeadDatastore(
            worksheet=open_worksheet(
                service_account_file=settings.google_service_account_file,
                sheet_id=settings.sheet_id,
            )
        )
        logger.info("datastore: Google Sheets (%s)", settings.sheet_id)
    else:
        datastore = InMemoryLeadDatastore()
        logger.info("datastore: in-memory (no sheet configured)")

    calendar: CalendarTool
    if settings.google_service_account_file and settings.calendar_id:
        from prime_estate.tools.google_calendar import GoogleCalendarTool, build_service

        calendar = GoogleCalendarTool(
            service=build_service(service_account_file=settings.google_service_account_file),
            calendar_id=settings.calendar_id,
            timezone=settings.timezone,
        )
        logger.info("calendar: Google Calendar (%s)", settings.calendar_id)
    else:
        calendar = InMemoryCalendar()
        logger.info("calendar: in-memory (no calendar configured)")

    notifier: LeadNotifier | None = None
    if settings.gmail_user and settings.gmail_app_password:
        notifier = GmailSmtpNotifier(
            user=settings.gmail_user, app_password=settings.gmail_app_password
        )
        logger.info("email: Gmail SMTP as %s", settings.gmail_user)
    else:
        logger.info("email: disabled (no Gmail credentials)")

    return Orchestrator(
        router=IntentRouter(model=model),
        registry=build_registry(model=model, calendar=calendar, datastore=datastore),
        sessions=SessionStore(ttl_seconds=settings.session_ttl_seconds),
        calendar=calendar,
        datastore=datastore,
        notifier=notifier,
    )


def create_app(*, orchestrator: Orchestrator | None = None) -> FastAPI:
    """Build the FastAPI app.

    ``orchestrator`` is injectable so API tests run against a scripted model
    with in-memory tools; production (arg omitted) wires from settings lazily
    on startup, keeping module import free of credential side effects.
    """
    state: dict[str, Orchestrator] = {}
    if orchestrator is not None:
        state["orchestrator"] = orchestrator

    @asynccontextmanager
    async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
        if "orchestrator" not in state:
            state["orchestrator"] = build_orchestrator_from_settings()
        yield

    application = FastAPI(
        title="Prime Estate Agent", docs_url=None, redoc_url=None, lifespan=_lifespan
    )
    # The chat frontend is also hosted on GitHub Pages (a different origin
    # from wherever this API is deployed), so the chat endpoint must accept
    # cross-origin POSTs. The endpoint is public and unauthenticated by
    # design — it is the front door of the business — so an open CORS policy
    # widens nothing.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type"],
    )
    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        reply = state["orchestrator"].handle(
            InboundMessage(
                session_id=request.session_id,
                text=request.message,
                received_at=datetime.now(UTC),
            )
        )
        return ChatResponse(reply=reply.text, is_final=reply.is_final)

    if _WEB_DIR.is_dir():
        application.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")

    return application


app = create_app()
