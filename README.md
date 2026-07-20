# Prime Estate Agent

A production agentic lead system for a real-estate agency's WhatsApp channel:
one inbound message goes in, one reply comes out, and somewhere in between a
language model holds a natural conversation while **every consequential action
passes through typed, testable Python boundaries**.

This repository also hosts the chat frontend (`web/index.html`, served by the
FastAPI app itself; `index.html` at the root is the same page deployed to
GitHub Pages at https://ai-with-abdullah.github.io/prime-estate-chat/) and the
original n8n workflow export (`prime-estate-chat.json`) that this Python
system is the engineered rewrite of. The frontend talks to this backend over
one endpoint: `POST /api/chat`.

This is a ground-up Python port of a system that ran in production as an n8n
workflow (9 agents, ~190 nodes). The port exists because the original's logic —
session policy, dedup rules, slot conflicts, lead scoring — deserved to live in
reviewable, unit-tested source instead of being spread across workflow nodes
and prompt text.

## The thesis: decisions inside boundaries

"Agentic" is easy to claim and easy to fake. The design test this codebase is
built around: *where does the model actually decide something, and what stops a
bad decision from becoming a bad outcome?*

Trace a message through the spine (`core/orchestrator.py`):

```
inbound message
   |
   v
[1] SessionStore.resolve          pure Python: TTL, override keywords,
   |                              sticky intent. Decides IF the model
   |                              even gets to classify this turn.
   v
[2] IntentRouter.classify         MODEL DECIDES - inside a closed enum.
   |                              Output is parsed defensively; garbage
   |                              degrades to GENERAL, never crashes.
   v
[3] agent.handle_turn             MODEL DECIDES - conversation wording,
   |                              which field to ask next, when to signal
   |                              completion, when to call a tool.
   |                              Completion payloads are re-validated in
   |                              Python; scores are computed, not trusted.
   v
[4] booking pipeline              pure Python: duplicate check, slot
   |                              conflict, calendar race re-check.
   v                              The model cannot bypass any of it.
reply
```

The model drives the conversation. It cannot invent a lead score, book a taken
slot, cancel a booking without two-factor identity verification, calculate a
meeting date, or persist a malformed record — because none of those paths run
on model output alone.

### Where that shows up in code

| Boundary | Mechanism | Module |
|---|---|---|
| Routing | closed `Intent` enum, defensive tag parse | `core/router.py`, `domain/models.py` |
| Continuity | sticky sessions short-circuit the router structurally | `core/session.py` |
| Tool access | `Protocol` seams; agents hold 3 calendar + 3 datastore capabilities, nothing more | `tools/base.py` |
| Completion | `*_DONE` payloads re-validated by regex validators before persisting | `agents/base.py`, `validation/validators.py` |
| Scoring | deterministic HOT/Warm/Cold, lifted out of prompts | `domain/scoring.py` |
| Dates | model selects from a precomputed Mon–Sat table, never derives | `utils/dates.py` |
| Destructive ops | cancel/reschedule require email AND phone to match a record | `agents/lookup.py`, `tools/datastore.py` |
| Booking | dup + slot guards + race re-check in `create_event` | `core/orchestrator.py`, `tools/calendar.py` |

## Architecture

```
src/prime_estate/
  core/        orchestrator.py  the spine (message -> reply)
               router.py        closed-enum intent classification
               session.py       TTL / override / sticky-intent policy
               intents.py       Intent -> agent registry (verified total)
  agents/      base.py          slot-filling base (seller/buyer/rent/investor)
               lookup.py        lookup-and-act base (cancel/reschedule/followup)
               general.py       tool-less fallback for unclassified turns
               seller.py buyer.py rent.py investor.py
               cancel.py reschedule.py followup.py
  llm/         groq_client.py   ChatModel protocol + Groq adapter w/ retry
  tools/       base.py          CalendarTool / LeadDatastore / LeadLookup protocols
               calendar.py      in-memory calendar (race-safe create)
               datastore.py     in-memory datastore (dedup + slot rules)
  domain/      models.py        Pydantic models crossing every boundary
               scoring.py       deterministic lead scoring
  validation/  validators.py    email/phone/date/time gates
  utils/       dates.py logging.py
config/        settings.py      pydantic-settings, .env-driven
tests/         unit + integration suite (no network required)
```

Two agent base classes, on purpose. Slot-filling agents **create** a record by
collecting a fixed field schema; lookup agents **act on** a record that already
exists and must verify it exists first. Forcing both shapes into one base would
have meant optional fields and a signal contract meaning two things — the
classic bloated-base failure. The orchestrator sees neither: it dispatches
through the narrow `ConversationAgent` protocol in `core/intents.py`.

## Running it

Everything deterministic runs with zero credentials — the in-memory tools are
first-class implementations of the same protocols the production tools satisfy,
not test stubs.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest          # full suite, no network, no keys
ruff check .
mypy
```

The LLM seam is a protocol (`ChatModel`), so the entire agent loop — routing,
slot-filling, tool rounds, booking — is exercised in tests with a scripted
fake. `tests/test_orchestrator.py` runs the real composition root end to end;
`tests/test_api.py` runs the HTTP transport in front of it.

## Running the web chat

The FastAPI app (`src/prime_estate/api.py`) is the composition root: it wires
the orchestrator from `.env` and serves both the API and the chat page.

```bash
cp .env.example .env      # fill in PRIME_GROQ_API_KEY (required)
pip install -e .          # add ".[google]" for the Sheets/Calendar tools
uvicorn prime_estate.api:app --reload
# open http://127.0.0.1:8000  — the chat UI, served from web/
```

Endpoints: `POST /api/chat` (`{"session_id": ..., "message": ...}` ->
`{"reply": ..., "is_final": ...}`) and `GET /api/health`.

Tool selection is `.env`-driven — with only the Groq key set, the app runs on
the in-memory calendar/datastore; setting the Google variables (see
`.env.example`) swaps in Google Sheets, Google Calendar, and Gmail
confirmations with zero code changes.

**GitHub Pages frontend.** The root `index.html` is the same chat page
deployed at https://ai-with-abdullah.github.io/prime-estate-chat/. Because
Pages cannot host the Python backend, deploy the backend somewhere public
(Railway/Render/Fly), then set `BACKEND_URL` at the top of the page's script
to that URL (or test ad hoc with `?api=https://your-backend`). The API ships
with CORS open on the chat endpoint for exactly this split.

## Production swap

Swapping the credential-free demo wiring for production is a composition-root
change only; no business logic moves:

- `InMemoryCalendar` -> `GoogleCalendarTool` (`tools/google_calendar.py`),
  selected when `PRIME_GOOGLE_SERVICE_ACCOUNT_FILE` + `PRIME_CALENDAR_ID` are
  set.
- `InMemoryLeadDatastore` -> `GoogleSheetsLeadDatastore`
  (`tools/google_sheets.py`), selected when the service-account file +
  `PRIME_SHEET_ID` are set. The row semantics (header offset, 1-based) are
  already sheet-shaped for exactly this swap.
- Booking confirmations -> `GmailSmtpNotifier` (`tools/email.py`), selected
  when `PRIME_GMAIL_USER` + `PRIME_GMAIL_APP_PASSWORD` are set.
- The model is `GroqChatModel` with a real client (`PRIME_GROQ_API_KEY`),
  already implemented with bounded retry.
- A WhatsApp webhook would normalise into `InboundMessage` and call
  `Orchestrator.handle` exactly as `/api/chat` does — the orchestrator
  neither knows nor cares about the transport.

Honest limitations of the current state: sessions and history are in-process
(Redis is the swap for multi-worker deployments), and the in-memory tools are
per-process by definition. Both live behind seams designed for that migration.
