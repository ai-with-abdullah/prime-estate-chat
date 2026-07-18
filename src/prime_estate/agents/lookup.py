"""Base class for the lookup-based agents (cancel, reschedule, followup).

These three flows are a fundamentally different shape from the slot-filling
verticals, which is why they get their own base instead of being forced into
:class:`~prime_estate.agents.base.SpecialistAgent`:

* a slot-filling agent *creates* a record from nothing, collecting a fixed
  field schema one question at a time;
* a lookup agent *acts on a record that already exists* — it must first prove
  the record exists (identity verification through the datastore), and only
  then perform a bounded action (cancel, move, or report status).

Bolting lookup semantics onto ``SpecialistAgent`` would have meant optional
fields, a conditional finalisation path, and a signal contract that means two
different things — the classic bloated-base failure. Two small bases, each
with one job, is the cleaner boundary.

The interaction contract with the model is a miniature tool loop:

1. The model converses until it has the client's identifying details, then
   emits ``LOOKUP:{...}`` on its own line. The base intercepts it, queries the
   datastore in Python, and feeds a ``[TOOL RESULT] ...`` message back — the
   model never fabricates a record, it can only react to what the datastore
   returned. Reschedule additionally gets ``CHECK_SLOT:{...}`` for calendar
   availability.
2. When the flow completes, the model emits a block of ``TAG: value`` lines
   (the exact tag scheme of the original workflow). The base parses them and
   hands them to the subclass's :meth:`_apply`, where the *action itself* —
   the datastore update, the calendar mutation — is plain Python operating on
   the record found in step 1, never on values the model invented.

That two-step design is the same bounded-autonomy story as the slot-filling
side: the model decides *when* to look up and *how* to talk, Python decides
*what actually happens*.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from prime_estate.domain.models import AgentReply, Intent, Session
from prime_estate.llm.groq_client import ChatMessage, ChatModel
from prime_estate.tools.base import CalendarTool, LeadRepository
from prime_estate.utils.dates import UpcomingDate, render_dates_table, upcoming_dates
from prime_estate.utils.logging import get_logger

logger = get_logger(__name__)

# One line like `LOOKUP:{"email":"a@b.c","phone":"+92..."}`. Anchored to a line
# start so tag names inside prose don't trigger tool dispatch.
_TOOL_SIGNAL_RE = re.compile(r"^\s*(LOOKUP|CHECK_SLOT):\s*(\{.*?\})\s*$", re.MULTILINE)


class LookupAgent(ABC):
    """Abstract base for agents that verify an existing lead, then act on it.

    Concrete subclasses declare :attr:`intent`, :attr:`persona`,
    :attr:`final_tags` (the tag names whose joint presence marks completion),
    and implement :meth:`_run_lookup` (interpret a LOOKUP payload against the
    datastore) and :meth:`_apply` (perform the real action once tags arrive).
    """

    #: Which intent this agent serves. Used by the orchestrator's registry.
    intent: Intent
    #: Persona/flow description injected at the top of the system prompt.
    persona: str
    #: Tag names that must ALL be present for the turn to count as final.
    #: An empty tuple means the flow is read-only and never finalises.
    final_tags: tuple[str, ...] = ()

    # Upper bound on model->tool->model rounds within one user turn. Two is the
    # honest maximum a healthy flow needs (a lookup, or a lookup then a slot
    # check); the cap exists so a confused model cannot spin the loop.
    _MAX_TOOL_ROUNDS = 3

    def __init__(self, *, model: ChatModel, calendar: CalendarTool, datastore: LeadRepository) -> None:
        self._model = model
        self._calendar = calendar
        self._datastore = datastore

    # --- contract subclasses must fill in -------------------------------------

    @abstractmethod
    def _run_lookup(self, *, payload: dict[str, Any], session: Session) -> str:
        """Execute a LOOKUP payload against the datastore.

        Returns the text fed back to the model as the tool result. On success
        implementations must stash the found row in ``session.collected`` so
        :meth:`_apply` acts on the verified record, not on model-echoed values.
        """

    @abstractmethod
    def _apply(self, *, tags: dict[str, str], session: Session, raw: str) -> AgentReply:
        """Perform the agent's real action once the final tag block arrives."""

    def _flow_rules(self) -> str:
        """Extra prompt rules specific to the subclass (e.g. CHECK_SLOT usage)."""
        return ""

    # --- shared machinery -----------------------------------------------------

    def build_system_prompt(self, *, today: str, dates: list[UpcomingDate]) -> str:
        """Assemble the persona prompt with the tool and tag contracts."""
        tag_block = "\n".join(f"{tag}: <value>" for tag in self.final_tags)
        finalisation = (
            "COMPLETION (client must NEVER see these lines):\n"
            "Once the flow is complete, send the client a warm one-line "
            "confirmation, then emit ALL of these tags, one per line, exactly "
            "once:\n" + tag_block
            if self.final_tags
            else "This flow is read-only: never emit tags, just answer the client."
        )
        return (
            f"{self.persona}\n"
            f"CURRENT DATE: {today}\n\n"
            "TOOL CONTRACT (client must NEVER see these lines):\n"
            "To look up the client's existing record, emit on its own line:\n"
            'LOOKUP:{"..."}  -- fields as described in your flow rules.\n'
            "You will receive a [TOOL RESULT] message with what the datastore "
            "returned. NEVER invent or assume a record: if the tool says no "
            "record was found, that is the truth.\n\n"
            f"{self._flow_rules()}\n"
            "UPCOMING DATES TABLE (use exactly as shown, YYYY-MM-DD, Mon-Sat):\n"
            f"{render_dates_table(dates)}\n\n"
            f"{finalisation}"
        )

    def handle_turn(
        self,
        *,
        message: str,
        session: Session,
        history: list[ChatMessage],
        today: str,
    ) -> AgentReply:
        """Process one user turn, running the bounded tool loop as needed."""
        dates = upcoming_dates()
        system = self.build_system_prompt(today=today, dates=dates)
        turn: list[ChatMessage] = [*history, ChatMessage(role="user", content=message)]

        raw = self._model.complete(system=system, messages=turn)
        for _ in range(self._MAX_TOOL_ROUNDS):
            signal = self._extract_tool_signal(raw)
            if signal is None:
                break
            name, payload = signal
            result = self._dispatch_tool(name=name, payload=payload, session=session)
            logger.info("%s tool %s -> %s", type(self).__name__, name, result[:80])
            turn.append(ChatMessage(role="assistant", content=raw))
            turn.append(ChatMessage(role="user", content=f"[TOOL RESULT] {result}"))
            raw = self._model.complete(system=system, messages=turn)

        tags = self._extract_final_tags(raw)
        if tags is None:
            return AgentReply(text=self._strip_machine_lines(raw), is_final=False)
        return self._apply(tags=tags, session=session, raw=raw)

    def _dispatch_tool(self, *, name: str, payload: dict[str, Any], session: Session) -> str:
        """Route an intermediate signal to its handler."""
        if name == "LOOKUP":
            return self._run_lookup(payload=payload, session=session)
        if name == "CHECK_SLOT":
            return self._run_slot_check(payload=payload)
        return "Unknown tool."  # unreachable given the signal regex

    def _run_slot_check(self, *, payload: dict[str, Any]) -> str:
        """Check calendar availability for a proposed (date, time).

        Offered to subclasses whose flow rules mention CHECK_SLOT (reschedule).
        When the slot is taken, up to three free alternatives at the same time
        on upcoming days are computed here — in Python, against the real
        calendar — so the model relays genuine availability instead of
        guessing.
        """
        iso_date = str(payload.get("date", ""))
        time_ = str(payload.get("time", ""))
        if self._calendar.is_slot_free(iso_date=iso_date, time=time_):
            return f"Slot {iso_date} {time_} is FREE."
        alternatives = [
            f"{d.iso_date} {time_}"
            for d in upcoming_dates()
            if d.iso_date != iso_date and self._calendar.is_slot_free(iso_date=d.iso_date, time=time_)
        ][:3]
        return (
            f"Slot {iso_date} {time_} is TAKEN. Free alternatives: "
            + ("; ".join(alternatives) if alternatives else "none in the next two weeks")
        )

    def _extract_tool_signal(self, raw: str) -> tuple[str, dict[str, Any]] | None:
        """Return (signal_name, payload) if *raw* contains a tool signal."""
        match = _TOOL_SIGNAL_RE.search(raw)
        if not match:
            return None
        try:
            return match.group(1), json.loads(match.group(2))
        except json.JSONDecodeError:
            logger.warning("%s emitted an unparseable %s payload", type(self).__name__, match.group(1))
            return None

    def _extract_final_tags(self, raw: str) -> dict[str, str] | None:
        """Parse the final tag block; ``None`` unless every declared tag is present.

        All-or-nothing on purpose: a partial tag block means the model finished
        prematurely or mangled the format, and acting on half a cancellation
        request is worse than asking the flow to continue.
        """
        if not self.final_tags:
            return None
        tags: dict[str, str] = {}
        for tag in self.final_tags:
            match = re.search(rf"^{re.escape(tag)}:\s*(.+)$", raw, re.MULTILINE)
            if match is None:
                return None
            tags[tag] = match.group(1).strip()
        return tags

    def _strip_machine_lines(self, raw: str) -> str:
        """Remove tool signals and tag lines so the client never sees them."""
        cleaned = _TOOL_SIGNAL_RE.sub("", raw)
        for tag in self.final_tags:
            cleaned = re.sub(rf"^{re.escape(tag)}:.*$", "", cleaned, flags=re.MULTILINE)
        return cleaned.strip()
