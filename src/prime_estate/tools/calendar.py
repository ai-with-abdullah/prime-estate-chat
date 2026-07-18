"""In-memory calendar tool.

Mirrors the calendar capabilities the original agents used through the Google
Calendar nodes: check availability before confirming a slot, create the event on
booking, delete it on cancellation. The production implementation swaps this for
a Google-Calendar-backed class satisfying the same
:class:`~prime_estate.tools.base.CalendarTool` Protocol; nothing else in the
system changes.

Availability is the interesting decision point. In the original design the
seller agent was instructed: "Before confirming any meeting slot, use the
calendar tool to check availability. If slot is taken: suggest 3 alternatives."
That check is a genuine tool call whose result changes the agent's next action —
which is exactly the "model making real decisions inside boundaries" behaviour
the architecture is built around.
"""

from __future__ import annotations

import uuid

from prime_estate.domain.models import Lead


class InMemoryCalendar:
    """A dict-backed calendar satisfying the ``CalendarTool`` protocol."""

    def __init__(self) -> None:
        # (iso_date, time) -> event_id
        self._slots: dict[tuple[str, str], str] = {}
        # event_id -> (iso_date, time), for O(1) deletion
        self._events: dict[str, tuple[str, str]] = {}

    def is_slot_free(self, *, iso_date: str, time: str) -> bool:
        """Return True if nothing is booked at *iso_date* / *time*."""
        return (iso_date, time) not in self._slots

    def create_event(self, *, lead: Lead) -> str:
        """Book *lead*'s meeting; return the new event id.

        Raises:
            SlotUnavailableError: if the slot was taken between the agent's
                availability check and this call. Re-checking here rather than
                trusting the earlier check closes the race where two concurrent
                conversations book the same slot.
        """
        key = (lead.meeting_date, lead.meeting_time)
        if key in self._slots:
            raise SlotUnavailableError(iso_date=lead.meeting_date, time=lead.meeting_time)
        event_id = uuid.uuid4().hex
        self._slots[key] = event_id
        self._events[event_id] = key
        return event_id

    def delete_event(self, *, event_id: str) -> bool:
        """Cancel the event with *event_id*, freeing its slot. Returns success."""
        key = self._events.pop(event_id, None)
        if key is None:
            return False
        self._slots.pop(key, None)
        return True

    def suggest_alternatives(
        self, *, candidates: list[tuple[str, str]], limit: int = 3
    ) -> list[tuple[str, str]]:
        """Return up to *limit* free (date, time) pairs from *candidates*.

        Supports the "suggest 3 alternative times" behaviour from the seller
        prompt. Kept as a separate method (not part of the Protocol) because it
        is a convenience over ``is_slot_free``, not a distinct capability.
        """
        free = [(d, t) for (d, t) in candidates if self.is_slot_free(iso_date=d, time=t)]
        return free[:limit]


class SlotUnavailableError(RuntimeError):
    """Raised when booking a slot that is already taken."""

    def __init__(self, *, iso_date: str, time: str) -> None:
        super().__init__(f"slot {iso_date} {time} is already booked")
        self.iso_date = iso_date
        self.time = time
