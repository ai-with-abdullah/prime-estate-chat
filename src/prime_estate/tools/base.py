"""Tool abstraction layer.

An "agent with tools" is only meaningfully bounded if the tools are a small,
typed, auditable surface — otherwise "autonomy" degenerates into "the model can
do anything". Every side-effecting capability an agent has (reading the
datastore, checking the calendar, booking a slot) is expressed as a Tool with a
narrow method signature. The agent chooses *which* tool to call and *with what
arguments*; it can never reach past this interface to the raw Google API. That
separation is the boundary.

The Protocol-based design also makes the whole system testable without network:
tests inject in-memory fakes that satisfy the same Protocol as the production
Google-backed implementations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from prime_estate.domain.models import Lead, SlotCheck


@runtime_checkable
class CalendarTool(Protocol):
    """Read/write access to the scheduling calendar.

    Kept minimal on purpose: an agent needs exactly three calendar capabilities
    — check whether a slot is free, book it, and release it. Anything broader
    (listing every event, editing arbitrary events) would widen the blast radius
    without serving a real conversational need.
    """

    def is_slot_free(self, *, iso_date: str, time: str) -> bool:
        """Return True if no event occupies *iso_date* at *time*."""
        ...

    def create_event(self, *, lead: Lead) -> str:
        """Book *lead*'s meeting and return the created event id."""
        ...

    def delete_event(self, *, event_id: str) -> bool:
        """Cancel the event with *event_id*. Returns success."""
        ...


@runtime_checkable
class LeadDatastore(Protocol):
    """Persistence for captured leads (Google Sheets in production)."""

    def check(self, *, lead: Lead) -> SlotCheck:
        """Run duplicate + slot-conflict detection for a candidate *lead*."""
        ...

    def save(self, *, lead: Lead) -> int:
        """Persist a new *lead*; return its row/record id."""
        ...

    def update(self, *, row: int, lead: Lead) -> None:
        """Overwrite the record at *row* with *lead*'s current state."""
        ...


@runtime_checkable
class LeadLookup(Protocol):
    """Read-side retrieval of existing leads.

    Kept separate from :class:`LeadDatastore` on interface-segregation grounds:
    the four slot-filling verticals never need to *fetch* a lead, only to check
    a candidate for collisions — so they are not handed retrieval capability
    they have no business using. Only the lookup-based agents (cancel,
    reschedule, followup) depend on this Protocol.
    """

    def find_by_contact(self, *, email: str, phone: str) -> tuple[int, Lead] | None:
        """Return (row, lead) where BOTH normalised email and phone match.

        Both-must-match is deliberate: this powers destructive flows (cancel,
        reschedule), so identity is verified on two independent factors before
        anything is deleted or moved.
        """
        ...

    def find_by_email_or_name(self, *, query: str) -> tuple[int, Lead] | None:
        """Return (row, lead) matching *query* against email or full name.

        Looser than :meth:`find_by_contact` because it powers the read-only
        followup flow, where the worst failure mode is telling someone their
        own booking status.
        """
        ...


@runtime_checkable
class LeadRepository(LeadDatastore, LeadLookup, Protocol):
    """Combined read + write datastore surface for agents that need both."""
