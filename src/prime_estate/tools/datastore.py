"""In-memory lead datastore implementing the real dedup + slot-conflict rules.

This is a faithful port of the ``Seller Dup + Slot Check`` node, promoted from
row-scanning JavaScript over a Google Sheet into a typed component. The *logic*
is identical — duplicate on email OR phone, slot taken when an active booking
shares the same date and time — but it now lives behind the
:class:`~prime_estate.tools.base.LeadDatastore` Protocol, so the production
Sheets-backed implementation and this in-memory one are interchangeable.

Keeping an in-memory implementation as a first-class citizen (not just a test
stub) means the whole agent loop can run and be demonstrated with zero external
credentials.
"""

from __future__ import annotations

from prime_estate.domain.models import Lead, LeadStage, SlotCheck


class InMemoryLeadDatastore:
    """A list-backed datastore satisfying the ``LeadDatastore`` protocol.

    Rows are stored 0-indexed internally but reported 1-based + header-offset to
    match the original spreadsheet semantics (``existing_row = i + 2``), so the
    contract callers see is unchanged from the n8n version.
    """

    _HEADER_OFFSET = 2  # row 1 is the sheet header; data starts at row 2

    def __init__(self) -> None:
        self._rows: list[Lead] = []

    def check(self, *, lead: Lead) -> SlotCheck:
        """Detect duplicates and slot conflicts for a candidate *lead*.

        Duplicate rule (ported verbatim): a match on **email OR phone** — after
        normalising both sides — is sufficient. Using OR rather than AND is a
        deliberate product decision from the original system: a returning client
        who books once by email and later by a different email but the same
        phone is still the same person, and vice versa.

        Slot rule (ported verbatim): a slot is taken when an existing record
        shares the same date and time and is not ``Cancelled`` — cancelled
        bookings free their slot.
        """
        cand_email, cand_phone = lead.dedup_key()

        is_duplicate = False
        existing_row: int | None = None
        for i, row in enumerate(self._rows):
            row_email, row_phone = row.dedup_key()
            email_matches = bool(cand_email and row_email and row_email == cand_email)
            phone_matches = bool(cand_phone and row_phone and row_phone == cand_phone)
            if email_matches or phone_matches:
                is_duplicate = True
                existing_row = i + self._HEADER_OFFSET
                break

        slot_taken = any(
            row.meeting_date == lead.meeting_date
            and row.meeting_time == lead.meeting_time
            and row.stage is not LeadStage.CANCELLED
            for row in self._rows
        )

        return SlotCheck(
            is_duplicate=is_duplicate,
            existing_row=existing_row,
            slot_taken=slot_taken,
        )

    def save(self, *, lead: Lead) -> int:
        """Append *lead* and return its 1-based, header-offset row id."""
        self._rows.append(lead)
        return len(self._rows) - 1 + self._HEADER_OFFSET

    def update(self, *, row: int, lead: Lead) -> None:
        """Overwrite the record at *row* (header-offset) with *lead*."""
        index = row - self._HEADER_OFFSET
        if not 0 <= index < len(self._rows):
            raise IndexError(f"row {row} out of range")
        self._rows[index] = lead

    def find_by_contact(self, *, email: str, phone: str) -> tuple[int, Lead] | None:
        """Return (header-offset row, lead) where BOTH email and phone match.

        Normalisation reuses :meth:`Lead.dedup_key` on both sides so the match
        rule is byte-identical to duplicate detection. Requiring both factors
        (unlike ``check``'s either/or) is the guard for destructive flows —
        knowing someone's email alone must not be enough to cancel their
        booking.
        """
        cand_email = (email or "").lower().strip()
        cand_phone = (phone or "").replace(" ", "").lstrip("+").strip()
        if not cand_email or not cand_phone:
            return None
        for i, row in enumerate(self._rows):
            row_email, row_phone = row.dedup_key()
            if row_email == cand_email and row_phone == cand_phone:
                return i + self._HEADER_OFFSET, row
        return None

    def find_by_email_or_name(self, *, query: str) -> tuple[int, Lead] | None:
        """Return (header-offset row, lead) matching *query* by email or name.

        Case-insensitive exact match on email, case-insensitive match on the
        full name. Read-only followup flow only — see the Protocol docstring
        for why this looser rule is acceptable there and nowhere else.
        """
        needle = (query or "").lower().strip()
        if not needle:
            return None
        for i, row in enumerate(self._rows):
            if row.email.lower().strip() == needle or row.full_name.lower().strip() == needle:
                return i + self._HEADER_OFFSET, row
        return None

    def all_leads(self) -> list[Lead]:
        """Return a snapshot copy of stored leads (diagnostics/tests)."""
        return list(self._rows)
