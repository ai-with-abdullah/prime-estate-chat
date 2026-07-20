"""Google Sheets implementation of the lead datastore.

This is the production persistence layer the in-memory datastore was shaped
after: header in row 1, data from row 2, one lead per row — which is why the
in-memory implementation always reported header-offset row ids. Here that
contract stops being a simulation and becomes literal sheet coordinates.

Two design decisions worth defending:

* **The dedup/slot/lookup rules are not re-implemented.** Every read-side
  check loads a snapshot of the sheet into an
  :class:`~prime_estate.tools.datastore.InMemoryLeadDatastore` and delegates.
  Duplicated business rules drift; delegated ones cannot. The cost is a full
  sheet read per check, which at a lead-agency's volume (hundreds of rows) is
  irrelevant next to an LLM turn.
* **The gspread worksheet is injected, not constructed.** The class holds an
  object with ``get_all_records`` / ``append_row`` / ``update`` — which a
  10-line fake satisfies in tests. Credentials only enter through the
  :func:`open_worksheet` factory at the composition root.
"""

from __future__ import annotations

import json
from typing import Any

from prime_estate.domain.models import Intent, Lead, LeadScore, LeadStage, SlotCheck
from prime_estate.tools.datastore import InMemoryLeadDatastore
from prime_estate.utils.logging import get_logger

logger = get_logger(__name__)

# Sheet header, column order fixed. `extra` is a JSON blob column so
# vertical-specific fields survive round-trips without a per-vertical schema.
COLUMNS: tuple[str, ...] = (
    "fullName",
    "email",
    "phone",
    "intent",
    "meetingDate",
    "meetingTime",
    "timezone",
    "stage",
    "score",
    "calendarEventId",
    "extra",
)

_HEADER_OFFSET = 2  # row 1 is the header; first data row is 2


def _lead_to_row(lead: Lead) -> list[str]:
    """Serialise a Lead into the fixed column order."""
    return [
        lead.full_name,
        lead.email,
        lead.phone,
        lead.intent.value,
        lead.meeting_date,
        lead.meeting_time,
        lead.timezone,
        lead.stage.value,
        lead.score.value if lead.score else "",
        lead.calendar_event_id or "",
        json.dumps(lead.extra) if lead.extra else "",
    ]


def _record_to_lead(record: dict[str, Any]) -> Lead:
    """Deserialise a sheet record defensively.

    Sheets are human-editable: someone WILL eventually hand-type a stage or
    delete a cell. Every enum parse falls back to a safe default instead of
    crashing the whole agent on one bad row.
    """
    def _enum(cls: Any, raw: Any, default: Any) -> Any:
        try:
            return cls(str(raw))
        except ValueError:
            return default

    raw_extra = str(record.get("extra", "") or "")
    try:
        extra = json.loads(raw_extra) if raw_extra else {}
    except json.JSONDecodeError:
        extra = {"_unparsed": raw_extra}

    score_raw = str(record.get("score", "") or "")
    return Lead(
        full_name=str(record.get("fullName", "")),
        email=str(record.get("email", "")),
        phone=str(record.get("phone", "")),
        intent=_enum(Intent, record.get("intent"), Intent.GENERAL),
        meeting_date=str(record.get("meetingDate", "")),
        meeting_time=str(record.get("meetingTime", "")),
        timezone=str(record.get("timezone", "") or "PKT"),
        stage=_enum(LeadStage, record.get("stage"), LeadStage.NEW),
        score=_enum(LeadScore, score_raw, None) if score_raw else None,
        calendar_event_id=str(record.get("calendarEventId", "") or "") or None,
        extra=extra,
    )


class GoogleSheetsLeadDatastore:
    """Sheet-backed datastore satisfying ``LeadDatastore`` + ``LeadLookup``."""

    def __init__(self, *, worksheet: Any) -> None:
        self._ws = worksheet

    def _snapshot(self) -> InMemoryLeadDatastore:
        """Load the sheet into an in-memory datastore and reuse its rules."""
        snapshot = InMemoryLeadDatastore()
        for record in self._ws.get_all_records():
            snapshot.save(lead=_record_to_lead(record))
        return snapshot

    def check(self, *, lead: Lead) -> SlotCheck:
        """Duplicate + slot-conflict detection, delegated to the shared rules."""
        return self._snapshot().check(lead=lead)

    def find_by_contact(self, *, email: str, phone: str) -> tuple[int, Lead] | None:
        """Two-factor lookup, delegated to the shared rules."""
        return self._snapshot().find_by_contact(email=email, phone=phone)

    def find_by_email_or_name(self, *, query: str) -> tuple[int, Lead] | None:
        """Loose read-only lookup, delegated to the shared rules."""
        return self._snapshot().find_by_email_or_name(query=query)

    def save(self, *, lead: Lead) -> int:
        """Append the lead as a new row; return its 1-based sheet row number."""
        self._ws.append_row(_lead_to_row(lead), value_input_option="RAW")
        row = len(self._ws.get_all_records()) - 1 + _HEADER_OFFSET
        logger.info("sheet: saved lead %s at row %d", lead.email, row)
        return row

    def update(self, *, row: int, lead: Lead) -> None:
        """Overwrite the sheet row (A..K) with the lead's current state."""
        end_col = chr(ord("A") + len(COLUMNS) - 1)
        self._ws.update(f"A{row}:{end_col}{row}", [_lead_to_row(lead)], value_input_option="RAW")
        logger.info("sheet: updated row %d (%s -> %s)", row, lead.email, lead.stage.value)


def open_worksheet(*, service_account_file: str, sheet_id: str, tab: str = "Leads") -> Any:
    """Open (creating if needed) the leads worksheet using a service account.

    The only place gspread and credentials are touched. Share the spreadsheet
    with the service account's ``client_email`` or every call will 403.
    """
    import gspread

    client = gspread.service_account(filename=service_account_file)
    spreadsheet = client.open_by_key(sheet_id)
    try:
        worksheet = spreadsheet.worksheet(tab)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=tab, rows=1000, cols=len(COLUMNS))
        worksheet.append_row(list(COLUMNS), value_input_option="RAW")
        logger.info("sheet: created tab %r with header row", tab)
    return worksheet
