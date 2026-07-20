"""Google Calendar implementation of the CalendarTool protocol.

The contract mirrors the in-memory tool exactly: three capabilities, one-hour
meetings, and — critically — the same race discipline. ``create_event``
re-checks availability immediately before inserting, because between the
agent's conversational availability check and the actual insert, another
conversation (or a human with the calendar open) may have taken the slot.
Google Calendar happily double-books unless you refuse to; the refusal lives
here, raising the same :class:`SlotUnavailableError` the orchestrator already
handles for the in-memory tool.

The API ``service`` object is injected; credentials only enter through the
:func:`build_service` factory at the composition root.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from prime_estate.domain.models import Lead
from prime_estate.tools.calendar import SlotUnavailableError
from prime_estate.utils.logging import get_logger

logger = get_logger(__name__)

_MEETING_MINUTES = 60


class GoogleCalendarTool:
    """Calendar-API-backed implementation of the ``CalendarTool`` protocol."""

    def __init__(self, *, service: Any, calendar_id: str, timezone: str = "Asia/Karachi") -> None:
        self._service = service
        self._calendar_id = calendar_id
        self._tz = timezone

    def _window(self, iso_date: str, time: str) -> tuple[str, str]:
        """Return the RFC3339 (start, end) window for a one-hour slot."""
        start = datetime.fromisoformat(f"{iso_date}T{time}:00").replace(tzinfo=ZoneInfo(self._tz))
        end = start + timedelta(minutes=_MEETING_MINUTES)
        return start.isoformat(), end.isoformat()

    def is_slot_free(self, *, iso_date: str, time: str) -> bool:
        """True when no non-cancelled event overlaps the one-hour window."""
        start, end = self._window(iso_date, time)
        events = (
            self._service.events()
            .list(
                calendarId=self._calendar_id,
                timeMin=start,
                timeMax=end,
                singleEvents=True,
                maxResults=1,
            )
            .execute()
        )
        return not events.get("items", [])

    def create_event(self, *, lead: Lead) -> str:
        """Insert the meeting; raise ``SlotUnavailableError`` if just taken."""
        if not self.is_slot_free(iso_date=lead.meeting_date, time=lead.meeting_time):
            raise SlotUnavailableError(iso_date=lead.meeting_date, time=lead.meeting_time)
        start, end = self._window(lead.meeting_date, lead.meeting_time)
        body = {
            "summary": f"Prime Estate meeting — {lead.full_name}",
            "description": (
                f"Intent: {lead.intent.value}\nPhone: {lead.phone}\nEmail: {lead.email}"
            ),
            "start": {"dateTime": start, "timeZone": self._tz},
            "end": {"dateTime": end, "timeZone": self._tz},
        }
        created = self._service.events().insert(calendarId=self._calendar_id, body=body).execute()
        event_id = str(created["id"])
        logger.info("calendar: created event %s (%s %s)", event_id, lead.meeting_date, lead.meeting_time)
        return event_id

    def delete_event(self, *, event_id: str) -> bool:
        """Delete the event; a missing/already-deleted event returns False."""
        try:
            self._service.events().delete(calendarId=self._calendar_id, eventId=event_id).execute()
        except Exception as exc:  # noqa: BLE001 — adapter maps any API failure to False
            logger.warning("calendar: delete of %s failed: %s", event_id, exc)
            return False
        return True


def build_service(*, service_account_file: str) -> Any:
    """Build an authenticated Calendar API client from a service account.

    Share the target calendar with the service account's ``client_email``
    (with "Make changes to events" permission) or every write will 403.
    """
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=["https://www.googleapis.com/auth/calendar"]
    )
    return build("calendar", "v3", credentials=credentials, cache_discovery=False)
