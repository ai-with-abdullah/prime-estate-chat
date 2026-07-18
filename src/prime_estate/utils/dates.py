"""Deterministic date handling for meeting scheduling.

The original agents were explicitly forbidden from calculating dates themselves
("NEVER calculate dates yourself — always use the UPCOMING DATES TABLE"). That
rule exists because LLMs reliably miscount weekday offsets, producing bookings
on the wrong day or — worse — on a Sunday when the office is closed. We keep the
same contract here: the table is computed in ordinary Python and injected into
the prompt, so the model only ever *selects* a date, never *derives* one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

# 6 == Sunday in Python's date.weekday() (Mon=0 .. Sun=6). The business is closed
# Sundays, so Sunday is never offered as a bookable slot.
_CLOSED_WEEKDAY = 6


@dataclass(frozen=True)
class UpcomingDate:
    """A single bookable calendar day."""

    day_name: str  # e.g. "Wednesday"
    iso_date: str  # e.g. "2026-07-22"


def upcoming_dates(*, days_ahead: int = 14, today: date | None = None) -> list[UpcomingDate]:
    """Return the next *days_ahead* bookable days (Mon–Sat), starting tomorrow.

    Args:
        days_ahead: How many calendar days forward to scan.
        today: Injectable "now" for deterministic tests. Defaults to the real
            system date.

    Returns:
        Bookable days in chronological order, Sundays omitted. Starting from
        *tomorrow* mirrors the original behaviour of never offering same-day
        slots.
    """
    anchor = today or date.today()
    out: list[UpcomingDate] = []
    for offset in range(1, days_ahead + 1):
        d = anchor + timedelta(days=offset)
        if d.weekday() == _CLOSED_WEEKDAY:
            continue
        out.append(UpcomingDate(day_name=d.strftime("%A"), iso_date=d.isoformat()))
    return out


def render_dates_table(dates: list[UpcomingDate]) -> str:
    """Render the dates as a compact table for prompt injection.

    Format is deliberately plain (``DayName: YYYY-MM-DD`` per line) so the model
    can copy the ISO value verbatim without reformatting.
    """
    return "\n".join(f"{d.day_name}: {d.iso_date}" for d in dates)


def is_bookable(iso_date: str, *, today: date | None = None) -> bool:
    """True if *iso_date* is a real future Mon–Sat date.

    Guards the case where a user free-types a date that bypassed the table: it
    must parse, fall on or after tomorrow, and not be a Sunday.
    """
    anchor = today or date.today()
    try:
        parsed = date.fromisoformat(iso_date)
    except ValueError:
        return False
    return parsed > anchor and parsed.weekday() != _CLOSED_WEEKDAY
