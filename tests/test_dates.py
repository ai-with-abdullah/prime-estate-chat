"""Tests for the deterministic upcoming-dates table."""

from __future__ import annotations

from datetime import date

from prime_estate.utils.dates import is_bookable, render_dates_table, upcoming_dates

# A fixed anchor: Wednesday 2030-01-02.
_ANCHOR = date(2030, 1, 2)


class TestUpcomingDates:
    def test_starts_tomorrow_never_today(self) -> None:
        dates = upcoming_dates(days_ahead=7, today=_ANCHOR)
        assert dates[0].iso_date == "2030-01-03"
        assert all(d.iso_date != _ANCHOR.isoformat() for d in dates)

    def test_never_contains_a_sunday(self) -> None:
        dates = upcoming_dates(days_ahead=21, today=_ANCHOR)
        assert all(date.fromisoformat(d.iso_date).weekday() != 6 for d in dates)
        # And Sundays were genuinely in range, so something was skipped.
        assert len(dates) == 18  # 21 days minus 3 Sundays

    def test_day_names_match_iso_dates(self) -> None:
        for d in upcoming_dates(days_ahead=10, today=_ANCHOR):
            assert date.fromisoformat(d.iso_date).strftime("%A") == d.day_name


class TestRenderTable:
    def test_one_line_per_date_iso_verbatim(self) -> None:
        dates = upcoming_dates(days_ahead=3, today=_ANCHOR)
        table = render_dates_table(dates)
        lines = table.splitlines()
        assert len(lines) == len(dates)
        assert lines[0] == f"{dates[0].day_name}: {dates[0].iso_date}"


class TestIsBookable:
    def test_future_weekday_is_bookable(self) -> None:
        assert is_bookable("2030-01-07", today=_ANCHOR)  # a Monday

    def test_sunday_is_not(self) -> None:
        assert not is_bookable("2030-01-06", today=_ANCHOR)  # a Sunday

    def test_today_and_past_are_not(self) -> None:
        assert not is_bookable("2030-01-02", today=_ANCHOR)
        assert not is_bookable("2029-12-28", today=_ANCHOR)

    def test_garbage_is_not(self) -> None:
        assert not is_bookable("next tuesday", today=_ANCHOR)
