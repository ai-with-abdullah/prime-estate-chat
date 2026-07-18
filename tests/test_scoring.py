"""Boundary tests for deterministic HOT/Warm/Cold lead scoring."""

from __future__ import annotations

import pytest

from prime_estate.domain.models import LeadScore
from prime_estate.domain.scoring import score_lead


class TestHot:
    @pytest.mark.parametrize("timeline", ["ASAP", "this week", "within 1 month", "urgent"])
    def test_near_term_with_clear_price(self, timeline: str) -> None:
        assert score_lead(timeline=timeline, price="45000000") is LeadScore.HOT

    def test_near_term_but_vague_price_is_not_hot(self) -> None:
        assert score_lead(timeline="ASAP", price="not sure yet") is not LeadScore.HOT

    def test_clear_price_but_slow_timeline_is_not_hot(self) -> None:
        assert score_lead(timeline="in about 3 months", price="45000000") is not LeadScore.HOT


class TestWarm:
    def test_mid_horizon(self) -> None:
        assert score_lead(timeline="2 months", price="2 crore") is LeadScore.WARM

    def test_unmatched_timeline_defaults_warm_leaning(self) -> None:
        # No pattern matches -> ambiguous 4-month bucket -> Warm, the
        # deliberate conservative default.
        assert score_lead(timeline="after my transfer", price="1.5M") is LeadScore.WARM

    def test_near_term_vague_price(self) -> None:
        assert score_lead(timeline="this month", price="negotiable") is LeadScore.WARM


class TestCold:
    @pytest.mark.parametrize("timeline", ["flexible", "no rush", "just looking", "whenever"])
    def test_vague_timeline(self, timeline: str) -> None:
        assert score_lead(timeline=timeline, price="45000000") is LeadScore.COLD

    def test_year_horizon(self) -> None:
        assert score_lead(timeline="in a year", price="45000000") is LeadScore.COLD

    def test_empty_signals(self) -> None:
        # Empty timeline falls to the ambiguous default (Warm), never Hot.
        assert score_lead(timeline="", price="") is not LeadScore.HOT
