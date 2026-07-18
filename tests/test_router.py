"""Tests for the intent router's closed-enum classification."""

from __future__ import annotations

import pytest

from prime_estate.core.router import IntentRouter
from prime_estate.domain.models import Intent
from prime_estate.llm.groq_client import ChatMessage
from tests.conftest import ScriptedChatModel


class TestTagParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("[[INTENT:seller]]", Intent.SELLER),
            ("[[INTENT:qualify]]", Intent.QUALIFY),
            ("[[INTENT:INVESTOR]]", Intent.INVESTOR),  # casing tolerated
            ("  [[intent:rent]]  ", Intent.RENT),
            ("The user wants to sell, so [[INTENT:seller]].", Intent.SELLER),  # prose-wrapped
            ("[[INTENT:fals]]", Intent.GENERAL),  # legacy alias
            ("[[INTENT:unknown-label]]", Intent.GENERAL),
            ("complete garbage with no tag", Intent.GENERAL),
            ("", Intent.GENERAL),
        ],
    )
    def test_from_tag(self, raw: str, expected: Intent) -> None:
        assert Intent.from_tag(raw) is expected


class TestRouter:
    def test_classifies_via_model(self) -> None:
        model = ScriptedChatModel(["[[INTENT:seller]]"])
        router = IntentRouter(model=model)
        assert router.classify(text="I want to sell my flat") is Intent.SELLER

    def test_garbled_model_output_degrades_to_general(self) -> None:
        model = ScriptedChatModel(["I think this user might be... hmm"])
        router = IntentRouter(model=model)
        assert router.classify(text="???") is Intent.GENERAL

    def test_history_is_forwarded_to_the_model(self) -> None:
        model = ScriptedChatModel(["[[INTENT:cancel]]"])
        router = IntentRouter(model=model)
        history = [
            ChatMessage(role="user", content="I booked a meeting yesterday"),
            ChatMessage(role="assistant", content="Great, how can I help?"),
        ]
        router.classify(text="I need to call it off", history=history)
        sent = model.calls[0]["messages"]
        assert isinstance(sent, list)
        # History precedes the current turn, and the current turn is last.
        assert len(sent) == 3
        assert sent[-1].content == "I need to call it off"
