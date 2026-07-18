"""Tests for the session store: TTL, override keywords, stickiness."""

from __future__ import annotations

from prime_estate.core.session import SESSION_TTL_SECONDS, SessionStore, has_override
from prime_estate.domain.models import Intent

_T0 = 1_000_000.0


class TestOverrideKeywords:
    def test_meta_commands_detected(self) -> None:
        assert has_override("I want to CANCEL my meeting")
        assert has_override("can we reschedule?")
        assert has_override("just a follow up on my appointment")

    def test_ordinary_text_is_not_an_override(self) -> None:
        assert not has_override("I want to sell my flat in DHA")


class TestSessionStore:
    def test_sticky_hit_within_ttl(self) -> None:
        store = SessionStore()
        store.remember(session_id="u1", intent=Intent.SELLER, now=_T0)
        assert store.resolve(session_id="u1", text="it is 5 marla", now=_T0 + 60) is Intent.SELLER

    def test_expired_session_is_evicted(self) -> None:
        store = SessionStore()
        store.remember(session_id="u1", intent=Intent.SELLER, now=_T0)
        late = _T0 + SESSION_TTL_SECONDS + 1
        assert store.resolve(session_id="u1", text="hello again", now=late) is None
        # Eviction is permanent: an immediate retry also misses.
        assert store.resolve(session_id="u1", text="hello again", now=late) is None

    def test_exactly_at_ttl_is_expired(self) -> None:
        store = SessionStore()
        store.remember(session_id="u1", intent=Intent.RENT, now=_T0)
        assert store.resolve(session_id="u1", text="hi", now=_T0 + SESSION_TTL_SECONDS) is None

    def test_override_keyword_bypasses_live_session(self) -> None:
        store = SessionStore()
        store.remember(session_id="u1", intent=Intent.SELLER, now=_T0)
        assert store.resolve(session_id="u1", text="actually, cancel my booking", now=_T0 + 5) is None

    def test_remember_refreshes_ttl(self) -> None:
        store = SessionStore()
        store.remember(session_id="u1", intent=Intent.SELLER, now=_T0)
        near_expiry = _T0 + SESSION_TTL_SECONDS - 10
        store.remember(session_id="u1", intent=Intent.SELLER, now=near_expiry)
        beyond_original = _T0 + SESSION_TTL_SECONDS + 60
        assert store.resolve(session_id="u1", text="hi", now=beyond_original) is Intent.SELLER

    def test_unknown_session_misses(self) -> None:
        store = SessionStore()
        assert store.resolve(session_id="ghost", text="hello", now=_T0) is None

    def test_clear_drops_session(self) -> None:
        store = SessionStore()
        store.remember(session_id="u1", intent=Intent.SELLER, now=_T0)
        store.clear(session_id="u1")
        assert store.resolve(session_id="u1", text="hi", now=_T0 + 1) is None

    def test_collected_state_survives_remember(self) -> None:
        # Lookup agents stash verified identity in session.collected between
        # turns; remember() must refresh the session without discarding it.
        store = SessionStore()
        session = store.remember(session_id="u1", intent=Intent.CANCEL, now=_T0)
        session.collected["verified_email"] = "a@b.co"
        refreshed = store.remember(session_id="u1", intent=Intent.CANCEL, now=_T0 + 30)
        assert refreshed.collected["verified_email"] == "a@b.co"
