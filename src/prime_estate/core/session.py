"""Session state management — the port of the ``Session Check`` node.

This component answers one question per inbound message: *"is this user already
mid-flow, so I should skip re-classification and stay in their current intent?"*
Getting that right is what makes the bot feel coherent instead of goldfish-brained
— without it, a user answering the seller agent's "what's your asking price?"
with "450000" would be re-routed as a general enquiry.

Three rules, ported verbatim from the original node:

1. **TTL** — a session older than 30 minutes since last activity is stale and
   discarded. Real-estate conversations are bursty; a 30-minute gap almost
   always means a new topic.
2. **Override keywords** — words like "cancel" or "reschedule" always bypass the
   cache and force re-routing, because they are meta-commands that must escape
   whatever flow the user is in.
3. **Sticky intent** — otherwise, a valid session's stored intent is reused.

The original stored sessions in n8n's global static data; here they live in an
injectable store so the policy is unit-testable and the backing store
(in-memory now, Redis later) is a swappable detail.
"""

from __future__ import annotations

from prime_estate.domain.models import Intent, Session
from prime_estate.utils.logging import get_logger

logger = get_logger(__name__)

# 30 minutes, matching the original SESSION_TIMEOUT_MS. Expressed in seconds to
# match ``time.time()`` used by the Session model's timestamps.
SESSION_TTL_SECONDS: float = 30 * 60

# Words that always bypass the cache and re-route. Ported verbatim; these are
# meta-commands that must interrupt any in-progress flow.
OVERRIDE_KEYWORDS: tuple[str, ...] = (
    "cancel",
    "reschedule",
    "follow up",
    "followup",
    "check my booking",
    "check status",
    "my appointment",
)


def has_override(text: str) -> bool:
    """True if *text* contains any override keyword (case-insensitive substring)."""
    lowered = text.lower().strip()
    return any(keyword in lowered for keyword in OVERRIDE_KEYWORDS)


class SessionStore:
    """Injectable session cache enforcing TTL + override + stickiness.

    The backing dict is an implementation detail; the public contract is
    :meth:`resolve` (read a usable cached intent, if any) and :meth:`remember`
    (persist an intent for subsequent turns).
    """

    def __init__(self, *, ttl_seconds: float = SESSION_TTL_SECONDS) -> None:
        self._sessions: dict[str, Session] = {}
        self._ttl = ttl_seconds

    def resolve(self, *, session_id: str, text: str, now: float) -> Intent | None:
        """Return the cached intent to reuse, or ``None`` to force re-routing.

        Returns ``None`` — meaning "the router must classify this turn" — when
        any of: no session exists, the message contains an override keyword, the
        stored session has no intent, or the session has expired. Expired
        sessions are evicted as a side effect, mirroring the original node's
        "clear stale session" branch.

        The explicit *now* parameter (rather than calling ``time.time()``
        internally) keeps TTL behaviour deterministic under test.
        """
        if has_override(text):
            logger.info("session %s: override keyword — forcing re-route", session_id)
            return None

        session = self._sessions.get(session_id)
        if session is None or session.intent is None:
            return None

        age = now - session.updated_at
        if age >= self._ttl:
            logger.info("session %s: expired (%.0fs old) — evicting", session_id, age)
            del self._sessions[session_id]
            return None

        logger.info("session %s: cache hit — reusing intent %s", session_id, session.intent)
        return session.intent

    def remember(self, *, session_id: str, intent: Intent, now: float) -> Session:
        """Persist *intent* for *session_id*, creating or refreshing the session.

        Called after the router classifies a fresh turn, so the next message
        from the same user stays sticky to this intent until it expires or an
        override fires.
        """
        session = self._sessions.get(session_id)
        if session is None:
            session = Session(session_id=session_id, intent=intent, created_at=now, updated_at=now)
        else:
            session.intent = intent
            session.updated_at = now
        self._sessions[session_id] = session
        return session

    def clear(self, *, session_id: str) -> None:
        """Drop a session entirely (e.g. after a completed booking flow)."""
        self._sessions.pop(session_id, None)
