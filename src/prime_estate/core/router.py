"""Intent router — the decision core of the system.

This is the component that makes the system *agentic* rather than a fixed script:
a model reads the user's message plus the conversation so far and decides which
of eight bounded specialist agents should take the turn. That decision changes
control flow. Everything downstream (which agent runs, which tools it may touch,
what it collects) follows from this one classification.

Two design choices carried over from the original ``AI Router Agent`` prompt and
made explicit here:

* **Context continuity is enforced structurally, not just prompted.** The
  original prompt begged the model to "continue classifying with that same
  intent" mid-flow; in this port the :class:`~prime_estate.core.session.SessionStore`
  short-circuits the router entirely when a live session exists, so continuity
  is guaranteed by architecture instead of hoped for from the LLM.
* **The router's output space is a closed enum.** It returns an
  :class:`~prime_estate.domain.models.Intent`, never a free string, so a garbled
  LLM response degrades to ``GENERAL`` rather than crashing dispatch.
"""

from __future__ import annotations

from prime_estate.domain.models import Intent
from prime_estate.llm.groq_client import ChatMessage, ChatModel
from prime_estate.utils.logging import get_logger

logger = get_logger(__name__)

# Ported from the original AI Router Agent system prompt. Trimmed to the
# classification contract; the model's sole job is to emit one intent tag.
ROUTER_SYSTEM_PROMPT = """\
ROLE: You are the intent classifier for Prime Estate.

TASK: Read the user's current message AND the conversation history. Output ONLY
the correct intent tag — nothing else.

INTENT CATEGORIES:
[[INTENT:qualify]]    -> buying, or any question related to buying property
[[INTENT:seller]]     -> selling a property, valuation, listing
[[INTENT:investor]]   -> investment, ROI, rental yield, buy-to-let, portfolio
[[INTENT:rent]]       -> renting out or taking a property on rent
[[INTENT:reschedule]] -> changing an existing meeting date or time
[[INTENT:cancel]]     -> cancelling or removing a booking entirely
[[INTENT:followup]]   -> checking the status of a previously submitted lead
[[INTENT:general]]    -> greetings, FAQs, anything uncategorised

OUTPUT FORMAT:
Return ONLY the tag. Correct: [[INTENT:seller]]
Wrong: "The user wants to sell, so [[INTENT:seller]]."
"""


class IntentRouter:
    """Classifies an inbound turn into a single :class:`Intent`.

    The router is intentionally stateless: continuity across turns is the
    :class:`SessionStore`'s responsibility. This one only answers "given this
    message and history, what is the user trying to do *right now*?"
    """

    def __init__(self, *, model: ChatModel) -> None:
        self._model = model

    def classify(self, *, text: str, history: list[ChatMessage] | None = None) -> Intent:
        """Return the :class:`Intent` for *text* in the context of *history*.

        The model is prompted to emit a single ``[[INTENT:x]]`` tag; parsing is
        defensive (:meth:`Intent.from_tag`) so any deviation — extra prose, wrong
        casing, an unknown label — collapses safely to ``GENERAL`` instead of
        misrouting or raising.
        """
        messages = list(history or [])
        messages.append(ChatMessage(role="user", content=text))

        raw = self._model.complete(system=ROUTER_SYSTEM_PROMPT, messages=messages)
        intent = Intent.from_tag(raw)
        logger.info("router classified %r -> %s", text[:60], intent)
        return intent
