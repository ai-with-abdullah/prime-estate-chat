"""Silent lead qualification.

In the original prompts each agent was told to score leads HOT / Warm / Cold
"silently — never show client". Leaving that judgement inside the LLM prompt
made it non-deterministic and unauditable: the same lead could score differently
across runs, and there was no way to unit-test the rule. Here the scoring is
lifted out of the prompt into deterministic code. The model's only job is to
capture the two signals (timeline and price clarity); the classification is a
pure function of those signals.
"""

from __future__ import annotations

import re

from prime_estate.domain.models import LeadScore

# Timeline phrasing → an approximate "months until they act" bucket. Ordered by
# specificity so the first match wins.
_MONTH_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b(asap|immediately|this week|urgent|right away)\b", re.I), 0.25),
    (re.compile(r"\b(within|in)\s*(a|1)\s*month\b", re.I), 1.0),
    (re.compile(r"\b(1|2|3|4)\s*months?\b", re.I), 3.0),
    (re.compile(r"\b(this|next)\s*month\b", re.I), 1.0),
    (re.compile(r"\b(6|six)\s*months?\b", re.I), 6.0),
    (re.compile(r"\b(year|12\s*months?)\b", re.I), 12.0),
    (re.compile(r"\b(flexible|no rush|just (looking|browsing)|whenever)\b", re.I), 99.0),
]


def _timeline_to_months(timeline: str) -> float:
    """Map a free-text timeline to an approximate month count.

    Returns a large sentinel (99) for vague/flexible timelines so they fall into
    the COLD bucket. Defaults to 4.0 (ambiguous → Warm-leaning) when nothing
    matches, which is the conservative choice: better to keep a borderline lead
    warm than to prematurely cool it.
    """
    text = timeline or ""
    for pattern, months in _MONTH_PATTERNS:
        if pattern.search(text):
            return months
    return 4.0


def _has_clear_price(price: str) -> bool:
    """True if the asking/budget figure contains a concrete number.

    A price like "around 2 crore" or "AED 1.5M" counts as clear; "not sure yet"
    or an empty string does not. Clarity of price is the second axis of intent —
    a seller who names a firm number is materially more serious than one who
    hedges.
    """
    if not price:
        return False
    if re.search(r"\b(not sure|dunno|don'?t know|flexible|negotiable|tbd|open)\b", price, re.I):
        return False
    return bool(re.search(r"\d", price))


def score_lead(*, timeline: str, price: str) -> LeadScore:
    """Classify a lead as HOT, Warm, or Cold from timeline + price clarity.

    Rules mirror the original prompt intent:

    * **HOT** — acting within ~1 month *and* a clear price/budget.
    * **Warm** — 2–4 month horizon, or a near-term timeline with a vague price.
    * **Cold** — flexible / very vague timeline.

    Encoding this as a table keeps it testable and consistent across every
    vertical, instead of relying on each agent's prompt to re-derive it.
    """
    months = _timeline_to_months(timeline)
    clear_price = _has_clear_price(price)

    if months <= 1.0 and clear_price:
        return LeadScore.HOT
    if months >= 12.0:
        return LeadScore.COLD
    if months <= 4.0:
        return LeadScore.WARM
    return LeadScore.COLD
