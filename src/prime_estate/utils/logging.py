"""Centralised logger factory.

A single configuration point keeps log format consistent across modules and
avoids each file re-implementing ``logging.basicConfig``. Structured, prefixed
logs matter here because the system is conversational and asynchronous — when a
booking goes wrong you need to trace one session id across the router, an agent,
and two tool calls.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def _configure_root() -> None:
    """Attach a single stderr handler to the package root logger once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger("prime_estate")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``prime_estate`` root."""
    _configure_root()
    return logging.getLogger(name)
