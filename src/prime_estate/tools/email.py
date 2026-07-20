"""Email notification tool — booking confirmations via Gmail SMTP.

Notification is deliberately a *fire-and-forget side effect*, not part of the
booking transaction: a lead that booked successfully must stay booked even if
Gmail is down, so the orchestrator calls the notifier inside a try/except and
only logs failures. That ordering decision (persist first, notify best-effort)
is the difference between "the client did not get an email" and "the booking
silently vanished".

Gmail is reached over SMTP with an app password rather than the Gmail API:
same delivery result, no OAuth consent screens, one secret in the
environment. The SMTP transport is injectable so the notifier is unit-testable
without a network.
"""

from __future__ import annotations

import smtplib
from collections.abc import Callable
from email.message import EmailMessage
from typing import Protocol, runtime_checkable

from prime_estate.domain.models import Lead
from prime_estate.utils.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class LeadNotifier(Protocol):
    """Outbound notification capability, kept to the one event that matters."""

    def booking_confirmed(self, *, lead: Lead) -> None:
        """Notify *lead* that their meeting is booked."""
        ...


class GmailSmtpNotifier:
    """Sends booking confirmations from a Gmail account via SMTP.

    Requires a Gmail *app password* (Google Account -> Security -> 2-Step
    Verification -> App passwords), not the normal account password.
    """

    def __init__(
        self,
        *,
        user: str,
        app_password: str,
        from_name: str = "Prime Estate",
        smtp_factory: Callable[[], smtplib.SMTP] | None = None,
    ) -> None:
        self._user = user
        self._password = app_password
        self._from_name = from_name
        # Injectable for tests; the default is the real Gmail endpoint.
        self._smtp_factory = smtp_factory or (lambda: smtplib.SMTP("smtp.gmail.com", 587, timeout=15))

    def booking_confirmed(self, *, lead: Lead) -> None:
        """Send the confirmation email. Raises on transport failure.

        The caller (orchestrator) decides what a failure means; this class
        only knows how to send.
        """
        message = EmailMessage()
        message["Subject"] = "Your Prime Estate meeting is confirmed"
        message["From"] = f"{self._from_name} <{self._user}>"
        message["To"] = lead.email
        message.set_content(
            f"Dear {lead.full_name},\n\n"
            f"Your meeting with Prime Estate is confirmed for "
            f"{lead.meeting_date} at {lead.meeting_time} ({lead.timezone}).\n\n"
            "If you need to change or cancel it, just reply in the chat with "
            "'reschedule' or 'cancel'.\n\n"
            "Warm regards,\nAhmed Raza\nPrime Estate"
        )
        with self._smtp_factory() as smtp:
            smtp.starttls()
            smtp.login(self._user, self._password)
            smtp.send_message(message)
        logger.info("booking confirmation sent to %s", lead.email)


class NullNotifier:
    """No-op notifier used when Gmail is not configured."""

    def booking_confirmed(self, *, lead: Lead) -> None:
        """Do nothing; the booking flow works identically without email."""
        logger.info("email not configured; skipping confirmation for %s", lead.email)
