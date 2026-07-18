"""Unit tests for the ported field validators."""

from __future__ import annotations

import pytest

from prime_estate.validation.validators import (
    validate_date,
    validate_email,
    validate_lead_fields,
    validate_phone,
    validate_time,
)


class TestEmail:
    @pytest.mark.parametrize(
        "value",
        ["a@b.co", "first.last+tag@sub.domain.org", "USER_9%x@ex-ample.io"],
    )
    def test_valid(self, value: str) -> None:
        assert validate_email(value)

    @pytest.mark.parametrize(
        "value",
        ["", "plainaddress", "a@b", "a@b.c", "@no-local.com", "spaces in@mail.com"],
    )
    def test_invalid(self, value: str) -> None:
        assert not validate_email(value)


class TestPhone:
    @pytest.mark.parametrize("value", ["+9230012345", "+12025550123", "+123456789012345"])
    def test_valid(self, value: str) -> None:
        assert validate_phone(value)

    @pytest.mark.parametrize(
        "value",
        ["", "03001234567", "+123456", "+1234567890123456", "+92 300 1234567", "0092300"],
    )
    def test_invalid(self, value: str) -> None:
        assert not validate_phone(value)


class TestDateAndTime:
    def test_date_format_only(self) -> None:
        assert validate_date("2030-01-07")
        assert not validate_date("07-01-2030")
        assert not validate_date("2030/01/07")
        assert not validate_date("tomorrow")

    def test_time_24h(self) -> None:
        assert validate_time("09:00")
        assert validate_time("23:59")
        assert not validate_time("24:00")
        assert not validate_time("9:00")
        assert not validate_time("09:60")


class TestLeadFields:
    def test_all_valid(self) -> None:
        result = validate_lead_fields(
            email="a@b.co",
            phone="+9230012345",
            meeting_date="2030-01-07",
            meeting_time="11:00",
        )
        assert result.ok
        assert result.errors == []
        assert result.as_client_prompt() is None

    def test_collects_every_error_in_one_pass(self) -> None:
        result = validate_lead_fields(
            email="nope", phone="12345", meeting_date="soon", meeting_time="9am"
        )
        assert not result.ok
        assert len(result.errors) == 4

    def test_client_prompt_names_the_failures(self) -> None:
        result = validate_lead_fields(
            email="nope",
            phone="+9230012345",
            meeting_date="2030-01-07",
            meeting_time="11:00",
        )
        prompt = result.as_client_prompt()
        assert prompt is not None
        assert "email" in prompt
