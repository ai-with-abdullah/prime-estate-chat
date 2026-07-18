"""Tests for datastore duplicate detection, slot conflicts, and lookup."""

from __future__ import annotations

from prime_estate.domain.models import LeadStage
from prime_estate.tools.datastore import InMemoryLeadDatastore
from tests.conftest import make_lead


class TestDuplicateDetection:
    def test_duplicate_on_email_only(self, datastore: InMemoryLeadDatastore) -> None:
        datastore.save(lead=make_lead(email="a@b.co", phone="+920000000001"))
        check = datastore.check(lead=make_lead(email="a@b.co", phone="+929999999999"))
        assert check.is_duplicate

    def test_duplicate_on_phone_only(self, datastore: InMemoryLeadDatastore) -> None:
        datastore.save(lead=make_lead(email="a@b.co", phone="+920000000001"))
        check = datastore.check(lead=make_lead(email="other@b.co", phone="+920000000001"))
        assert check.is_duplicate

    def test_no_duplicate_when_neither_matches(self, datastore: InMemoryLeadDatastore) -> None:
        datastore.save(lead=make_lead(email="a@b.co", phone="+920000000001"))
        check = datastore.check(
            lead=make_lead(email="other@b.co", phone="+929999999999", meeting_time="15:00")
        )
        assert not check.is_duplicate

    def test_normalisation_case_space_and_plus(self, datastore: InMemoryLeadDatastore) -> None:
        datastore.save(lead=make_lead(email="A@B.Co", phone="+92 300 0000001"))
        check = datastore.check(lead=make_lead(email="a@b.co", phone="923000000001"))
        assert check.is_duplicate

    def test_header_offset_row_reporting(self, datastore: InMemoryLeadDatastore) -> None:
        datastore.save(lead=make_lead(email="first@b.co", phone="+920000000001"))
        datastore.save(lead=make_lead(email="second@b.co", phone="+920000000002"))
        check = datastore.check(lead=make_lead(email="second@b.co", phone="+929999999999"))
        # Second stored lead -> sheet row 3 (row 1 is the header).
        assert check.existing_row == 3


class TestSlotConflicts:
    def test_same_date_time_is_taken(self, datastore: InMemoryLeadDatastore) -> None:
        datastore.save(lead=make_lead(meeting_date="2030-01-07", meeting_time="11:00"))
        check = datastore.check(
            lead=make_lead(
                email="other@b.co",
                phone="+929999999999",
                meeting_date="2030-01-07",
                meeting_time="11:00",
            )
        )
        assert check.slot_taken

    def test_same_date_different_time_is_free(self, datastore: InMemoryLeadDatastore) -> None:
        datastore.save(lead=make_lead(meeting_date="2030-01-07", meeting_time="11:00"))
        check = datastore.check(
            lead=make_lead(
                email="other@b.co",
                phone="+929999999999",
                meeting_date="2030-01-07",
                meeting_time="12:00",
            )
        )
        assert not check.slot_taken

    def test_cancelled_booking_frees_its_slot(self, datastore: InMemoryLeadDatastore) -> None:
        row = datastore.save(lead=make_lead(meeting_date="2030-01-07", meeting_time="11:00"))
        cancelled = make_lead(
            meeting_date="2030-01-07", meeting_time="11:00", stage=LeadStage.CANCELLED
        )
        datastore.update(row=row, lead=cancelled)
        check = datastore.check(
            lead=make_lead(
                email="other@b.co",
                phone="+929999999999",
                meeting_date="2030-01-07",
                meeting_time="11:00",
            )
        )
        assert not check.slot_taken


class TestLookup:
    def test_find_by_contact_requires_both_factors(self, datastore: InMemoryLeadDatastore) -> None:
        datastore.save(lead=make_lead(email="a@b.co", phone="+920000000001"))
        assert datastore.find_by_contact(email="a@b.co", phone="+920000000001") is not None
        # Email alone, phone alone, or a mismatched pair must all fail.
        assert datastore.find_by_contact(email="a@b.co", phone="+929999999999") is None
        assert datastore.find_by_contact(email="wrong@b.co", phone="+920000000001") is None
        assert datastore.find_by_contact(email="a@b.co", phone="") is None

    def test_find_by_contact_normalises(self, datastore: InMemoryLeadDatastore) -> None:
        datastore.save(lead=make_lead(email="A@B.Co", phone="+92 300 0000001"))
        found = datastore.find_by_contact(email="a@b.co", phone="923000000001")
        assert found is not None
        row, lead = found
        assert row == 2
        assert lead.email == "A@B.Co"

    def test_find_by_email_or_name(self, datastore: InMemoryLeadDatastore) -> None:
        datastore.save(lead=make_lead(email="a@b.co", full_name="Ayesha Khan"))
        assert datastore.find_by_email_or_name(query="a@b.co") is not None
        assert datastore.find_by_email_or_name(query="ayesha khan") is not None
        assert datastore.find_by_email_or_name(query="nobody") is None
        assert datastore.find_by_email_or_name(query="") is None
