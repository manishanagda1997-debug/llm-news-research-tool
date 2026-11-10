"""
Tests for the event label check (no network).

The check reads a filled event sheet and lists cases for the labeller to look
at. It never changes a label. Findings:
    missing_event       - an empty event, or a name that is not lower case
                          words joined by "_" (for example "Iveco Offer")
    possible_typo       - two different names that are almost the same
                          ("iveco_offer_raised" and "iveco_offer_raise")
    possible_duplicate  - very similar headlines under different event names
    possible_wrong_merge- two articles under one event name that share no
                          content word in their title and description
    single_article      - an event with one article (optional review: some
                          developments only get one report)
"""

import pytest

from src import event_check as ec


def _row(n, event, title, description=""):
    return {"row": str(n), "event": event, "title": title, "description": description,
            "source": "S", "published": "2026-10-20T08:00:00Z", "content_first_200_chars": "",
            "url": f"https://x.com/{n}"}


def _kinds(findings):
    return sorted({f["kind"] for f in findings})


CLEAN = [
    _row(1, "iveco_offer_raised", "Tata Motors raises Iveco takeover offer to 14.40 euro per share",
         "Tata Motors raised its offer for Iveco."),
    _row(2, "iveco_offer_raised", "Tata Motors raises Iveco takeover offer price to 14.40 euro",
         "The Iveco offer is raised to 14.40 euro."),
    _row(3, "navratri_sales", "Car registrations plunge 22% this Navratri", "Registrations fall during the festival."),
    _row(4, "navratri_sales", "Car sales fall 22% this Navratri on a high base", "Navratri registrations fall."),
]


def test_a_clean_sheet_has_no_findings():
    assert ec.check_events(CLEAN) == []


@pytest.mark.parametrize("event", ["", "  ", "Iveco Offer", "iveco-offer", "iveco__offer", "_iveco"])
def test_a_missing_or_badly_written_event_name_is_found(event):
    findings = ec.check_events(CLEAN + [_row(5, event, "Jaguar opens a Delhi showroom", "JLR retail.")])
    missing = [f for f in findings if f["kind"] == "missing_event"]
    assert [f["rows"] for f in missing] == [["5"]]


def test_almost_identical_names_are_a_possible_typo():
    rows = CLEAN + [_row(5, "iveco_offer_raise", "Iveco investors weigh the higher Tata bid", "Investors weigh it.")]
    typo = [f for f in ec.check_events(rows) if f["kind"] == "possible_typo"]
    assert len(typo) == 1 and set(typo[0]["events"]) == {"iveco_offer_raised", "iveco_offer_raise"}


def test_different_stages_with_similar_names_are_not_typos():
    rows = [_row(1, "iveco_offer_raised", "Tata Motors raises Iveco offer", "Offer raised."),
            _row(2, "iveco_shareholders_approve", "Iveco shareholders approve the Tata bid", "Vote passes.")]
    assert "possible_typo" not in _kinds(ec.check_events(rows))


def test_very_similar_headlines_under_different_names_are_a_possible_duplicate():
    rows = CLEAN + [_row(5, "iveco_bid_up", "Tata Motors raises Iveco takeover offer to 14.40 euro per share - Reuters",
                         "Tata Motors raised its offer for Iveco.")]
    duplicate = [f for f in ec.check_events(rows) if f["kind"] == "possible_duplicate"]
    assert len(duplicate) == 1 and "5" in duplicate[0]["rows"]


def test_two_articles_sharing_no_words_under_one_name_are_a_possible_wrong_merge():
    rows = CLEAN + [_row(5, "navratri_sales", "Stellantis signs a vehicle platform agreement", "A new partnership.")]
    merge = [f for f in ec.check_events(rows) if f["kind"] == "possible_wrong_merge"]
    assert merge and all("5" in f["rows"] for f in merge)


def test_a_single_article_event_is_listed_for_optional_review_only():
    rows = CLEAN + [_row(5, "jlr_delhi_showroom", "Jaguar Land Rover opens a Delhi showroom", "JLR retail grows.")]
    single = [f for f in ec.check_events(rows) if f["kind"] == "single_article"]
    assert single == [{"kind": "single_article", "rows": ["5"], "events": ["jlr_delhi_showroom"],
                       "detail": "optional review: some developments get only one report"}]


def test_the_check_never_changes_the_rows():
    rows = CLEAN + [_row(5, "Iveco Offer", "Tata Motors raises Iveco takeover offer", "x")]
    before = [dict(r) for r in rows]
    ec.check_events(rows)
    assert rows == before


def test_findings_come_in_a_fixed_order_most_important_first():
    rows = CLEAN + [_row(5, "", "Jaguar showroom", "x"),
                    _row(6, "iveco_offer_raise", "Iveco investors weigh the higher Tata bid", "y")]
    kinds = [f["kind"] for f in ec.check_events(rows)]
    order = [ec.KINDS.index(k) for k in kinds]
    assert order == sorted(order)
    assert ec.KINDS == ["missing_event", "possible_typo", "possible_duplicate", "possible_wrong_merge",
                        "single_article"]


def test_spaces_around_a_name_are_ignored():
    rows = [_row(1, " iveco_offer_raised ", "Tata Motors raises Iveco offer", "Offer raised."),
            _row(2, "iveco_offer_raised", "Tata Motors raises Iveco offer price", "Offer raised.")]
    assert ec.check_events(rows) == []
