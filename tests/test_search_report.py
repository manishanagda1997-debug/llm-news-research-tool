"""
Tests, group 3: reporting per search configuration, slot use and new-only evaluation.

Configurations: A, A+B, C, C+B (after merging and removing exact duplicates).
Every configuration reports its budget (requests and article slots), so a gain
from a bigger budget is not mistaken for a gain from better search terms.
"""

import pytest

from src import search_report as sr


def _article(n, title=None, url=None):
    return {"url": url or f"https://x.com/{n}", "title": title or f"Story number {n} about something"}


def _response(numbers, page_size=30):
    return {"articles": [_article(n) for n in numbers], "request": {"page_size": page_size}}


RESPONSES = {
    "A": _response([1, 2, 3]),          # 1, 2 relevant
    "B": _response([3, 4, 5]),          # 4 relevant, 5 a listing
    "C": _response([1, 2, 6]),          # 6 relevant
}
LABELS = {f"https://x.com/{n}": label for n, label in
          [(1, "direct"), (2, "indirect"), (3, "not"), (4, "direct"), (5, "not"), (6, "indirect")]}


def _is_listing(article):
    return article["url"] == "https://x.com/5"


# ---------- configurations ----------

def test_configurations_are_the_agreed_four():
    assert sr.CONFIGURATIONS == {"A": ["A"], "A+B": ["A", "B"], "C": ["C"], "C+B": ["C", "B"]}


def test_configuration_scores_count_budget_and_relevant_articles():
    result = sr.configuration_scores(RESPONSES, LABELS, is_listing=_is_listing)
    assert result["pool_relevant"] == 4                      # 1, 2, 4, 6 found by any search
    a, ab = result["configurations"]["A"], result["configurations"]["A+B"]
    assert (a["requests"], a["slots"], a["returned"], a["relevant"]) == (1, 30, 3, 2)
    assert (ab["requests"], ab["slots"], ab["returned"], ab["relevant"]) == (2, 60, 5, 3)
    assert ab["pool_recall"] == pytest.approx(3 / 4)
    assert ab["precision"] == pytest.approx(3 / 5)
    assert ab["relevant_per_request"] == pytest.approx(3 / 2)


def test_configuration_scores_count_slots_used_by_listings():
    result = sr.configuration_scores(RESPONSES, LABELS, is_listing=_is_listing)
    assert result["configurations"]["A+B"]["listings"] == 1
    assert result["configurations"]["A"]["listings"] == 0


def test_configuration_scores_show_what_each_search_adds():
    result = sr.configuration_scores(RESPONSES, LABELS, is_listing=_is_listing)
    assert result["added_by"]["A+B"] == {"B": ["https://x.com/4"]}   # relevant articles only B found
    assert result["added_by"]["C+B"] == {"B": ["https://x.com/4"]}


def test_configuration_scores_support_strict_relevance():
    result = sr.configuration_scores(RESPONSES, LABELS, is_listing=_is_listing, strict=True)
    assert result["pool_relevant"] == 2                      # 1 and 4 are direct
    assert result["configurations"]["C"]["relevant"] == 1


def test_configuration_scores_stop_when_an_article_has_no_label():
    labels = dict(LABELS)
    del labels["https://x.com/6"]
    with pytest.raises(sr.SearchReportError):
        sr.configuration_scores(RESPONSES, labels, is_listing=_is_listing)


def test_a_skipped_search_costs_nothing():
    responses = dict(RESPONSES, B=None)
    result = sr.configuration_scores(responses, LABELS, is_listing=_is_listing)
    assert result["configurations"]["A+B"]["requests"] == 1


# ---------- listing checks ----------

def test_listing_check_scores_report_each_check_overlap_and_mistakes():
    articles = [_article(n) for n in range(1, 5)]
    signals = {"https://x.com/1": {"pattern": True, "site": True},
               "https://x.com/2": {"pattern": True, "site": False},
               "https://x.com/3": {"pattern": False, "site": True},
               "https://x.com/4": {"pattern": False, "site": False}}
    labels = {"https://x.com/1": "not", "https://x.com/2": "direct",
              "https://x.com/3": "not", "https://x.com/4": "not"}
    result = sr.listing_check_scores(articles, labels, signals=lambda a: signals[a["url"]])
    assert result["pattern"] == {"rejected": 2, "relevant_rejected": 1}
    assert result["site"] == {"rejected": 2, "relevant_rejected": 0}
    assert result["both"] == 1
    assert result["either"] == {"rejected": 3, "relevant_rejected": 1}


# ---------- new-only evaluation ----------

def test_unseen_drops_articles_seen_under_another_web_address():
    seen = [_article(1, url="https://x.com/story")]
    new = [_article(1, url="https://www.x.com/story/?utm_source=feed"), _article(2)]
    assert [a["url"] for a in sr.unseen(new, seen)] == ["https://x.com/2"]


def test_unseen_drops_the_same_title_from_another_publisher():
    seen = [_article(1, title="Tata Motors raises Iveco takeover offer to 14.40 euro per share")]
    new = [_article(2, title="Tata Motors raises Iveco takeover offer to 14.40 euro per share")]
    assert sr.unseen(new, seen) == []


def test_unseen_drops_near_duplicate_titles():
    seen = [_article(1, title="Tata Motors raises Iveco takeover offer to 14.40 euro per share")]
    new = [_article(2, title="Tata Motors raises Iveco takeover offer to 14.4 euro per share")]
    assert sr.unseen(new, seen) == []


def test_unseen_keeps_genuinely_new_articles():
    seen = [_article(1, title="Tata Motors raises Iveco takeover offer to 14.40 euro per share")]
    new = [_article(2, title="Jaguar Land Rover opens new plant in Slovakia for electric models")]
    assert len(sr.unseen(new, seen)) == 1
