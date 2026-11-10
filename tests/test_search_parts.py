"""
Tests, group 1: separate searches, saved raw results, merging.

Search A: the company name only                 "Tata Motors"
Search B: the subsidiaries (Jaguar Land Rover)  ("Jaguar Land Rover" OR JLR OR (Jaguar AND (car ...)))
Search C: the company and its own names         ("Tata Motors" OR TMPV OR ... OR TMCV)

Each search has its own article slots, so one search cannot push the
articles of another out of the results.
"""

import dataclasses
from pathlib import Path

import pytest

import src.news_fetcher as news_fetcher
import src.query as query
from src import multi_search as ms

ALIAS_FILE = Path(__file__).resolve().parent.parent / "data" / "company_aliases.json"
JAGUAR_WITH_CONTEXT = ('(Jaguar AND (car OR cars OR carmaker OR carmakers OR "car maker" OR automaker OR '
                       'automakers OR vehicle OR vehicles OR EV OR EVs OR electric OR SUV OR sedan OR "Land Rover"))')
OWN_NAMES = ('("Tata Motors" OR "Tata Motors Passenger Vehicles" OR TMPV OR '
             '"Tata Motors Commercial Vehicles" OR TMCV)')
SUBSIDIARIES = '("Jaguar Land Rover" OR JLR OR ' + JAGUAR_WITH_CONTEXT + ')'


@pytest.fixture
def aliases():
    return query.load_aliases(ALIAS_FILE)


# ---------- search parts ----------

def test_own_names_part_has_the_company_names_only(aliases):
    plan = query.build_search_plan("Tata Motors", aliases=aliases, part="own")
    assert plan["search_query"] == OWN_NAMES
    assert "JLR" not in plan["search_query"] and "Jaguar" not in plan["search_query"]


def test_subsidiaries_part_has_the_subsidiary_names_only(aliases):
    plan = query.build_search_plan("Tata Motors", aliases=aliases, part="subsidiaries")
    assert plan["search_query"] == SUBSIDIARIES
    assert "Tata Motors" not in plan["search_query"]


def test_subsidiaries_part_keeps_excluded_words(aliases):
    plan = query.build_search_plan('"Tata Motors" -Jaguar', aliases=aliases, part="subsidiaries")
    assert plan["search_query"] == '("Jaguar Land Rover" OR JLR) NOT Jaguar'


def test_all_part_is_unchanged(aliases):
    whole = query.build_search_plan("Tata Motors", aliases=aliases)
    assert whole == query.build_search_plan("Tata Motors", aliases=aliases, part="all")
    assert "TMPV" in whole["search_query"] and "JLR" in whole["search_query"]


def test_subsidiaries_part_is_empty_when_the_company_has_none():
    plan = query.build_search_plan("Infosys", aliases=None, part="subsidiaries")
    assert plan["search_query"] == ""
    assert plan["note"]


def test_unknown_part_is_refused(aliases):
    with pytest.raises(query.QueryError):
        query.build_search_plan("Tata Motors", aliases=aliases, part="everything")


def test_fetch_news_sends_the_chosen_part_and_records_it(monkeypatch, aliases):
    sent = {}

    def fake_get_everything(self, **kwargs):
        sent.update(kwargs)
        return {"status": "ok", "totalResults": 0, "articles": []}

    monkeypatch.setattr(news_fetcher, "settings",
                        dataclasses.replace(news_fetcher.settings, newsapi_key="test-key"))
    monkeypatch.setattr(news_fetcher.NewsApiClient, "get_everything", fake_get_everything)
    response = news_fetcher.fetch_news("Tata Motors", aliases=aliases, part="subsidiaries", save=False)
    assert sent["q"] == SUBSIDIARIES
    assert response["request"]["search_part"] == "subsidiaries"


# ---------- running the searches ----------

def _response(urls, from_date="2026-10-17", page_size=30):
    return {"status": "ok", "totalResults": len(urls),
            "articles": [{"url": url, "title": f"Title {url}"} for url in urls],
            "request": {"from_date": from_date, "page_size": page_size}}


class FakeFetch:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, company, **options):
        self.calls.append(options)
        return self.responses[options.get("part", "all") if options.get("aliases") else "none"]


def test_the_three_searches_use_the_agreed_parts_and_identical_settings(aliases):
    fake = FakeFetch({"none": _response(["a"]), "subsidiaries": _response(["b"]), "own": _response(["c"])})
    responses = ms.run_searches("Tata Motors", aliases, fetch=fake)
    assert set(responses) == {"A", "B", "C"}
    assert [call.get("part", "all") if call["aliases"] else "none" for call in fake.calls] == ["none", "subsidiaries", "own"]
    for key in ("page_size", "days_back", "sort_by", "exact_phrase", "title_only"):
        assert len({call[key] for call in fake.calls}) == 1


def test_searches_stop_when_their_dates_differ(aliases):
    fake = FakeFetch({"none": _response(["a"], "2026-10-17"), "subsidiaries": _response(["b"], "2026-10-18"),
                      "own": _response(["c"], "2026-10-17")})
    with pytest.raises(ms.MultiSearchError):
        ms.run_searches("Tata Motors", aliases, fetch=fake)


def test_search_b_is_skipped_without_a_request_when_there_are_no_subsidiaries():
    fake = FakeFetch({"none": _response(["a"])})
    responses = ms.run_searches("Infosys", None, names=("A", "B"), fetch=fake)
    assert responses["B"] is None
    assert len(fake.calls) == 1


# ---------- merging ----------

def test_merge_keeps_each_article_once_and_records_which_searches_found_it():
    responses = {"A": _response(["https://x.com/1", "https://x.com/2"]),
                 "B": _response(["https://www.x.com/2/", "https://x.com/3"])}
    merged = ms.merge(responses, ["A", "B"])
    assert [a["url"] for a in merged] == ["https://x.com/1", "https://x.com/2", "https://x.com/3"]
    assert [a["found_by"] for a in merged] == [["A"], ["A", "B"], ["B"]]


def test_merge_keeps_the_order_of_the_searches_given():
    responses = {"A": _response(["https://x.com/1"]), "B": _response(["https://x.com/2"])}
    assert [a["url"] for a in ms.merge(responses, ["B", "A"])] == ["https://x.com/2", "https://x.com/1"]


def test_merge_ignores_a_skipped_search():
    responses = {"A": _response(["https://x.com/1"]), "B": None}
    assert len(ms.merge(responses, ["A", "B"])) == 1


def test_merge_does_not_change_the_raw_responses():
    responses = {"A": _response(["https://x.com/1"])}
    ms.merge(responses, ["A"])
    assert "found_by" not in responses["A"]["articles"][0]


def test_budget_counts_requests_and_article_slots():
    responses = {"A": _response(["a"]), "B": _response(["b"]), "C": None}
    assert ms.budget(responses, ["A", "B", "C"]) == {"requests": 2, "slots": 60}
