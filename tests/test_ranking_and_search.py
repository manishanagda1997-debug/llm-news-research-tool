"""
Tests for Proposal 1b (agreed): ranking by where the company is mentioned,
and adding verified aliases to the search sent to NewsAPI.

Ranking: title match first, then description, then content only. Inside a
level, NewsAPI's order is kept. Ranking happens after filtering and before
the article limit, so the limit keeps the best-placed articles.

Search plan: aliases are added to the NewsAPI search only when the query is
one name or phrase, with optional excluded words. Excluded words apply to the
whole group: ("Tata Motors" OR TMPV OR TMCV) NOT Jaguar. Anything more
complex is sent unchanged and the reason is recorded. The local relevance
check works from the user's own query, so brackets added by the search plan
never switch it off.

Run with:  pytest
"""

from pathlib import Path

import pytest

import src.news_fetcher as news_fetcher
import src.query as query
from src.preprocess import clean_articles

ALIAS_FILE = Path(__file__).resolve().parent.parent / "data" / "company_aliases.json"
JAGUAR_WITH_CONTEXT = ('(Jaguar AND (car OR cars OR carmaker OR carmakers OR "car maker" OR automaker OR '
                       'automakers OR vehicle OR vehicles OR EV OR EVs OR electric OR SUV OR sedan OR "Land Rover"))')
TATA_GROUP = ('("Tata Motors" OR "Tata Motors Passenger Vehicles" OR TMPV OR '
              '"Tata Motors Commercial Vehicles" OR TMCV OR "Jaguar Land Rover" OR JLR OR '
              + JAGUAR_WITH_CONTEXT + ')')
TATA_GROUP_WITHOUT_JAGUAR = ('("Tata Motors" OR "Tata Motors Passenger Vehicles" OR TMPV OR '
                             '"Tata Motors Commercial Vehicles" OR TMCV OR "Jaguar Land Rover" OR JLR)')


@pytest.fixture
def aliases():
    return query.load_aliases(ALIAS_FILE)


def make_article(title, description="Some description.", content=None, url=None):
    return {
        "source": {"id": None, "name": "Example News"},
        "author": "Reporter",
        "title": title,
        "description": description,
        "url": url or f"https://example.com/{abs(hash(title))}",
        "urlToImage": None,
        "publishedAt": "2026-10-16T08:00:00Z",
        "content": content,
    }


CONTENT_ONLY = make_article("Car sales rise in October", "Festive demand lifted sales.",
                            "Makers such as Maruti and Tata Motors reported more bookings.")
DESCRIPTION_ONLY = make_article("Auto stocks rally", "Tata Motors and Mahindra led the gains on Monday.")
TITLE_ONE = make_article("Tata Motors raises Iveco offer")
TITLE_TWO = make_article("Tata Motors pips Mahindra in car sales")


# ---------- Ranking ----------

def test_mention_ranking_puts_title_then_description_then_content():
    raw = [CONTENT_ONLY, DESCRIPTION_ONLY, TITLE_ONE]
    articles, _ = clean_articles(raw, query="Tata Motors", ranking="mention")
    assert [a["title"] for a in articles] == [TITLE_ONE["title"], DESCRIPTION_ONLY["title"], CONTENT_ONLY["title"]]
    assert [a["mention_level"] for a in articles] == ["title", "description", "content"]


def test_numbers_follow_the_new_order():
    articles, _ = clean_articles([CONTENT_ONLY, TITLE_ONE], query="Tata Motors", ranking="mention")
    assert [a["id"] for a in articles] == [1, 2]
    assert articles[0]["title"] == TITLE_ONE["title"]


def test_newsapi_order_is_kept_inside_a_level():
    raw = [TITLE_TWO, CONTENT_ONLY, TITLE_ONE]
    articles, _ = clean_articles(raw, query="Tata Motors", ranking="mention")
    assert [a["title"] for a in articles[:2]] == [TITLE_TWO["title"], TITLE_ONE["title"]]


def test_limit_is_applied_after_ranking():
    raw = [CONTENT_ONLY, DESCRIPTION_ONLY, TITLE_ONE]
    articles, stats = clean_articles(raw, max_articles=1, query="Tata Motors", ranking="mention")
    assert [a["title"] for a in articles] == [TITLE_ONE["title"]]
    assert stats["over_limit"] == 2


def test_default_ranking_keeps_newsapi_order():
    raw = [CONTENT_ONLY, DESCRIPTION_ONLY, TITLE_ONE]
    articles, _ = clean_articles(raw, query="Tata Motors")
    assert [a["title"] for a in articles] == [CONTENT_ONLY["title"], DESCRIPTION_ONLY["title"], TITLE_ONE["title"]]


def test_alias_in_title_counts_as_title_mention(aliases):
    raw = [CONTENT_ONLY, make_article("TMPV shares hit 52-week low")]
    articles, _ = clean_articles(raw, query="Tata Motors", aliases=aliases, ranking="mention")
    assert articles[0]["title"] == "TMPV shares hit 52-week low"
    assert articles[0]["mention_level"] == "title"


def test_no_query_means_no_mention_level_and_same_order():
    raw = [CONTENT_ONLY, TITLE_ONE]
    articles, _ = clean_articles(raw, ranking="mention")
    assert [a["title"] for a in articles] == [CONTENT_ONLY["title"], TITLE_ONE["title"]]
    assert all(a["mention_level"] is None for a in articles)


def test_unknown_ranking_name_is_an_error():
    with pytest.raises(ValueError):
        clean_articles([TITLE_ONE], query="Tata Motors", ranking="best")


def test_fewer_articles_than_the_limit_is_fine():
    articles, stats = clean_articles([TITLE_ONE], max_articles=20, query="Tata Motors", ranking="mention")
    assert len(articles) == 1
    assert stats["over_limit"] == 0


# ---------- Search plan sent to NewsAPI ----------

def test_plain_company_search_gets_alias_group(aliases):
    plan = query.build_search_plan("Tata Motors", aliases=aliases)
    assert plan["search_query"] == TATA_GROUP
    assert plan["aliases_added"] is True
    assert plan["original_query"] == "Tata Motors"


def test_exclusion_applies_to_the_whole_group(aliases):
    plan = query.build_search_plan('"Tata Motors" -Jaguar', aliases=aliases)
    assert plan["search_query"] == TATA_GROUP_WITHOUT_JAGUAR + " NOT Jaguar"


def test_not_keyword_exclusion_also_applies_to_the_group(aliases):
    plan = query.build_search_plan('"Tata Motors" NOT Jaguar', aliases=aliases)
    assert plan["search_query"] == TATA_GROUP_WITHOUT_JAGUAR + " NOT Jaguar"


def test_excluded_alias_is_left_out_of_the_group(aliases):
    plan = query.build_search_plan('"Tata Motors" -TMPV', aliases=aliases)
    assert "OR TMPV" not in plan["search_query"]
    assert plan["search_query"].endswith("NOT TMPV")


@pytest.mark.parametrize(
    "user_query, exact_phrase, expected",
    [
        ("Tata Motors", False, "Tata AND Motors"),        # all words: not one name, no aliases
        ("Microsoft OR Google", True, "Microsoft OR Google"),
        ("Tata AND (EV OR hybrid)", True, "Tata AND (EV OR hybrid)"),
        ("Infosys", True, "Infosys"),                     # no alias group
    ],
)
def test_other_queries_are_sent_unchanged(aliases, user_query, exact_phrase, expected):
    plan = query.build_search_plan(user_query, exact_phrase=exact_phrase, aliases=aliases)
    assert plan["search_query"] == expected
    assert plan["aliases_added"] is False


def test_no_alias_file_means_no_change():
    plan = query.build_search_plan("Tata Motors")
    assert plan["search_query"] == '"Tata Motors"'
    assert plan["aliases_added"] is False


def test_too_long_search_is_not_expanded_and_says_why(aliases):
    long_exclusions = " ".join(f"-word{i}" for i in range(10, 65))   # 55 excluded words
    user_query = f'"Tata Motors" {long_exclusions}'
    assert len(user_query) <= 500          # the user's own search is allowed ...
    plan = query.build_search_plan(user_query, aliases=aliases)      # ... the expanded one would not be
    assert plan["aliases_added"] is False
    assert "500" in plan["note"]
    assert plan["search_query"] == user_query


def test_local_check_still_runs_when_search_has_brackets(aliases):
    raw = [TITLE_ONE, make_article("Tata Motors unveils new Jaguar EV")]
    articles, stats = clean_articles(raw, query='"Tata Motors" -Jaguar', aliases=aliases)
    assert stats["relevance_check"] == "applied"
    assert [a["title"] for a in articles] == [TITLE_ONE["title"]]


# ---------- fetch_news uses the plan and records it ----------

def test_fetch_news_sends_expanded_search_and_records_both(monkeypatch, aliases):
    import dataclasses

    sent = {}

    def fake_get_everything(self, **kwargs):
        sent.update(kwargs)
        return {"status": "ok", "totalResults": 0, "articles": []}

    monkeypatch.setattr(news_fetcher, "settings",
                        dataclasses.replace(news_fetcher.settings, newsapi_key="test-key"))
    monkeypatch.setattr(news_fetcher.NewsApiClient, "get_everything", fake_get_everything)

    response = news_fetcher.fetch_news('"Tata Motors" -Jaguar', aliases=aliases, save=False)
    assert sent["q"] == TATA_GROUP_WITHOUT_JAGUAR + " NOT Jaguar"
    assert response["request"]["query"] == '"Tata Motors" -Jaguar'
    assert response["request"]["search_query"] == TATA_GROUP_WITHOUT_JAGUAR + " NOT Jaguar"
    assert response["request"]["aliases_added"] is True
