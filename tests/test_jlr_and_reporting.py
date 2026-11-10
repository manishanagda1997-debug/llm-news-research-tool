"""
Tests for Proposal 3 (agreed): Jaguar Land Rover names, and stricter reporting.

JLR: "Jaguar Land Rover" and "JLR" are verified names of a separate company
owned by Tata Motors Passenger Vehicles. "Jaguar" alone is ambiguous (the
animal, a fighter jet, a sports team), so it counts only when the same text
also contains a general car word such as carmaker, vehicle or EV.

Reporting: results are given for strict relevance (direct only) and broad
relevance (direct + indirect). Precision@k divides by the number of
articles actually shown when fewer than k exist.

Run with:  pytest
"""

import json
from pathlib import Path

import pytest

import src.evaluate as evaluate
import src.query as query
from src.preprocess import clean_articles

ALIAS_FILE = Path(__file__).resolve().parent.parent / "data" / "company_aliases.json"


@pytest.fixture
def aliases():
    return query.load_aliases(ALIAS_FILE)


def make_article(title, description="Some description.", content=None):
    return {
        "source": {"id": None, "name": "Example News"},
        "author": "Reporter",
        "title": title,
        "description": description,
        "url": f"https://example.com/{abs(hash(title))}",
        "urlToImage": None,
        "publishedAt": "2026-10-16T08:00:00Z",
        "content": content,
    }


# ---------- Alias file ----------

def test_jlr_is_a_separate_identity_owned_by_the_passenger_company():
    data = json.loads(ALIAS_FILE.read_text(encoding="utf-8"))
    jlr = data["identities"]["jaguar_land_rover"]
    assert jlr["parent"] == "tata_motors_pv"
    assert set(jlr["names"]) == {"Jaguar Land Rover", "JLR"}
    assert jlr["sources"]
    assert "Jaguar" not in jlr["names"]                    # never a plain alias
    context = {entry["name"]: entry for entry in jlr["names_needing_context"]}
    assert "Jaguar" in context
    assert "carmaker" in context["Jaguar"]["context_words"]
    assert "jaguar_land_rover" in data["search_groups"]["tata motors"]["identities"]
    for identity in ("tata_motors_pv", "tata_motors_cv"):
        assert "JLR" not in data["identities"][identity]["names"]


def test_plain_names_include_jlr_but_not_jaguar(aliases):
    names = query.alias_names("Tata Motors", aliases)
    assert {"jaguar land rover", "jlr"} <= set(names)
    assert "jaguar" not in names


# ---------- Matching ----------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Luxury carmaker Jaguar has launched its new all-electric vehicle.", True),
        ("Jaguar's reinvention as an electric-only luxury car maker has taken more than a decade", True),
        ("JLR sales rise in China", True),
        ("Jaguar Land Rover posts quarterly profit", True),
        ("A jaguar was spotted near the river in the national park", False),
        ("Jacksonville Jaguars beat the Titans on Sunday", False),
        ("The air force will retire its Jaguar fighter jets next year", False),
        ("Jaguar", False),
    ],
)
def test_jaguar_needs_car_context(aliases, text, expected):
    rule = query.parse_query("Tata Motors", aliases=aliases)
    assert query.matches(text, rule) is expected


def test_excluding_jaguar_blocks_jaguar_articles(aliases):
    rule = query.parse_query('"Tata Motors" -Jaguar', aliases=aliases)
    assert query.matches("Luxury carmaker Jaguar launched an electric vehicle", rule) is False
    assert query.matches("Tata Motors raises Iveco offer", rule) is True


def test_search_plan_includes_jlr_and_jaguar_with_context(aliases):
    plan = query.build_search_plan("Tata Motors", aliases=aliases)
    search = plan["search_query"]
    assert '"Jaguar Land Rover"' in search
    assert " OR JLR" in search
    assert "(Jaguar AND (" in search
    assert len(search) <= 500


def test_search_plan_leaves_out_jaguar_when_excluded(aliases):
    plan = query.build_search_plan('"Tata Motors" -Jaguar', aliases=aliases)
    assert "(Jaguar AND (" not in plan["search_query"]
    assert plan["search_query"].endswith("NOT Jaguar")


# ---------- Which company an article is about ----------

@pytest.mark.parametrize(
    "title, expected",
    [
        ("TMCV truck sales rise", ["tata_motors_cv"]),
        ("JLR posts quarterly profit", ["jaguar_land_rover"]),
        ("Luxury carmaker Jaguar launches an EV", ["jaguar_land_rover"]),
        ("TMPV and TMCV shares diverge", ["tata_motors_pv", "tata_motors_cv"]),
        ("Tata Motors raises Iveco offer", []),
    ],
)
def test_matched_identities(aliases, title, expected):
    assert query.matched_identities(title, "Tata Motors", aliases) == expected


def test_clean_articles_records_matched_identities(aliases):
    raw = [make_article("JLR posts quarterly profit"), make_article("Tata Motors raises Iveco offer")]
    articles, _ = clean_articles(raw, query="Tata Motors", aliases=aliases)
    assert [a["matched_identities"] for a in articles] == [["jaguar_land_rover"], []]


def test_matched_identities_empty_without_aliases():
    articles, _ = clean_articles([make_article("JLR posts quarterly profit")], query="JLR")
    assert articles[0]["matched_identities"] == []


# ---------- Reporting ----------

def test_strict_relevance_counts_only_direct():
    assert evaluate.is_relevant("direct", strict=True) is True
    assert evaluate.is_relevant("indirect", strict=True) is False
    assert evaluate.is_relevant("indirect") is True


def test_precision_at_k_divides_by_articles_shown():
    relevant = {"a": True, "b": False, "c": True}
    assert evaluate.precision_at_k(["a", "b", "c"], relevant, k=15) == pytest.approx(2 / 3)
