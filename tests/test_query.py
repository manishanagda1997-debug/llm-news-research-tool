"""
Tests for the search rules in src/query.py.

The rules (agreed before coding):
  - supported: "phrases", AND, OR, NOT, +word, -word (operators in capitals only)
  - a plain multi-word query is an exact phrase when exact_phrase=True,
    and "all words, any order" when exact_phrase=False
  - brackets, AND mixed with OR, or NOT mixed with OR are not supported:
    the local check is skipped instead of guessed
  - terms of any length are kept; matching is whole-word
  - the check looks only at the text we receive (title, description,
    first 200 characters), not the full article

Run with:  pytest
"""

import pytest

from src.query import QueryError, build_search_query, matches, parse_query


# ---------- What is sent to NewsAPI ----------

def test_single_word_is_sent_as_typed():
    assert build_search_query("Infosys") == "Infosys"


def test_exact_phrase_on_adds_quotes():
    assert build_search_query("Tata Motors", exact_phrase=True) == '"Tata Motors"'


def test_exact_phrase_off_joins_words_with_and():
    assert build_search_query("Tata Motors EV sales", exact_phrase=False) == "Tata AND Motors AND EV AND sales"


@pytest.mark.parametrize("query", ["Microsoft OR Google", "Tata Motors -Jaguar", '"Tata Motors" AND results'])
def test_advanced_query_is_sent_as_typed(query):
    assert build_search_query(query, exact_phrase=True) == query
    assert build_search_query(query, exact_phrase=False) == query


def test_hyphenated_name_is_not_an_exclusion():
    rule = parse_query("Rolls-Royce")
    assert rule.include == ["rolls-royce"]
    assert rule.exclude == []


def test_words_that_start_like_operators_are_normal_words():
    rule = parse_query("Oracle Andhra", exact_phrase=False)
    assert rule.include == ["oracle", "andhra"]
    assert rule.mode == "all"


def test_unclosed_quote_is_an_error():
    with pytest.raises(QueryError):
        build_search_query('Tata "Motors')


# ---------- Does an article's visible text match? ----------

@pytest.mark.parametrize(
    "query, exact_phrase, text, expected",
    [
        ("Microsoft OR Google", True, "Google cloud sales grow", True),
        ("Microsoft OR Google", True, "Apple results beat estimates", False),
        ("Tata Motors -Jaguar", True, "Tata Motors profit rises", True),
        ("Tata Motors -Jaguar", True, "Tata Motors launches new Jaguar EV", False),
        ("Tata Motors NOT Jaguar", True, "Tata Motors profit rises", True),
        ("Tata Motors NOT Jaguar", True, "Tata Motors launches new Jaguar EV", False),
        ("AI AND healthcare", True, "AI-powered healthcare tools launched", True),
        ("AI AND healthcare", True, "Innovation in healthcare, said the minister", False),
        ("EV OR hybrid", True, "New EV launch next month", True),
        ("EV OR hybrid", True, "Every car maker is worried", False),
        ("C++", True, "Demand for C++ developers rises", True),
        ("+Infosys results", True, "Infosys results beat estimates", True),
        ("+Infosys results", True, "Infosys signs a new deal", False),
        ("Tata Motors", True, "Tata Motors shares jump", True),
        ("Tata Motors", True, "Tata Steel and Ashok Motors", False),
        ("Tata Motors EV sales", False, "EV sales of Tata Motors rose", True),
        ("Tata Motors EV sales", False, "Tata Motors sales rose", False),
        ("X", True, "X Corp changes its policy", True),
    ],
)
def test_matching_rules(query, exact_phrase, text, expected):
    rule = parse_query(query, exact_phrase=exact_phrase)
    assert rule.supported
    assert matches(text, rule) is expected


# ---------- Queries the local check does not try to guess ----------

@pytest.mark.parametrize(
    "query",
    [
        "Tata AND (EV OR hybrid)",        # brackets
        "Tata AND EV OR hybrid",          # AND mixed with OR
        "Tata Motors NOT Jaguar OR EV",   # NOT mixed with OR
        "Tata AND",                       # operator with nothing after it
    ],
)
def test_unsupported_queries_are_marked_not_guessed(query):
    rule = parse_query(query)
    assert rule.supported is False
    assert rule.reason  # a short message the app can show
