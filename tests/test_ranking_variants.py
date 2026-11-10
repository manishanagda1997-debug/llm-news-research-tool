"""Tests for the round 1 ranking variants (src/ranking_variants.py). All data is made up."""

import json

import pytest

from src import ranking_variants as rv
from src.query import QueryRule


def _c(key, level, text=""):
    return {"key": key, "title": f"Story {key}", "description": f"About {key}", "mention_level": level,
            "check": None, "text": text}


RULE = QueryRule(include=["tata motors", "tata motors passenger vehicles", "tmpv"], mode="any",
                 include_with_context=[("jaguar", ["car", "cars"])])


# ---------- counting ----------

def test_names_are_counted_as_whole_words_without_case():
    assert rv.name_count("Tata Motors said. TATA MOTORS again; tata motorsport is not it.", RULE) == 2


def test_the_longest_name_counts_once_without_overlap():
    assert rv.name_count("Tata Motors Passenger Vehicles and TMPV", RULE) == 2


def test_a_context_name_counts_only_with_its_context_word():
    assert rv.name_count("Jaguar and Jaguar", RULE) == 0
    assert rv.name_count("Jaguar sells a new car; Jaguar grows", RULE) == 2


def test_an_excluded_term_gives_zero():
    rule = QueryRule(include=["tata motors"], exclude=["tata steel"], mode="all")
    assert rv.name_count("Tata Motors and Tata Steel", rule) == 0


def test_the_full_text_is_counted_and_the_description_without_it():
    assert rv.counted_text({"text": "full", "description": "short"}) == "full"
    assert rv.counted_text({"text": "", "description": "short"}) == "short"


# ---------- the variants ----------

POOL = [_c("a", "content", "Tata Motors. Tata Motors. Tata Motors."), _c("b", "title"), _c("c", "description"),
        _c("d", "title"), _c("e", "content", "Tata Motors once")]


def test_o1_is_the_frozen_order():
    assert [c["key"] for c in rv.order_variant(POOL, "O1", RULE)] == ["b", "d", "c", "a", "e"]


def test_the_retrieval_variants_keep_the_first_surviving_candidates_on_top():
    assert [c["key"] for c in rv.order_variant(POOL, "O1R1", RULE)] == ["a", "b", "d", "c", "e"]
    assert [c["key"] for c in rv.order_variant(POOL, "O1R2", RULE)] == ["a", "b", "d", "c", "e"]
    survivors = POOL[1:]                                      # the first raw article removed by Stage 1
    assert [c["key"] for c in rv.order_variant(survivors, "O1R1", RULE)] == ["b", "d", "c", "e"]


def test_a_mention_variant_promotes_an_article_that_names_the_company_often():
    assert [c["key"] for c in rv.order_variant(POOL, "O1M3", RULE)] == ["a", "b", "d", "c", "e"]
    assert [c["key"] for c in rv.order_variant(POOL, "O1M5", RULE)] == ["b", "d", "c", "a", "e"]


def test_every_variant_keeps_every_candidate():
    for name in rv.VARIANTS:
        assert sorted(c["key"] for c in rv.order_variant(POOL, name, RULE)) == sorted(c["key"] for c in POOL)


def test_an_unknown_variant_is_refused():
    with pytest.raises(ValueError):
        rv.order_variant(POOL, "O9", RULE)


# ---------- one day and the choice ----------

def _day(n_title=12, body_relevant_text="Tata Motors. Tata Motors. Tata Motors. Tata Motors. Tata Motors."):
    """One body-only direct article first in the retrieval order, then title articles that are not relevant."""
    cands = [_c("x", "content", body_relevant_text)] + [_c(f"t{i}", "title") for i in range(n_title)]
    labels = {"x": "direct", **{f"t{i}": "not" for i in range(n_title)}}
    events = {"x": "big_news"}
    return cands, labels, events


def test_a_variant_that_rescues_a_direct_event_passes_the_day():
    cands, labels, events = _day()
    day = rv.day_result(cands, labels, events, RULE)
    assert day["checks"]["O1R1"]["passed"] and day["checks"]["O1M5"]["passed"]
    assert day["measures"]["O1"]["10"]["direct_events_covered"] == []
    assert day["measures"]["O1R1"]["10"]["direct_events_covered"] == ["big_news"]


def test_a_variant_that_loses_a_direct_event_fails_the_day():
    cands = [_c("n", "content", "x")] + [_c("d1", "title")] + [_c(f"t{i}", "title") for i in range(9)]
    labels = {"n": "not", "d1": "direct", **{f"t{i}": "not" for i in range(9)}}
    events = {"d1": "results"}
    cands = [cands[0]] + cands[2:] + [cands[1]]               # the direct article is 11th in the retrieval order
    day = rv.day_result(cands, labels, events, RULE)
    assert day["measures"]["O1"]["10"]["direct_events_covered"] == ["results"]
    assert not day["checks"]["O1R1"]["passed"] and day["checks"]["O1R1"]["events"]["direct_lost"] == ["results"]


def test_a_tie_is_not_lower():
    cands = [_c(f"t{i}", "title") for i in range(12)]
    labels = {f"t{i}": "not" for i in range(12)}
    day = rv.day_result(cands, labels, {}, RULE)
    assert all(day["checks"][name]["passed"] for name in rv.VARIANTS)


def test_a_variant_must_pass_every_day_and_the_ranking_uses_pooled_counts():
    good = rv.day_result(*_day(), RULE)
    flat = rv.day_result([_c(f"t{i}", "title") for i in range(12)], {f"t{i}": "not" for i in range(12)}, {}, RULE)
    choice = rv.choose({"d1": good, "d2": flat, "d3": flat})
    assert choice["passing"] == list(rv.VARIANTS)
    assert choice["chosen"] == "O1R1"                          # equal counts: the simplest wins
    assert choice["pooled"]["O1R1"]["direct_events_covered"] == 1 and choice["pooled"]["O1"]["direct_events_covered"] == 0


def test_with_no_passing_variant_the_baseline_stays():
    cands = [_c("n", "content", "x")] + [_c(f"t{i}", "title") for i in range(9)] + [_c("d1", "title")]
    labels = {"n": "not", "d1": "direct", **{f"t{i}": "not" for i in range(9)}}
    day = rv.day_result(cands, labels, {"d1": "results"}, RULE)
    assert not day["checks"]["O1R1"]["passed"]                 # the retrieval head pushes the direct article out
    blocked = dict(day, checks={n: {**c, "passed": False} for n, c in day["checks"].items()})
    assert rv.choose({"d1": blocked})["chosen"] == "O1+S1"


def test_k_5_and_15_never_decide():
    assert rv.RULE["reported_only"] == "k = 5 and k = 15" and rv.PRIMARY_K == 10


# ---------- the plan and the run guards ----------

FP = {"labels": "l", "events": "e", "aliases": "a", "raw_files": {"r": "1"}, "candidates": "c", "code": "k"}


def test_the_plan_records_the_rule_the_variants_and_every_fingerprint():
    plan = rv.plan_record(FP, {"2026-10-14": ["r"]}, "abc")
    assert set(plan["variants"]) == set(rv.VARIANTS) and plan["rule"] == rv.RULE
    assert plan["fingerprints"] == FP and plan["tag"] == rv.PLAN_TAG and plan["source_commit"] == "abc"
    assert "src/ranking.py" in plan["code_files"] and "src/ranking_variants.py" in plan["code_files"]


def test_every_piece_of_code_that_affects_the_result_is_fingerprinted():
    for name in ("src/ranking_variants.py", "src/ranking.py", "src/ranking_eval.py", "src/preprocess.py",
                 "src/query.py", "src/fulltext.py", "src/fulltext_eval.py", "src/evaluate.py", "src/summary_run.py"):
        assert name in rv.CODE_FILES


@pytest.mark.parametrize("name", ["labels", "events", "aliases", "raw_files", "candidates", "code"])
def test_the_run_is_refused_when_anything_changed_since_the_plan(name):
    plan = {"fingerprints": FP}
    assert rv.run_problems("p", "p", plan, FP, False) == []
    assert rv.run_problems("p", "p", plan, dict(FP, **{name: "other"}), False) == [f"changed since the plan: {name}"]


def test_the_run_is_refused_without_the_tagged_plan_or_a_second_time():
    plan = {"fingerprints": FP}
    assert rv.run_problems("p", None, plan, FP, False)
    assert rv.run_problems("p", "older", plan, FP, False)
    assert rv.run_problems("p", "p", plan, FP, True) == [f"{rv.RECORD_FILE} already exists: the run is made once"]


def test_the_candidates_fingerprint_changes_with_any_text_or_field():
    days = {"d": [_c("a", "title", "text")]}
    first = rv.candidates_fingerprint(days)
    assert rv.candidates_fingerprint({"d": [_c("a", "title", "text!")]}) != first
    assert rv.candidates_fingerprint({"d": [_c("a", "content", "text")]}) != first


def test_the_week1_check_reports_only_places():
    cands = [_c("x", "content", "Tata Motors " * 6)] + [_c(f"t{i}", "title") for i in range(12)]
    result = rv.week1_check(cands, "O1M5", RULE, ["x"])
    assert result["articles"]["x"] == {"retrieval": 1, "O1": 13, "O1M5": 1, "in_top_10": True}
