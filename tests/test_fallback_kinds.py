"""
Tests for the kind of every fallback (fake models only, no API calls).

A fallback keeps its `error` field as before and adds `kind`, so a report can
tell an outage from a quota:

    server          5xx or a temporary service error
    rate_limit      requests or tokens a minute
    daily_limit     the daily quota, only when the provider's message says so
    timeout         the request takes too long
    authentication  a wrong or refused key
    invalid_output  the model answers, but the answer fails validation twice
    too_large       one article alone is too large for the limit (its text is never cut)
    unknown         anything else, and an error without a kind: never a guess
"""

import json

import pytest

from src import llm as llm_module
from src import model_compare as mc
from src import relevance_check as rc
from src.summarizer import LLMUnavailable

COMPANY = "Tata Motors"
ITEMS = [{"id": "A1", "title": "t1", "passage": "p1"}, {"id": "A2", "title": "t2", "passage": "p2"}]


class Failing:
    model_name, cache_tag = "m", ""

    def __init__(self, error):
        self.error = error

    def __call__(self, prompt):
        raise self.error


def _unavailable(kind):
    error = LLMUnavailable("message")
    error.kind = kind
    return error


@pytest.mark.parametrize("kind, expected", [
    ("server", "server"), ("rate_limit", "rate_limit"), ("daily_limit", "daily_limit"), ("timeout", "timeout"),
    ("auth", "authentication"), ("other", "unknown"),
])
def test_the_fallback_names_the_kind_of_failure(kind, expected):
    result = rc.classify(ITEMS, Failing(_unavailable(kind)), COMPANY)
    assert {r["kind"] for r in result.values()} == {expected}
    assert all(r["error"] == "LLMUnavailable" and r["source"] == "fallback" for r in result.values())


def test_an_error_without_a_kind_is_unknown_not_a_guess():
    result = rc.classify(ITEMS, Failing(LLMUnavailable("429 quota")), COMPANY)
    assert {r["kind"] for r in result.values()} == {"unknown"}


def test_a_python_timeout_is_a_timeout():
    result = rc.classify(ITEMS, Failing(TimeoutError("t")), COMPANY)
    assert {r["kind"] for r in result.values()} == {"timeout"} and result["A1"]["error"] == "timeout"


def test_an_answer_that_fails_validation_twice_is_invalid_output():
    class Bad:
        model_name, cache_tag = "m", ""

        def __call__(self, prompt):
            return "{not json"

    result = rc.classify(ITEMS, Bad(), COMPANY)
    assert {r["kind"] for r in result.values()} == {"invalid_output"}


def test_one_article_too_large_on_its_own_is_too_large():
    class TooLarge:
        model_name, cache_tag = "m", ""

        def __call__(self, prompt):
            raise llm_module.RequestTooLarge("too large")

    result = rc.classify(ITEMS[:1], TooLarge(), COMPANY)
    assert result["A1"]["kind"] == "too_large"


@pytest.mark.parametrize("message, kind", [
    ("429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests, limit: 20, "
     "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier", "daily_limit"),
    ("429 RESOURCE_EXHAUSTED. Quota exceeded, quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
     "rate_limit"),
    ("Rate limit reached for model on tokens per day (TPD): Limit 200000.", "daily_limit"),
])
def test_google_and_groq_daily_quota_messages_are_daily_limits(message, kind):
    assert llm_module.error_kind(Exception(message)) == kind


def test_model_measures_count_fallbacks_by_kind():
    answers = {"A": {"label": None, "certainty": None, "reason": "", "source": "fallback", "error": "LLMUnavailable",
                     "kind": "server"},
               "B": {"label": None, "certainty": None, "reason": "", "source": "fallback", "error": "LLMUnavailable",
                     "kind": "daily_limit"},
               "C": {"label": None, "certainty": None, "reason": "", "source": "fallback", "error": "LLMUnavailable"}}
    m = mc.model_measures(answers, {"A": "direct", "B": "not", "C": "not"})
    assert m["fallbacks"] == {"server": 1, "daily_limit": 1, "unknown": 1}


# ---------- the coverage rule for runs (agreed before any quality result) ----------

def test_a_run_is_valid_for_quality_only_at_90_percent_coverage():
    assert mc.MIN_COVERAGE == 0.9
    assert mc.valid_for_quality(answered=61, articles=67) and not mc.valid_for_quality(answered=60, articles=67)


def test_a_second_run_is_allowed_only_after_a_provider_failure_below_90_percent():
    assert mc.second_run_allowed({"coverage": 5 / 67, "failure_kinds": {"server": 37, "daily_limit": 25}})
    assert not mc.second_run_allowed({"coverage": 0.95, "failure_kinds": {"server": 3}})
    assert not mc.second_run_allowed({"coverage": 0.5, "failure_kinds": {"invalid_output": 33}})   # the model's fault


def test_a_second_run_never_overwrites_the_first():
    assert mc.run_file_name("gemini-3.8-flash", attempt=1) == "run_gemini-3.8-flash.json"
    assert mc.run_file_name("gemini-3.8-flash", attempt=2) == "run_gemini-3.8-flash_attempt2.json"


def _summary(coverage, passes=3, events=10):
    return {"runs_passed": passes, "events_k10": events, "wrong_promotions": 0, "f1": 0.9, "fallbacks": 0,
            "seconds": 100.0, "provider": "groq", "complete": True, "coverage": coverage}


def test_a_run_under_90_percent_coverage_is_never_chosen():
    assert mc.choose({"a": _summary(0.07, events=30), "b": _summary(0.95, events=10)}) == "b"
    assert mc.choose({"a": _summary(0.89)}) == "O1"


def test_the_summary_carries_the_coverage_and_the_failure_kinds():
    from src import ranking as rk
    from src.ranking_eval import evaluate_run
    keys = ["K0", "K1", "K2"]
    labels = {"K0": "direct", "K1": "not", "K2": "indirect"}
    events = {"K0": "e0", "K2": "e2"}
    fallback = {"label": None, "certainty": None, "reason": "", "source": "fallback", "error": "LLMUnavailable",
                "kind": "server"}
    answers = {"K0": {"label": "direct", "certainty": "sure", "reason": "", "source": "model"},
               "K1": fallback, "K2": fallback}
    candidates = [{"key": k, "title": k, "mention_level": "title", "check": answers[k]} for k in keys]
    report = evaluate_run(candidates, labels, events, seconds=10.0)
    measures = mc.model_measures(answers, labels)
    s = mc.summarize("gemini", {"r": report}, {"r": mc.wrong_promotions(report, labels)}, measures, {"r": 10.0},
                     complete=True)
    assert s["coverage"] == pytest.approx(1 / 3) and s["failure_kinds"] == {"server": 2}
    assert mc.second_run_allowed(s)


def test_old_records_take_the_kind_from_their_query_only_when_it_is_one_kind():
    old = {"label": None, "certainty": None, "reason": "", "source": "fallback", "error": "LLMUnavailable"}
    ok = {"label": "not", "certainty": "sure", "reason": "", "source": "model"}
    runs = {"q1": {"answers": {"A": dict(old), "B": dict(ok)}, "llm_stats": {"failures": {"server": 2}}},
            "q2": {"answers": {"C": dict(old)}, "llm_stats": {"failures": {"daily_limit": 3}}},
            "q3": {"answers": {"D": dict(old)}, "llm_stats": {"failures": {"server": 1, "daily_limit": 1}}}}
    fixed = mc.with_fallback_kinds(runs)
    assert fixed["q1"]["answers"]["A"]["kind"] == "server" and "kind" not in fixed["q1"]["answers"]["B"]
    assert fixed["q2"]["answers"]["C"]["kind"] == "daily_limit"
    assert fixed["q3"]["answers"]["D"]["kind"] == "unknown"                 # mixed: never a guess
    assert "kind" not in runs["q1"]["answers"]["A"]                          # the saved record is not changed


def test_a_kind_already_recorded_is_kept():
    new = {"label": None, "certainty": None, "reason": "", "source": "fallback", "error": "LLMUnavailable",
           "kind": "timeout"}
    runs = {"q": {"answers": {"A": new}, "llm_stats": {"failures": {"server": 1}}}}
    assert mc.with_fallback_kinds(runs)["q"]["answers"]["A"]["kind"] == "timeout"
