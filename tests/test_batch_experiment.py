"""
Tests for the batch-size experiment (fake models only, no API calls).

A separate development experiment, agreed after the first 120B runs: does
sending 3, 4 or 5 articles per request finish the 25-article query faster
under Groq's 8,000 tokens a minute? Smaller requests repeat the fixed
instructions more often, but may fit the minute better.

    fixed        model gpt-oss-120b, its comparison settings, prompt version 3,
                 the same articles and text: the 25-article query of 14 October
    sizes        3, 4 and 5 articles per request, each run once
    measured     wall-clock seconds, tokens, waits, 429 errors, fallbacks by
                 kind, and quality (the model measures on that query)
    separate     its own files (batch_...), never read by the comparison
                 report; the comparison run (batch size 10) is shown beside it
                 for reference and never changed
"""

import json

import pytest

from src import model_compare as mc


def test_the_experiment_is_fixed_before_it_runs():
    assert mc.BATCH_SIZES == (3, 4, 5)
    assert mc.BATCH_CONFIG == "gpt-oss-120b" and mc.BATCH_QUERY == "2026-10-14"


def test_its_files_never_mix_with_the_comparison_runs():
    name = mc.batch_file_name(3)
    assert name == "batch_gpt-oss-120b_size3.json" and not name.startswith("run_")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeLLM:
    model_name, cache_tag, provider, generation = "m", "t", "groq", {}

    def __init__(self):
        self.batches = []
        self.stats = {"per_call": [], "waits": []}

    def __call__(self, prompt):
        ids = [p.split('"')[0] for p in prompt.split('<article id="')[1:]]
        self.batches.append(len(ids))
        return json.dumps({"results": [{"id": i, "label": "not", "certainty": "sure", "reason": "r"} for i in ids]})


def _candidates(n):
    return [{"key": f"A{i}", "title": f"Tata Motors item {i}", "text": f"Tata Motors text {i}.",
             "description": f"d {i}", "mention_level": "title"} for i in range(n)]


@pytest.mark.parametrize("size, batches", [(3, [3, 3, 3, 3]), (4, [4, 4, 4]), (5, [5, 5, 2])])
def test_a_query_run_sends_the_batch_size_it_is_given(size, batches):
    llm = FakeLLM()
    result = mc.run_query(_candidates(12), llm, "Tata Motors", None, clock=Clock(), batch_size=size)
    assert sorted(llm.batches, reverse=True) == sorted(batches, reverse=True)
    assert result["batch_size"] == size and len(result["answers"]) == 12


def test_the_comparison_runs_keep_batch_size_10():
    from src.relevance_check import BATCH_SIZE
    llm = FakeLLM()
    result = mc.run_query(_candidates(12), llm, "Tata Motors", None, clock=Clock())
    assert BATCH_SIZE == 10 and result["batch_size"] == 10 and sorted(llm.batches) == [2, 10]


def _record(size, seconds, kinds=None, answered=25):
    answers = {f"A{i}": {"label": "direct" if i < answered else None, "certainty": "sure", "reason": "",
                         "source": "model" if i < answered else "fallback",
                         **({} if i < answered else {"error": "LLMUnavailable", "kind": "server"})}
               for i in range(25)}
    stats = {"calls": 9, "rate_limited": 1, "splits": 0, "waited_seconds": 300.0,
             "per_call": [{"input": 2000, "output": 500, "reasoning": 300, "cached": 500}] * 9}
    return {"configuration": "gpt-oss-120b", "batch_size": size, "query": "2026-10-14",
            "result": {"wall_seconds": seconds, "answers": answers, "llm_stats": stats}}


def test_each_size_gets_one_row_with_speed_tokens_errors_and_quality():
    labels = {f"A{i}": "direct" for i in range(25)}
    rows = mc.batch_rows([_record(3, 170.0), _record(4, 150.0, answered=24)], labels)
    assert [r["batch_size"] for r in rows] == [3, 4]
    first = rows[0]
    assert first["wall_seconds"] == 170.0 and first["calls"] == 9 and first["rate_limited"] == 1
    assert first["tokens_counted"] == 9 * (2000 - 500 + 500) and first["waited_seconds"] == 300.0
    assert first["meets_180_s"] is True and first["coverage"] == 1.0 and first["f1"] == pytest.approx(1.0)
    assert rows[1]["fallbacks"] == {"server": 1} and rows[1]["coverage"] == pytest.approx(24 / 25)


def test_a_size_over_the_time_limit_is_marked():
    labels = {f"A{i}": "direct" for i in range(25)}
    assert mc.batch_rows([_record(5, 181.0)], labels)[0]["meets_180_s"] is False
