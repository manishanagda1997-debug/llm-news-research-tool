"""
Tests for the experiment record (no network, no model calls).

The record states everything that decides a model result: provider, model,
generation settings, relevance prompt version, batch size, workers, passage
limit and the ranking settings. The development record saves it. The fresh
score builds it again and refuses to run when any part differs, so a model
name changed in .env (outside the freeze) cannot slip into the fresh test.
The commit is recorded but not compared here: the freeze check covers it.
"""

import pytest

from src import experiment as ex
from src import relevance_check as rc


class FakeLLM:
    def __init__(self, provider="groq", model="openai/gpt-oss-120b", generation=None):
        self.provider, self.model_name = provider, model
        self.generation = generation or {"temperature": 0, "reasoning_effort": "medium"}


RANKING = {"threshold": 0.5, "ks": [5, 10, 15], "primary_k": 10, "order": ["O1+S1", "O1+S2"]}


def test_the_record_holds_everything_that_decides_a_result():
    record = ex.build_record(FakeLLM(), RANKING, commit="abc123")
    assert record == {
        "provider": "groq", "model": "openai/gpt-oss-120b",
        "generation": {"temperature": 0, "reasoning_effort": "medium"},
        "relevance": {"prompt_version": rc.PROMPT_VERSION, "batch_size": rc.BATCH_SIZE, "workers": rc.WORKERS,
                      "passage_limit": rc.MAX_PASSAGE_CHARS},
        "ranking": RANKING, "commit": "abc123"}


def test_the_same_setup_gives_no_problems_even_on_another_commit():
    frozen = ex.build_record(FakeLLM(), RANKING, commit="abc")
    assert ex.record_problems(frozen, ex.build_record(FakeLLM(), RANKING, commit="def")) == []


@pytest.mark.parametrize("llm, ranking, part", [
    (FakeLLM(provider="gemini", model="gemini-x", generation={"temperature": 0}), RANKING, "provider"),
    (FakeLLM(model="openai/gpt-oss-20b"), RANKING, "model"),
    (FakeLLM(generation={"temperature": 0, "reasoning_effort": "low"}), RANKING, "generation"),
    (FakeLLM(), {**RANKING, "threshold": 0.6}, "ranking"),
])
def test_any_difference_is_named(llm, ranking, part):
    frozen = ex.build_record(FakeLLM(), RANKING, commit="abc")
    problems = ex.record_problems(frozen, ex.build_record(llm, ranking, commit="abc"))
    assert any(p.startswith(f"{part}:") for p in problems)


def test_a_missing_frozen_record_is_a_problem():
    assert ex.record_problems(None, ex.build_record(FakeLLM(), RANKING, commit="x")) == [
        "no experiment record in the development record"]


def test_the_record_never_holds_a_key():
    class KeyLLM(FakeLLM):
        api_key = "gsk_SECRET"

    assert "gsk_SECRET" not in repr(ex.build_record(KeyLLM(), RANKING, commit="x"))


# ---------- where the record is saved and checked ----------

def test_dev_and_fresh_runs_describe_the_ranking_the_same_way():
    from src import ranking_eval as re_
    settings = re_.ranking_settings(0.5)
    assert settings == {"threshold": 0.5, "ks": list(re_.KS), "primary_k": re_.PRIMARY_K,
                        "order": list(__import__("src.ranking", fromlist=["SIMPLICITY"]).SIMPLICITY)}


def test_the_fresh_score_refuses_a_different_setup():
    from src import fresh_eval as fe
    frozen = ex.build_record(FakeLLM(), RANKING, commit="abc")
    assert fe.experiment_problems({"experiment": frozen}, ex.build_record(FakeLLM(), RANKING, commit="zzz")) == []
    assert fe.experiment_problems({}, ex.build_record(FakeLLM(), RANKING, commit="x"))
    other = ex.build_record(FakeLLM(model="openai/gpt-oss-20b"), RANKING, commit="abc")
    assert fe.experiment_problems({"experiment": frozen}, other)


def test_the_runtime_record_keeps_the_token_use_and_errors():
    from src import ranking_eval as re_

    class Counting(FakeLLM):
        cache_tag = ""
        stats = {"calls": 2, "rate_limited": 1, "retries": 1, "failures": {}, "input_tokens": 10,
                 "output_tokens": 4, "reasoning_tokens": 2, "per_call": []}

        def __call__(self, prompt):
            import json
            ids = [p.split('"')[0] for p in prompt.split('<article id="')[1:]]
            return json.dumps({"results": [{"id": i, "label": "direct", "certainty": "sure", "reason": "r"}
                                           for i in ids]})

    candidates = [{"key": "A1", "title": "Tata Motors t", "text": "Tata Motors text", "description": "d"}]
    answers, runtime = re_._ask_model(candidates, "Tata Motors", None, llm=Counting(), cache=None)
    assert answers["A1"]["source"] == "model"
    assert runtime["provider"] == "groq" and runtime["model"] == "openai/gpt-oss-120b"
    assert runtime["llm_stats"]["rate_limited"] == 1 and runtime["llm_stats"]["input_tokens"] == 10
