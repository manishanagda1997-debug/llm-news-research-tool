"""
Tests for version 3 of the stage 2 relevance check (fake model only, no API calls).

Prompt version 3 adds one general rule: first decide what the article is
mainly about, then judge that main topic against the conditions. A passing
or historical mention of the company does not make an article `not` when
its main topic is itself a development in the company's markets and
segments. The existing `not` cases stay.

Batches can be sent several at a time (a fixed number of workers, so the
provider's rate limits are respected). Results keep their article ids, a
failed or timed-out batch falls back on its own, and the answers are the
same as when the batches are sent one by one.
"""

import json
import threading
import time

import pytest

from src import relevance_check as rc
from src.summarizer import LLMUnavailable

COMPANY = "Tata Motors"
ITEMS = [{"id": f"A{n}", "title": f"title {n}", "passage": f"passage {n}"} for n in range(7)]


# ---------- prompt version 3 ----------

def test_prompt_version_is_3():
    assert rc.PROMPT_VERSION == "3"


def test_prompt_asks_for_the_main_topic_first():
    prompt = rc.build_prompt(ITEMS[:1], COMPANY).lower()
    assert "mainly about" in prompt and "main topic" in prompt
    assert prompt.index("main topic") < prompt.index("labels:")


def test_a_passing_or_historical_mention_does_not_decide_when_the_topic_is_the_companys_market():
    prompt = rc.build_prompt(ITEMS[:1], COMPANY).lower()
    assert "historical mention" in prompt and "does not make" in prompt


def test_the_existing_not_cases_stay():
    prompt = rc.build_prompt(ITEMS[:1], COMPANY).lower()
    for rule in ("past job", "results calendar", "share price or percentage change", "another company's service"):
        assert rule in prompt


# ---------- parallel batches ----------

def _answer_for(prompt):
    ids = [part.split('"')[0] for part in prompt.split('<article id="')[1:]]
    return json.dumps({"results": [{"id": i, "label": "indirect", "certainty": "sure", "reason": f"r {i}"}
                                   for i in ids]})


class SlowLLM:
    """Answers every article; records how many calls run at the same time."""
    model_name = "fake-model"

    def __init__(self, fail_on=None, error=None):
        self.lock, self.running, self.most, self.calls = threading.Lock(), 0, 0, 0
        self.fail_on, self.error = fail_on, error

    def __call__(self, prompt):
        with self.lock:
            self.running += 1
            self.calls += 1
            self.most = max(self.most, self.running)
        try:
            time.sleep(0.05)
            if self.fail_on and f'<article id="{self.fail_on}">' in prompt:
                raise self.error
            return _answer_for(prompt)
        finally:
            with self.lock:
                self.running -= 1


def test_parallel_and_one_by_one_give_the_same_answers():
    one_by_one = rc.classify(ITEMS, SlowLLM(), COMPANY, batch_size=2, workers=1)
    parallel = rc.classify(ITEMS, SlowLLM(), COMPANY, batch_size=2, workers=3)
    assert parallel == one_by_one
    assert list(parallel) == [item["id"] for item in ITEMS]                 # the original order of ids


def test_no_more_batches_run_at_once_than_the_workers_allow():
    llm = SlowLLM()
    rc.classify(ITEMS, llm, COMPANY, batch_size=1, workers=3)
    assert llm.most <= 3 and llm.calls == 7


def test_one_unavailable_batch_falls_back_alone():
    llm = SlowLLM(fail_on="A2", error=LLMUnavailable("429 rate limit"))
    result = rc.classify(ITEMS, llm, COMPANY, batch_size=2, workers=3)
    assert result["A2"]["source"] == "fallback" and result["A3"]["source"] == "fallback"   # same batch
    assert all(result[f"A{n}"]["source"] == "model" for n in (0, 1, 4, 5, 6))


def test_a_timeout_falls_back_for_its_batch_instead_of_stopping_the_run():
    llm = SlowLLM(fail_on="A4", error=TimeoutError("read timed out"))
    result = rc.classify(ITEMS, llm, COMPANY, batch_size=2, workers=2)
    assert result["A4"] == {"label": None, "certainty": None, "reason": "", "source": "fallback",
                            "error": "timeout", "kind": "timeout"}
    assert result["A0"]["source"] == "model"


def test_progress_counts_every_batch_once_in_parallel():
    calls = []
    rc.classify(ITEMS, SlowLLM(), COMPANY, batch_size=2, workers=3, progress=lambda *a: calls.append(a))
    assert sorted(done for done, _, _ in calls) == [1, 2, 3, 4] and all(total == 4 for _, total, _ in calls)


def test_saved_answers_are_the_same_as_fresh_ones(tmp_path):
    cache = rc.CheckCache(tmp_path)
    fresh = rc.classify(ITEMS, SlowLLM(), COMPANY, batch_size=2, workers=3, cache=cache)
    llm = SlowLLM()
    saved = rc.classify(ITEMS, llm, COMPANY, batch_size=2, workers=3, cache=cache)
    assert saved == fresh and llm.calls == 0


def test_the_default_number_of_workers_is_small():
    assert 1 <= rc.WORKERS <= 4


def test_workers_must_be_at_least_one():
    with pytest.raises(ValueError):
        rc.classify(ITEMS, SlowLLM(), COMPANY, workers=0)
