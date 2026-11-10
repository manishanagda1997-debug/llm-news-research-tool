"""
Tests for batch splitting and cache keys in the relevance check (fake model only, no API calls).

A batch too large for the provider's token limit raises RequestTooLarge
before it is sent. The batch then splits in two, again and again if needed.
One article that is too large on its own falls back; its text is never cut.
Every article keeps exactly one answer, in the original order, and a failed
article stays in the results as a fallback, so it counts in every score.

Saved answers are keyed by company, model, prompt version, scope, title,
passage and the provider's cache tag. The Gemini tag is empty, so the keys of
answers saved before the Groq provider stay the same.
"""

import hashlib
import json

from src import relevance_check as rc
from src.llm import RequestTooLarge
from src.summarizer import LLMUnavailable

COMPANY = "Tata Motors"


def _items(n, size=10):
    return [{"id": f"A{i}", "title": f"title {i}", "passage": "p" * size} for i in range(n)]


def _ids(prompt):
    return [part.split('"')[0] for part in prompt.split('<article id="')[1:]]


class LimitedLLM:
    """Refuses any prompt with more than `limit` articles, or with one of the `huge` articles."""

    def __init__(self, limit, huge=(), tag=""):
        self.limit, self.huge, self.prompts = limit, set(huge), []
        self.model_name, self.cache_tag = "fake-model", tag

    def __call__(self, prompt):
        ids = _ids(prompt)
        if len(ids) > self.limit or self.huge & set(ids):
            raise RequestTooLarge("too large")
        self.prompts.append(ids)
        return json.dumps({"results": [{"id": i, "label": "direct", "certainty": "sure", "reason": "r"} for i in ids]})


def test_a_batch_too_large_splits_until_it_fits():
    items = _items(10)
    llm = LimitedLLM(limit=3)
    results = rc.classify(items, llm, COMPANY, batch_size=10)
    assert list(results) == [i["id"] for i in items]
    assert all(r["source"] == "model" for r in results.values())
    assert all(len(p) <= 3 for p in llm.prompts)
    assert sorted(i for p in llm.prompts for i in p) == sorted(i["id"] for i in items)   # each article asked once


def test_one_article_too_large_on_its_own_falls_back_and_the_rest_are_answered():
    items = _items(6)
    llm = LimitedLLM(limit=10, huge={"A4"})
    results = rc.classify(items, llm, COMPANY, batch_size=6)
    assert results["A4"]["source"] == "fallback" and results["A4"]["error"] == "too large"
    assert all(results[i]["source"] == "model" for i in results if i != "A4")
    assert list(results) == [i["id"] for i in items]


def test_a_failed_batch_keeps_every_article_as_a_fallback():
    class Down:
        model_name, cache_tag = "fake-model", ""

        def __call__(self, prompt):
            raise LLMUnavailable("daily limit")

    results = rc.classify(_items(5), Down(), COMPANY, batch_size=2)
    assert len(results) == 5 and all(r["source"] == "fallback" for r in results.values())


def test_the_cache_key_is_unchanged_for_gemini():
    item = {"id": "A1", "title": "t", "passage": "p"}
    old = hashlib.sha256(json.dumps([COMPANY, "gemini-x", rc.PROMPT_VERSION, "t", "p", "scope"],
                                    ensure_ascii=False).encode("utf-8")).hexdigest()
    assert rc._key(item, COMPANY, "gemini-x", "scope") == old
    assert rc._key(item, COMPANY, "gemini-x", "scope", tag="") == old


def test_the_cache_key_separates_provider_settings_model_and_text():
    item = {"id": "A1", "title": "t", "passage": "p"}
    base = rc._key(item, COMPANY, "m", "scope", tag="groq|reasoning=medium")
    assert base != rc._key(item, COMPANY, "m", "scope", tag="groq|reasoning=low")
    assert base != rc._key(item, COMPANY, "m", "scope")
    assert base != rc._key(item, COMPANY, "m2", "scope", tag="groq|reasoning=medium")
    assert base != rc._key({**item, "passage": "p2"}, COMPANY, "m", "scope", tag="groq|reasoning=medium")


def test_saved_answers_of_one_provider_are_not_used_for_another(tmp_path):
    cache = rc.CheckCache(tmp_path)
    items = _items(2)
    rc.classify(items, LimitedLLM(limit=10, tag=""), COMPANY, cache=cache)
    groq = LimitedLLM(limit=10, tag="groq|reasoning=medium")
    rc.classify(items, groq, COMPANY, cache=cache)
    assert groq.prompts                                       # asked again, not read from Gemini's answers
    again = LimitedLLM(limit=10, tag="groq|reasoning=medium")
    rc.classify(items, again, COMPANY, cache=cache)
    assert again.prompts == []                                # its own saved answers are used
