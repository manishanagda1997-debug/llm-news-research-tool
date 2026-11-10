"""
Tests for src/research.py: the analyst pipeline (Stage 5), with fake searches, fake pages and fake models.

    search       searches A and B for the last `days` days; the window, the counts and the latest article date
                 are shown; a NewsAPI failure is an error, never an empty list shown as a success
    provisional  the O1M5+S1 order from NewsAPI data only, labelled provisional (Stage 1 keeps or drops articles
                 using the full text, so this list is not the tested result)
    enrich       full text within a time limit for the whole step; a page not fetched in time keeps its short
                 text and is listed
    final        the default: Stage 1 with the full text, then O1M5+S1, the same as the evaluation code
    summary      fresh, saved (exact match only), model unavailable, or removed by the checks: always told apart
    review       model labels beside each article; the order does not change
    apply        the O3 order (title protection), labelled experimental; no article is removed; back to default
"""

import json

import pytest

from src import research as rs
from src import ranking as rk
from src.query import parse_query
from src.ranking_eval import stage1_candidates
from src.ranking_variants import order_variant
from src.summarizer import LLMUnavailable

COMPANY = "Tata Motors"


def _article(i, title, description="", content="", day="2026-10-28"):
    return {"url": f"https://news.example/{i}", "title": title, "description": description, "content": content,
            "publishedAt": f"{day}T10:00:00Z", "source": {"name": f"Site {i}"}}


ARTICLES_A = [
    _article(1, "Tata Motors raises its Iveco offer", "Tata Motors raises the offer price."),
    _article(2, "Car sales rise in October", "Carmakers report higher sales.", "Tata Motors sold more cars."),
    _article(3, "Markets today", "Stocks fall.", day="2026-10-30"),
    _article(4, "Tata Motors shares rise", "Tata Motors shares rise after results."),
]
ARTICLES_B = [_article(5, "Jaguar Land Rover opens a showroom", "JLR opens a new showroom in Delhi.")]


class FakeNews:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def __call__(self, company, aliases=None, part="all", **settings):
        """The same signature and the same allowed values as src.news_fetcher.fetch_news."""
        from src.news_fetcher import NewsFetchError
        from src.query import SEARCH_PARTS
        if part not in SEARCH_PARTS:
            raise ValueError(f"part must be one of {SEARCH_PARTS}")
        self.calls.append((None if aliases is None else part, settings))
        if self.fail:
            raise NewsFetchError(self.fail)
        articles = ARTICLES_A if aliases is None else ARTICLES_B
        return {"articles": [dict(a) for a in articles],
                "request": {"from_date": "2026-10-27", "to_date": None, "page_size": settings["page_size"]}}


ALIASES = json.load(open("data/company_aliases.json", encoding="utf-8"))


class FakePages:
    """Full text for each page; a page in `slow` takes `seconds` on the fake clock."""

    def __init__(self, clock, texts=None, slow=(), seconds=0.0):
        self.clock, self.texts, self.slow, self.seconds, self.asked = clock, texts or {}, set(slow), seconds, []

    def fetch(self, url):
        self.asked.append(url)
        if url in self.slow:
            self.clock.now += self.seconds
        text = self.texts.get(url)
        return {"url": url, "status": "ok" if text else "blocked_by_robots", "text": text or ""}


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


# ---------- search ----------

def test_search_runs_a_and_b_with_the_given_days_and_reports_window_counts_and_latest_date():
    news = FakeNews()
    found = rs.search(COMPANY, ALIASES, days=7, fetch=news)
    assert found["status"] == "ok"
    assert [part for part, _ in news.calls] == [None, "subsidiaries"]
    assert all(settings["days_back"] == 7 for _, settings in news.calls)
    assert found["counts"] == {"A": 4, "B": 1} and len(found["articles"]) == 5
    assert found["window"]["from"] == "2026-10-27" and found["latest_published"] == "2026-10-30T10:00:00Z"


def test_a_company_without_aliases_runs_search_a_only_and_says_so():
    news = FakeNews()
    found = rs.search("Some Company", ALIASES, fetch=news)
    assert [part for part, _ in news.calls] == [None] and found["counts"]["B"] is None
    assert found["notes"]


def test_a_newsapi_failure_is_an_error_not_an_empty_success():
    found = rs.search(COMPANY, ALIASES, fetch=FakeNews(fail="Could not reach NewsAPI. Check the internet connection."))
    assert found["status"] == "failed" and found["articles"] == []
    assert "Could not reach NewsAPI" in found["error"]
    with pytest.raises(rs.ResearchError):
        rs.provisional(found, COMPANY, ALIASES)


# ---------- provisional and final lists ----------

def _found():
    return rs.search(COMPANY, ALIASES, fetch=FakeNews())


def test_the_provisional_list_uses_newsapi_data_only_and_is_labelled_provisional():
    result = rs.provisional(_found(), COMPANY, ALIASES, k=2)
    assert result["state"] == "provisional" and "Provisional" in result["label"]
    assert result["ordering"] == "O1M5+S1" and len(result["selected"]) == 2


def test_the_final_list_equals_the_evaluation_code_on_the_same_input():
    found, clock = _found(), Clock()
    texts = {"https://news.example/3": "Markets fall. " + "Tata Motors shares fall after results. " * 20}
    enriched = rs.enrich(found, FakePages(clock, texts), clock=clock)
    final = rs.final(found, enriched, COMPANY, ALIASES, k=3)
    candidates = stage1_candidates(found["articles"], enriched["texts"], COMPANY, ALIASES)
    expected = order_variant(candidates, "O1M5", parse_query(COMPANY, aliases=ALIASES))
    assert [c["key"] for c in final["ranked"]] == [c["key"] for c in expected]
    assert final["state"] == "final" and final["label"] == rs.DEFAULT_LABEL and final["ordering"] == "O1M5+S1"
    assert [c["key"] for c in final["selected"]] == [c["key"] for c in expected[:3]]
    assert final["selected"] + final["additional"] == final["ranked"]


def test_provisional_and_final_lists_are_told_apart_when_full_text_changes_the_list():
    found, clock = _found(), Clock()
    texts = {"https://news.example/3": "Markets fall. " + "Tata Motors shares fall after results. " * 20}
    first = rs.provisional(found, COMPANY, ALIASES)
    last = rs.final(found, rs.enrich(found, FakePages(clock, texts), clock=clock), COMPANY, ALIASES)
    assert first["state"] != last["state"]
    assert rs.list_changes(first, last)["added"] == ["Markets today"]


# ---------- enrichment time limit ----------

def test_a_slow_site_cannot_hold_the_list_past_the_time_limit():
    found, clock = _found(), Clock()
    pages = FakePages(clock, slow={"https://news.example/1"}, seconds=95)
    enriched = rs.enrich(found, pages, budget=90, clock=clock)
    assert pages.asked == ["https://news.example/1"]                  # nothing new starts after the limit
    assert len(enriched["not_in_time"]) == 4
    assert all(enriched["status"][url] == "not_fetched_in_time" for url in enriched["not_in_time"])


# ---------- the summary ----------

class FakeSummaryModel:
    model_name, cache_tag, provider = "fake-model", "fake|{}", "fake"

    def __init__(self, fail=None):
        self.prompts, self.fail = [], fail
        self.stats = {"per_call": [], "requests": []}

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            error = LLMUnavailable(self.fail)
            error.kind = "daily_limit"
            raise error
        ids = [int(part.split('"')[0]) for part in prompt.split('<article id="')[1:]]
        return json.dumps({"article_briefs": [{"id": i, "brief": "Tata Motors news.", "enough_text": True} for i in ids],
                           "key_points": [{"text": "Tata Motors raises its offer.", "citations": [ids[0]]}],
                           "industry_context": [], "not_used": [{"id": i, "reason": "Covered."} for i in ids[1:]]})


def _final(k=2):
    found, clock = _found(), Clock()
    return rs.final(found, rs.enrich(found, FakePages(clock), clock=clock), COMPANY, ALIASES, k=k)


def test_a_fresh_summary_then_the_same_request_reuses_it_and_says_so(tmp_path):
    from src.summarizer import SummaryCache
    cache, llm = SummaryCache(tmp_path), FakeSummaryModel()
    first = rs.summarize_selection(_final(), COMPANY, ALIASES, llm, design="direct", cache=cache)
    again = rs.summarize_selection(_final(), COMPANY, ALIASES, llm, design="direct", cache=cache)
    assert first["origin"] == "fresh" and again["origin"] == "saved" and len(llm.prompts) == 1
    assert first["summary_status"] == "ready"


def test_a_changed_order_or_scope_never_reuses_a_saved_summary(tmp_path):
    from src.summarizer import SummaryCache
    cache, llm = SummaryCache(tmp_path), FakeSummaryModel()
    result = _final()
    rs.summarize_selection(result, COMPANY, ALIASES, llm, design="direct", cache=cache)
    swapped = dict(result, selected=list(reversed(result["selected"])))
    assert rs.summarize_selection(swapped, COMPANY, ALIASES, llm, design="direct", cache=cache)["origin"] == "fresh"
    other_scope = json.loads(json.dumps(ALIASES))
    other_scope["identities"]["tata_motors_cv"]["business"] = "Trucks only"
    assert rs.summarize_selection(result, COMPANY, other_scope, llm, design="direct", cache=cache)["origin"] == "fresh"


def test_an_unavailable_model_gives_headlines_and_the_wait_without_the_organisation():
    message = ("Rate limit reached in organization `org_01abcDEF` on tokens per day (TPD): Limit 200000, "
               "Used 199000, Requested 6000. Please try again in 1h2m3s.")
    out = rs.summarize_selection(_final(), COMPANY, ALIASES, FakeSummaryModel(fail=message), design="direct")
    assert out["summary_status"] == "model_unavailable" and out["wait_seconds"] == 3723
    assert out["headlines"] and "org_01abcDEF" not in json.dumps(out)
    assert "Rate limit reached" not in json.dumps(out)


def test_a_summary_removed_by_the_checks_is_reported_as_such():
    class Wrong(FakeSummaryModel):
        def __call__(self, prompt):
            self.prompts.append(prompt)
            if prompt.startswith("You correct one sentence"):
                return json.dumps({"text": "Tata Motors raises its offer to 99 euro."})
            ids = [int(part.split('"')[0]) for part in prompt.split('<article id="')[1:]]
            return json.dumps({"article_briefs": [{"id": i, "brief": "News.", "enough_text": True} for i in ids],
                               "key_points": [{"text": "Tata Motors raises its offer to 99 euro.", "citations": [ids[0]]}],
                               "industry_context": [], "not_used": [{"id": i, "reason": "Covered."} for i in ids[1:]]})
    out = rs.summarize_selection(_final(), COMPANY, ALIASES, Wrong(), design="direct")
    assert out["summary_status"] == "failed_checks" and out["headlines"]


# ---------- model review and apply updates ----------

class FakeReviewModel:
    model_name, cache_tag, provider = "fake-review", "fake|{}", "fake"

    def __init__(self, labels, fail_ids=()):
        self.labels, self.fail_ids = labels, set(fail_ids)

    def __call__(self, prompt):
        ids = [part.split('"')[0] for part in prompt.split('<article id="')[1:]]
        if any(i in self.fail_ids for i in ids):
            error = LLMUnavailable("timeout")
            error.kind = "timeout"
            raise error
        return json.dumps({"results": [{"id": i, "label": self.labels.get(i, "direct"), "certainty": "sure",
                                        "reason": "Reason."} for i in ids]})


def test_a_review_adds_labels_but_never_changes_the_order():
    result = _final(k=3)
    keys = [c["key"] for c in result["ranked"]]
    reviewed = rs.review(result, COMPANY, ALIASES, FakeReviewModel({keys[0]: "not"}), cache=None)
    assert [c["key"] for c in reviewed["ranked"]] == keys and reviewed["ordering"] == "O1M5+S1"
    assert reviewed["ranked"][0]["check"]["label"] == "not"


def test_apply_updates_gives_the_o3_order_labelled_experimental_and_keeps_every_article():
    result = _final(k=3)
    keys = [c["key"] for c in result["ranked"]]
    reviewed = rs.review(result, COMPANY, ALIASES, FakeReviewModel({keys[0]: "not", keys[-1]: "direct"}), cache=None)
    applied = rs.apply_updates(reviewed)
    assert applied["ordering"] == "O3+S1" and applied["label"] == rs.EXPERIMENTAL_LABEL
    assert sorted(c["key"] for c in applied["ranked"]) == sorted(keys)
    assert [c["key"] for c in applied["ranked"]] == [c["key"] for c in rk.order(reviewed["ranked"], "O3")]
    back = rs.back_to_default(applied)
    assert [c["key"] for c in back["ranked"]] == keys and back["label"] == rs.DEFAULT_LABEL
    assert back["ordering"] == "O1M5+S1"


def test_a_partly_failed_review_keeps_the_evaluated_fallback_for_unanswered_articles():
    result = _final(k=3)
    keys = [c["key"] for c in result["ranked"]]
    reviewed = rs.review(result, COMPANY, ALIASES, FakeReviewModel({}, fail_ids={keys[1]}), cache=None, batch_size=1)
    unanswered = [c for c in reviewed["ranked"] if c["check"]["source"] != "model"]
    assert [c["key"] for c in unanswered] == [keys[1]]
    assert reviewed["review_counts"] == {"answered": len(keys) - 1, "unanswered": 1}
    applied = rs.apply_updates(reviewed)
    assert keys[1] in [c["key"] for c in applied["ranked"]]       # kept, in the middle group (evaluated fallback)


def test_apply_updates_needs_a_review_first():
    with pytest.raises(rs.ResearchError):
        rs.apply_updates(_final())


# ---------- messages ----------

def test_messages_never_show_keys_organisations_or_paths(monkeypatch):
    import dataclasses
    from src import llm as llm_module
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(llm_module.settings, groq_api_key="gsk_SECRET1"))
    text = rs.safe_text("failed with gsk_SECRET1 in org_ABC123 at C:\\Users\\manis\\projects_9\\x.py")
    assert "gsk_SECRET1" not in text and "org_ABC123" not in text and "C:\\Users" not in text


def test_the_fake_search_has_the_real_signature():
    import inspect
    from src.news_fetcher import fetch_news
    assert inspect.signature(fetch_news).parameters["part"].default == "all"


# ---------- the default order is O1M5, not O1 ----------

FIVE = ("Tata Motors reports record exports. Tata Motors expands its plant. Tata Motors hires engineers. "
        "Tata Motors opens a centre. Tata Motors signs a supplier. " * 3)


def _five_mention_case():
    """Retrieval order: a headline without the company but five names in its text comes after a description-only
    article; O1 puts it last, O1M5 lifts it into the title group."""
    articles = [_article(1, "Tata Motors raises its Iveco offer", "Tata Motors raises the offer price."),
                _article(2, "Auto stocks in focus", "Tata Motors among the movers today."),
                _article(3, "India's biggest export push", "A carmaker plans new exports.", FIVE)]
    found = {"status": "ok", "articles": [dict(a) for a in articles], "window": {}, "counts": {}, "notes": []}
    texts = {"https://news.example/3": FIVE}
    return found, texts


def test_the_default_order_follows_o1m5_where_o1_and_o1m5_differ():
    found, texts = _five_mention_case()
    final = rs.final(found, {"texts": texts}, COMPANY, ALIASES, k=2)
    candidates = stage1_candidates(found["articles"], texts, COMPANY, ALIASES)
    o1 = [c["key"] for c in rk.order(candidates, "O1")]
    o1m5 = [c["key"] for c in order_variant(candidates, "O1M5", parse_query(COMPANY, aliases=ALIASES))]
    assert o1 != o1m5                                   # the case separates the two orders
    assert [c["key"] for c in final["ranked"]] == o1m5 and final["ordering"] == "O1M5+S1"
    assert [c["key"] for c in final["selected"]] == o1m5[:2] != o1[:2]
    lifted = next(c for c in final["ranked"] if c["title"] == "India's biggest export push")
    assert final["ranked"].index(lifted) == 1           # second, in the title group, in retrieval order


def test_the_recorded_ordering_is_the_order_that_was_run(monkeypatch):
    found, texts = _five_mention_case()
    calls = []
    real = rs.default_order

    def spy(candidates, company, aliases):
        calls.append(company)
        return real(candidates, company, aliases)
    monkeypatch.setattr(rs, "default_order", spy)
    result = rs.final(found, {"texts": texts}, COMPANY, ALIASES, k=2)
    assert calls == [COMPANY] and result["ordering"] == rs.DEFAULT_ORDERING == f"{rs.DEFAULT_ORDER}+S1"
    assert rs.DEFAULT_ORDER == "O1M5" and rs.DEFAULT_ORDERING in rs.DEFAULT_LABEL


def test_the_provisional_list_also_uses_o1m5_on_the_short_text():
    found, _ = _five_mention_case()
    result = rs.provisional(found, COMPANY, ALIASES, k=2)
    candidates = stage1_candidates(found["articles"], {}, COMPANY, ALIASES)
    expected = order_variant(candidates, "O1M5", parse_query(COMPANY, aliases=ALIASES))
    assert [c["key"] for c in result["ranked"]] == [c["key"] for c in expected]


def test_back_to_default_restores_the_o1m5_order_after_the_experimental_order():
    found, texts = _five_mention_case()
    final = rs.final(found, {"texts": texts}, COMPANY, ALIASES, k=2)
    keys = [c["key"] for c in final["ranked"]]
    reviewed = rs.review(final, COMPANY, ALIASES, FakeReviewModel({keys[0]: "not"}), cache=None)
    back = rs.back_to_default(rs.apply_updates(reviewed))
    assert [c["key"] for c in back["ranked"]] == keys and back["ordering"] == "O1M5+S1"
