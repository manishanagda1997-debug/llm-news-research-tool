"""
Tests for src/app_flow.py: what the Streamlit app does, step by step, without a screen or any API.

    get_news      the provisional list is drawn before the full text starts loading; then the loading
                  progress; then the final default list and what changed. A failed search shows its error and
                  no list. The same company and window within FRESH_MINUTES reuses the search (NewsAPI's free
                  plan allows 100 requests a day); another company or window is a new context.
    summary       saved with the fingerprint of its exact selection; after a change of order it belongs to the
                  earlier selection and says so
    review        optional; the order and the selection stay the same
    experimental  the O3 order, labelled; back to default restores the exact original order
    evidence      each card says whether the full text is used, or why only the short text is
    citations     a summary's article numbers point to the matching cards
    about         each stage's result from the project records; a missing record is "pending"
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src import app_flow as af

NOW = datetime(2026, 11, 4, 9, 0, tzinfo=timezone.utc)


def _card(key, title=None):
    return {"key": key, "url": f"https://news.example/{key}", "title": title or f"Story {key}", "source": "S",
            "description": "d", "mention_level": "title", "check": None, "text": ""}


class FakeScreen:
    def __init__(self):
        self.calls = []

    def notice(self, kind, text):
        self.calls.append(("notice", kind, text))

    def provisional(self, result):
        self.calls.append(("provisional", [c["key"] for c in result["ranked"]]))

    def progress(self, done, total):
        self.calls.append(("progress", done, total))

    def final(self, context):
        self.calls.append(("final", [c["key"] for c in context["result"]["ranked"]]))

    def names(self):
        return [c[0] for c in self.calls]


class FakePipeline:
    """Records every step; enrich reports progress per page, like the real one."""

    def __init__(self, found_status="ok"):
        self.steps, self.found_status = [], found_status

    def search(self, company, days):
        self.steps.append(("search", company, days))
        if self.found_status != "ok":
            return {"status": "failed", "error": "NewsAPI did not answer.", "notes": []}
        return {"status": "ok", "company": company, "articles": [{"url": "u1"}, {"url": "u2"}, {"url": "u3"}],
                "counts": {"A": 3, "B": 0}, "window": {"from": "2026-10-28", "days": days},
                "latest_published": "2026-11-03T10:00:00Z", "notes": ["No subsidiary names are known."]}

    def provisional(self, found, company):
        self.steps.append(("provisional",))
        cards = [_card("k1"), _card("k2"), _card("k3")]
        return {"company": company, "k": 2, "ordering": "O1+S1", "ranked": cards, "selected": cards[:2],
                "additional": cards[2:], "state": "provisional", "label": "Provisional"}

    def enrich(self, found, progress):
        self.steps.append(("enrich",))
        for done in range(1, 4):
            progress(done, 3)
        return {"texts": {}, "status": {"https://news.example/k1": "ok"}, "not_in_time": [],
                "not_in_time_titles": [], "seconds": 1.0}

    def final(self, found, enriched, company):
        self.steps.append(("final",))
        cards = [_card("k2"), _card("k1"), _card("k3")]
        return {"company": company, "k": 2, "ordering": "O1+S1", "ranked": cards, "selected": cards[:2],
                "additional": cards[2:], "state": "final", "label": "Default"}

    def changes(self, first, last):
        return {"added": [], "removed": []}

    def summarize(self, result, company):
        self.steps.append(("summarize", tuple(c["key"] for c in result["selected"])))
        return {"summary_status": "ready", "removed": [], "message": "",
                "summary": {"key_points": [{"text": "Tata Motors raises its offer for Iveco.", "citations": [1, 2]}],
                            "industry_context": [{"text": "Truck demand in Europe stays weak.", "citations": [2]}],
                            "article_briefs": [{"id": 1, "brief": "The offer rises to 14.40 euro a share."},
                                               {"id": 2, "brief": "Investors holding 20.4% have not tendered."}],
                            "not_used": []},
                "headlines": [{"url": c["url"], "title": c["title"]} for c in result["selected"]]}

    def review(self, result, company):
        self.steps.append(("review",))
        ranked = [dict(c, check={"label": "not", "reason": "r", "source": "model"}) for c in result["ranked"]]
        by_key = {c["key"]: c for c in ranked}
        return {**result, "ranked": ranked, "selected": [by_key[c["key"]] for c in result["selected"]],
                "additional": [by_key[c["key"]] for c in result["additional"]], "reviewed": True}

    def apply(self, result):
        ranked = list(reversed(result["ranked"]))
        return {**result, "ranked": ranked, "selected": ranked[:2], "additional": ranked[2:], "ordering": "O3+S1",
                "label": "Experimental", "default_order": [c["key"] for c in result["ranked"]]}

    def back(self, result):
        by_key = {c["key"]: c for c in result["ranked"]}
        ranked = [by_key[k] for k in result["default_order"]]
        return {**result, "ranked": ranked, "selected": ranked[:2], "additional": ranked[2:], "ordering": "O1+S1",
                "label": "Default"}


def _search(state=None, company="Tata Motors", days=7, now=NOW, pipeline=None, screen=None):
    state = {} if state is None else state
    pipeline, screen = pipeline or FakePipeline(), screen or FakeScreen()
    return af.get_news(company, days, screen, pipeline, state, now), state, pipeline, screen


# ---------- Get News ----------

def test_the_provisional_list_is_drawn_before_the_full_text_starts_loading():
    context, _, pipeline, screen = _search()
    order = screen.names()
    assert order.index("provisional") < order.index("progress") < order.index("final")
    assert [s[0] for s in pipeline.steps] == ["search", "provisional", "enrich", "final"]
    assert ("progress", 3, 3) in screen.calls
    assert context["result"]["state"] == "final"


def test_a_failed_search_shows_its_error_and_no_list():
    context, state, _, screen = _search(pipeline=FakePipeline(found_status="failed"))
    assert context is None and ("notice", "error", "NewsAPI did not answer.") in screen.calls
    assert "provisional" not in screen.names() and "final" not in screen.names()
    assert af.active(state) is None


def test_an_empty_company_asks_for_a_name_and_searches_nothing():
    context, _, pipeline, screen = _search(company="   ")
    assert context is None and pipeline.steps == [] and screen.calls[0][:2] == ("notice", "warning")


def test_the_search_notes_and_the_untested_company_note_are_shown():
    _, _, _, screen = _search(company="Infosys")
    texts = [c[2] for c in screen.calls if c[0] == "notice"]
    assert "No subsidiary names are known." in texts
    assert any("tested on Tata Motors" in t for t in texts)
    _, _, _, screen = _search(company="tata motors")
    assert not any("tested on Tata Motors" in c[2] for c in screen.calls if c[0] == "notice")


def test_the_same_search_within_the_fresh_time_reuses_the_results():
    first, state, pipeline, _ = _search()
    again, _, _, screen = _search(state=state, company=" TATA motors ", now=NOW + timedelta(minutes=10),
                                  pipeline=pipeline)
    assert again is first and [s[0] for s in pipeline.steps].count("search") == 1
    assert screen.names() == ["notice", "final"]


def test_an_old_search_another_company_or_another_window_is_a_new_context():
    first, state, pipeline, _ = _search()
    later, _, _, _ = _search(state=state, now=NOW + timedelta(minutes=af.FRESH_MINUTES + 1), pipeline=pipeline)
    other, _, _, _ = _search(state=state, company="Reliance Industries", pipeline=pipeline)
    window, _, _, _ = _search(state=state, days=3, pipeline=pipeline)
    assert len({id(first), id(later), id(other), id(window)}) == 4
    assert af.active(state) is window and window["summary"] is None


# ---------- the summary belongs to its selection ----------

def test_a_summary_belongs_to_its_exact_selection():
    context, _, pipeline, _ = _search()
    af.make_summary(context, pipeline)
    assert af.summary_view(context)["state"] == "current"
    af.run_review(context, pipeline)
    af.experimental(context, pipeline)
    view = af.summary_view(context)
    assert view["state"] == "earlier_selection" and view["outcome"]["summary_status"] == "ready"
    af.default(context, pipeline)
    assert af.summary_view(context)["state"] == "current"


def test_no_summary_is_none():
    context, _, _, _ = _search()
    assert af.summary_view(context) == {"state": "none", "outcome": None}


def test_the_fingerprint_depends_on_the_ordering_and_the_selected_articles():
    a = {"ordering": "O1+S1", "selected": [_card("k1"), _card("k2")]}
    assert af.selection_fingerprint(a) == af.selection_fingerprint(dict(a))
    assert af.selection_fingerprint(a) != af.selection_fingerprint({**a, "selected": [_card("k2"), _card("k1")]})
    assert af.selection_fingerprint(a) != af.selection_fingerprint({**a, "ordering": "O3+S1"})


# ---------- review and the experimental order ----------

def test_the_review_keeps_the_order_and_the_summary():
    context, _, pipeline, _ = _search()
    af.make_summary(context, pipeline)
    before = [c["key"] for c in context["result"]["ranked"]]
    af.run_review(context, pipeline)
    assert [c["key"] for c in context["result"]["ranked"]] == before and context["result"]["reviewed"]
    assert af.summary_view(context)["state"] == "current"


def test_the_experimental_order_needs_the_review_and_back_to_default_restores_the_exact_order():
    context, _, pipeline, _ = _search()
    with pytest.raises(af.FlowError):
        af.experimental(context, pipeline)
    original = [c["key"] for c in context["result"]["ranked"]]
    af.run_review(context, pipeline)
    af.experimental(context, pipeline)
    assert context["result"]["ordering"] == "O3+S1"
    af.default(context, pipeline)
    assert [c["key"] for c in context["result"]["ranked"]] == original and context["result"]["ordering"] == "O1+S1"


# ---------- evidence and citations ----------

@pytest.mark.parametrize("text, status, kind, reason", [
    ("full text", "ok", "full_text", None),
    ("", "ok", "short_text", "the page's text did not pass the check"),
    ("", "ok_teaser", "short_text", "the page shows only a teaser of the article"),
    ("", "ok_wrong_page", "short_text", "the link leads to a page that does not hold this article"),
    ("", "not_fetched_in_time", "short_text", "the page was not loaded within the time limit"),
    ("", "blocked_by_robots", "short_text", "the publisher does not allow automatic reading"),
    ("", "http_error", "short_text", "the page could not be loaded"),
    ("", None, "short_text", "the page could not be loaded"),
])
def test_each_card_says_which_evidence_it_has(text, status, kind, reason):
    card = dict(_card("k1"), text=text)
    reasons = {"ok_teaser": ["teaser"], "ok_wrong_page": ["description_missing", "title_low"]}.get(status)
    status = "ok" if reasons else status
    enriched = {"status": {card["url"]: status} if status else {},
                "check_reasons": {card["url"]: reasons} if reasons else {}}
    assert af.evidence(card, enriched) == {"kind": kind, "reason": reason}


def test_citations_point_to_the_matching_cards():
    context, _, pipeline, _ = _search()
    af.make_summary(context, pipeline)
    outcome = context["summary"]["outcome"]
    assert af.citation_cards(outcome, context["result"]) == {1: "k2", 2: "k1"}


def test_citations_are_not_linked_when_the_summary_belongs_to_another_selection():
    context, _, pipeline, _ = _search()
    af.make_summary(context, pipeline)
    outcome = context["summary"]["outcome"]
    af.run_review(context, pipeline)
    af.experimental(context, pipeline)
    assert af.citation_cards(outcome, context["result"]) == {}


# ---------- the about page ----------

def test_a_missing_record_is_pending_and_a_present_one_is_read(tmp_path):
    (tmp_path / "fresh_2026-10-24_30").mkdir()
    (tmp_path / "fresh_2026-10-24_30" / "fresh_ranking_record.json").write_text(
        '{"default": "O1+S1"}', encoding="utf-8")
    stages = {s["stage"]: s for s in af.about_status(tmp_path)}
    assert stages["ranking"]["status"] == "done" and "O1+S1" in stages["ranking"]["result"]
    assert stages["summary"]["status"] == "pending" and stages["transfer"]["status"] == "pending"


def test_the_transfer_test_is_pending_until_its_summary_is_scored(tmp_path):
    folder = tmp_path / "transfer_reliance_2026-10-26_11-01"
    folder.mkdir()
    (folder / "transfer_score.json").write_text('{"summary": {"status": "pending"}, "ranking": {"passed": true}}',
                                                encoding="utf-8")
    assert {s["stage"]: s for s in af.about_status(tmp_path)}["transfer"]["status"] == "pending"


def test_cards_get_a_clear_date_from_the_search_results():
    found = {"articles": [{"url": "u1", "publishedAt": "2026-11-04T10:05:00Z"}, {"url": "u2", "publishedAt": None},
                          {"url": "u3", "publishedAt": "not a date"}]}
    assert af.published_dates(found) == {"u1": "4 Nov 2026, 10:05 UTC"}


def test_dates_are_readable_and_a_missing_date_is_none():
    assert af.readable_date("2026-11-04T10:05:00Z") == "4 Nov 2026, 10:05 UTC"
    assert af.readable_date(None) is None and af.readable_date("soon") is None



# ---------- what the summary section shows ----------

def _outcome(status="ready", key_points=True, not_used=()):
    summary = {"key_points": [{"text": "A finding.", "citations": [1]}] if key_points else [],
               "industry_context": [], "article_briefs": [], "not_used": list(not_used)}
    return {"summary_status": status, "summary": summary,
            "headlines": [{"url": "https://news.example/k1", "title": "First story"},
                          {"url": "https://news.example/k2", "title": "Second story"}]}


@pytest.mark.parametrize("status, key_points, label, kind", [
    ("ready", True, "Summary ready · Automated checks passed", "success"),
    ("ready_with_removed_points", True, "Summary ready · Automated checks passed after removing some points",
     "warning"),
    ("ready", False, "No summary: no key finding passes the checks", "error"),
    ("ready_with_removed_points", False, "No summary: no key finding passes the checks", "error"),
    ("model_unavailable", True, "No summary: the model is not available right now", "error"),
    ("failed_checks", True, "No summary: the answer does not pass the checks", "error"),
])
def test_ready_is_shown_only_for_a_summary_with_key_findings(status, key_points, label, kind):
    assert af.summary_status_line(_outcome(status, key_points)) == (label, kind)


def test_articles_not_used_are_listed_with_their_titles_and_reasons():
    outcome = _outcome(not_used=[{"id": 2, "reason": "About another company."}])
    assert af.not_used(outcome) == [{"number": 2, "title": "Second story", "reason": "About another company."}]


def test_the_overview_counts_come_from_the_result():
    context, _, _, _ = _search()
    context["result"]["ranked"][0]["text"] = "full text"
    assert af.overview(context) == {"found": 3, "kept": 3, "selected": 2, "full_text": 1, "short_text": 2}


def test_the_window_shows_its_real_dates():
    found = {"window": {"from": "2026-10-28T09:00:00", "days": 7}}
    assert af.window_text(found, NOW) == "28 Oct to 4 Nov 2026"
    assert af.window_text({"window": None}, NOW) == "dates not given"



def test_a_note_says_when_the_summary_rests_on_short_descriptions():
    context, _, _, _ = _search()
    assert af.evidence_note(context) == ("This summary relies on short article descriptions: the full text of the "
                                         "selected articles is not available.")
    context["result"]["selected"][0]["text"] = "full text"
    assert af.evidence_note(context) == "1 of the 2 selected articles relies on its short description only."
    context["result"]["selected"][1]["text"] = "full text"
    assert af.evidence_note(context) is None



# ---------- text from the model and from publishers is shown as written ----------

@pytest.mark.parametrize("raw, shown", [
    ("valued at $4.4 billion, up from $3.8 billion", "valued at \\$4.4 billion, up from \\$3.8 billion"),
    ("*Exclusive*: Tata_Motors [update]", "\\*Exclusive\\*: Tata\\_Motors \\[update\\]"),
    ("plain text", "plain text"),
])
def test_markdown_characters_are_escaped(raw, shown):
    assert af.md(raw) == shown


def test_the_same_search_can_be_forced_to_run_again():
    first, state, pipeline, _ = _search()
    again, _, _, _ = _search(state=state, now=NOW + timedelta(minutes=5), pipeline=pipeline)
    assert again is first
    fresh = af.get_news("Tata Motors", 7, FakeScreen(), pipeline, state, NOW + timedelta(minutes=6), refresh=True)
    assert fresh is not first and [s[0] for s in pipeline.steps].count("search") == 2


def test_the_real_pipeline_makes_each_model_once_per_session(monkeypatch):
    made = []
    pipeline = af.Pipeline.__new__(af.Pipeline)
    monkeypatch.setattr(af, "_summary_model", lambda: made.append("summary") or object())
    monkeypatch.setattr(af, "_review_model", lambda: made.append("review") or object())
    assert pipeline.summary_model() is pipeline.summary_model()
    assert pipeline.review_model() is pipeline.review_model()
    assert made == ["summary", "review"]



# ---------- slow model steps run in view, with a visible message ----------

def test_a_requested_model_step_runs_once_and_is_then_cleared():
    context, _, pipeline, _ = _search()
    state = {"pending": "summary"}
    assert af.take_pending(state) == "summary" and af.take_pending(state) is None


def test_the_more_articles_are_shown_ten_at_a_time():
    assert af.pages(30, 10) == [(0, 10), (10, 20), (20, 30)]
    assert af.pages(23, 10) == [(0, 10), (10, 20), (20, 23)]
    assert af.pages(4, 10) == [(0, 4)]
    assert af.pages(0, 10) == []



def test_the_free_plan_limits_are_stated():
    assert "about one day after publication" in af.SOURCE_NOTE and "one month" in af.SOURCE_NOTE
    assert af.DAY_CHOICES == (3, 7, 14, 28) and af.DEFAULT_DAYS == 7


def test_every_offered_day_choice_is_accepted_by_the_news_fetcher():
    """The app never offers a period the fetcher refuses (the free plan allows at most MAX_DAYS_BACK days)."""
    from src.news_fetcher import MAX_DAYS_BACK, _check_inputs
    assert max(af.DAY_CHOICES) <= MAX_DAYS_BACK and af.DEFAULT_DAYS in af.DAY_CHOICES
    for days in af.DAY_CHOICES:
        _check_inputs("Tata Motors", 30, days, "relevancy")                     # raises on a refused value



# ---------- aggregator links ----------

STUB = ("Tata Motors sales rise 23% to 75,426 in October\n"
        "breakingthenews.net/Article/Tata-Motors-sales-rise-23-to-75426-in-October/67451651\n"
        "Tata Motors Passenger Vehicles Limited announced on Sunday a 23% increase in total sales in October.")


def test_an_aggregator_entry_gives_the_original_articles_address():
    assert af.original_link("https://biztoc.com/x/46321cc87434187e", STUB) == \
        "https://breakingthenews.net/Article/Tata-Motors-sales-rise-23-to-75426-in-October/67451651"


@pytest.mark.parametrize("url, text", [
    ("https://biztoc.com/x/1", "Headline only\nno address on this line\nsnippet"),
    ("https://biztoc.com/x/1", ""),
    ("https://www.business-standard.com/a", STUB),          # only aggregator pages are read this way
])
def test_no_original_address_is_made_up(url, text):
    assert af.original_link(url, text) is None


def test_the_final_list_carries_the_original_addresses_of_aggregator_cards():
    context, _, _, _ = _search()
    context["enriched"]["texts"] = {"https://biztoc.com/x/46321cc87434187e": STUB}
    card = dict(_card("k9"), url="https://biztoc.com/x/46321cc87434187e")
    assert af.card_link(card, context["enriched"]) == {
        "url": "https://breakingthenews.net/Article/Tata-Motors-sales-rise-23-to-75426-in-October/67451651",
        "via": "Biztoc", "note": None}
    plain = _card("k1")
    assert af.card_link(plain, context["enriched"]) == {"url": plain["url"], "via": None, "note": None}
    lost = dict(_card("k8"), url="https://biztoc.com/x/0000")
    assert af.card_link(lost, context["enriched"]) == {
        "url": "https://biztoc.com/x/0000", "via": "Biztoc",
        "note": "Biztoc is an aggregator; its link may not open the article."}


@pytest.mark.parametrize("url", ["javascript:alert(1)", " JavaScript:alert(1)", "data:text/html,x", "ftp://a/b",
                                 "/relative/path", "", None])
def test_only_an_http_or_https_address_is_ever_linked(url):
    context, _, _, _ = _search()
    card = dict(_card("k1"), url=url)
    assert af.card_link(card, context["enriched"]) == {
        "url": None, "via": None, "note": "The article's address is not a web link, so the title is not linked."}
    assert af.web_link("https://a.com/x") and af.web_link("HTTP://a.com/x")


# ---------- an empty industry section ----------

def _industry_outcome(points, removed=()):
    return {"summary_status": "ready", "removed": list(removed),
            "summary": {"key_points": [{"text": "a", "citations": [1]}], "industry_context": points}}


def test_an_empty_industry_section_says_the_articles_do_not_report_on_the_wider_industry():
    assert af.NO_INDUSTRY == "The selected articles do not report on the wider industry."
    assert af.industry_note(_industry_outcome([])) == af.NO_INDUSTRY


def test_no_note_when_the_section_has_points():
    assert af.industry_note(_industry_outcome([{"text": "Truck demand in Europe stays weak.", "citations": [2]}])) == ""


def test_no_note_when_the_checks_removed_the_industry_points():
    """Then the articles may report on the industry; the removed points are listed with their reason instead."""
    removed = [{"section": "industry_context", "text": "Truck demand fell 9%.", "reason": "numbers"}]
    assert af.industry_note(_industry_outcome([], removed)) == ""
    brief_only = [{"section": "article_briefs", "id": 1, "text": "x", "reason": "numbers"}]
    assert af.industry_note(_industry_outcome([], brief_only)) == af.NO_INDUSTRY


def test_the_ranking_update_line_reads_the_round_1_record(tmp_path):
    assert {s["stage"]: s for s in af.about_status(tmp_path)}["ranking_update"]["status"] == "pending"
    (tmp_path / "ranking_variants_dev_record.json").write_text('{"choice": {"chosen": "O1M5"}}', encoding="utf-8")
    line = {s["stage"]: s for s in af.about_status(tmp_path)}["ranking_update"]
    assert line["status"] == "done" and "Default O1M5+S1" in line["result"]
    assert line["result"].endswith("The fresh-week test is pending.")          # no week 2 record: pending, not old text


def test_the_transfer_line_says_which_ranking_condition_is_not_met(tmp_path):
    folder = tmp_path / "transfer_reliance_2026-10-26_11-01"
    folder.mkdir()
    score = {"summary": {"status": "scored", "passed": True},
             "ranking": {"passed": False, "conditions": {"no_direct_event_lost": False, "event_recall_not_lower": True,
                                                         "precision_not_lower": True}}}
    (folder / "transfer_score.json").write_text(json.dumps(score), encoding="utf-8")
    line = {s["stage"]: s for s in af.about_status(tmp_path)}["transfer"]["result"]
    assert line == "Ranking: 2 of 3 conditions met (a direct event is lost); summary pass levels met."


def test_the_about_lines_match_the_saved_records():
    from src.config import EVAL_DIR
    lines = {s["stage"]: s["result"] for s in af.about_status(EVAL_DIR)}
    chosen = json.loads((EVAL_DIR / "ranking_variants_dev_record.json").read_text(encoding="utf-8"))["choice"]["chosen"]
    from src import research as rs
    assert f"Default {chosen}+S1" in lines["ranking_update"] and f"{chosen}+S1" == rs.DEFAULT_ORDERING


# ---------- the about page: round 3 and the two fresh weeks ----------

def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _week2_ranking(second_passed, primary_passed):
    return {"second": {"passed": second_passed}, "primary": {"passed": primary_passed}}


def _week2_score(passed, covered, total):
    return {"summary": {"status": "scored", "passed": passed,
                        "checks": {"produced": True, "event_coverage": covered / total >= 0.8},
                        "values": {"events": {"covered": covered, "total": total}}}}


def test_the_ranking_update_line_reports_both_fresh_weeks_from_their_records(tmp_path):
    _write(tmp_path / "ranking_variants_dev_record.json", {"choice": {"chosen": "O1M5", "passing": ["O1M5"]}})
    folders = list(af.WEEK2_FOLDERS.values())
    _write(tmp_path / folders[0] / "week2_ranking.json", _week2_ranking(True, False))
    line = {s["stage"]: s for s in af.about_status(tmp_path)}["ranking_update"]["result"]
    assert line.endswith("The fresh-week test is pending.")                    # one week missing: still pending
    _write(tmp_path / folders[1] / "week2_ranking.json", _week2_ranking(True, False))
    line = {s["stage"]: s for s in af.about_status(tmp_path)}["ranking_update"]["result"]
    assert line == ("Default O1M5+S1: the only variant that passes on all three development days. On two fresh weeks "
                    "it beats O1+S1 both times and fails against the retrieval order both times.")
    _write(tmp_path / folders[1] / "week2_ranking.json", _week2_ranking(False, True))
    line = {s["stage"]: s for s in af.about_status(tmp_path)}["ranking_update"]["result"]
    assert "beats O1+S1 in 1 of 2 weeks and fails against the retrieval order in 1 of 2 weeks." in line


def test_the_summary_design_line_says_whether_round_3_is_scored(tmp_path):
    freeze = {"design": "direct", "ks": [5], "settings": {"prompt_version": "4"}}
    _write(tmp_path / "summary_freeze.json", freeze)
    line = {s["stage"]: s for s in af.about_status(tmp_path)}["summary"]["result"]
    assert line == ("Direct summary design frozen for k = 5; the three-run comparison of the prompts (Round 3) "
                    "is pending.")
    _write(tmp_path / "summary_v2_r3_judging_score.json", {"choice": {"prompt": "4"}})
    line = {s["stage"]: s for s in af.about_status(tmp_path)}["summary"]["result"]
    assert line == ("Direct summary design frozen for k = 5; Prompt 4 kept after the three-run comparison with "
                    "Prompt 6 (Round 3).")


def test_the_week2_line_is_pending_until_both_weeks_are_scored_and_kept_apart_from_ranking(tmp_path):
    folders = list(af.WEEK2_FOLDERS.values())
    line = {s["stage"]: s for s in af.about_status(tmp_path)}["week2"]
    assert line["status"] == "pending" and line["result"] == "Not scored yet."
    _write(tmp_path / folders[0] / "week2_summary_score.json", _week2_score(False, 3, 4))
    assert {s["stage"]: s for s in af.about_status(tmp_path)}["week2"]["status"] == "pending"
    _write(tmp_path / folders[1] / "week2_summary_score.json", _week2_score(True, 2, 2))
    line = {s["stage"]: s for s in af.about_status(tmp_path)}["week2"]
    assert line["status"] == "done"
    assert line["result"] == "Summary test: Tata Motors not passed (events 3 of 4); Reliance Industries passed."
    assert "O1" not in line["result"]                                         # the ranking result is a separate line


def test_the_new_about_lines_match_the_saved_records():
    from src.config import EVAL_DIR
    lines = {s["stage"]: s for s in af.about_status(EVAL_DIR)}
    assert lines["ranking_update"]["result"] == (
        "Default O1M5+S1: the only variant that passes on all three development days. On two fresh weeks it beats "
        "O1+S1 both times and fails against the retrieval order both times.")
    assert lines["summary"]["result"] == ("Direct summary design frozen for k = 5; Prompt 4 kept after the three-run "
                                          "comparison with Prompt 6 (Round 3).")
    assert lines["week2"]["status"] == "done"
    assert lines["week2"]["result"] == ("Summary test: Tata Motors not passed (events 3 of 4); "
                                        "Reliance Industries passed.")


def test_the_stage_3_line_states_the_earlier_choice_and_the_next_line_the_default():
    from src.config import EVAL_DIR
    lines = {s["stage"]: s["result"] for s in af.about_status(EVAL_DIR)}
    assert lines["ranking"] == ("O1+S1 chosen and confirmed on a fresh week labelled before any model ran; "
                                "model-based orders remain experimental.")
    assert lines["ranking_update"].startswith("Default O1M5+S1")
