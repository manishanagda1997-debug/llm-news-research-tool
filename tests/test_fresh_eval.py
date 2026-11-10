"""
Tests for the fresh evaluation (fake fetch, fake git, no network, no model calls).

One fresh run, with a stable id, in its own folder. Its stages are recorded,
so a stopped run resumes without new requests and never mixes two fetches.

fetch    - searches A and B for the window 2026-10-24T00:00:00Z to
           2026-10-30T23:59:59Z, refused before 2026-11-01 (UTC), because
           the free plan shows articles a day late. Request settings are
           saved without keys or secrets.
overlap  - exact overlaps with development data (same tidy link) and
           articles published outside the window are excluded and logged.
           Near-identical titles are not excluded automatically: they go to
           a review sheet, where `same` means the same article or a repost,
           not just the same news event.
validate - before scoring: every candidate has a valid label, every relevant
           candidate has an event, labels and events match the run's
           articles, the frozen threshold is present, no confirmed
           development article is left. All problems are reported together.
freeze   - the frozen implementation (code, tests, requirements, aliases,
           guides, the development ranking record, the run's labels, events
           and protocol) must equal the freeze tag; new raw files, saved
           model answers, the run state and evaluation outputs are allowed.
"""

import json
from datetime import datetime, timezone

import pytest

from src import fresh_eval as fe


def _raw(n, title, published="2026-10-26T08:00:00Z", url=None):
    return {"source": {"id": None, "name": f"Source {n}"}, "author": "A", "title": title,
            "description": f"Description {n}.", "content": "", "url": url or f"https://fresh.com/{n}",
            "urlToImage": None, "publishedAt": published}


AFTER = datetime(2026, 11, 1, 6, 0, tzinfo=timezone.utc)


class FakeFetch:
    def __init__(self):
        self.calls = []

    def __call__(self, company, **options):
        self.calls.append(options)
        part = options.get("part", "all") if options.get("aliases") else "none"
        article = _raw(len(self.calls), f"{part} article about Tata Motors")
        return {"status": "ok", "totalResults": 1, "articles": [article],
                "request": {"query": company, "search_part": part, "from_date": options["from_time"],
                            "to_date": options["to_time"], "apiKey": "SECRET", "Authorization": "Bearer X"}}


def _saver(folder):
    def save(response):
        path = folder / f"raw_{len(list(folder.glob('raw_*.json')))}.json"
        path.write_text(json.dumps(response), encoding="utf-8")
        return path
    return save


# ---------- the run and its window ----------

def test_the_run_has_a_fixed_id_window_and_searches():
    assert fe.RUN_ID == "fresh_2026-10-24_30"
    assert (fe.WINDOW_FROM, fe.WINDOW_TO) == ("2026-10-24T00:00:00Z", "2026-10-30T23:59:59Z")
    assert fe.FETCH_NOT_BEFORE == datetime(2026, 11, 1, 6, 0, tzinfo=timezone.utc)   # 6 hours after the delay
    assert fe.SEARCHES == ("A", "B")


def test_fetching_before_the_window_is_complete_is_refused(tmp_path):
    fetch = FakeFetch()
    with pytest.raises(fe.FreshEvalError):
        fe.fetch_stage(fetch, now=datetime(2026, 10, 31, 23, 0, tzinfo=timezone.utc), run_dir=tmp_path,
                       aliases={"x": 1}, save=_saver(tmp_path))
    assert fetch.calls == []


def test_fetch_makes_searches_a_and_b_with_the_window(tmp_path):
    fetch = FakeFetch()
    state = fe.fetch_stage(fetch, now=AFTER, run_dir=tmp_path, aliases={"x": 1}, save=_saver(tmp_path))
    assert len(fetch.calls) == 2
    assert fetch.calls[0].get("aliases") is None                               # A: the name only
    assert fetch.calls[1]["part"] == "subsidiaries"                            # B
    assert all(c["from_time"] == fe.WINDOW_FROM and c["to_time"] == fe.WINDOW_TO for c in fetch.calls)
    assert state["stages"]["fetch"]["done"] is True and set(state["stages"]["fetch"]["raw_files"]) == {"A", "B"}


def test_saved_request_settings_hold_no_secrets(tmp_path):
    fe.fetch_stage(FakeFetch(), now=AFTER, run_dir=tmp_path, aliases={"x": 1}, save=_saver(tmp_path))
    saved = (tmp_path / fe.STATE_FILE).read_text(encoding="utf-8")
    assert "SECRET" not in saved and "Bearer" not in saved
    assert "apiKey" not in saved and "Authorization" not in saved


def test_a_resumed_run_reuses_the_first_fetch(tmp_path):
    fe.fetch_stage(FakeFetch(), now=AFTER, run_dir=tmp_path, aliases={"x": 1}, save=_saver(tmp_path))
    again = FakeFetch()
    state = fe.fetch_stage(again, now=AFTER, run_dir=tmp_path, aliases={"x": 1}, save=_saver(tmp_path))
    assert again.calls == [] and state["stages"]["fetch"]["done"] is True


def test_a_half_finished_fetch_is_not_mixed_with_a_new_one(tmp_path):
    (tmp_path / fe.STATE_FILE).write_text(json.dumps(
        {"run_id": fe.RUN_ID, "stages": {"fetch": {"done": False, "raw_files": {"A": "raw_0.json"}}}}),
        encoding="utf-8")
    with pytest.raises(fe.FreshEvalError):
        fe.fetch_stage(FakeFetch(), now=AFTER, run_dir=tmp_path, aliases={"x": 1}, save=_saver(tmp_path))


def test_a_fetch_that_stopped_before_any_response_starts_again(tmp_path):
    (tmp_path / fe.STATE_FILE).write_text(json.dumps(
        {"run_id": fe.RUN_ID, "stages": {"fetch": {"done": False, "raw_files": {}, "requests": {}}}}),
        encoding="utf-8")
    fetch = FakeFetch()
    state = fe.fetch_stage(fetch, now=AFTER, run_dir=tmp_path, aliases={"x": 1}, save=_saver(tmp_path))
    assert len(fetch.calls) == 2 and state["stages"]["fetch"]["done"] is True


def test_safe_request_drops_secret_fields():
    assert fe.safe_request({"query": "Tata Motors", "apiKey": "x", "api_key": "y", "Authorization": "z",
                            "token": "t", "page_size": 30}) == {"query": "Tata Motors", "page_size": 30}


# ---------- overlap with development data ----------

DEV = [_raw(90, "Tata Motors raises Iveco takeover offer to 14.40 euro per share",
            published="2026-10-20T08:00:00Z", url="https://dev.com/iveco")]


def test_an_exact_link_overlap_is_excluded_and_logged():
    fresh = [_raw(1, "Some other headline", url="https://www.dev.com/iveco/?utm_source=x")]
    result = fe.overlap(fresh, DEV)
    assert result["kept"] == [] and result["excluded"][0]["reason"] == "development_link"


@pytest.mark.parametrize("published, reason", [
    ("2026-10-23T23:59:59Z", "outside_window"), ("2026-10-31T00:00:00Z", "outside_window"), ("", "no_date")])
def test_articles_outside_the_window_are_excluded_and_logged(published, reason):
    result = fe.overlap([_raw(1, "Tata Motors news", published=published)], DEV)
    assert result["kept"] == [] and result["excluded"][0]["reason"] == reason


@pytest.mark.parametrize("published", ["2026-10-24T00:00:00Z", "2026-10-30T23:59:59Z"])
def test_the_window_edges_are_inside(published):
    assert len(fe.overlap([_raw(1, "Tata Motors news", published=published)], DEV)["kept"]) == 1


def test_a_near_identical_title_goes_to_review_not_out():
    fresh = [_raw(1, "Tata Motors raises Iveco takeover offer to 14.40 euro per share - Reuters")]
    result = fe.overlap(fresh, DEV)
    assert len(result["kept"]) == 1 and result["excluded"] == []
    assert result["review"][0]["fresh_url"] == "https://fresh.com/1"
    assert result["review"][0]["dev_url"] == "https://dev.com/iveco"


def test_the_same_event_with_a_different_headline_is_neither_excluded_nor_reviewed():
    fresh = [_raw(1, "Iveco shareholders weigh Tata's sweetened bid as deadline nears")]
    result = fe.overlap(fresh, DEV)
    assert len(result["kept"]) == 1 and result["review"] == []


def test_review_decisions_exclude_only_confirmed_reposts():
    fresh = [_raw(1, "Tata Motors raises Iveco takeover offer to 14.40 euro per share - Reuters"),
             _raw(2, "Tata Motors news item")]
    result = fe.overlap(fresh, DEV)
    decided = fe.apply_review(result, [{"fresh_url": "https://fresh.com/1", "decision": "same"}])
    assert [a["url"] for a in decided["kept"]] == ["https://fresh.com/2"]
    assert decided["excluded"][-1]["reason"] == "confirmed_repost"


@pytest.mark.parametrize("decision", ["", "maybe", "same event"])
def test_a_missing_or_invalid_review_decision_stops(decision):
    result = fe.overlap([_raw(1, "Tata Motors raises Iveco takeover offer to 14.40 euro per share")], DEV)
    with pytest.raises(fe.FreshEvalError):
        fe.apply_review(result, [{"fresh_url": "https://fresh.com/1", "decision": decision}])


def test_every_reviewed_pair_needs_a_decision():
    result = fe.overlap([_raw(1, "Tata Motors raises Iveco takeover offer to 14.40 euro per share")], DEV)
    with pytest.raises(fe.FreshEvalError):
        fe.apply_review(result, [])


# ---------- validation before scoring ----------

CANDS = [{"key": "A1", "url": "https://fresh.com/1"}, {"key": "A2", "url": "https://fresh.com/2"},
         {"key": "A3", "url": "https://fresh.com/3"}]
GOOD = {"labels": {"A1": "direct", "A2": "not", "A3": "indirect"}, "events": {"A1": "e1", "A3": "e2"},
        "article_ids": {"A1", "A2", "A3"}, "threshold": 0.5, "dev_links": {"https://dev.com/iveco"}}


def test_valid_inputs_pass():
    assert fe.validation_problems(CANDS, **GOOD) == []


@pytest.mark.parametrize("change, problem", [
    ({"labels": {"A1": "direct", "A3": "indirect"}}, "no label"),
    ({"labels": {"A1": "direct", "A2": "maybe", "A3": "indirect"}}, "invalid label"),
    ({"events": {"A1": "e1"}}, "no event"),
    ({"article_ids": {"A1", "A2"}}, "not in the run"),
    ({"threshold": None}, "threshold"),
    ({"dev_links": {"https://fresh.com/2"}}, "development"),
])
def test_each_problem_is_reported(change, problem):
    problems = fe.validation_problems(CANDS, **{**GOOD, **change})
    assert any(problem in p for p in problems)


def test_labels_for_articles_outside_the_run_are_a_problem():
    problems = fe.validation_problems(CANDS, **{**GOOD, "labels": {**GOOD["labels"], "A9": "direct"}})
    assert any("A9" in p for p in problems)


def test_all_problems_are_reported_together():
    problems = fe.validation_problems(CANDS, **{**GOOD, "events": {}, "threshold": None})
    assert len(problems) >= 3


# ---------- the freeze ----------

class FakeGit:
    def __init__(self, changed=(), untracked=(), tag=True, head="abc123", tag_commit="def456"):
        self.changed, self.untracked, self.tag, self.head, self.tag_commit = changed, untracked, tag, head, tag_commit

    def __call__(self, *args):
        if args[:2] == ("rev-parse", "--verify"):
            if not self.tag:
                raise fe.FreshEvalError("no tag")
            return self.tag_commit
        if args == ("rev-parse", "HEAD"):
            return self.head
        if args[0] == "diff":
            return "\n".join(self.changed)
        if args[0] == "ls-files":
            return "\n".join(self.untracked)
        raise AssertionError(args)


def test_new_data_files_do_not_break_the_freeze():
    git = FakeGit(changed=["data/eval/relevance_checks/abc.json"],
                  untracked=["data/raw/20261101_tata-motors.json", "data/eval/fresh_2026-10-24_30/state.json",
                             "data/eval/fresh_2026-10-24_30/fresh_ranking_record.json"])
    freeze = fe.freeze_check(git)
    assert freeze["ok"] is True and freeze["problems"] == []
    assert freeze["head"] == "abc123" and freeze["tag_commit"] == "def456"


@pytest.mark.parametrize("path", ["src/ranking.py", "tests/test_ranking.py", "requirements.txt",
                                  "data/company_aliases.json", "data/eval/LABELLING_GUIDE.md",
                                  "data/eval/SUMMARY_JUDGING_GUIDE.md", "data/eval/ranking_dev_record.json",
                                  "pytest.ini", "data/eval/fresh_2026-10-24_30/labels.csv",
                                  "data/eval/fresh_2026-10-24_30/events.csv",
                                  "data/eval/fresh_2026-10-24_30/fresh_protocol.json"])
def test_a_changed_frozen_file_breaks_the_freeze(path):
    freeze = fe.freeze_check(FakeGit(changed=[path]))
    assert freeze["ok"] is False and path in freeze["problems"]


def test_a_new_file_in_the_code_breaks_the_freeze():
    freeze = fe.freeze_check(FakeGit(untracked=["src/new_helper.py"]))
    assert freeze["ok"] is False


def test_a_missing_tag_breaks_the_freeze():
    freeze = fe.freeze_check(FakeGit(tag=False))
    assert freeze["ok"] is False and "freeze-fresh-eval" in " ".join(freeze["problems"])


def test_the_record_lists_package_versions():
    versions = fe.package_versions()
    assert {"streamlit", "langchain-core", "langchain-google-genai", "trafilatura", "pandas"} <= set(versions)
    assert {"langchain-groq", "groq"} <= set(versions)                  # the provider of the model run



def test_a_fetch_at_0559_utc_on_1_november_is_still_refused(tmp_path):
    with pytest.raises(fe.FreshEvalError):
        fe.fetch_stage(FakeFetch(), datetime(2026, 11, 1, 5, 59, tzinfo=timezone.utc), tmp_path, None,
                       lambda r: tmp_path / "x.json", seen_files=[])


def test_articles_are_counted_for_every_day_of_the_window_including_empty_days():
    articles = [{"publishedAt": "2026-10-24T08:00:00Z"}, {"publishedAt": "2026-10-24T22:00:00Z"},
                {"publishedAt": "2026-10-29T01:00:00Z"}, {"publishedAt": "bad"}]
    counts = fe.day_counts(articles)
    assert list(counts) == [f"2026-10-{d}" for d in range(24, 31)]
    assert counts["2026-10-24"] == 2 and counts["2026-10-29"] == 1 and counts["2026-10-30"] == 0


# ---------- the label sheet: the text shown is the text the tool uses ----------

def _article(n, title, description="", content=""):
    return {"url": f"https://fresh.com/{n}", "title": title, "description": description, "content": content,
            "source": {"name": f"Source {n}"}, "publishedAt": "2026-10-26T08:00:00Z"}


LONG = "\n".join(["Tata Motors completes the Iveco takeover after shareholders tender 89.42% of the shares."] * 6)


def test_the_sheet_shows_the_short_text_for_a_teaser_a_wrong_page_and_a_blocked_page():
    full = _article(1, "Tata Motors completes the Iveco takeover", "Tata Motors completes the Iveco takeover.")
    teaser_desc = "Most Iveco shareholders tendered their shares for the Tata Motors takeover offer of Iveco."
    teaser = _article(2, "Tata Motors clears Iveco hurdle", teaser_desc, teaser_desc + " More at wsj.com today.")
    blocked = _article(3, "Tata Motors bid crosses 80%", "Acceptances reach 87.81% for the Iveco offer.")
    results = {full["url"]: {"status": "ok", "text": LONG, "html_cleaning": "done"},
               teaser["url"]: {"status": "ok", "text": "Tata Motors clears Iveco hurdle\nwsj.com", "html_cleaning": "done"},
               blocked["url"]: {"status": "blocked_by_robots"}}
    sheet = fe.build_sheet([full, teaser, blocked], results, "Tata Motors", aliases=None)
    status = {item["url"]: item["status"] for item in sheet["items"]}
    assert status[full["url"]] == "complete"
    assert status[teaser["url"]] == fe.STATUS_TEASER
    assert status[blocked["url"]] == fe.STATUS_NOT_FETCHED
    evidence = sheet["evidence"]
    assert evidence == {fe.article_id(full["url"]): "full text", fe.article_id(teaser["url"]): "short text: teaser",
                        fe.article_id(blocked["url"]): "short text: page not fetched"}


def test_the_sheet_shows_the_short_text_for_a_page_that_holds_another_story():
    page = "\n".join(["Market Summary", "Gold slips as dollar firms", "Oil majors rally in early trade",
                      "Asian shares mixed ahead of data"] * 5)
    a = _article(1, "Jaguar unveils a new electric grand tourer in New York",
                 "Jaguar has unveiled its new all-electric four-door grand tourer at an event in New York.")
    sheet = fe.build_sheet([a], {a["url"]: {"status": "ok", "text": page, "html_cleaning": "done"}},
                           "Tata Motors", aliases=None)
    assert sheet["items"][0]["status"] == fe.STATUS_ANOTHER_STORY
    assert sheet["evidence"][fe.article_id(a["url"])] == "short text: the page holds another story"


def test_the_sheet_rows_keep_the_fixed_shuffle():
    articles = [_article(n, f"Tata Motors news {n}") for n in range(10)]
    results = {a["url"]: {"status": "blocked_by_robots"} for a in articles}
    first = fe.build_sheet(articles, results, "Tata Motors", aliases=None)
    again = fe.build_sheet(list(articles), results, "Tata Motors", aliases=None)
    assert [r["article_id"] for r in first["rows"]] == [r["article_id"] for r in again["rows"]]
    assert [r["row"] for r in first["rows"]] == list(range(1, 11))


def test_the_progress_list_hides_the_search_order_and_the_fetch_order_is_kept():
    articles = [_article(n, f"Tata Motors news {n}") for n in range(12)]
    before = list(articles)
    shown = fe.progress_order(articles)
    assert articles == before                                       # the fetch order itself is unchanged
    assert sorted(a["url"] for a in shown) == sorted(a["url"] for a in articles)
    assert [a["url"] for a in shown] != [a["url"] for a in articles]


def test_the_run_records_that_its_first_progress_list_showed_the_search_order():
    assert any("search order" in note for note in fe.RUN_NOTES)


def test_an_empty_label_sheet_can_be_made_again_but_a_started_one_cannot(tmp_path):
    path = tmp_path / fe.LABEL_FILE
    path.write_text("row,article_id,title,url,label,unsure,note\n1,A1,T,u,,,\n", encoding="utf-8")
    assert fe.label_sheet_is_empty(path) is True
    path.write_text("row,article_id,title,url,label,unsure,note\n1,A1,T,u,not,,\n", encoding="utf-8")
    assert fe.label_sheet_is_empty(path) is False


# ---------- the eligible set comes from development only ----------

def _dev_record(passes):
    """passes: {configuration: [passed on run 1, run 2, run 3]}"""
    runs = {}
    for i in range(3):
        runs[f"run{i}"] = {"acceptance": {name: {"passed": flags[i]} for name, flags in passes.items()}}
    return {"reports": runs}


def test_a_configuration_passing_two_of_three_development_runs_is_eligible_and_wins_development():
    result = fe.eligible_set(_dev_record({"O1+S2": [True, True, False], "O2+S1": [False, False, False]}), "O1")
    assert result["eligible"] == ["O1+S1", "O1+S2"] and result["dev_winner"] == "O1+S2"


def test_one_pass_of_three_is_not_enough():
    result = fe.eligible_set(_dev_record({"O1+S2": [True, False, False]}), "O1")
    assert result["eligible"] == ["O1+S1"] and result["dev_winner"] == "O1+S1"


def test_a_model_configuration_is_eligible_only_when_the_model_comparison_chose_it():
    passes = {"O1+S2": [False, False, False], "O2+S1": [True, True, True]}
    assert fe.eligible_set(_dev_record(passes), "O1")["eligible"] == ["O1+S1"]
    assert fe.eligible_set(_dev_record(passes), "O2+S1")["eligible"] == ["O1+S1", "O2+S1"]


def test_the_fresh_test_confirms_or_rejects_the_development_winner_only():
    fresh = {"acceptance": {"O1+S2": {"passed": False}, "O2+S1": {"passed": True}}}
    assert fe.default_configuration("O1+S1", fresh) == "O1+S1"        # a fresh pass never promotes a model
    assert fe.default_configuration("O1+S2", fresh) == "O1+S1"        # the development winner fails: baseline
    assert fe.default_configuration("O1+S2", {"acceptance": {"O1+S2": {"passed": True}}}) == "O1+S2"


# ---------- collecting the model answers: a fixed budget of passes ----------

def _state_with_sheet(tmp_path):
    fe.save_state(tmp_path, {"run_id": fe.RUN_ID, "stages": {"sheet": {"done": True}}})


def test_the_pass_budget_survives_a_restart_and_a_crashed_pass_counts(tmp_path):
    _state_with_sheet(tmp_path)
    now = datetime(2026, 11, 3, 10, 0, tzinfo=timezone.utc)
    assert fe.start_pass(tmp_path, now, cached_before=0) == 1          # pass 1 starts and then crashes
    assert fe.start_pass(tmp_path, now, cached_before=0) == 2          # a restart reads the saved count
    fe.finish_pass(tmp_path, 2, {"asked": 5, "answered": 3, "missing_after": 2, "failures": {"server": 2},
                                 "failed_keys": {"A1": "server", "A2": "server"}, "seconds": 30.0})
    assert fe.start_pass(tmp_path, now, cached_before=0) == 3
    with pytest.raises(fe.FreshEvalError):
        fe.start_pass(tmp_path, now, cached_before=0)                  # no fourth pass
    stage = fe.load_state(tmp_path)["stages"]["collect"]
    assert [p["status"] for p in stage["passes"]] == ["started", "done", "started"]
    assert stage["passes"][1]["failures"] == {"server": 2}             # failures stay on record


def test_no_pass_starts_once_every_answer_is_in(tmp_path):
    _state_with_sheet(tmp_path)
    now = datetime(2026, 11, 3, 10, 0, tzinfo=timezone.utc)
    fe.start_pass(tmp_path, now, cached_before=0)
    fe.finish_pass(tmp_path, 1, {"asked": 5, "answered": 5, "missing_after": 0, "failures": {}, "failed_keys": {},
                                 "seconds": 50.0})
    with pytest.raises(fe.FreshEvalError):
        fe.start_pass(tmp_path, now, cached_before=0)


def test_the_pass_record_counts_answers_and_failures_by_kind():
    answers = {"A1": {"source": "model"}, "A2": {"source": "fallback", "kind": "rate_limit"},
               "A3": {"source": "fallback", "kind": "server"}}
    record = fe.pass_record(["A1", "A2", "A3"], answers, missing_after=2, seconds=12.5, stats={"tokens": 900})
    assert record["asked"] == 3 and record["answered"] == 1 and record["missing_after"] == 2
    assert record["failures"] == {"rate_limit": 1, "server": 1}
    assert record["failed_keys"] == {"A2": "rate_limit", "A3": "server"} and record["tokens"] == 900


@pytest.mark.parametrize("passes, complete", [
    ([], False),
    ([{"status": "done", "missing_after": 0}], True),
    ([{"status": "done", "missing_after": 3}], False),
    ([{"status": "done", "missing_after": 3}, {"status": "started"}], False),
    ([{"status": "done", "missing_after": 3}, {"status": "done", "missing_after": 2}, {"status": "started"}], True),
])
def test_collection_ends_when_every_answer_is_in_or_the_budget_is_used(passes, complete):
    assert fe.collection_complete({"passes": passes}) is complete


# ---------- the first pass and the live runtime ----------

def _collect(seconds, missing_after, cached_before=0, status="done"):
    return {"cached_before": cached_before,
            "passes": [{"status": status, "seconds": seconds, "missing_after": missing_after}]}


def test_a_fast_live_first_pass_with_enough_answers_counts_for_the_runtime():
    runtime = fe.first_pass_runtime(_collect(150.0, 2), candidates=40)
    assert runtime["coverage"] == 0.95 and runtime["seconds_for_rule"] == 150.0 and runtime["kind"] == "live"


def test_a_fast_first_pass_with_many_failures_does_not_count():
    runtime = fe.first_pass_runtime(_collect(40.0, 20), candidates=40)
    assert runtime["coverage"] == 0.5 and runtime["seconds_for_rule"] is None


def test_a_first_pass_that_starts_with_cached_answers_is_mixed_and_does_not_count():
    runtime = fe.first_pass_runtime(_collect(30.0, 0, cached_before=12), candidates=40)
    assert runtime["kind"] == "mixed" and runtime["seconds_for_rule"] is None


def test_a_crashed_first_pass_does_not_count():
    runtime = fe.first_pass_runtime(_collect(None, None, status="started"), candidates=40)
    assert runtime["seconds_for_rule"] is None


# ---------- the score reads saved answers only ----------

def test_saved_answers_are_used_and_missing_ones_fall_back_with_their_last_failure():
    saved = {"A1": {"label": "direct", "source": "model"}}
    collect = {"passes": [{"status": "done", "failed_keys": {"A2": "server"}},
                          {"status": "done", "failed_keys": {"A2": "daily_limit"}}]}
    answers = fe.final_answers(["A1", "A2", "A3"], saved, collect)
    assert answers["A1"]["source"] == "model"
    assert answers["A2"]["source"] == "fallback" and answers["A2"]["kind"] == "daily_limit"
    assert answers["A3"]["source"] == "fallback" and answers["A3"]["kind"] == "unknown"


# ---------- the frozen protocol ----------

def test_a_file_fingerprint_ignores_windows_line_endings(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_bytes(b"row,label\r\n1,not\r\n")
    b.write_bytes(b"row,label\n1,not\n")
    assert fe.file_fingerprint(a) == fe.file_fingerprint(b)


def test_the_protocol_names_a_changed_label_or_event_file(tmp_path):
    labels, events = tmp_path / "labels.csv", tmp_path / "events.csv"
    labels.write_text("row,label\n1,not\n", encoding="utf-8")
    events.write_text("row,event\n1,x\n", encoding="utf-8")
    protocol = {"labels_sha256": fe.file_fingerprint(labels), "events_sha256": fe.file_fingerprint(events)}
    assert fe.protocol_problems(protocol, labels, events) == []
    labels.write_text("row,label\n1,direct\n", encoding="utf-8")
    assert fe.protocol_problems(protocol, labels, events) == ["labels.csv differs from the frozen protocol"]


def test_the_labels_events_and_protocol_of_the_run_are_frozen():
    for name in (fe.LABEL_FILE, fe.EVENTS_FILE, fe.PROTOCOL_FILE):
        assert fe._frozen(f"data/eval/{fe.RUN_ID}/{name}")
    assert not fe._frozen(f"data/eval/{fe.RUN_ID}/state.json")          # collection keeps writing the state


def test_the_protocol_keeps_the_source_commit_apart_from_the_freeze_tag():
    protocol = fe.protocol_record(source_commit="abc123", dev_record={"threshold": {"chosen": 0.3},
                                  "experiment": {"model": "m"}}, eligibility={"eligible": ["O1+S1"],
                                  "dev_winner": "O1+S1"}, compare_choice="O1", llm_identity={"model": "m",
                                  "cache_tag": "groq|t"}, labels_sha256="l", events_sha256="e", dev_record_sha256="d",
                                  created="2026-11-03T10:00:00+00:00")
    assert protocol["source_commit"] == "abc123" and "freeze_commit" not in protocol
    assert protocol["collection"]["max_passes"] == fe.MAX_COLLECTION_PASSES == 3
    assert protocol["acceptance"]["min_first_pass_coverage"] == fe.MIN_MODEL_COVERAGE == 0.9
    assert protocol["experimental"] == [c for c in protocol["configurations"] if c not in ["O1+S1"]]


def test_collection_rebuilds_the_frozen_model_setting(monkeypatch):
    seen = {}
    monkeypatch.setattr("src.llm.make_llm", lambda provider=None, **kw: seen.update(provider=provider, **kw) or "llm")
    experiment = {"provider": "groq", "model": "openai/gpt-oss-120b",
                  "generation": {"temperature": 1.0, "reasoning_effort": "medium"}}
    assert fe.llm_from_experiment(experiment) == "llm"
    assert seen == {"provider": "groq", "model": "openai/gpt-oss-120b",
                    "generation": {"temperature": 1.0, "reasoning_effort": "medium"}}
