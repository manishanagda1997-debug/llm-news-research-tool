"""Tests for src/week2_summary.py: the week 2 summary run, its judging, the lock and the score (fake model)."""

import csv
import json
from datetime import datetime, timedelta, timezone

import pytest

from src import summary_round3 as r3
from src import week2_eval as w2
from src import week2_summary as ws

UTC = timezone.utc
BEFORE_CUTOFF = datetime(2026, 11, 10, 12, tzinfo=UTC)
FITS = {"fits": True, "logged_tokens": 0, "other_tokens": 0, "need": 45_000, "limit": 200_000}


# ---------- the guards before the run ----------

def test_without_the_round_3_score_the_design_is_not_decided(tmp_path):
    assert ws.round3_prompt(tmp_path) is None
    assert any("round 3 is not scored" in f for f in ws.freeze_found(None))


def test_the_freeze_must_accept_k5_and_hold_the_prompt_round_3_chose(monkeypatch):
    from src import summary_freeze as sf

    def refused(k, run_git=None):
        raise sf.FreezeError("The frozen summary cannot be used:\nthe tag is missing")
    monkeypatch.setattr(sf, "frozen_summary", refused)
    assert any("tag is missing" in f for f in ws.freeze_found("4"))
    monkeypatch.setattr(sf, "frozen_summary", lambda k, run_git=None: {"design": "direct",
                                                                        "settings": {"prompt_version": "4"}})
    assert ws.freeze_found("4") == []
    assert any("round 3 chose prompt 6" in f for f in ws.freeze_found("6"))
    calls = []
    monkeypatch.setattr(sf, "frozen_summary", lambda k, run_git=None: calls.append(k) or
                        {"design": "direct", "settings": {"prompt_version": "6"}})
    assert ws.freeze_found("6") == [] and calls == [5]


@pytest.fixture
def week(tmp_path, monkeypatch):
    w = w2.week("tata")
    w["run_dir"] = tmp_path / "week"
    w["run_dir"].mkdir()
    for name, text in ((w2.PROTOCOL_FILE, "{}"), (w2.LABEL_FILE, "x"), (w2.EVENTS_FILE, "y"), (w2.PAIRS_FILE, "z")):
        (w["run_dir"] / name).write_text(text, encoding="utf-8")
    (w["run_dir"] / w2.SEAL_FILE).write_text(json.dumps(w2.seal_record(w["run_dir"])), encoding="utf-8")
    monkeypatch.setattr(w2, "_tag_text", lambda week: "{}")
    from src import fulltext as ft
    monkeypatch.setattr(ft, "FULLTEXT_DIR", tmp_path / "fulltext")       # never the real passages of a real run
    return w, tmp_path / "fulltext"


def test_every_guard_is_checked_before_the_run(week):
    w, fulltext = week
    assert ws.run_problems(w, BEFORE_CUTOFF, "4", [], FITS, fulltext) == []
    late = datetime(2026, 11, 10, 16, 31, tzinfo=UTC)          # 22:01 IST
    assert any("cut-off" in f for f in ws.run_problems(w, late, "4", [], FITS, fulltext))
    exactly = datetime(2026, 11, 10, 16, 30, tzinfo=UTC)       # 22:00 IST: not started by then, so refused
    assert any("cut-off" in f for f in ws.run_problems(w, exactly, "4", [], FITS, fulltext))
    just_before = datetime(2026, 11, 10, 16, 29, 59, tzinfo=UTC)
    assert not any("cut-off" in f for f in ws.run_problems(w, just_before, "4", [], FITS, fulltext))
    tight = dict(FITS, fits=False, logged_tokens=170_000, earliest_ist="2026-11-10T20:00:00+05:30")
    assert any("do not fit" in f and "fits from" in f for f in ws.run_problems(w, BEFORE_CUTOFF, "4", [], tight, fulltext))
    assert ws.run_problems(w, BEFORE_CUTOFF, "4", ["freeze problem"], FITS, fulltext) == ["freeze problem"]
    (w["run_dir"] / w2.EVENTS_FILE).write_text("changed", encoding="utf-8")
    assert any("changed after the seal" in f for f in ws.run_problems(w, BEFORE_CUTOFF, "4", [], FITS, fulltext))


def test_the_cutoff_is_22_00_ist_on_10_november():
    assert ws.RUN_CUTOFF == datetime(2026, 11, 10, 16, 30, tzinfo=UTC)
    assert ws.JUDGING_CUTOFF == datetime(2026, 11, 11, 0, 0, tzinfo=UTC)


def test_an_unsealed_week_or_a_changed_protocol_is_refused(week, monkeypatch):
    w, fulltext = week
    monkeypatch.setattr(w2, "_tag_text", lambda week: None)
    assert any("differs from the tag" in f for f in ws.run_problems(w, BEFORE_CUTOFF, "4", [], FITS, fulltext))
    (w["run_dir"] / w2.SEAL_FILE).unlink()
    assert any("not sealed" in f for f in ws.week_found(w))


# ---------- the run, with a fake model ----------

ARTICLES = [{"id": 1, "key": "Ka", "title": "Tata Motors raises its Iveco stake", "description": "Tata Motors buys 89%.",
             "content": "", "url": "https://news.example/a"},
            {"id": 2, "key": "Kb", "title": "Car sales rise", "description": "Carmakers sell more; Tata grows 21%.",
             "content": "", "url": "https://news.example/b"}]
ANSWER = json.dumps({"article_briefs": [{"id": 1, "brief": "Tata Motors buys 89% of Iveco.", "enough_text": True},
                                        {"id": 2, "brief": "Carmakers sell more cars.", "enough_text": True}],
                     "key_points": [{"text": "Tata Motors buys 89% of Iveco.", "citations": [1]}],
                     "industry_context": [{"text": "Carmakers sell more cars.", "citations": [2]}], "not_used": []})


class FakeLLM:
    model_name = "openai/gpt-oss-120b"
    output_limit = 4096

    def __init__(self, marker, answer=ANSWER):
        self.marker, self.answer, self.prompts = marker, answer, []
        self.stats = {"per_call": [], "requests": []}

    def __call__(self, prompt):
        assert self.marker.exists()                         # the marker comes before any model request
        self.prompts.append(prompt)
        self.stats["per_call"].append({"sent_at": 1.0, "input": 900, "output": 300, "finish_reason": "stop"})
        return self.answer


def test_the_run_writes_the_marker_first_saves_the_passages_and_runs_once(week):
    w, fulltext = week
    llm = FakeLLM(ws.paths(w, fulltext)["attempt"])
    record = ws.execute(w, ARTICLES, llm, "4", 0, FITS, "abc", fulltext, ranking_fp="rankfp")
    assert len(llm.prompts) == 1 and ws.produced(record) and record["relevance_check"].startswith("not run")
    assert record["tokens"] == 1200 and not record["allowance_exceeded"] and record["prompt"] == "4"
    marker = json.loads(ws.paths(w, fulltext)["attempt"].read_text(encoding="utf-8"))
    assert marker["attempt_id"] == record["attempt_id"] and marker["budget"] == FITS
    saved = json.loads(ws.paths(w, fulltext)["passages"].read_text(encoding="utf-8"))
    assert ws.saved_passages(record, saved) == {"Ka": "Tata Motors buys 89%.", "Kb": "Carmakers sell more; Tata grows 21%."}
    with pytest.raises(FileExistsError):
        ws.execute(w, ARTICLES, FakeLLM(ws.paths(w, fulltext)["attempt"]), "4", 0, FITS, "abc", fulltext, ranking_fp="rankfp")
    assert any("already exists" in f for f in ws.run_problems(w, BEFORE_CUTOFF, "4", [], FITS, fulltext))


def test_a_refused_run_sends_nothing(week, monkeypatch):
    from src import summary_v2 as sv
    w, _ = week
    monkeypatch.setattr(ws, "round3_prompt", lambda eval_dir=None: None)
    monkeypatch.setattr(sv, "summary_llm", lambda: (_ for _ in ()).throw(AssertionError("no model may be made")))
    monkeypatch.setattr(ws, "log_runs", lambda: [])
    monkeypatch.setattr(ws, "_daily_limit", lambda: 200_000)
    with pytest.raises(ws.Week2SummaryError, match="nothing was sent"):
        ws._run_command(w, 0)
    assert not (w["run_dir"] / ws.ATTEMPT_FILE).exists()


def test_a_changed_passage_is_found(week):
    w, fulltext = week
    record = ws.execute(w, ARTICLES, FakeLLM(ws.paths(w, fulltext)["attempt"]), "4", 0, FITS, "abc", fulltext, ranking_fp="rankfp")
    saved = json.loads(ws.paths(w, fulltext)["passages"].read_text(encoding="utf-8"))
    saved["passages"][0]["description"] = "something else"
    with pytest.raises(ws.Week2SummaryError):
        ws.saved_passages(record, saved)
    with pytest.raises(ws.Week2SummaryError, match="another attempt"):
        ws.saved_passages(dict(record, attempt_id="x"), json.loads(ws.paths(w, fulltext)["passages"].read_text()))


def test_the_token_log_counts_week_2_runs_and_a_started_run_without_a_record(tmp_path, monkeypatch):
    for name in w2.WEEKS:
        (tmp_path / w2.WEEKS[name]["run_id"]).mkdir()
    monkeypatch.setattr(w2, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(r3, "_load_round", lambda: ({}, {}, None))
    monkeypatch.setattr(r3, "_log_runs", lambda records, markers: [])
    tata, rel = w2.week("tata"), w2.week("reliance")
    (tata["run_dir"] / ws.ATTEMPT_FILE).write_text(json.dumps({"started_at": "2026-11-10T10:00:00+00:00",
                                                               "clock_start": "2026-11-10T10:00:00+00:00"}))
    (tata["run_dir"] / ws.RECORD_FILE).write_text(json.dumps({"summary_record": {"records": [
        {"result": {"usage": {"per_call": [{"sent_at": 5, "input": 1000, "output": 500}]}}}]}}))
    (rel["run_dir"] / ws.ATTEMPT_FILE).write_text(json.dumps({"started_at": "2026-11-10T11:00:00+00:00",
                                                              "clock_start": "2026-11-10T11:00:00+00:00"}))
    log = r3.token_log(ws.log_runs())
    assert log == [(datetime(2026, 11, 10, 10, 0, 5, tzinfo=UTC), 1500),
                   (datetime(2026, 11, 10, 11, 0, tzinfo=UTC), r3.ALLOWANCE)]


# ---------- judging, lock and score ----------

def _judging_week(week):
    w, fulltext = week
    with open(w["run_dir"] / w2.LABEL_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row", "article_id", "title", "url", "label", "unsure", "note"])
        writer.writeheader()
        writer.writerows([{"row": 1, "article_id": "a", "title": "t", "url": ARTICLES[0]["url"], "label": "direct"},
                          {"row": 2, "article_id": "b", "title": "t", "url": ARTICLES[1]["url"], "label": "indirect"}])
    from src.summary_run import EVENT_COLUMNS
    with open(w["run_dir"] / w2.EVENTS_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        writer.writerows([{"row": 1, "event": "iveco_stake", "url": ARTICLES[0]["url"]},
                          {"row": 2, "event": "october_sales", "url": ARTICLES[1]["url"]}])
    with open(w["run_dir"] / w2.PAIRS_FILE, "w", newline="", encoding="utf-8") as handle:
        handle.write("expectation_event,outcome_event,note\noctober_sales,iveco_stake,\nother,iveco_stake,\n")
    (w["run_dir"] / w2.SEAL_FILE).write_text(json.dumps(w2.seal_record(w["run_dir"])), encoding="utf-8")
    record = ws.execute(w, ARTICLES, FakeLLM(ws.paths(w, fulltext)["attempt"]), "4", 0, FITS, "abc", fulltext, ranking_fp="rankfp")
    return w, fulltext, record


def test_the_sheets_hold_the_industry_row_the_items_the_events_and_only_pairs_present_in_the_summary(week):
    w, fulltext, record = _judging_week(week)
    passages = ws.saved_passages(record, json.loads(ws.paths(w, fulltext)["passages"].read_text(encoding="utf-8")))
    sheets = ws.build(w, record, passages)
    assert len(sheets["industry"]) == 1 and len(sheets["events"]) == 2
    assert [r["pair"] for r in sheets["pairs"]] == ["october_sales -> iveco_stake"]
    assert {r["kind"] for r in sheets["items"]} >= {"company point", "industry point"}


def _filled(sheets):
    items = [dict(r, judgement="supported", contradicted="no", central="") for r in sheets["items"]]
    events = [dict(r, covered="yes") for r in sheets["events"]]
    industry = [dict(r, industry_evidence="yes", note="article 2 reports carmakers' sales") for r in sheets["industry"]]
    pairs = [dict(r, outlook_stated="no", kept_apart="no", note="no outlook in the points") for r in sheets["pairs"]]
    return items, events, industry, pairs


def test_lock_needs_complete_sheets_and_any_later_change_is_found(week, monkeypatch):
    w, fulltext, record = _judging_week(week)
    monkeypatch.setattr(ws, "paths", lambda week, fulltext_dir=None: {
        "record": w["run_dir"] / ws.RECORD_FILE, "attempt": w["run_dir"] / ws.ATTEMPT_FILE,
        "passages": fulltext / f"week2_passages_{w['run_id']}.json", "page": fulltext / "page.html"})
    passages = ws.saved_passages(record, json.loads(ws.paths(w)["passages"].read_text(encoding="utf-8")))
    sheets = ws.build(w, record, passages)
    with pytest.raises(ws.Week2SummaryError, match="not complete"):
        ws.lock_record(w, (sheets["items"], sheets["events"], sheets["industry"], sheets["pairs"]), sheets["key"])
    no_note = _filled(sheets)[:3] + ([dict(sheets["pairs"][0], outlook_stated="no", kept_apart="yes")],)
    with pytest.raises(ws.Week2SummaryError, match="every no needs a note"):
        ws.lock_record(w, no_note, sheets["key"])
    filled = _filled(sheets)
    lock = ws.lock_record(w, filled, sheets["key"])
    assert ws.lock_changes(lock, w, filled, sheets["key"]) == []
    changed = (filled[0], filled[1], [dict(filled[2][0], industry_evidence="no")], filled[3])
    assert any("industry" in f for f in ws.lock_changes(lock, w, changed, sheets["key"]))
    (w["run_dir"] / w2.LABEL_FILE).write_text("changed", encoding="utf-8")
    assert any("labels" in f for f in ws.lock_changes(lock, w, filled, sheets["key"]))


def test_the_score_uses_the_week_1_pass_levels_and_reports_the_industry_and_pairs(week):
    w, fulltext, record = _judging_week(week)
    passages = ws.saved_passages(record, json.loads(ws.paths(w, fulltext)["passages"].read_text(encoding="utf-8")))
    sheets = ws.build(w, record, passages)
    filled = _filled(sheets)
    result = ws.score_result(w, record, filled, sheets["key"], {"after_judging_cutoff": False})
    summary = result["summary"]
    assert result["status"] == "judged" and summary["passed"]
    assert set(summary["checks"]) == {"produced", "event_coverage", "contradicted", "no_central_contradiction"}
    assert result["industry"]["checks"] == {"no_unsupported_industry_point": True,
                                            "industry_section_matches_evidence": True}
    assert result["pairs"]["rows"][0]["kept_apart"] == "no" and result["relevance_check"] == "not run"


def test_a_summary_not_locked_by_the_judging_cutoff_claims_no_result(week):
    w, fulltext, record = _judging_week(week)
    assert ws.score_result(w, record, None, None, None)["status"] == "run, not judged"
    late = ws.score_result(w, record, None, None, {"after_judging_cutoff": True})
    assert late["status"] == "run, not judged" and "no result is claimed" in late["summary"]["reason"]


def test_a_run_without_a_summary_fails_produced_and_is_not_judged(week):
    w, fulltext = week
    llm = FakeLLM(ws.paths(w, fulltext)["attempt"], answer="not json")
    record = ws.execute(w, ARTICLES, llm, "4", 0, FITS, "abc", fulltext, ranking_fp="rankfp")
    assert not ws.produced(record)
    result = ws.score_result(w, record, None, None, None)
    assert result["status"] == "run, no summary" and result["summary"]["checks"] == {"produced": False}


def test_the_pass_levels_are_the_protocols():
    assert ws.pass_levels()["pass_levels"]["summary"] == {**w2.SUMMARY_RULE["pass_levels"], "k": 5}


def test_this_module_is_outside_every_frozen_or_locked_list():
    import json as _json
    from src import transfer_eval as te
    from src import summary_prompt6 as p6
    from src.config import EVAL_DIR
    assert "src/week2_summary.py" not in te.JUDGING_CODE and "src/week2_summary.py" not in r3.CODE_FILES
    plan = _json.loads((EVAL_DIR / r3.PLAN_FILE).read_text(encoding="utf-8"))
    assert plan["code_fingerprint"] == p6.code_fingerprint(r3.CODE_FILES)


def test_the_summary_articles_must_be_the_ranking_tests_top_5(week):
    w, _ = week
    assert "run the ranking test first" in ws.rank_mismatch(w, ["a"])     # no ranking test: refused
    (w["run_dir"] / w2.RANK_FILE).write_text(json.dumps({"orders": {w2.DEFAULT: list("abcdefg")}}), encoding="utf-8")
    assert ws.rank_mismatch(w, list("abcde")) == ""
    assert "not the ranking test's top 5" in ws.rank_mismatch(w, list("abced"))


def test_a_week_without_its_ranking_test_is_refused_before_any_model_is_made(week, monkeypatch):
    from src import summary_v2 as sv
    w, _ = week
    monkeypatch.setattr(ws, "round3_prompt", lambda eval_dir=None: "4")
    monkeypatch.setattr(ws, "freeze_found", lambda chosen: [])
    monkeypatch.setattr(ws, "log_runs", lambda: [])
    monkeypatch.setattr(ws, "_daily_limit", lambda: 200_000)
    monkeypatch.setattr(ws, "RUN_CUTOFF", datetime(2100, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(ws, "top5", lambda week, aliases: ([], [{"key": k} for k in "abcde"]))
    monkeypatch.setattr(sv, "summary_llm", lambda: (_ for _ in ()).throw(AssertionError("no model may be made")))
    with pytest.raises(ws.Week2SummaryError, match="run the ranking test first"):
        ws._run_command(w, 0)
    assert not (w["run_dir"] / ws.ATTEMPT_FILE).exists()


# ---------- the ranking result is kept in the record and checked later ----------

def _with_ranking(w, top=list("abcdefg")):
    (w["run_dir"] / w2.RANK_FILE).write_text(json.dumps({"orders": {w2.DEFAULT: top}}), encoding="utf-8")
    return ws.ranking_fingerprint(w)


def test_a_malformed_ranking_file_is_a_clean_refusal(week):
    w, _ = week
    (w["run_dir"] / w2.RANK_FILE).write_text("{not json", encoding="utf-8")
    assert "cannot be read" in ws.rank_mismatch(w, list("abcde"))
    (w["run_dir"] / w2.RANK_FILE).write_text(json.dumps({"other": 1}), encoding="utf-8")
    assert "cannot be read" in ws.rank_mismatch(w, list("abcde"))
    (w["run_dir"] / w2.RANK_FILE).write_text(json.dumps({"orders": {w2.DEFAULT: 5}}), encoding="utf-8")
    assert "refused" in ws.rank_mismatch(w, list("abcde"))
    (w["run_dir"] / w2.RANK_FILE).write_text("[]", encoding="utf-8")
    assert "cannot be read" in ws.rank_mismatch(w, list("abcde"))


def test_the_record_keeps_the_ranking_fingerprint_and_a_later_change_blocks_judging_lock_and_score(week):
    w, fulltext = week
    fp = _with_ranking(w)
    record = ws.execute(w, ARTICLES, FakeLLM(ws.paths(w, fulltext)["attempt"]), "4", 0, FITS, "abc", fulltext,
                        ranking_fp=fp)
    marker = json.loads(ws.paths(w, fulltext)["attempt"].read_text(encoding="utf-8"))
    assert record["ranking_fingerprint"] == marker["ranking_fingerprint"] == fp
    assert ws.attempt_found(w, record, fulltext) == []
    _with_ranking(w, top=list("gfedcba"))                      # the ranking file changes after the run
    assert any("changed after the run" in f for f in ws.attempt_found(w, record, fulltext))
    assert any("no ranking fingerprint" in f for f in ws.attempt_found(w, dict(record, ranking_fingerprint=None),
                                                                        fulltext))


def test_the_lock_fingerprints_the_ranking_file(week):
    w, _ = week
    _with_ranking(w)
    before = ws.judging_inputs(w)["ranking"]
    _with_ranking(w, top=list("gfedcba"))
    assert before is not None and ws.judging_inputs(w)["ranking"] != before


def test_judge_lock_and_score_all_refuse_when_the_ranking_file_changed(week, monkeypatch):
    w, fulltext = week
    fp = _with_ranking(w)
    monkeypatch.setattr(ws, "paths", lambda week, fulltext_dir=None: {
        "record": w["run_dir"] / ws.RECORD_FILE, "attempt": w["run_dir"] / ws.ATTEMPT_FILE,
        "passages": fulltext / f"week2_passages_{w['run_id']}.json", "page": fulltext / "page.html"})
    ws.execute(w, ARTICLES, FakeLLM(ws.paths(w)["attempt"]), "4", 0, FITS, "abc", ranking_fp=fp)
    _with_ranking(w, top=list("gfedcba"))
    for command in (ws._judge_command, ws._lock_command, ws._score_command):
        with pytest.raises(ws.Week2SummaryError, match="changed after the run"):
            command(w)


def test_the_fingerprint_comes_from_the_same_bytes_that_were_checked(week, monkeypatch):
    from pathlib import Path as P
    from src.fresh_eval import file_fingerprint
    w, _ = week
    _with_ranking(w)
    problem, fp = ws.checked_ranking(w, list("abcde"))
    assert problem == "" and fp == file_fingerprint(w["run_dir"] / w2.RANK_FILE) == ws.ranking_fingerprint(w)
    reads = []
    real = P.read_bytes

    def counting(self):
        reads.append(self.name)
        return real(self)
    monkeypatch.setattr(P, "read_bytes", counting)
    ws.checked_ranking(w, list("abcde"))
    assert reads.count(w2.RANK_FILE) == 1                      # one read for both the check and the fingerprint
    assert ws.checked_ranking(w, list("abced"))[1] is None     # a failed check gives no fingerprint


def test_the_run_command_saves_the_fingerprint_of_the_checked_file(week, monkeypatch):
    from src import summary_v2 as sv
    w, fulltext = week
    fp = _with_ranking(w)
    seen = {}
    monkeypatch.setattr(ws, "round3_prompt", lambda eval_dir=None: "4")
    monkeypatch.setattr(ws, "freeze_found", lambda chosen: [])
    monkeypatch.setattr(ws, "log_runs", lambda: [])
    monkeypatch.setattr(ws, "_daily_limit", lambda: 200_000)
    monkeypatch.setattr(ws, "RUN_CUTOFF", datetime(2100, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(ws, "top5", lambda week, aliases: ([], [{"key": k} for k in "abcde"]))
    monkeypatch.setattr(sv, "passage_articles", lambda chosen, aliases, company: ARTICLES)
    monkeypatch.setattr(sv, "summary_llm", lambda: "model")
    monkeypatch.setattr(w2, "_git", lambda *args: "abc")

    def fake_execute(week, articles, llm, chosen, other, budget, commit, fulltext_dir=None, *, ranking_fp):
        seen["fp"] = ranking_fp
        raise RuntimeError("stop before any request")
    monkeypatch.setattr(ws, "execute", fake_execute)
    with pytest.raises(RuntimeError):
        ws._run_command(w, 0)
    assert seen["fp"] == fp


def test_a_lock_without_exactly_four_sheet_fingerprints_is_refused(week):
    w, _ = week
    sheets = ([], [], [], [])
    assert any("exactly 4" in f for f in ws.lock_changes({"sheets": ["a", "b", "c"]}, w, sheets, {}))
    assert any("exactly 4" in f for f in ws.lock_changes({"sheets": []}, w, sheets, {}))
    assert any("exactly 4" in f for f in ws.lock_changes({}, w, sheets, {}))
