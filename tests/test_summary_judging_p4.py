"""
Tests for src/summary_judging_p4.py: judging of the prompt version 4 rerun (SUMMARY_JUDGING_GUIDE_PART4.md).
No model calls.

    sheet   new or changed items only, shuffled; an item with exactly the same section, text and cited articles
            as a locked direct item keeps its locked judgement and is not shown; an event sheet; one industry
            evidence row per summary; a key; a reading page with the industry evidence first
    lock    every mark present and valid, then fingerprints saved
    score   the Part 4 pass rule, in total and per run
"""

import json

import pytest

from src import summary_judging as sj
from src import summary_judging_p4 as p4


def _headline(i, run):
    return {"id": i, "title": f"Story {run} {i}", "source": "S", "published_at": "", "url": f"https://x.com/{run}/{i}"}


def _entry(run, design, points, industry, briefs=None, status="ok", repairs=()):
    briefs = briefs or [f"Brief text {run} {i}." for i in range(1, 4)]
    summary = {"article_briefs": [{"id": i, "brief": b, "enough_text": True, "hidden": False}
                                  for i, b in enumerate(briefs, start=1)],
               "key_points": [{"text": t, "citations": c} for t, c in points],
               "industry_context": [{"text": t, "citations": c} for t, c in industry], "not_used": []}
    return {"run": run, "design": design, "k": 3,
            "articles": [{"id": i, "key": f"K{run}{i}", "title": f"Story {run} {i}"} for i in range(1, 4)],
            "result": {"status": status, "summary": summary if status == "ok" else None,
                       "headlines": [_headline(i, run) for i in range(1, 4)], "usage": {"calls": 2},
                       "repairs": list(repairs)}}


LOCKED_RECORD = {"records": [
    _entry("r1", "direct", [("Offer raised.", [1]), ("Vote held.", [2])], [("Sector slows.", [3])]),
    _entry("r1", "brief_first", [("Offer raised.", [1, 2])], [("Sector note.", [3])]),
    _entry("r2", "direct", [("Sales rose.", [1])], [("Market grows.", [2])]),
]}
PASSAGES = {f"K{run}{i}": f"Passage text {run} {i}." for run in ("r1", "r2") for i in range(1, 4)}
EVENTS = {"https://x.com/r1/1": "offer", "https://x.com/r1/2": "vote", "https://x.com/r1/3": "sector",
          "https://x.com/r2/1": "sales", "https://x.com/r2/2": "sales", "https://x.com/r2/3": "other"}
LABELS = {"https://x.com/r1/1": "direct", "https://x.com/r1/2": "direct", "https://x.com/r1/3": "not",
          "https://x.com/r2/1": "direct", "https://x.com/r2/2": "indirect", "https://x.com/r2/3": "direct"}


def _locked():
    """The locked comparison: every direct item supported except the r1 industry point (not supported)."""
    sheets = sj.build_sheets(LOCKED_RECORD, PASSAGES, EVENTS, LABELS, seed=7)
    for row in sheets["items"]:
        entry = sheets["key"]["items"][str(row["row"])]
        bad = entry["section"] == "industry_context" and entry["run"] == "r1"
        row.update(judgement="not_supported" if bad else "supported", contradicted="no")
    return sheets["items"], sheets["key"]


# the rerun: r1 keeps "Offer raised." [1] and its briefs, rewrites "Vote held.", drops its industry point;
# r2 changes one brief and keeps a reworded industry point
P4_RECORD = {"prompt_version": "4", "designs": ["direct"], "records": [
    _entry("r1", "direct", [("Offer raised.", [1]), ("The vote is held.", [2])], []),
    _entry("r2", "direct", [("Sales rose.", [1])], [("The market grows.", [2])],
           briefs=["Brief text r2 1.", "A new brief r2 2.", "Brief text r2 3."]),
]}


def _build(record=P4_RECORD):
    items, key = _locked()
    return p4.build_sheets(record, PASSAGES, EVENTS, LABELS, p4.locked_items(items, key), seed=5)


# ---------- the carry-over ----------

def test_only_exactly_identical_direct_items_keep_their_locked_judgement():
    sheets = _build()
    carried = {(c["run"], c["section"], c["index"]) for c in sheets["key"]["carried"]}
    # r1: "Offer raised." [1] and its three briefs; r2: "Sales rose." [1] and briefs 1 and 3
    assert carried == {("r1", "key_points", 0), ("r1", "article_briefs", 0), ("r1", "article_briefs", 1),
                       ("r1", "article_briefs", 2), ("r2", "key_points", 0), ("r2", "article_briefs", 0),
                       ("r2", "article_briefs", 2)}
    texts = {row["text"] for row in sheets["items"]}
    assert texts == {"The vote is held.", "The market grows.", "A new brief r2 2."}


def test_the_same_text_with_other_citations_is_judged_again():
    record = json.loads(json.dumps(P4_RECORD))
    record["records"][0]["result"]["summary"]["key_points"][0]["citations"] = [1, 2]
    assert "Offer raised." in {row["text"] for row in _build(record)["items"]}


def test_the_same_text_in_another_section_is_judged_again():
    record = json.loads(json.dumps(P4_RECORD))
    record["records"][1]["result"]["summary"]["industry_context"] = [{"text": "Sales rose.", "citations": [1]}]
    rows = [r for r in _build(record)["items"] if r["text"] == "Sales rose."]
    assert [r["kind"] for r in rows] == ["industry point"]


def test_a_text_that_differs_only_in_spacing_or_case_is_judged_again():
    record = json.loads(json.dumps(P4_RECORD))
    record["records"][0]["result"]["summary"]["key_points"][0]["text"] = "Offer raised. "
    record["records"][1]["result"]["summary"]["key_points"][0]["text"] = "sales rose."
    texts = {row["text"] for row in _build(record)["items"]}
    assert {"Offer raised. ", "sales rose."} <= texts


def test_brief_first_items_are_never_carried_over():
    items, key = _locked()
    assert all(key["items"][n]["design"] == "direct" for n in p4.locked_items(items, key).values())
    record = json.loads(json.dumps(P4_RECORD))
    record["records"][0]["result"]["summary"]["key_points"][0]["citations"] = [1, 2]    # brief_first's citation
    assert "Offer raised." in {row["text"] for row in _build(record)["items"]}


def test_a_locked_signature_with_two_different_judgements_is_judged_again():
    items, key = _locked()
    record = {"records": LOCKED_RECORD["records"] + [_entry("r3", "direct", [("Offer raised.", [1])], [])]}
    sheets = sj.build_sheets(record, PASSAGES, EVENTS, LABELS, seed=7)
    for row in sheets["items"]:
        row.update(judgement="supported", contradicted="no")
    # the same signature cannot repeat across runs (article keys differ); a clash inside one locked sheet is forced
    first = next(r for r in sheets["items"] if r["text"] == "Offer raised." and
                 sheets["key"]["items"][str(r["row"])]["design"] == "direct" and
                 sheets["key"]["items"][str(r["row"])]["run"] == "r1")
    twin = dict(first, row=str(len(sheets["items"]) + 1), judgement="partly_supported")
    sheets["items"].append(twin)
    sheets["key"]["items"][twin["row"]] = dict(sheets["key"]["items"][str(first["row"])])
    assert ("key_points", "Offer raised.", "Kr11") not in p4.locked_items(sheets["items"], sheets["key"])


def test_a_record_with_another_design_is_refused():
    with pytest.raises(p4.JudgingError):
        _build(LOCKED_RECORD)


# ---------- the sheets and the page ----------

def test_the_new_items_are_numbered_shuffled_with_a_fixed_seed_and_start_empty():
    first, second = _build(), _build()
    assert first["items"] == second["items"]
    assert [row["row"] for row in first["items"]] == [1, 2, 3]
    assert all(row["judgement"] == row["contradicted"] == row["central"] == row["note"] == "" for row in first["items"])


def test_one_industry_row_per_summary_and_one_event_row_per_relevant_event():
    sheets = _build()
    assert [row["summary"] for row in sheets["industry"]] == sorted(sheets["key"]["summaries"])
    assert all(row["industry_evidence"] == row["note"] == "" for row in sheets["industry"])
    assert len(sheets["industry"]) == 2 and len(sheets["events"]) == 4      # r1: offer, vote; r2: sales, other


def test_nothing_shown_names_a_design_a_run_a_repair_or_a_carried_judgement():
    sheets = _build()
    shown = (json.dumps(sheets["items"]) + json.dumps(sheets["events"]) + json.dumps(sheets["industry"])
             + sheets["page"]).lower()
    for word in ("direct", "brief_first", "design", "prompt", "repair", '"run"', "locked", "carried", "supported"):
        assert word not in shown


def test_the_page_shows_the_industry_evidence_before_the_items_and_the_summaries():
    page = _build()["page"]
    assert page.index("1. Industry evidence") < page.index("2. Items") < page.index("3. Event coverage")
    evidence = page[page.index("1. Industry evidence"):page.index("2. Items")]
    assert "Passage text r1 3." in evidence and "Offer raised." not in evidence     # articles only, no summary
    assert "(none)" in page                                                          # r1 has no industry point


# ---------- the checks and the lock ----------

def _judged(sheets, item="supported", covered="yes", evidence=("no", "yes")):
    for row in sheets["items"]:
        row.update(judgement=item, contradicted="no", note="n" if item != "supported" else "")
    for row in sheets["events"]:
        row.update(covered=covered)
    by_run = {sheets["key"]["summaries"][r["summary"]]["run"]: r for r in sheets["industry"]}
    for run, value in zip(("r1", "r2"), evidence):
        by_run[run].update(industry_evidence=value, note="reason")
    return sheets


def test_the_industry_rows_need_yes_or_no_and_a_note():
    sheets = _build()
    found = p4.industry_problems(sheets["industry"], sheets["key"]["industry"])
    assert any("yes or no" in f for f in found) and any("note" in f for f in found)
    assert p4.problems(*[_judged(sheets)[k] for k in ("items", "events", "industry", "key")]) == []


def test_the_lock_refuses_an_incomplete_sheet_and_notices_a_later_change():
    sheets = _build()
    with pytest.raises(p4.JudgingError):
        p4.lock_record(sheets["items"], sheets["events"], sheets["industry"], sheets["key"], "guide", "record")
    sheets = _judged(sheets)
    lock = p4.lock_record(sheets["items"], sheets["events"], sheets["industry"], sheets["key"], "guide", "record")
    p4.check_lock(lock, sheets["items"], sheets["events"], sheets["industry"], sheets["key"])
    sheets["industry"][0]["industry_evidence"] = "yes" if sheets["industry"][0]["industry_evidence"] == "no" else "no"
    with pytest.raises(p4.JudgingError):
        p4.check_lock(lock, sheets["items"], sheets["events"], sheets["industry"], sheets["key"])


def test_the_lock_notices_a_change_of_the_key():
    sheets = _judged(_build())
    lock = p4.lock_record(sheets["items"], sheets["events"], sheets["industry"], sheets["key"], "guide", "record")
    sheets["key"]["carried"][0]["locked_row"] = 999
    with pytest.raises(p4.JudgingError, match="key"):
        p4.check_lock(lock, sheets["items"], sheets["events"], sheets["industry"], sheets["key"])


def test_the_lock_notices_a_change_of_the_judging_code(monkeypatch):
    sheets = _judged(_build())
    lock = p4.lock_record(sheets["items"], sheets["events"], sheets["industry"], sheets["key"], "guide", "record")
    monkeypatch.setattr(p4, "_code_fingerprint", lambda: "changed")
    with pytest.raises(p4.JudgingError, match="code"):
        p4.check_lock(lock, sheets["items"], sheets["events"], sheets["industry"], sheets["key"])


def test_the_code_fingerprint_covers_the_shared_judging_helpers(monkeypatch, tmp_path):
    before = p4._code_fingerprint()
    copy = tmp_path / "summary_judging.py"
    copy.write_bytes(open(sj.__file__, "rb").read() + b"\n# changed\n")
    monkeypatch.setattr(sj, "__file__", str(copy))
    assert p4._code_fingerprint() != before


# ---------- the score ----------

def _score(sheets, record=P4_RECORD, baseline=None, runs=None):
    items, key = _locked()
    baseline = baseline or p4.locked_baseline(items, key)
    runs = runs or ["r1", "r2"]
    return p4.score(sheets["items"], sheets["events"], sheets["industry"], sheets["key"], record, items, baseline, runs)


def test_the_locked_baseline_counts_the_direct_key_points_and_briefs():
    items, key = _locked()
    assert p4.locked_baseline(items, key) == {"supported": 9, "total": 9}   # r1: 2 + 3, r2: 1 + 3


def test_carried_items_count_with_their_locked_judgement():
    result = _score(_judged(_build()))
    assert result["total"]["carried"] == 7
    assert result["key_points_and_briefs"]["supported"] == 9 and result["key_points_and_briefs"]["total"] == 9
    assert result["per_run"]["r1"]["industry"]["produced"] == 0
    assert result["total"]["industry"] == {"produced": 1, "total": 1, "supported": 1, "partly_supported": 0,
                                           "not_supported": 0, "supported_share": 1.0, "contradicted": 0}


def test_everything_passes_when_the_empty_section_has_no_evidence_and_nothing_drops():
    result = _score(_judged(_build()))
    assert result["passes"] and all(result["checks"].values())


def test_an_empty_section_fails_when_an_article_reports_on_the_wider_industry():
    result = _score(_judged(_build(), evidence=("yes", "yes")))
    assert result["empty_section_with_evidence"] == ["r1"]
    assert not result["checks"]["empty_section_only_without_evidence"] and not result["passes"]


def test_an_unsupported_industry_point_fails():
    sheets = _judged(_build())
    for row in sheets["items"]:
        if row["kind"] == "industry point":
            row.update(judgement="not_supported", note="n")
    assert not _score(sheets)["checks"]["no_unsupported_industry_point"]


def test_no_industry_point_at_all_fails_unless_every_section_is_empty_without_evidence():
    record = json.loads(json.dumps(P4_RECORD))
    record["records"][1]["result"]["summary"]["industry_context"] = []
    sheets = _judged(_build(record), evidence=("no", "no"))
    assert _score(sheets, record)["checks"]["a_supported_industry_point_or_all_empty_without_evidence"]
    sheets = _judged(_build(), evidence=("no", "no"))
    for row in sheets["items"]:
        if row["kind"] == "industry point":
            row.update(judgement="partly_supported", note="n")
    assert not _score(sheets)["checks"]["a_supported_industry_point_or_all_empty_without_evidence"]


def test_one_unsupported_key_point_or_brief_fails():
    sheets = _judged(_build())
    next(r for r in sheets["items"] if r["kind"] == "article brief").update(judgement="not_supported", note="n")
    assert not _score(sheets)["checks"]["no_unsupported_key_point_or_brief"]


def test_the_threshold_is_the_unrounded_locked_share_minus_five_points():
    sheets = _judged(_build())
    for row in sheets["items"]:
        if row["kind"] != "industry point":
            row.update(judgement="partly_supported", note="n")      # 2 of 9 key points and briefs now partly
    result = _score(sheets, baseline={"supported": 29, "total": 38})
    kb = result["key_points_and_briefs"]
    assert kb["threshold"] == pytest.approx(29 / 38 - 0.05) and kb["threshold"] > 0.7131
    assert kb["supported"] == 7 and kb["total"] == 9                  # 77.8%: passes against 71.3%
    assert result["checks"]["key_points_and_briefs_within_tolerance"]
    # 27 of 38 (71.05%) is below 71.32%: it fails even though it rounds to 71.1%
    assert 27 / 38 < 29 / 38 - 0.05 <= 28 / 38


def test_event_coverage_below_80_percent_fails():
    result = _score(_judged(_build(), covered="no"))
    assert result["total"]["events"]["coverage"] == 0.0 and not result["checks"]["event_coverage"]


def test_a_run_without_a_summary_fails():
    record = json.loads(json.dumps(P4_RECORD))
    record["records"].append(_entry("r3", "direct", [], [], status="unavailable"))
    result = _score(_judged(_build(record)), record, runs=["r1", "r2", "r3"])
    assert not result["checks"]["every_development_run_has_one_summary"]


def test_a_record_missing_a_development_run_fails():
    result = _score(_judged(_build()), runs=["r1", "r2", "r3"])
    assert not result["checks"]["every_development_run_has_one_summary"] and not result["passes"]


def test_a_run_recorded_twice_fails():
    record = json.loads(json.dumps(P4_RECORD))
    record["records"].append(json.loads(json.dumps(record["records"][1])))
    assert not _score(_judged(_build()), record)["checks"]["every_development_run_has_one_summary"]


def test_industry_points_fail_when_no_article_reports_on_the_wider_industry():
    result = _score(_judged(_build(), evidence=("no", "no")))         # r2 has an industry point
    assert result["points_without_evidence"] == ["r2"]
    assert not result["checks"]["industry_points_only_with_evidence"] and not result["passes"]


def test_repaired_items_are_counted_apart():
    record = json.loads(json.dumps(P4_RECORD))
    record["records"][0]["result"]["repairs"] = [{"section": "key_points", "after": "The vote is held.", "ok": True}]
    result = _score(_judged(_build(record)), record)
    assert result["total"]["repaired"]["total"] == 1 and result["total"]["repaired"]["supported"] == 1


def test_a_share_below_the_threshold_fails():
    sheets = _judged(_build())
    for row in sheets["items"]:
        if row["kind"] != "industry point":
            row.update(judgement="partly_supported", note="n")      # 7 of 9 = 77.8%
    result = _score(sheets, baseline={"supported": 31, "total": 37})  # 83.8% - 5 = 78.8%
    assert not result["checks"]["key_points_and_briefs_within_tolerance"] and not result["passes"]
