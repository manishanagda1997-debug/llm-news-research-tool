"""
Tests for src/summary_judging_p5.py: the prompt version 5 rerun judged and scored with Part 4 unchanged
(SUMMARY_JUDGING_GUIDE_PART5.md). No model calls.

    carried items     exact section, text and cited articles of a locked direct item of Part 3 or Part 4;
                      a clash between different locked judgements is judged again
    carried evidence  the locked Part 4 industry decision of each run, not judged again
    lock and score    both sheets, the key and the code (this module, Part 4 and the shared helpers); the Part 4
                      pass rule unchanged
"""

import json

import pytest

from src import summary_judging as sj
from src import summary_judging_p4 as p4
from src import summary_judging_p5 as p5
from tests.test_summary_judging_p4 import EVENTS, LABELS, LOCKED_RECORD, P4_RECORD, PASSAGES, _entry, _locked


def _p4_locked():
    """A locked Part 4 sheet on P4_RECORD: every new item supported; evidence r1 no, r2 yes."""
    items, key = _locked()
    sheets = p4.build_sheets(P4_RECORD, PASSAGES, EVENTS, LABELS, p4.locked_items(items, key), seed=5)
    for row in sheets["items"]:
        row.update(judgement="supported", contradicted="no")
    by_run = {sheets["key"]["summaries"][r["summary"]]["run"]: r for r in sheets["industry"]}
    by_run["r1"].update(industry_evidence="no", note="only deal parties")
    by_run["r2"].update(industry_evidence="yes", note="a rival's sales")
    return items, key, sheets


# the version 5 rerun: r1 keeps "Offer raised." [1] (Part 3) and "The vote is held." [2] (Part 4), adds an
# industry point; r2 keeps "The market grows." [2] (Part 4) and rewrites "Sales rose."
P5_RECORD = {"prompt_version": "5", "designs": ["direct"], "records": [
    _entry("r1", "direct", [("Offer raised.", [1]), ("The vote is held.", [2])], [("Rivals cut prices.", [3])]),
    _entry("r2", "direct", [("Sales rose 5%.", [1])], [("The market grows.", [2])],
           briefs=["Brief text r2 1.", "A new brief r2 2.", "Brief text r2 3."]),
]}


def _sources():
    p3_items, p3_key, p4_sheets = _p4_locked()
    locked, sheet = p5.locked_sources(p3_items, p3_key, p4_sheets["items"], p4_sheets["key"])
    evidence = p5.carried_evidence(p4_sheets["industry"], p4_sheets["key"])
    return locked, sheet, evidence, (p3_items, p3_key)


def _build(record=P5_RECORD):
    locked, sheet, evidence, _ = _sources()
    return p5.build_sheets(record, PASSAGES, EVENTS, LABELS, locked, evidence, seed=9)


# ---------- the carried judgements ----------

def test_items_carry_over_from_either_locked_sheet_only_on_an_exact_match():
    sheets = _build()
    sources = {c["locked_row"].split(":")[0] for c in sheets["key"]["carried"]}
    assert sources == {"p3", "p4"}
    texts = {row["text"] for row in sheets["items"]}
    assert texts == {"Rivals cut prices.", "Sales rose 5%."}                 # only the new and changed items
    carried_texts = {c["locked_row"] for c in sheets["key"]["carried"]}
    assert len(carried_texts) == len(sheets["key"]["carried"])


def test_the_carried_rows_point_at_the_locked_rows_with_the_same_text():
    locked, sheet, _, _ = _sources()
    rows = {r["row"]: r for r in sheet}
    for sig, name in locked.items():
        assert rows[name]["text"] == sig[1] and rows[name]["evidence"] == sig[2]


def test_brief_first_items_of_part_3_never_carry_over():
    _, sheet, _, _ = _sources()
    p3_items, p3_key = _locked()
    brief_first = {n for n, e in p3_key["items"].items() if e["design"] == "brief_first"}
    assert not any(r["row"] == f"p3:{n}" for r in sheet for n in brief_first)


def test_different_locked_judgements_for_the_same_item_are_judged_again():
    p3_items, p3_key, p4_sheets = _p4_locked()
    offer = next(r for r in p4_sheets["items"] if r["text"] == "The vote is held.")
    twin = dict(offer, row=str(len(p4_sheets["items"]) + 1), judgement="partly_supported", note="n")
    p4_sheets["items"].append(twin)
    p4_sheets["key"]["items"][twin["row"]] = dict(p4_sheets["key"]["items"][str(offer["row"])])
    locked, _ = p5.locked_sources(p3_items, p3_key, p4_sheets["items"], p4_sheets["key"])
    assert ("key_points", "The vote is held.", "Kr12") not in locked


def test_a_clash_between_part_3_and_part_4_is_judged_again():
    p3_items, p3_key, p4_sheets = _p4_locked()
    p4_sheets["key"]["items"]["99"] = {"design": "direct", "run": "r1", "section": "key_points", "index": 0,
                                       "ids": [1], "repaired": False, "fingerprint": ""}
    p4_sheets["items"].append({"row": "99", "kind": "company point", "text": "Offer raised.", "evidence": "Kr11",
                               "judgement": "not_supported", "contradicted": "no", "central": "", "note": "n"})
    locked, _ = p5.locked_sources(p3_items, p3_key, p4_sheets["items"], p4_sheets["key"])
    assert ("key_points", "Offer raised.", "Kr11") not in locked


# ---------- the carried evidence ----------

def test_industry_evidence_is_carried_by_run_and_not_shown_for_judging():
    sheets = _build()
    carried = sheets["key"]["industry_carried"]
    assert {run: c["industry_evidence"] for run, c in carried.items()} == {"r1": "no", "r2": "yes"}
    assert "industry" not in sheets and "industry" not in sheets["key"]
    rows = p5.industry_rows(sheets["key"])
    assert {sheets["key"]["summaries"][r["summary"]]["run"]: r["industry_evidence"] for r in rows} == \
        {"r1": "no", "r2": "yes"}


def test_a_run_without_a_locked_evidence_decision_is_refused():
    locked, _, evidence, _ = _sources()
    evidence.pop("r2")
    with pytest.raises(p5.JudgingError):
        p5.build_sheets(P5_RECORD, PASSAGES, EVENTS, LABELS, locked, evidence, seed=9)


def test_the_page_has_the_items_and_the_events_but_no_evidence_section():
    page = _build()["page"]
    assert "Industry evidence" not in page and "Part 5" in page
    assert page.index("1. Items") < page.index("2. Event coverage")


def test_nothing_shown_names_a_design_a_repair_or_a_carried_judgement():
    sheets = _build()
    shown = (json.dumps(sheets["items"]) + json.dumps(sheets["events"]) + sheets["page"]).lower()
    for word in ("direct", "design", "prompt", "repair", '"run"', "locked", "carried", "p3:", "p4:"):
        assert word not in shown


# ---------- lock and score ----------

def _judged(sheets, item="supported", covered="yes"):
    for row in sheets["items"]:
        row.update(judgement=item, contradicted="no", note="n" if item != "supported" else "")
    for row in sheets["events"]:
        row.update(covered=covered)
    return sheets


def test_the_lock_needs_every_mark_and_notices_changes_of_sheets_key_and_code(monkeypatch):
    sheets = _build()
    with pytest.raises(p5.JudgingError):
        p5.lock_record(sheets["items"], sheets["events"], sheets["key"], ["g4", "g5"], "record")
    sheets = _judged(sheets)
    lock = p5.lock_record(sheets["items"], sheets["events"], sheets["key"], ["g4", "g5"], "record")
    p5.check_lock(lock, sheets["items"], sheets["events"], sheets["key"])
    assert len(lock["guide_fingerprints"]) == 2
    changed = json.loads(json.dumps(sheets["key"]))
    changed["industry_carried"]["r1"]["industry_evidence"] = "yes"
    with pytest.raises(p5.JudgingError, match="key"):
        p5.check_lock(lock, sheets["items"], sheets["events"], changed)
    monkeypatch.setattr(p5, "_code_fingerprint", lambda: "changed")
    with pytest.raises(p5.JudgingError, match="code"):
        p5.check_lock(lock, sheets["items"], sheets["events"], sheets["key"])


def test_the_code_fingerprint_covers_part_4_and_the_shared_helpers(monkeypatch, tmp_path):
    before = p5._code_fingerprint()
    for module in (p4, sj):
        copy = tmp_path / f"{module.__name__}.py"
        copy.write_bytes(open(module.__file__, "rb").read() + b"\n# changed\n")
        monkeypatch.setattr(module, "__file__", str(copy))
        assert p5._code_fingerprint() != before
        monkeypatch.undo()


def _score(sheets, record=P5_RECORD, runs=("r1", "r2")):
    _, sheet, _, (p3_items, p3_key) = _sources()
    return p5.score(sheets["items"], sheets["events"], sheets["key"], record, sheet,
                    p4.locked_baseline(p3_items, p3_key), list(runs))


def test_carried_items_count_with_their_locked_judgement_from_either_sheet():
    result = _score(_judged(_build()))
    assert result["total"]["carried"] == len(_build()["key"]["carried"])
    assert result["industry_evidence"] == "carried from the locked Part 4 sheet"


def test_an_industry_point_in_a_run_whose_evidence_is_no_fails():
    result = _score(_judged(_build()))                  # r1 has an industry point, its evidence is no
    assert result["points_without_evidence"] == ["r1"]
    assert not result["checks"]["industry_points_only_with_evidence"] and not result["passes"]


def test_an_empty_section_in_a_run_whose_evidence_is_yes_fails():
    record = json.loads(json.dumps(P5_RECORD))
    record["records"][0]["result"]["summary"]["industry_context"] = []
    record["records"][1]["result"]["summary"]["industry_context"] = []
    result = _score(_judged(_build(record)), record)
    assert result["empty_section_with_evidence"] == ["r2"] and not result["passes"]


def test_the_part_4_pass_rule_passes_when_every_condition_holds():
    record = json.loads(json.dumps(P5_RECORD))
    record["records"][0]["result"]["summary"]["industry_context"] = []          # r1: evidence no, empty
    result = _score(_judged(_build(record)), record)
    assert result["passes"] and all(result["checks"].values())


def test_the_threshold_is_exactly_29_of_38_minus_5_points_unrounded():
    record = json.loads(json.dumps(P5_RECORD))
    record["records"][0]["result"]["summary"]["industry_context"] = []
    sheets = _judged(_build(record))
    _, sheet, _, _ = _sources()
    result = p5.score(sheets["items"], sheets["events"], sheets["key"], record, sheet,
                      {"supported": 29, "total": 38}, ["r1", "r2"])
    assert result["key_points_and_briefs"]["threshold"] == 29 / 38 - 0.05
    assert result["key_points_and_briefs"]["threshold"] != 0.7132


def test_a_missing_development_run_fails():
    assert not _score(_judged(_build()), runs=("r1", "r2", "r3"))["checks"]["every_development_run_has_one_summary"]


def test_a_record_with_another_design_is_refused():
    locked, _, evidence, _ = _sources()
    with pytest.raises(sj.JudgingError):
        p5.build_sheets(LOCKED_RECORD, PASSAGES, EVENTS, LABELS, locked, evidence, seed=9)
