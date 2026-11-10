"""
Tests for src/summary_judging_fresh.py: judging and scoring the fresh-week summary
(SUMMARY_JUDGING_GUIDE_PART6.md). No model calls.

    sheet   every item judged from scratch (no carry-over), one industry row, the events of the selected
            articles, a key with the repairs, a reading page with the industry evidence first
    lock    the sheets, the key, the guides (Parts 1-3, 4 and 6), the freeze record, the fresh record, the
            fresh labels and events, and the judging code (this module, Part 4 and the shared helpers)
    score   refuses when any locked input changed; pass checks: produced, event coverage >= 80%, contradicted
            items / all items <= 5% unrounded, no central contradiction, no unsupported industry point, the
            industry section matches the evidence; the supported share of key points and briefs is reported
            next to the development result (32 of 40) and never decides the result
"""

import json

import pytest

from src import summary_judging as sj
from src import summary_judging_fresh as fr
from tests.test_summary_judging_p4 import EVENTS, LABELS, PASSAGES, _entry


def _record(industry=(("Rivals cut prices.", [3]),), status="ok"):
    return {"run": "r1", "design": "direct", "records": [
        _entry("r1", "direct", [("Offer raised.", [1]), ("The vote is held.", [2])], list(industry), status=status)]}


def _build(record=None):
    return fr.build_sheets(record or _record(), PASSAGES, EVENTS, LABELS, seed=3)


def _judged(sheets, evidence="yes", item="supported", covered="yes"):
    for row in sheets["items"]:
        row.update(judgement=item, contradicted="no", note="n" if item != "supported" else "")
    for row in sheets["events"]:
        row.update(covered=covered)
    sheets["industry"][0].update(industry_evidence=evidence, note="reason")
    return sheets


def _score(sheets, record=None):
    return fr.score(sheets["items"], sheets["events"], sheets["industry"], sheets["key"], record or _record(),
                    {"supported": 32, "total": 40}, "r1")


# ---------- the sheets ----------

def test_every_item_is_judged_from_scratch():
    sheets = _build()
    assert sheets["key"]["carried"] == [] and len(sheets["items"]) == 6            # 3 briefs, 2 points, 1 industry
    assert all(row["judgement"] == "" for row in sheets["items"])


def test_one_industry_row_and_the_events_of_the_selected_articles():
    sheets = _build()
    assert len(sheets["industry"]) == 1 and sheets["industry"][0]["industry_evidence"] == ""
    assert sorted(row["event"] for row in sheets["events"]) == ["offer", "vote"]


def test_the_page_shows_the_industry_evidence_first_and_hides_repairs_and_settings():
    page = _build()["page"]
    assert page.index("1. Industry evidence") < page.index("2. Items") < page.index("3. Event coverage")
    for word in ("direct", "repair", "prompt", "carried", "locked"):
        assert word not in page.lower()


def test_a_record_with_more_than_the_fresh_run_is_refused():
    record = _record()
    record["records"].append(_entry("r2", "direct", [("Sales rose.", [1])], []))
    with pytest.raises(fr.JudgingError):
        _build(record)


# ---------- the lock ----------

INPUTS = {"guides": ["g123", "g4", "g6"], "freeze": "frz", "record": "rec", "passages": "pas", "labels": "lab",
          "events_data": "evd", "development_score": "dev"}


def test_the_lock_needs_every_mark_and_saves_every_input():
    sheets = _build()
    with pytest.raises(fr.JudgingError):
        fr.lock_record(sheets["items"], sheets["events"], sheets["industry"], sheets["key"], INPUTS)
    sheets = _judged(sheets)
    lock = fr.lock_record(sheets["items"], sheets["events"], sheets["industry"], sheets["key"], INPUTS)
    assert set(lock["inputs"]) == {"guides", "freeze", "record", "passages", "labels", "events_data",
                                   "development_score"}
    assert fr.lock_problems(lock, sheets["items"], sheets["events"], sheets["industry"], sheets["key"], INPUTS) == []


@pytest.mark.parametrize("name, value", [("guides", ["g123", "g4", "CHANGED"]), ("freeze", "x"), ("record", "x"),
                                         ("passages", "x"), ("labels", "x"), ("events_data", "x"),
                                         ("development_score", "x")])
def test_the_score_refuses_when_any_locked_input_changes(name, value):
    sheets = _judged(_build())
    lock = fr.lock_record(sheets["items"], sheets["events"], sheets["industry"], sheets["key"], INPUTS)
    found = fr.lock_problems(lock, sheets["items"], sheets["events"], sheets["industry"], sheets["key"],
                             dict(INPUTS, **{name: value}))
    assert found and name in found[0]


def test_the_score_refuses_when_a_sheet_the_key_or_the_code_changes(monkeypatch):
    sheets = _judged(_build())
    lock = fr.lock_record(sheets["items"], sheets["events"], sheets["industry"], sheets["key"], INPUTS)
    changed = json.loads(json.dumps(sheets))
    changed["items"][0]["judgement"] = "not_supported"
    assert fr.lock_problems(lock, changed["items"], sheets["events"], sheets["industry"], sheets["key"], INPUTS)
    changed_key = dict(sheets["key"], seed=99)
    assert fr.lock_problems(lock, sheets["items"], sheets["events"], sheets["industry"], changed_key, INPUTS)
    monkeypatch.setattr(fr, "_code_fingerprint", lambda: "changed")
    assert any("code" in p for p in
               fr.lock_problems(lock, sheets["items"], sheets["events"], sheets["industry"], sheets["key"], INPUTS))


def test_the_code_fingerprint_covers_every_file_that_builds_the_sheets_and_the_score(monkeypatch, tmp_path):
    assert set(fr.CODE_FILES) >= {"src/summary_judging_fresh.py", "src/summary_judging_p4.py", "src/summary_judging.py",
                                  "src/summary_fresh.py", "src/summary_v2.py", "src/fresh_eval.py", "src/evaluate.py",
                                  "src/summary_run.py", "src/preprocess.py"}
    real = fr._code_files()
    before = fr._code_fingerprint()
    for index, path in enumerate(real):
        copy = tmp_path / f"{index}.py"
        copy.write_bytes(path.read_bytes() + b"\n# changed\n")
        changed = list(real)
        changed[index] = copy
        monkeypatch.setattr(fr, "_code_files", lambda changed=changed: changed)
        assert fr._code_fingerprint() != before, path
    monkeypatch.setattr(fr, "_code_files", lambda: real)


# ---------- the score ----------

def test_everything_passes_when_every_check_holds():
    result = _score(_judged(_build()))
    assert result["passes"] and all(result["checks"].values())
    assert result["key_points_and_briefs"]["supported"] == 5 and result["key_points_and_briefs"]["total"] == 5


def test_one_contradicted_item_in_fewer_than_twenty_fails_unrounded():
    sheets = _judged(_build())
    sheets["items"][0].update(judgement="partly_supported", contradicted="yes", central="no", note="n")
    result = _score(sheets)
    assert result["contradicted"] == {"count": 1, "total": 6, "share": 1 / 6}
    assert not result["checks"]["contradicted_at_most_5_percent"] and not result["passes"]


def test_the_threshold_is_compared_without_rounding():
    assert fr.contradiction_ok(1, 20) is True and fr.contradiction_ok(1, 19) is False      # 5.0% / 5.26%


def test_a_central_contradiction_fails():
    sheets = _judged(_build())
    sheets["items"][0].update(judgement="not_supported", contradicted="yes", central="yes", note="n")
    assert not _score(sheets)["checks"]["no_central_contradiction"]


def test_an_unsupported_industry_point_fails():
    sheets = _judged(_build())
    for row in sheets["items"]:
        if row["kind"] == "industry point":
            row.update(judgement="not_supported", note="n")
    assert not _score(sheets)["checks"]["no_unsupported_industry_point"]


def test_industry_points_need_evidence_and_an_empty_section_needs_none():
    assert not _score(_judged(_build(), evidence="no"))["checks"]["industry_section_matches_evidence"]
    empty = _record(industry=())
    assert not _score(_judged(_build(empty), evidence="yes"), empty)["checks"]["industry_section_matches_evidence"]
    assert _score(_judged(_build(empty), evidence="no"), empty)["checks"]["industry_section_matches_evidence"]


def test_event_coverage_below_80_percent_fails():
    assert not _score(_judged(_build(), covered="no"))["checks"]["event_coverage_at_least_80_percent"]


def test_no_summary_is_a_failed_result_with_nothing_to_judge():
    record = _record(status="unavailable")
    with pytest.raises(fr.JudgingError, match="no summary"):
        _build(record)
    result = fr.no_summary_result(record)
    assert result["passes"] is False and result["checks"] == {"summary_produced": False}


def test_the_supported_share_is_reported_and_never_decides_the_result():
    sheets = _judged(_build())
    for row in sheets["items"]:
        if row["kind"] != "industry point":
            row.update(judgement="partly_supported", note="n")      # 0 of 5 supported: far below 32 of 40
    result = _score(sheets)
    assert result["passes"]
    kb = result["key_points_and_briefs"]
    assert (kb["supported"], kb["total"], kb["development"]) == (0, 5, "32/40 (80.0%)")
    assert "key_points_and_briefs_within_tolerance" not in result["checks"]
    assert result["reported_only"]["within_development_tolerance"] is False


def test_the_sheet_command_stops_without_writing_anything_when_there_is_no_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(fr, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(fr, "_record_and_inputs", lambda: (_record(status="unavailable"), {}))
    with pytest.raises(fr.JudgingError, match="no summary"):
        fr._sheet_command()
    assert list(tmp_path.iterdir()) == []


def test_one_in_twenty_contradicted_passes_only_when_every_other_check_holds():
    assert fr.contradiction_ok(1, 20) is True
    sheets = _judged(_build(), covered="no")
    assert not _score(sheets)["passes"]                     # coverage fails, so the result fails



# ---------- the evidence is the one the model received ----------

def _saved(record):
    from src.summary_fresh import passage_entry
    return [passage_entry({"id": a["id"], "key": a["key"], "title": a["title"], "description": PASSAGES[a["key"]],
                           "content": ""}) for a in record["records"][0]["articles"]]


def _with_prints(record):
    record = json.loads(json.dumps(record))
    record["passage_fingerprints"] = [{"id": p["id"], "key": p["key"], "sha256": p["sha256"]} for p in _saved(record)]
    return record


def test_the_page_uses_the_saved_passages_when_they_match_the_record():
    record = _with_prints(_record())
    assert fr.passages_from_record(record, _saved(record)) == {k: v for k, v in PASSAGES.items() if k.startswith("Kr1")}


def test_a_changed_saved_passage_is_refused():
    record = _with_prints(_record())
    saved = _saved(record)
    saved[1]["description"] = "A rebuilt passage."
    with pytest.raises(fr.JudgingError, match="differs from the one the model received"):
        fr.passages_from_record(record, saved)


def test_passages_in_another_order_or_with_other_ids_are_refused():
    record = _with_prints(_record())
    with pytest.raises(fr.JudgingError, match="IDs or order"):
        fr.passages_from_record(record, list(reversed(_saved(record))))
    other = _saved(record)
    other[0]["id"] = 99
    with pytest.raises(fr.JudgingError, match="IDs or order"):
        fr.passages_from_record(record, other)


# ---------- the full 1-in-20 case ----------

def _twenty():
    points = [(f"Point {n}.", [1]) for n in range(16)]         # 3 briefs + 16 company points + 1 industry point = 20
    return {"run": "r1", "design": "direct",
            "records": [_entry("r1", "direct", points, [("Rivals cut prices.", [3])])]}


def test_twenty_items_with_one_contradicted_pass_when_every_other_check_holds():
    record = _twenty()
    sheets = _judged(_build(record))
    assert len(sheets["items"]) == 20
    sheets["items"][0].update(judgement="partly_supported", contradicted="yes", central="no", note="n")
    result = _score(sheets, record)
    assert result["contradicted"] == {"count": 1, "total": 20, "share": 0.05}
    assert result["passes"] and all(result["checks"].values())
    sheets["events"][0]["covered"] = "no"                       # one event of two: coverage 50%
    failed = _score(sheets, record)
    assert failed["checks"]["contradicted_at_most_5_percent"] and not failed["passes"]
