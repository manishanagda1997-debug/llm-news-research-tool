"""
Tests for src/summary_judging.py: the blind sheets for comparing the two summary designs
(SUMMARY_JUDGING_GUIDE.md, Part 3). No model calls.

    sheet          items (company points, industry points, article briefs) from every produced summary,
                   shuffled; the evidence column holds article IDs only; an event sheet with one row per
                   relevant event of each summary, under a random letter; a key; a reading page (outside Git)
    lock           every judgement present and valid, then fingerprints saved before anything is un-blinded
    score          refuses without a lock or after a change; per design: produced, support, contradictions,
                   central contradictions, event coverage (human and by citation), the acceptance rule and
                   the choice order
    rejudge        20% of the items, new order, new row numbers, no first judgements; only after the lock
"""

import json

import pytest

from src import summary_judging as sj


def _headline(i, run):
    return {"id": i, "title": f"Story {run} {i}", "source": "S", "published_at": "", "url": f"https://x.com/{run}/{i}"}


def _summary(run, points, briefs=3):
    return {"article_briefs": [{"id": i, "brief": f"Brief text {run} {i}.", "enough_text": True, "hidden": False}
                               for i in range(1, briefs + 1)],
            "key_points": [{"text": t, "citations": c} for t, c in points],
            "industry_context": [{"text": f"Sector note {run}.", "citations": [3]}],
            "not_used": []}


def _record_entry(run, design, status, points, calls):
    summary = _summary(run, points) if status in ("ok", "partial") else None
    return {"run": run, "design": design, "k": 3,
            "articles": [{"id": i, "key": f"K{run}{i}", "title": f"Story {run} {i}"} for i in range(1, 4)],
            "result": {"status": status, "summary": summary, "headlines": [_headline(i, run) for i in range(1, 4)],
                       "usage": {"calls": calls}}}


RECORD = {"prompt_version": "3", "records": [
    _record_entry("r1", "direct", "ok", [("Offer raised.", [1]), ("Vote held.", [2])], 2),
    _record_entry("r1", "brief_first", "ok", [("Offer raised again.", [1, 2])], 7),
    _record_entry("r2", "direct", "unavailable", [], 2),
    _record_entry("r2", "brief_first", "ok", [("Sales rose.", [1])], 6),
]}
PASSAGES = {f"K{run}{i}": f"Passage text {run} {i}." for run in ("r1", "r2") for i in range(1, 4)}
EVENTS = {"https://x.com/r1/1": "offer", "https://x.com/r1/2": "vote", "https://x.com/r1/3": "sector",
          "https://x.com/r2/1": "sales", "https://x.com/r2/2": "sales", "https://x.com/r2/3": "other"}
LABELS = {"https://x.com/r1/1": "direct", "https://x.com/r1/2": "direct", "https://x.com/r1/3": "not",
          "https://x.com/r2/1": "direct", "https://x.com/r2/2": "indirect", "https://x.com/r2/3": "direct"}


def _build():
    return sj.build_sheets(RECORD, PASSAGES, EVENTS, LABELS, seed=7)


# ---------- the sheets ----------

def test_every_item_of_every_produced_summary_is_a_row():
    sheets = _build()
    # r1 direct: 3 briefs + 2 points + 1 industry; r1 brief_first: 3 + 1 + 1; r2 brief_first: 3 + 1 + 1
    assert len(sheets["items"]) == 6 + 5 + 5
    kinds = {row["kind"] for row in sheets["items"]}
    assert kinds == {"company point", "industry point", "article brief"}
    assert [row["row"] for row in sheets["items"]] == list(range(1, 17))


def test_hidden_briefs_are_not_judged():
    record = json.loads(json.dumps(RECORD))
    record["records"][0]["result"]["summary"]["article_briefs"][0].update(brief="", hidden=True)
    assert len(sj.build_sheets(record, PASSAGES, EVENTS, LABELS, seed=7)["items"]) == 15


def test_the_evidence_column_holds_article_ids_only():
    sheets = _build()
    for row in sheets["items"]:
        assert all(part in PASSAGES for part in row["evidence"].split("; "))
        assert "Passage text" not in json.dumps(row)


def test_the_judgement_columns_start_empty():
    for row in _build()["items"]:
        assert row["judgement"] == row["contradicted"] == row["central"] == row["note"] == ""


def test_the_items_are_shuffled_with_a_fixed_seed():
    first, second = _build(), _build()
    assert first["items"] == second["items"]
    designs = [first["key"]["items"][str(row["row"])]["design"] for row in first["items"]]
    switches = sum(1 for a, b in zip(designs, designs[1:]) if a != b)
    assert switches > 2                     # not grouped by summary: three summaries in blocks give at most 2


def test_nothing_shown_names_a_design_or_its_settings():
    sheets = _build()
    shown = (json.dumps(sheets["items"]) + json.dumps(sheets["events"]) + sheets["page"]).lower()
    for word in ("direct", "brief_first", "brief first", "design", "prompt", "repair", '"run"', "calls"):
        assert word not in shown


def test_the_event_sheet_has_one_row_per_relevant_event_of_each_produced_summary():
    sheets = _build()
    by_letter = {}
    for row in sheets["events"]:
        by_letter.setdefault(row["summary"], []).append(row["event"])
    assert sorted(len(events) for events in by_letter.values()) == [2, 2, 2]   # r1: offer, vote; r2: sales, other
    assert all(row["covered"] == "" for row in sheets["events"])
    letters = {sheets["key"]["summaries"][letter]["run"] for letter in by_letter}
    assert letters == {"r1", "r2"}


def test_an_event_row_lists_the_titles_of_its_articles_in_that_summary():
    sheets = _build()
    sales = [row for row in sheets["events"] if row["event"] == "sales"][0]
    assert sales["articles"] == "Story r2 1 | Story r2 2"


def test_the_page_shows_each_item_with_its_evidence_text_and_each_summarys_points():
    sheets = _build()
    assert "Passage text r1 1" in sheets["page"]
    for letter in {row["summary"] for row in sheets["events"]}:
        assert f"Summary {letter}" in sheets["page"]


# ---------- the lock: complete and valid before un-blinding ----------

def _judged(sheets, **override):
    items = [dict(row, judgement="supported", contradicted="no") for row in sheets["items"]]
    events = [dict(row, covered="yes") for row in sheets["events"]]
    for row_number, values in override.get("items", {}).items():
        items[row_number - 1].update(values)
    return items, events


def test_a_complete_valid_sheet_has_no_problems():
    sheets = _build()
    items, events = _judged(sheets)
    assert sj.problems(items, events, sheets["key"]) == []


@pytest.mark.parametrize("change, message", [
    ({"judgement": ""}, "judgement"),
    ({"judgement": "maybe"}, "judgement"),
    ({"contradicted": ""}, "contradicted"),
    ({"contradicted": "yes", "judgement": "not_supported", "note": "wrong vote"}, "central"),
    ({"contradicted": "no", "central": "yes"}, "central"),
    ({"contradicted": "yes", "central": "no", "judgement": "supported", "note": "x"}, "never supported"),
    ({"judgement": "partly_supported"}, "note"),
    ({"text": "changed text"}, "changed"),
])
def test_an_incomplete_or_invalid_item_is_refused(change, message):
    sheets = _build()
    items, events = _judged(sheets, items={1: change})
    found = sj.problems(items, events, sheets["key"])
    assert found and message in found[0]


def test_an_empty_coverage_mark_is_refused_and_never_read_as_no():
    sheets = _build()
    items, events = _judged(sheets)
    events[0]["covered"] = ""
    assert any("covered" in p for p in sj.problems(items, events, sheets["key"]))


def test_a_missing_or_extra_row_is_refused():
    sheets = _build()
    items, events = _judged(sheets)
    assert sj.problems(items[1:], events, sheets["key"])
    assert sj.problems(items, events + [dict(events[0], row=99)], sheets["key"])


def test_the_fingerprint_ignores_spacing_and_case_but_not_judgements():
    sheets = _build()
    items, _ = _judged(sheets)
    spaced = [dict(row, judgement=f"  {row['judgement'].upper()} ") for row in items]
    assert sj.sheet_fingerprint(items) == sj.sheet_fingerprint(spaced)
    changed = [dict(row) for row in items]
    changed[0]["judgement"] = "not_supported"
    assert sj.sheet_fingerprint(items) != sj.sheet_fingerprint(changed)


def test_score_refuses_a_sheet_changed_after_the_lock():
    sheets = _build()
    items, events = _judged(sheets)
    lock = sj.lock_record(items, events, sheets["key"], guide_text="guide", record_text="record")
    changed = [dict(row) for row in items]
    changed[0]["judgement"] = "not_supported"
    with pytest.raises(sj.JudgingError):
        sj.check_lock(lock, changed, events)
    sj.check_lock(lock, items, events)          # unchanged: no error


def test_the_lock_records_every_fingerprint_needed_to_repeat_the_result():
    sheets = _build()
    items, events = _judged(sheets)
    lock = sj.lock_record(items, events, sheets["key"], guide_text="guide", record_text="record")
    for name in ("items_fingerprint", "events_fingerprint", "guide_fingerprint", "record_fingerprint",
                 "code_fingerprint", "scoring_version", "locked_at"):
        assert lock[name]


# ---------- the score ----------

def _row_of(sheets, design, run, section, index=None):
    for number, entry in sheets["key"]["items"].items():
        if (entry["design"], entry["run"], entry["section"]) == (design, run, section) and \
                (index is None or entry["index"] == index):
            return int(number)
    raise KeyError


def test_the_score_counts_each_design_on_its_own():
    sheets = _build()
    contradiction = {"judgement": "not_supported", "contradicted": "yes", "central": "yes", "note": "wrong offer"}
    items, events = _judged(sheets, items={_row_of(sheets, "direct", "r1", "key_points", 0): contradiction})
    score = sj.score(items, events, sheets["key"], RECORD)
    direct, brief = score["designs"]["direct"], score["designs"]["brief_first"]
    assert direct["produced"] == {"count": 1, "total": 2, "share": 0.5}
    assert direct["items"]["total"] == 6 and direct["items"]["contradicted"] == 1
    assert direct["items"]["central_contradicted"] == 1
    assert brief["produced"] == {"count": 2, "total": 2, "share": 1.0}
    assert brief["items"]["supported_share"] == 1.0


def test_event_coverage_uses_the_human_marks_and_reports_citations_apart():
    sheets = _build()
    items, events = _judged(sheets)
    for row in events:
        if sheets["key"]["summaries"][row["summary"]]["design"] == "brief_first" and row["event"] == "vote":
            row["covered"] = "no"
    score = sj.score(items, events, sheets["key"], RECORD)
    brief = score["designs"]["brief_first"]["events"]
    assert brief["covered"] == 3 and brief["total"] == 4 and brief["coverage"] == 0.75
    assert brief["cited"] == 4        # r1: offer and vote by one point; r2: sales by a point, "other" by the industry point


def test_the_acceptance_rule_and_the_choice_order():
    sheets = _build()
    items, events = _judged(sheets)
    score = sj.score(items, events, sheets["key"], RECORD)
    assert score["designs"]["direct"]["passes"] is False          # 1 of 2 produced
    assert score["designs"]["direct"]["checks"]["produced"] is False
    assert score["designs"]["brief_first"]["passes"] is True
    assert score["choice"] == "brief_first"


def test_no_design_is_chosen_when_none_passes():
    sheets = _build()
    contradiction = {"judgement": "not_supported", "contradicted": "yes", "central": "yes", "note": "wrong"}
    items, events = _judged(sheets, items={_row_of(sheets, "brief_first", "r2", "key_points"): contradiction})
    score = sj.score(items, events, sheets["key"], RECORD)
    assert score["designs"]["brief_first"]["checks"]["no_central_contradiction"] is False
    assert score["choice"] is None


def test_a_tie_goes_to_fewer_calls():
    assert sj.choose({"direct": {"passes": True, "items": {"supported_share": 0.9}, "events": {"coverage": 0.9},
                                 "calls": 2.0},
                      "brief_first": {"passes": True, "items": {"supported_share": 0.9},
                                      "events": {"coverage": 0.9}, "calls": 7.0}}) == "direct"


# ---------- the consistency check ----------

def test_the_rejudge_sheet_holds_a_fifth_of_the_items_in_a_new_order_without_judgements():
    sheets = _build()
    rows, rejudge_key = sj.rejudge_rows(sheets["items"], seed=11)
    assert len(rows) == 4                                       # 20% of 16, rounded up
    assert [row["row"] for row in rows] == [1, 2, 3, 4]
    assert all(row["judgement"] == "" for row in rows)
    originals = {int(v) for v in rejudge_key.values()}
    assert originals <= {row["row"] for row in sheets["items"]}
    first = {row["row"]: row["text"] for row in sheets["items"]}
    assert all(first[rejudge_key[str(row["row"])]] == row["text"] for row in rows)


def test_agreement_reports_the_share_and_the_three_by_three_table():
    sheets = _build()
    items, _ = _judged(sheets)
    rows, rejudge_key = sj.rejudge_rows(sheets["items"], seed=11)
    second = [dict(row, judgement="supported", contradicted="no") for row in rows]
    second[0]["judgement"] = "partly_supported"
    second[0]["note"] = "extra date"
    result = sj.agreement(items, second, rejudge_key)
    assert result["items"] == 4 and result["judgement_agreement"] == 0.75
    assert result["table"]["supported"]["partly_supported"] == 1
    assert result["contradicted_agreement"] == 1.0


def test_the_rejudge_never_opens_the_key_or_shows_first_judgements(tmp_path, monkeypatch):
    """The rejudge reads the lock and the items sheet only; with no key file present it still works."""
    sheets = _build()
    items, events = _judged(sheets)
    items[0].update(judgement="partly_supported", note="first note")
    eval_dir, fulltext_dir = tmp_path / "eval", tmp_path / "fulltext"
    eval_dir.mkdir()
    fulltext_dir.mkdir()
    lock = sj.lock_record(items, events, sheets["key"], guide_text="guide", record_text="record",
                          record_file="record.json")
    (eval_dir / sj.LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    (eval_dir / sj.ITEMS_FILE).write_text(sj._csv_text(items, sj.ITEM_COLUMNS), encoding="utf-8")
    (eval_dir / "record.json").write_text(json.dumps(RECORD), encoding="utf-8")
    monkeypatch.setattr(sj, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(sj, "_passages", lambda record: PASSAGES)
    monkeypatch.setattr("src.fulltext.FULLTEXT_DIR", fulltext_dir)
    sj._rejudge_command()
    assert not (eval_dir / sj.KEY_FILE).exists()
    rows = sj._read_csv(eval_dir / sj.REJUDGE_FILE)
    assert list(rows[0]) == sj.ITEM_COLUMNS
    assert all(r["judgement"] == r["contradicted"] == r["central"] == r["note"] == "" for r in rows)
    page = (fulltext_dir / sj.REJUDGE_PAGE_FILE).read_text(encoding="utf-8").lower()
    assert "first note" not in page and "partly_supported" not in page


# ---------- repaired points: marked in the key only, scored on their own ----------

def _record_with_repair():
    record = json.loads(json.dumps(RECORD))
    entry = record["records"][1]                                   # r1 brief_first: one company point
    entry["result"]["repairs"] = [{"section": "key_points", "index": 0, "ok": True,
                                   "before": "Offer raised again to 99.", "after": "Offer raised again."},
                                  {"section": "key_points", "index": 5, "ok": False, "before": "x", "after": None}]
    return record


def test_a_repaired_point_is_marked_in_the_key_but_never_on_the_sheet():
    sheets = sj.build_sheets(_record_with_repair(), PASSAGES, EVENTS, LABELS, seed=7)
    repaired = [n for n, e in sheets["key"]["items"].items() if e["repaired"]]
    assert len(repaired) == 1
    row = next(r for r in sheets["items"] if str(r["row"]) == repaired[0])
    assert row["text"] == "Offer raised again."
    assert "repair" not in json.dumps(sheets["items"]).lower() and "repair" not in sheets["page"].lower()


def test_the_score_reports_repaired_points_on_their_own():
    record = _record_with_repair()
    sheets = sj.build_sheets(record, PASSAGES, EVENTS, LABELS, seed=7)
    repaired = next(int(n) for n, e in sheets["key"]["items"].items() if e["repaired"])
    items, events = _judged(sheets, items={repaired: {"judgement": "partly_supported", "note": "extra claim"}})
    score = sj.score(items, events, sheets["key"], record)
    part = score["designs"]["brief_first"]["repaired_items"]
    assert part == {"total": 1, "supported": 0, "supported_share": 0.0, "contradicted": 0}
    assert score["designs"]["direct"]["repaired_items"]["total"] == 0
