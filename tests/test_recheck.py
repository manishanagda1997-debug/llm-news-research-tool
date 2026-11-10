"""
Tests for the blind relevance re-check of the guide version 3 labels (no network, no model calls).

The re-check labels a set of already judged articles a second time, without
the first label, and measures how well the two labellings agree.

    pool      the 74 articles judged on their own text: 71 from full text and
              3 from short text because the publisher blocks the page
    Part A    40 articles drawn from the pool with a fixed seed, in the same
              label mix as the pool; agreement and kappa use Part A only
    Part B    the 8 articles where the model and the label disagree; reported
              on their own
    sheet     Part A and Part B together, each article once, shuffled; no
              column shows the part or the first label
    page      the same text the first labelling uses, checked by fingerprint
    order     sheet (not before 1 November) -> check -> score -> adjudication
              -> checked label file; the version 3 label file never changes
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import recheck as rc

ROOT = Path(__file__).resolve().parent.parent
V3_FILE = ROOT / "data" / "eval" / "v3_dev_tata_motors.csv"
TRACKED = ROOT / "data" / "eval" / "stage2_tracked_v1.json"


def _v3_rows():
    with open(V3_FILE, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _row(article_id, label, basis="full text", title=None):
    return {"article_id": article_id, "title": title or f"Title {article_id}", "url": f"https://x.com/{article_id}",
            "label": label, "unsure": "", "note": "", "basis": basis, "label_v2": label, "changed": "",
            "reason": "", "guide_version": "3"}


# ---------- the pool, Part A and Part B ----------

def test_the_pool_is_the_74_articles_judged_on_their_own_text():
    pool = rc.recheck_pool(_v3_rows())
    assert len(pool) == 74
    bases = {r["basis"] for r in pool}
    assert bases == {"full text", "short text (page blocked by robots.txt)"}


def test_sale_listings_and_copies_stay_out_of_the_pool():
    rows = [_row("A1", "not", "short text (sale listing, not fetched)"),
            _row("A2", "direct", "short text (copy of a fetched article)"), _row("A3", "direct")]
    assert [r["article_id"] for r in rc.recheck_pool(rows)] == ["A3"]


def test_part_b_is_the_seven_tracked_misses_and_the_waymo_article():
    tracked = json.loads(TRACKED.read_text(encoding="utf-8"))
    assert list(rc.PART_B[:7]) == tracked["false_negatives"]
    assert rc.PART_B[7] == "Aac357bfe35" and len(set(rc.PART_B)) == 8


def test_every_part_b_article_is_in_the_pool():
    pool_ids = {r["article_id"] for r in rc.recheck_pool(_v3_rows())}
    assert set(rc.PART_B) <= pool_ids


def test_part_a_has_40_articles_in_the_label_mix_of_the_pool():
    pool = rc.recheck_pool(_v3_rows())
    chosen = rc.part_a(pool)
    labels = {r["article_id"]: r["label"] for r in pool}
    mix = {name: sum(1 for i in chosen if labels[i] == name) for name in ("direct", "indirect", "not")}
    assert len(chosen) == 40 and mix == {"direct": 13, "indirect": 10, "not": 17}


def test_part_a_is_the_same_every_time_and_depends_on_the_seed():
    pool = rc.recheck_pool(_v3_rows())
    assert rc.part_a(pool) == rc.part_a(list(reversed(pool)))          # the file order gives no difference
    assert rc.part_a(pool) != rc.part_a(pool, seed=rc.SEED + 1)


def test_part_a_can_hold_part_b_articles_so_hard_cases_are_not_removed():
    pool = [_row(f"A{n}", "direct" if n < 5 else "not") for n in range(10)]
    chosen = rc.part_a(pool, n=10)
    assert chosen == {f"A{n}" for n in range(10)}


def test_the_mix_uses_the_largest_remainder_and_always_gives_n():
    pool = [_row(f"D{n}", "direct") for n in range(24)] + [_row(f"I{n}", "indirect") for n in range(19)] \
        + [_row(f"N{n}", "not") for n in range(31)]
    assert rc.stratum_sizes(pool, 40) == {"direct": 13, "indirect": 10, "not": 17}
    assert sum(rc.stratum_sizes(pool, 7).values()) == 7


# ---------- the sheet ----------

def test_the_sheet_lists_each_article_of_both_parts_once():
    pool = rc.recheck_pool(_v3_rows())
    rows = rc.sheet_rows(pool)
    ids = [r["article_id"] for r in rows]
    assert len(ids) == len(set(ids)) == len(rc.amended_part_a(pool) | set(rc.PART_B))


def test_the_sheet_hides_the_first_label_and_the_part():
    rows = rc.sheet_rows(rc.recheck_pool(_v3_rows()))
    assert list(rows[0]) == rc.SHEET_COLUMNS == ["row", "article_id", "title", "url", "label", "unsure", "note"]
    assert all(r["label"] == r["unsure"] == r["note"] == "" for r in rows)


def test_the_sheet_is_shuffled_the_same_way_every_time():
    pool = rc.recheck_pool(_v3_rows())
    first = [r["article_id"] for r in rc.sheet_rows(pool)]
    assert first == [r["article_id"] for r in rc.sheet_rows(pool)]
    assert first[-8:] != list(rc.PART_B) and sorted(first) != first
    assert [r["row"] for r in rc.sheet_rows(pool)] == list(range(1, len(first) + 1))


def test_the_sheet_is_refused_before_1_november():
    with pytest.raises(rc.RecheckError):
        rc.check_start(datetime(2026, 10, 31, 23, 59, tzinfo=timezone.utc))
    rc.check_start(datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc))


# ---------- the same evidence ----------

OLD_PAGE = (
    "<html><body><h1>Full-text labelling</h1>"
    '<section id="A1"><h2>Row 1 — One</h2><p class=\'meta\'>Article id: A1 | S | d</p>'
    "<p class='status'>Text: complete</p><p>First &amp; paragraph.</p><p>Second paragraph.</p></section>"
    '<section id="A2"><h2>Row 2 — Two</h2><p class=\'meta\'>Article id: A2 | S | d</p>'
    "<p class='status'>Text: complete</p><p>Only paragraph.</p></section>"
    "</body></html>")


def test_the_old_page_is_read_into_paragraphs_per_article():
    assert rc.parse_page(OLD_PAGE) == {"A1": ["First & paragraph.", "Second paragraph."], "A2": ["Only paragraph."]}


def test_a_fingerprint_is_the_sha256_of_the_paragraphs():
    expected = hashlib.sha256("First & paragraph.\nSecond paragraph.".encode("utf-8")).hexdigest()
    assert rc.fingerprint(["First & paragraph.", "Second paragraph."]) == expected
    assert rc.fingerprint(["First & paragraph."]) != expected


def test_blocked_articles_use_the_short_text_they_were_labelled_from():
    short = {"title": "T", "description": "D.", "content_first_200_chars": "C"}
    assert rc.short_text_paragraphs(short) == ["T", "D.", "C"]
    assert rc.short_text_paragraphs({"title": "T", "description": "", "content_first_200_chars": "C"}) == ["T", "C"]


def _items():
    return [{"row": 1, "article_id": "A2", "title": "Two", "url": "https://x.com/A2", "source": "S",
             "published": "d", "status": "complete", "paragraphs": ["Only paragraph."]},
            {"row": 2, "article_id": "A1", "title": "One", "url": "https://x.com/A1", "source": "S",
             "published": "d", "status": "complete", "paragraphs": ["First & paragraph.", "Second paragraph."]}]


def test_the_new_page_gives_back_exactly_the_old_text():
    page = rc.recheck_page(_items())
    assert rc.parse_page(page) == rc.parse_page(OLD_PAGE)


def test_the_new_page_shows_no_label_and_no_part():
    page = rc.recheck_page(_items()).lower()
    for word in ("part a", "part b", "direct", "indirect", "kappa"):
        assert word not in page


def test_the_evidence_check_passes_when_every_text_matches():
    old = {i: rc.fingerprint(p) for i, p in rc.parse_page(OLD_PAGE).items()}
    assert rc.evidence_problems(old, rc.recheck_page(_items())) == []


def test_the_evidence_check_finds_a_changed_or_missing_text():
    old = {i: rc.fingerprint(p) for i, p in rc.parse_page(OLD_PAGE).items()}
    items = _items()
    items[1]["paragraphs"] = ["First & paragraph."]
    problems = rc.evidence_problems(old, rc.recheck_page(items[1:]))
    assert any("A1" in p for p in problems) and any("A2" in p for p in problems)


# ---------- scoring ----------

def test_kappa_matches_a_worked_example():
    first = ["direct"] * 5 + ["not"] * 5
    second = ["direct"] * 4 + ["not"] * 6
    # agreement 0.9; chance agreement 0.5 * 0.4 + 0.5 * 0.6 = 0.5; kappa (0.9 - 0.5) / 0.5 = 0.8
    assert rc.cohen_kappa(first, second) == pytest.approx(0.8)


def test_kappa_is_undefined_when_both_labellings_use_one_label_only():
    assert rc.cohen_kappa(["not"] * 3, ["not"] * 3) is None


def _filled(labels):
    return [{"row": str(n), "article_id": i, "title": "", "url": "", "label": label, "unsure": "", "note": ""}
            for n, (i, label) in enumerate(labels.items(), start=1)]


def test_scoring_uses_part_a_only_and_reports_part_b_on_its_own():
    original = {"A1": "direct", "A2": "not", "A3": "indirect", "B1": "indirect"}
    recheck = _filled({"A1": "direct", "A2": "direct", "A3": "indirect", "B1": "not"})
    result = rc.score(original, recheck, part_a_ids={"A1", "A2", "A3"}, part_b_ids=("B1",))
    assert result["part_a"]["articles"] == 3 and result["part_a"]["agreement"] == pytest.approx(2 / 3)
    assert result["part_a"]["relevant_agreement"] == pytest.approx(2 / 3)
    assert result["part_b"] == [{"article_id": "B1", "first_label": "indirect", "recheck_label": "not",
                                 "exposure": "blind"}]


def test_scoring_stops_while_a_row_has_no_label_or_a_wrong_label():
    original = {"A1": "direct", "A2": "not"}
    for bad in ("", "  ", "relevant", "maybe"):
        with pytest.raises(rc.RecheckError):
            rc.score(original, _filled({"A1": "direct", "A2": bad}), part_a_ids={"A1", "A2"}, part_b_ids=())


def test_labels_are_read_without_case_or_spaces():
    result = rc.score({"A1": "direct"}, _filled({"A1": " Direct "}), part_a_ids={"A1"}, part_b_ids=())
    assert result["part_a"]["agreement"] == 1.0


def test_scoring_stops_when_the_sheet_misses_an_article_or_adds_one():
    original = {"A1": "direct", "A2": "not"}
    with pytest.raises(rc.RecheckError):
        rc.score(original, _filled({"A1": "direct"}), part_a_ids={"A1", "A2"}, part_b_ids=())
    with pytest.raises(rc.RecheckError):
        rc.score(original, _filled({"A1": "direct", "A2": "not", "A9": "not"}), part_a_ids={"A1", "A2"},
                 part_b_ids=())


def test_the_score_record_names_its_label_file_and_guide_version():
    original = {"A1": "direct", "A2": "not"}
    result = rc.score(original, _filled({"A1": "direct", "A2": "not"}), part_a_ids={"A1", "A2"}, part_b_ids=())
    assert result["label_file"] == "v3_dev_tata_motors.csv" and result["guide_version"] == "3"
    assert result["seed"] == rc.SEED


# ---------- adjudication and the checked label file ----------

def test_the_adjudication_sheet_lists_only_the_articles_whose_labels_differ():
    original = {"A1": "direct", "A2": "not", "A3": "indirect"}
    recheck = _filled({"A1": "direct", "A2": "indirect", "A3": "not"})
    rows = rc.adjudication_rows(original, recheck, titles={"A2": "Two", "A3": "Three"})
    assert [r["article_id"] for r in rows] == ["A2", "A3"]
    assert list(rows[0]) == rc.ADJUDICATION_COLUMNS == ["article_id", "title", "first_label", "recheck_label",
                                                         "final_label", "reason"]
    assert all(r["final_label"] == r["reason"] == "" for r in rows)


def test_the_adjudication_sheet_needs_the_score_record_first(tmp_path):
    with pytest.raises(rc.RecheckError):
        rc.require_score(tmp_path / "recheck_score.json")


def _adjudicated(*rows):
    return [dict(zip(rc.ADJUDICATION_COLUMNS, r)) for r in rows]


def test_the_checked_file_changes_only_the_adjudicated_labels_and_keeps_every_other_value():
    v3 = [_row("A1", "direct"), _row("A2", "not"), _row("A3", "indirect")]
    before = [dict(r) for r in v3]
    decided = _adjudicated(("A2", "Two", "not", "indirect", "indirect", "Tata Motors is a named party."),
                           ("A3", "Three", "indirect", "not", "indirect", "The first label follows rule 4."))
    checked, corrections = rc.apply_adjudication(v3, decided, when="2026-11-03")
    assert v3 == before                                                    # the version 3 rows never change
    assert [r["label"] for r in checked] == ["direct", "indirect", "indirect"]
    assert [{k: v for k, v in r.items() if k != "label"} for r in checked] == \
           [{k: v for k, v in r.items() if k != "label"} for r in before]
    assert corrections == [{"article_id": "A2", "title": "Two", "old_label": "not", "new_label": "indirect",
                            "reason": "Tata Motors is a named party.", "decided": "2026-11-03"}]


def test_every_adjudicated_row_needs_a_valid_final_label_and_a_reason():
    v3 = [_row("A2", "not")]
    for final, reason in (("", "x"), ("maybe", "x"), ("indirect", ""), ("indirect", "   ")):
        with pytest.raises(rc.RecheckError):
            rc.apply_adjudication(v3, _adjudicated(("A2", "Two", "not", "indirect", final, reason)), when="d")


def test_an_adjudicated_article_must_exist_in_the_label_file():
    with pytest.raises(rc.RecheckError):
        rc.apply_adjudication([_row("A1", "direct")],
                              _adjudicated(("A9", "Nine", "not", "direct", "direct", "x")), when="d")


def test_the_output_files_never_replace_the_version_3_file():
    assert rc.CHECKED_FILE == "v3_dev_tata_motors_checked.csv"
    assert rc.CORRECTIONS_FILE == "v3_dev_tata_motors_corrections.csv"
    assert V3_FILE.name not in (rc.CHECKED_FILE, rc.CORRECTIONS_FILE)


# ---------- safeguards: the pool, overlap, the source text, decisions ----------

def test_the_pool_fingerprint_is_stable_and_changes_with_any_label_or_article():
    pool = rc.recheck_pool(_v3_rows())
    first = rc.pool_fingerprint(pool)
    assert first == rc.pool_fingerprint(list(reversed(pool)))
    changed = [dict(r) for r in pool]
    changed[0]["label"] = "not" if changed[0]["label"] != "not" else "direct"
    assert rc.pool_fingerprint(changed) != first
    assert rc.pool_fingerprint(pool[1:]) != first


def test_the_sheet_record_holds_the_settings_and_dates_but_not_part_a():
    pool = rc.recheck_pool(_v3_rows())
    record = rc.sheet_record(pool, evidence={"A1": "f1"}, created="2026-11-06T09:00:00Z")
    assert record["pool_fingerprint"] == rc.pool_fingerprint(pool)
    assert record["seed"] == rc.SEED and record["part_a_size"] == 40 and record["part_b"] == list(rc.PART_B)
    assert record["label_file"] == "v3_dev_tata_motors.csv" and record["evidence"] == {"A1": "f1"}
    assert record["first_labelling"] == rc.FIRST_LABELLING and record["last_seen"] == rc.LAST_SEEN
    assert record["sheet_created"] == "2026-11-06T09:00:00Z"
    text = json.dumps(record)
    assert not any(i in text for i in rc.amended_part_a(pool) - set(rc.PART_B))


def test_scoring_stops_when_the_pool_differs_from_the_sheet_record():
    pool = rc.recheck_pool(_v3_rows())
    record = rc.sheet_record(pool, evidence={}, created="x")
    rc.check_pool(record, pool)
    changed = [dict(r) for r in pool]
    changed[3]["label"] = "not" if changed[3]["label"] != "not" else "direct"
    with pytest.raises(rc.RecheckError):
        rc.check_pool(record, changed)


def test_an_article_in_both_parts_counts_once_in_each():
    original = {"A1": "direct", "A2": "not"}
    result = rc.score(original, _filled({"A1": "not", "A2": "not"}), part_a_ids={"A1", "A2"}, part_b_ids=("A1",))
    assert result["part_a"]["articles"] == 2 and result["part_a"]["agreement"] == 0.5
    assert [r["article_id"] for r in result["part_b"]] == ["A1"]
    rows = rc.adjudication_rows(original, _filled({"A1": "not", "A2": "not"}), titles={})
    assert [r["article_id"] for r in rows] == ["A1"]


def test_the_score_says_part_a_is_a_stratified_sample():
    result = rc.score({"A1": "direct"}, _filled({"A1": "direct"}), part_a_ids={"A1"}, part_b_ids=())
    assert result["part_a"]["sample"] == ("stratified by first label, proportional, fixed seed, "
                                          "from the articles still blind")


def test_an_undefined_kappa_is_saved_as_none():
    result = rc.score({"A1": "not", "A2": "not"}, _filled({"A1": "not", "A2": "not"}), part_a_ids={"A1", "A2"},
                      part_b_ids=())
    assert result["part_a"]["kappa"] is None and result["part_a"]["agreement"] == 1.0


def test_page_text_is_checked_against_the_saved_source_text():
    page = {"A1": ["First & paragraph.", "Second paragraph."], "A2": ["Only paragraph."]}
    saved = {"A1": "First & paragraph.\n\n  Second paragraph.  \n", "A2": "Only paragraph, changed."}
    problems = rc.source_problems(page, saved)
    assert len(problems) == 1 and "A2" in problems[0]
    assert any("A3" in p for p in rc.source_problems(page | {"A3": ["x"]}, saved))


def test_blocked_short_text_must_be_the_same_in_every_sheet_that_holds_it():
    same = [{"title": "T", "description": "D", "content_first_200_chars": "C"}] * 2
    assert rc.one_short_text("A1", same) == ["T", "D", "C"]
    with pytest.raises(rc.RecheckError):
        rc.one_short_text("A1", same + [{"title": "T", "description": "D2", "content_first_200_chars": "C"}])
    with pytest.raises(rc.RecheckError):
        rc.one_short_text("A1", [])


def test_a_decision_that_keeps_the_first_label_is_recorded_but_is_not_a_correction():
    v3 = [_row("A1", "indirect"), _row("A2", "not")]
    decided = _adjudicated(("A1", "One", "indirect", "not", "indirect", "Rule 4: a named party."),
                           ("A2", "Two", "not", "direct", "direct", "Tata Motors is the main subject."))
    checked, corrections = rc.apply_adjudication(v3, decided, when="2026-11-07")
    assert [c["article_id"] for c in corrections] == ["A2"]
    assert rc.decision_counts(decided) == {"decisions": 2, "first_label_kept": 1, "recheck_label_taken": 1,
                                           "third_label": 0, "corrections": 1}


def test_the_folders_resolve():
    eval_dir, fulltext_dir = rc._paths()
    assert eval_dir.name == "eval" and fulltext_dir.name == "fulltext"


# ---------- a saved page that holds another story ----------

def test_a_saved_text_that_fails_the_wrong_page_check_is_replaced_by_the_short_text():
    title = "Jaguar unveils $130,500 electric vehicle two years after controversial rebrand"
    description = "Jaguar Land Rover shows the Type 01 electric grand tourer, priced from $130,500 in the US market."
    other_story = ("Crude climbed to its highest level since April as investors weighed renewed fighting. "
                   "The Dow fell 1.12%, the S&P 500 lost 0.43% and the Nasdaq slipped 0.13% as oil surged.\\n") * 5
    assert rc.saved_text_is_another_story(title, description, other_story, None) is True
    same_story = ("Jaguar unveils its Type 01 electric grand tourer, priced from $130,500 in the US market, two "
                  "years after the controversial rebrand of Jaguar Land Rover.\\n") * 3
    assert rc.saved_text_is_another_story(title, description, same_story, None) is False


def test_the_sheet_record_names_every_evidence_override_with_its_reason():
    pool = rc.recheck_pool(_v3_rows())
    record = rc.sheet_record(pool, evidence={}, created="x", overrides={"A1": "the saved page holds another story"})
    assert record["evidence_overrides"] == {"A1": "the saved page holds another story"}


# ---------- the amendment of 2 November: Part A only from articles still blind ----------

LABEL_SHOWN = {"A0ae8a2cfab", "Aaa3d8d3035", "A639826ac02", "A690a6ec804", "A02b25c8c80", "A58b7f6f1f1",
               "A43ea61cb65", "Aac357bfe35", "A7595fd1c77", "Ab5f09d2bd1", "A7830fe9dbf", "A20c83edcbc"}


def test_the_exposed_articles_and_the_two_levels_of_exposure():
    assert len(rc.EXPOSED) == 26
    shown = {i for i, kind in rc.EXPOSED.items() if kind == rc.LABEL_SHOWN}
    sheet = {i for i, kind in rc.EXPOSED.items() if kind == rc.FACT_SHEET}
    assert shown == LABEL_SHOWN and len(sheet) == 14                    # an article in both counts as label shown
    facts = {r["article_id"] for r in csv.DictReader(open(rc._paths()[0] / "fact_check_facts.csv", encoding="utf-8-sig"))}
    assert facts <= set(rc.EXPOSED)
    assert set(rc.EXPOSED) <= {r["article_id"] for r in rc.recheck_pool(_v3_rows())}


def test_amended_part_a_is_40_articles_from_the_48_still_blind_in_their_own_mix():
    pool = rc.recheck_pool(_v3_rows())
    blind = rc.blind_pool(pool)
    assert len(blind) == 48 and not {r["article_id"] for r in blind} & set(rc.EXPOSED)
    chosen = rc.amended_part_a(pool)
    assert len(chosen) == 40 and not chosen & set(rc.EXPOSED)
    by_label = {label: sum(1 for r in blind if r["article_id"] in chosen and r["label"] == label) for label in rc.LABELS}
    assert by_label == rc.stratum_sizes(blind, 40)


def test_the_original_part_a_rule_is_kept_unchanged():
    pool = rc.recheck_pool(_v3_rows())
    assert rc.part_a(pool) == rc.part_a(pool, exclude=())               # the first plan stays reproducible
    assert rc.part_a(pool) != rc.amended_part_a(pool)


def test_the_sheet_uses_the_amended_part_a_and_keeps_all_of_part_b():
    pool = rc.recheck_pool(_v3_rows())
    ids = {r["article_id"] for r in rc.sheet_rows(pool)}
    assert ids == rc.amended_part_a(pool) | set(rc.PART_B)


def test_the_record_states_the_amendment_the_exposure_and_the_new_last_seen_date():
    pool = rc.recheck_pool(_v3_rows())
    record = rc.sheet_record(pool, {}, "2026-11-03T04:00:00+00:00")
    assert record["last_seen"] == rc.LAST_SEEN == "2026-11-02"
    assert record["exposed"] == rc.EXPOSED
    amendment = record["amendment"]
    assert amendment["date"] == "2026-11-02" and amendment["before_labelling"] is True
    assert "74" in amendment["original_plan"] and "48" in amendment["amended_plan"]


def test_the_score_names_what_it_measures_with_per_label_agreement_and_part_b_exposure():
    original = {"A1": "direct", "A2": "direct", "A3": "not", "A4": "indirect", "A02b25c8c80": "indirect"}
    recheck = _filled({"A1": "direct", "A2": "indirect", "A3": "not", "A4": "indirect", "A02b25c8c80": "not"})
    result = rc.score(original, recheck, part_a_ids={"A1", "A2", "A3", "A4"}, part_b_ids=("A02b25c8c80",))
    a = result["part_a"]
    assert a["measures"] == "agreement on the unexposed development subset"
    assert a["per_label"] == {"direct": {"articles": 2, "same": 1, "agreement": 0.5},
                              "indirect": {"articles": 1, "same": 1, "agreement": 1.0},
                              "not": {"articles": 1, "same": 1, "agreement": 1.0}}
    assert "long relevant articles" in a["limits"]
    assert result["part_b"][0]["exposure"] == rc.LABEL_SHOWN
    assert "not combined" in result["part_b_note"]
