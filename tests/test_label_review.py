"""Tests for the blind relabelling of the development set under guide version 2."""

import csv

import pytest

from src import label_review as lr

V1_COLUMNS = ["row", "label", "unsure", "note", "title", "source", "published",
              "description", "content_first_200_chars", "url", "found_in_files"]


def _write(path, rows, columns=V1_COLUMNS):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    return path


def _row(n, label="direct", url=None):
    return {"row": n, "label": label, "unsure": "yes" if n == 1 else "", "note": "old note",
            "title": f"Title {n}", "source": "S", "published": "2026-10-20",
            "description": f"Description {n}", "content_first_200_chars": f"Content {n}",
            "url": url or f"https://news.example.com/story-{n}", "found_in_files": "a.json"}


@pytest.fixture
def v1_files(tmp_path):
    first = _write(tmp_path / "dev_a.csv", [_row(n) for n in range(1, 11)])
    # story-10 appears in both files, with www and a tracking code in the second
    second = _write(tmp_path / "dev_b.csv",
                    [_row(10, url="https://www.news.example.com/story-10?utm_source=x")]
                    + [_row(n) for n in range(11, 16)],
                    columns=V1_COLUMNS[:-1])
    return [first, second]


# ---------- 1. the blind sheet ----------

def test_recheck_sheet_lists_every_article_once(v1_files):
    rows = lr.recheck_rows(v1_files)
    assert len(rows) == 15


def test_recheck_sheet_hides_the_old_label_note_and_files(v1_files):
    rows = lr.recheck_rows(v1_files)
    assert list(rows[0].keys()) == lr.SHEET_COLUMNS
    assert all(r["label"] == "" and r["unsure"] == "" and r["note"] == "" for r in rows)
    assert "found_in_files" not in rows[0]


def test_recheck_sheet_keeps_the_visible_text(v1_files):
    rows = {r["title"]: r for r in lr.recheck_rows(v1_files)}
    assert rows["Title 3"]["description"] == "Description 3"
    assert rows["Title 3"]["content_first_200_chars"] == "Content 3"


def test_recheck_sheet_is_shuffled_but_repeatable(v1_files):
    first = [r["title"] for r in lr.recheck_rows(v1_files)]
    assert first == [r["title"] for r in lr.recheck_rows(v1_files)]
    assert first != [f"Title {n}" for n in range(1, 16)]
    assert [r["row"] for r in lr.recheck_rows(v1_files)] == list(range(1, 16))


# ---------- 2. agreement between the two versions ----------

def test_cohen_kappa_known_value():
    first = ["direct", "direct", "indirect", "not"]
    second = ["direct", "indirect", "indirect", "not"]
    # agreement 0.75, chance agreement 0.3125 -> (0.75 - 0.3125) / (1 - 0.3125)
    assert lr.cohen_kappa(first, second) == pytest.approx(0.6364, abs=1e-4)


def test_cohen_kappa_is_one_for_full_agreement():
    labels = ["direct", "not", "indirect", "not"]
    assert lr.cohen_kappa(labels, labels) == pytest.approx(1.0)


def test_cohen_kappa_when_everything_has_one_label():
    assert lr.cohen_kappa(["not", "not"], ["not", "not"]) == pytest.approx(1.0)


def test_compare_counts_agreement_and_lists_changes():
    old = {"u1": "direct", "u2": "indirect", "u3": "not", "u4": "indirect"}
    new = {"u1": "direct", "u2": "not", "u3": "not", "u4": "indirect"}
    result = lr.compare_labels(old, new)
    assert result["articles"] == 4
    assert result["agreement"] == pytest.approx(0.75)
    assert result["changes"] == [("u2", "indirect", "not")]
    assert result["moves"] == {("indirect", "not"): 1}


def test_compare_stops_when_the_articles_differ():
    with pytest.raises(lr.LabelReviewError):
        lr.compare_labels({"u1": "direct"}, {"u2": "direct"})


# ---------- 3. the differences sheet ----------

def test_differences_sheet_shows_both_labels_and_empty_decision_columns():
    old = {"u1": "direct", "u2": "indirect"}
    new = {"u1": "direct", "u2": "not"}
    rows = lr.difference_rows(old, new, titles={"u1": "One", "u2": "Two"})
    assert rows == [{"title": "Two", "url": "u2", "label_v1": "indirect", "label_v2_blind": "not",
                     "final_label": "", "reason": "", "unsure": ""}]


# ---------- 4. the version 2 label file ----------

def test_final_labels_use_the_decision_for_changed_rows():
    blind = {"u1": "direct", "u2": "not"}
    decisions = {"u2": "indirect"}
    assert lr.final_labels(blind, decisions) == {"u1": "direct", "u2": "indirect"}


def test_final_labels_stop_when_a_difference_has_no_decision():
    with pytest.raises(lr.LabelReviewError):
        lr.final_labels({"u1": "direct", "u2": "not"}, {"u2": ""}, changed={"u2"})


def test_final_labels_reject_unknown_labels():
    with pytest.raises(lr.LabelReviewError):
        lr.final_labels({"u1": "direct"}, {"u1": "maybe"})
