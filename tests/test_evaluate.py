"""
Tests for src/evaluate.py: the numbers used to compare versions of the
relevance rules against hand-made labels.

Run with:  pytest
"""

import csv

import pytest

import src.evaluate as evaluate


def test_confusion_counts():
    relevant = {"a": True, "b": True, "c": False, "d": False, "e": True}
    kept = {"a", "c"}
    counts = evaluate.confusion(relevant, kept)
    assert counts == {"tp": 1, "fp": 1, "fn": 2, "tn": 1}


def test_scores_from_counts():
    scores = evaluate.scores({"tp": 8, "fp": 1, "fn": 7, "tn": 4})
    assert scores["precision"] == pytest.approx(8 / 9)
    assert scores["recall"] == pytest.approx(8 / 15)
    assert scores["f1"] == pytest.approx(2 * (8 / 9) * (8 / 15) / ((8 / 9) + (8 / 15)))


def test_scores_never_divide_by_zero():
    scores = evaluate.scores({"tp": 0, "fp": 0, "fn": 0, "tn": 5})
    assert scores == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_precision_and_recall_at_k():
    relevant = {"a": True, "b": False, "c": True, "d": True}
    ranked = ["a", "b", "c", "d"]
    assert evaluate.precision_at_k(ranked, relevant, k=2) == pytest.approx(1 / 2)
    assert evaluate.recall_at_k(ranked, relevant, k=2) == pytest.approx(1 / 3)
    assert evaluate.recall_at_k(ranked, relevant, k=10) == pytest.approx(3 / 3)


def test_load_labels_reads_direct_and_indirect_as_relevant(tmp_path):
    path = tmp_path / "labels.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["row", "label", "note", "title", "url"])
        writer.writeheader()
        writer.writerow({"row": 1, "label": "direct", "note": "", "title": "A", "url": "https://x.com/1"})
        writer.writerow({"row": 2, "label": "indirect", "note": "", "title": "B", "url": "https://x.com/2"})
        writer.writerow({"row": 3, "label": "not", "note": "", "title": "C", "url": "https://x.com/3"})
    labels = evaluate.load_labels(path)
    assert labels["https://x.com/1"] == "direct"
    assert evaluate.is_relevant(labels["https://x.com/2"]) is True
    assert evaluate.is_relevant(labels["https://x.com/3"]) is False


def test_load_labels_rejects_unknown_label(tmp_path):
    path = tmp_path / "labels.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["row", "label", "url"])
        writer.writeheader()
        writer.writerow({"row": 1, "label": "relevent", "url": "https://x.com/1"})
    with pytest.raises(ValueError):
        evaluate.load_labels(path)
