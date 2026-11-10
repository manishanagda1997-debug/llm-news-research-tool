"""
Tests for the fresh evaluation run (searches A, B and C on a new day).

fetch:  make the three searches, save each raw response, save a record, and
        write a blind labelling sheet of every article without a label yet.
        The record freezes which raw files existed before the run: only those
        count as "seen" when the new-only results are measured later.
score:  after labelling, report each configuration on the whole pool and on
        unseen articles only, the listing checks, and the relevance filter.
"""

import csv
import json

import pytest

from src import fresh_run as fr
from src.retrieval_check import SHEET_COLUMNS


TOPICS = ["quarterly profit rises", "new electric SUV launched", "plant expansion in Pune",
          "Iveco takeover offer raised", "truck sales fall in September", "board appoints new CFO",
          "export orders grow strongly", "battery supply deal signed", "dealer network reaches 1,000",
          "share price hits record high"]


def _article(n, title=None, url=None, text="Tata Motors news"):
    # clearly different titles, so the near-duplicate rule does not join them
    return {"source": {"id": None, "name": "News"}, "author": "A",
            "title": title or f"Tata Motors: {TOPICS[n % 10]} ({n})",
            "description": text, "content": text,
            "url": url or f"https://x.com/{n}", "urlToImage": None,
            "publishedAt": "2026-10-23T10:00:00Z"}


def _response(articles, part=None):
    return {"status": "ok", "totalResults": len(articles), "articles": articles,
            "request": {"query": "Tata Motors", "search_part": part, "from_date": "2026-10-17",
                        "page_size": 30, "sort_by": "relevancy"}}


def _write_raw(folder, name, articles):
    path = folder / name
    path.write_text(json.dumps(_response(articles)), encoding="utf-8")
    return path


# ---------- what counts as seen ----------

def test_seen_articles_come_from_the_listed_raw_files_only(tmp_path):
    _write_raw(tmp_path, "old.json", [_article(1)])
    _write_raw(tmp_path, "new.json", [_article(2)])
    seen = fr.seen_articles(["old.json"], tmp_path)
    assert [a["url"] for a in seen] == ["https://x.com/1"]


def test_record_freezes_the_raw_files_that_existed_before_the_run(tmp_path):
    _write_raw(tmp_path, "20261014_old.json", [_article(1)])
    record = fr.fresh_record("Tata Motors", {"A": "C:\\data\\raw\\a.json", "B": None, "C": "/data/raw/c.json"},
                             seen_files=fr.existing_raw_files(tmp_path))
    assert record == {"query": "Tata Motors",
                      "raw_files": {"A": "a.json", "B": None, "C": "c.json"},
                      "seen_files": ["20261014_old.json"]}


# ---------- the blind sheet ----------

def test_sheet_lists_every_article_without_a_label_once():
    responses = {"A": _response([_article(1), _article(2)]),
                 "B": _response([_article(2), _article(3)]),
                 "C": _response([_article(1), _article(4)])}
    labelled = {"https://x.com/4"}
    rows = fr.sheet_rows(responses, labelled)
    assert sorted(r["url"] for r in rows) == ["https://x.com/1", "https://x.com/2", "https://x.com/3"]


def test_sheet_is_blind_and_shuffled():
    responses = {"A": _response([_article(n) for n in range(10)]),
                 "B": _response([_article(n) for n in range(10, 20)]),
                 "C": None}
    rows = fr.sheet_rows(responses, set())
    assert list(rows[0].keys()) == SHEET_COLUMNS
    assert [r["url"] for r in rows] != [f"https://x.com/{n}" for n in range(20)]
    assert [r["url"] for r in rows] == [r["url"] for r in fr.sheet_rows(responses, set())]


def test_sheet_skips_removed_articles():
    responses = {"A": _response([_article(1, title="[Removed]"), _article(2)]), "B": None, "C": None}
    assert [r["url"] for r in fr.sheet_rows(responses, set())] == ["https://x.com/2"]


# ---------- scoring ----------

LABELS = {"https://x.com/1": "direct", "https://x.com/2": "indirect",
          "https://x.com/3": "not", "https://x.com/4": "direct"}


def _responses():
    return {"A": _response([_article(1), _article(2), _article(3)]),
            "B": _response([_article(4, text="Jaguar Land Rover news")]),
            "C": _response([_article(1), _article(2)])}


def test_new_only_drops_articles_seen_before_in_every_search():
    seen = [_article(1, url="https://www.x.com/1/")]
    new_only = fr.new_only(_responses(), seen)
    assert [a["url"] for a in new_only["A"]["articles"]] == ["https://x.com/2", "https://x.com/3"]
    assert [a["url"] for a in new_only["C"]["articles"]] == ["https://x.com/2"]
    assert new_only["A"]["request"]["page_size"] == 30          # the budget does not change


def test_score_reports_whole_pool_and_new_only():
    result = fr.score(_responses(), LABELS, seen=[_article(1)], query="Tata Motors", aliases=None)
    assert result["whole"]["configurations"]["A"]["relevant"] == 2
    assert result["new_only"]["configurations"]["A"]["relevant"] == 1
    assert result["counts"] == {"whole": 4, "new_only": 3}


def test_score_ignores_removed_articles():
    responses = _responses()
    responses["A"]["articles"].append(_article(9, title="[Removed]"))
    result = fr.score(responses, LABELS, seen=[], query="Tata Motors", aliases=None)
    assert result["counts"]["whole"] == 4


def test_score_includes_the_relevance_filter_for_each_listing_mode():
    result = fr.score(_responses(), LABELS, seen=[], query="Tata Motors", aliases=None)
    table = result["whole"]["filter"]
    assert set(table) == {"A", "A+B", "C", "C+B"}
    assert set(table["A"]) == {"off", "patterns", "sites", "both"}
    a_off = table["A"]["off"]
    # all three A articles mention Tata Motors, so all are kept: 2 relevant, 1 not
    assert (a_off["tp"], a_off["fp"], a_off["fn"], a_off["tn"]) == (2, 1, 0, 0)
    assert set(a_off) >= {"precision", "recall", "f1", "kept"}


def test_filter_counts_a_dropped_duplicate_as_found_when_its_copy_is_kept():
    copy_a = _article(1, title="Tata Motors raises Iveco offer to 14.40 euro per share today")
    copy_b = _article(5, title="Tata Motors raises Iveco offer to 14.40 euro per share today")
    responses = {"A": _response([copy_a, copy_b]), "B": None, "C": None}
    labels = {"https://x.com/1": "direct", "https://x.com/5": "direct"}
    result = fr.score(responses, labels, seen=[], query="Tata Motors", aliases=None,
                      configurations={"A": ["A"]})
    off = result["whole"]["filter"]["A"]["off"]
    assert (off["tp"], off["fn"]) == (2, 0)


def test_score_stops_when_an_article_has_no_label():
    labels = dict(LABELS)
    del labels["https://x.com/4"]
    with pytest.raises(fr.FreshRunError):
        fr.score(_responses(), labels, seen=[], query="Tata Motors", aliases=None)


def test_write_sheet_uses_a_new_file_each_time(tmp_path):
    rows = fr.sheet_rows({"A": _response([_article(1)]), "B": None, "C": None}, set())
    first = fr.write_fresh_sheet(rows, tmp_path, "20261024")
    second = fr.write_fresh_sheet(rows, tmp_path, "20261024")
    assert first != second and first.name.startswith("dev_tata_motors_fresh_20261024")
    with open(first, newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == SHEET_COLUMNS
