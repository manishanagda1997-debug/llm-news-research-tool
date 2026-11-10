"""
Tests for the event sheet used by the ranking evaluation (no network, no model calls).

Event recall needs an event name for every relevant article the tool can
find, not only for the articles a summary run picks. So the sheet lists
every article labelled relevant (direct or indirect) under guide version 3,
each once, shuffled, with an empty `event` column. Labels are not shown:
grouping by event is a separate task from judging relevance.
"""

import pytest

from src import summary_run as sr


def _raw(n, url=None, label_title=None):
    return {"source": {"id": None, "name": f"Source {n}"}, "author": "A",
            "title": label_title or f"Tata Motors news item {n}", "description": f"Description {n}.",
            "content": "x" * 300, "url": url or f"https://x.com/{n}", "urlToImage": None,
            "publishedAt": f"2026-10-{10 + n:02d}T10:00:00Z"}


ARTICLES = [_raw(n) for n in range(1, 13)]
LABELS = {f"https://x.com/{n}": ("not" if n % 4 == 0 else "direct" if n % 2 else "indirect") for n in range(1, 13)}
RELEVANT = sorted(f"https://x.com/{n}" for n in range(1, 13) if n % 4 != 0)


def test_every_relevant_article_is_listed_once():
    rows = sr.relevant_event_rows(ARTICLES + [_raw(1, url="https://www.x.com/1/?utm_source=feed")], LABELS)
    assert sorted(row["url"] for row in rows) == RELEVANT


def test_not_relevant_articles_are_left_out():
    rows = sr.relevant_event_rows(ARTICLES, LABELS)
    assert "https://x.com/4" not in {row["url"] for row in rows}


def test_rows_have_the_event_columns_an_empty_event_and_no_label():
    rows = sr.relevant_event_rows(ARTICLES, LABELS)
    assert all(list(row) == sr.EVENT_COLUMNS for row in rows)
    assert all(row["event"] == "" for row in rows)
    assert "label" not in sr.EVENT_COLUMNS
    assert [row["row"] for row in rows] == list(range(1, len(rows) + 1))


def test_rows_hold_the_text_used_for_grouping():
    row = next(r for r in sr.relevant_event_rows(ARTICLES, LABELS) if r["url"] == "https://x.com/3")
    assert row["title"] == "Tata Motors news item 3" and row["description"] == "Description 3."
    assert row["source"] == "Source 3" and row["published"] == "2026-10-13T10:00:00Z"
    assert row["content_first_200_chars"] == "x" * 200


def test_rows_are_shuffled_the_same_way_every_time():
    first = [row["url"] for row in sr.relevant_event_rows(ARTICLES, LABELS)]
    again = [row["url"] for row in sr.relevant_event_rows(ARTICLES, LABELS)]
    assert first == again
    assert first != [a["url"] for a in ARTICLES if a["url"] in RELEVANT]      # position gives no hint


def test_an_article_without_a_label_stops_the_sheet():
    labels = dict(LABELS)
    del labels["https://x.com/2"]
    with pytest.raises(sr.SummaryRunError):
        sr.relevant_event_rows(ARTICLES, labels)


def test_the_sheet_uses_the_guide_version_3_labels():
    assert sr.EVENT_LABEL_FILE == "v3_dev_tata_motors.csv"
