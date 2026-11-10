"""
Tests for the summary evaluation run (fake model only).

run      - for each saved search and each k: clean and rank the articles with
           the frozen method, take the first k, summarise, save the result
           with its fingerprint (a saved result with the same fingerprint is
           reused, so a rerun makes no new model calls)
judging  - one blind sheet: every combined point from every run, shuffled, with
           k and the search hidden; the key that links each row to its run is
           kept in a separate file; article briefs are judged once per article
coverage - per run: share of `direct` articles cited by a combined point,
           `not` articles cited, and share of events covered (events are
           grouped by hand from the labels, before any summary is read)
"""

import csv
import json

import pytest

from src import summary_run as sr
from src.summarizer import SummaryCache


def _raw(numbers):
    return {"articles": [
        {"source": {"id": None, "name": "News"}, "author": "A",
         "title": f"Tata Motors news item {n}: {['profit', 'plant', 'offer', 'sales', 'board', 'exports'][n % 6]}",
         "description": f"Tata Motors reported item {n}.", "content": "",
         "url": f"https://x.com/{n}", "urlToImage": None, "publishedAt": "2026-10-20T10:00:00Z"}
        for n in numbers]}


@pytest.fixture
def raw_dir(tmp_path):
    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "s1.json").write_text(json.dumps(_raw(range(1, 7))), encoding="utf-8")
    (folder / "s2.json").write_text(json.dumps(_raw(range(4, 10))), encoding="utf-8")
    return folder


def _answer_for(prompt):
    """A fake model that cites every article it is given in one key point."""
    ids = [int(part.split('"')[0]) for part in prompt.split('<article id="')[1:]]
    return json.dumps({
        "article_briefs": [{"id": i, "brief": f"Tata Motors reported item.", "enough_text": True} for i in ids],
        "key_points": [{"text": "Tata Motors reported several items.", "citations": ids}],
        "industry_context": [],
        "not_used": [],
    })


class FakeLLM:
    provider, model_name = "fake", "fake-model"

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        return _answer_for(prompt)


# ---------- the run ----------

def test_the_run_settings_are_the_agreed_ones():
    assert sr.KS == (5, 10, 15)
    assert sr.CLEANING == {"remove_junk": True, "ranking": "mention", "drop_listings": "off"}
    assert len(sr.RUN_SEARCHES) == 4


def test_select_articles_uses_the_frozen_cleaning_and_takes_the_first_k(raw_dir):
    articles = sr.select_articles(raw_dir / "s1.json", "Tata Motors", aliases=None, k=5)
    assert [a["id"] for a in articles] == [1, 2, 3, 4, 5]


def test_run_makes_one_call_per_search_and_k(raw_dir, tmp_path):
    llm = FakeLLM()
    runs = sr.run_summaries(["s1.json", "s2.json"], ks=(2, 4), llm=llm, cache=SummaryCache(tmp_path / "c"),
                            query="Tata Motors", aliases=None, raw_dir=raw_dir)
    assert llm.calls == 4
    assert {(r["search"], r["k"]) for r in runs} == {("s1.json", 2), ("s1.json", 4), ("s2.json", 2), ("s2.json", 4)}
    assert all(r["result"]["status"] == "ok" and r["result"]["origin"] == "live" for r in runs)


def test_a_rerun_reuses_saved_results_without_calls(raw_dir, tmp_path):
    cache = SummaryCache(tmp_path / "c")
    sr.run_summaries(["s1.json"], ks=(2,), llm=FakeLLM(), cache=cache, query="Tata Motors",
                     aliases=None, raw_dir=raw_dir)
    again = FakeLLM()
    runs = sr.run_summaries(["s1.json"], ks=(2,), llm=again, cache=cache, query="Tata Motors",
                            aliases=None, raw_dir=raw_dir)
    assert again.calls == 0 and runs[0]["result"]["origin"] == "saved"


def test_each_run_records_what_is_needed_to_repeat_it(raw_dir, tmp_path):
    runs = sr.run_summaries(["s1.json"], ks=(3,), llm=FakeLLM(), cache=SummaryCache(tmp_path / "c"),
                            query="Tata Motors", aliases=None, raw_dir=raw_dir)
    run = runs[0]
    assert set(run) >= {"search", "k", "fingerprint", "model", "prompt_version", "articles", "result", "prompt_chars"}
    assert run["model"] == "fake-model" and len(run["articles"]) == 3 and run["prompt_chars"] > 0


# ---------- the blind judging sheet ----------

def _runs(raw_dir, tmp_path):
    return sr.run_summaries(["s1.json", "s2.json"], ks=(2, 4), llm=FakeLLM(), cache=SummaryCache(tmp_path / "c"),
                            query="Tata Motors", aliases=None, raw_dir=raw_dir)


def test_judging_rows_hide_k_and_search(raw_dir, tmp_path):
    rows, key = sr.judging_rows(_runs(raw_dir, tmp_path))
    assert list(rows[0].keys()) == sr.JUDGING_COLUMNS
    for column in sr.JUDGING_COLUMNS:
        assert column not in ("k", "search", "fingerprint")
    assert all(row["judgement"] == "" and row["contradicted"] == "" for row in rows)


def test_every_combined_point_is_a_row_and_the_key_links_it_back(raw_dir, tmp_path):
    runs = _runs(raw_dir, tmp_path)
    rows, key = sr.judging_rows(runs)
    point_rows = [row for row in rows if row["kind"] != "article brief"]
    assert len(point_rows) == 4                        # one key point in each of the 4 runs
    assert {key[row["item"]]["k"] for row in point_rows} == {2, 4}
    assert all(key[row["item"]]["search"] in ("s1.json", "s2.json") for row in rows)


def test_article_briefs_are_judged_once_per_article(raw_dir, tmp_path):
    rows, _ = sr.judging_rows(_runs(raw_dir, tmp_path))
    brief_rows = [row for row in rows if row["kind"] == "article brief"]
    # s1 gives articles 1-4 at k=4, s2 gives articles 4-7; article 4 appears in both searches
    assert len(brief_rows) == len({row["evidence"] for row in brief_rows})


def test_each_row_shows_the_evidence_the_model_had(raw_dir, tmp_path):
    rows, _ = sr.judging_rows(_runs(raw_dir, tmp_path))
    point = [row for row in rows if row["kind"] == "company point"][0]
    assert "Tata Motors reported item" in point["evidence"] and "[1]" in point["evidence"]


def test_judging_order_is_shuffled_but_repeatable(raw_dir, tmp_path):
    runs = _runs(raw_dir, tmp_path)
    first, _ = sr.judging_rows(runs)
    second, _ = sr.judging_rows(runs)
    assert [r["text"] + r["evidence"] for r in first] == [r["text"] + r["evidence"] for r in second]
    assert [r["kind"] for r in first] != sorted(r["kind"] for r in first) or len(first) < 3


def test_write_judging_files_keeps_the_key_separate(raw_dir, tmp_path):
    rows, key = sr.judging_rows(_runs(raw_dir, tmp_path))
    sheet, key_file = sr.write_judging_files(rows, key, tmp_path / "eval", "20261023")
    assert sheet.suffix == ".csv" and key_file.suffix == ".json" and sheet.parent == key_file.parent
    with open(sheet, newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == sr.JUDGING_COLUMNS
    assert "k" not in sheet.read_text(encoding="utf-8").splitlines()[0].split(",")


# ---------- coverage ----------

def test_direct_coverage_and_not_articles_cited():
    articles = [{"id": 1, "url": "https://x.com/1"}, {"id": 2, "url": "https://x.com/2"},
                {"id": 3, "url": "https://x.com/3"}]
    summary = {"key_points": [{"text": "a", "citations": [1, 3]}], "industry_context": []}
    labels = {"https://x.com/1": "direct", "https://x.com/2": "direct", "https://x.com/3": "not"}
    result = sr.citation_coverage(articles, summary, labels)
    assert result == {"direct_given": 2, "direct_cited": 1, "direct_coverage": 0.5, "not_cited": 1}


def test_event_coverage_counts_events_not_articles():
    articles = [{"id": n, "url": f"https://x.com/{n}"} for n in (1, 2, 3, 4)]
    events = {"https://x.com/1": "iveco", "https://x.com/2": "iveco", "https://x.com/3": "iveco",
              "https://x.com/4": "stellantis"}
    summary = {"key_points": [{"text": "Iveco deal", "citations": [2]}], "industry_context": []}
    result = sr.event_coverage(articles, summary, events)
    assert result == {"events_given": 2, "events_covered": 1, "event_coverage": 0.5}


def test_coverage_is_empty_for_a_run_without_summary():
    articles = [{"id": 1, "url": "https://x.com/1"}]
    assert sr.citation_coverage(articles, None, {"https://x.com/1": "direct"})["direct_cited"] == 0


# ---------- the event sheet (made before any summary is read) ----------

LABELS = {f"https://x.com/{n}": label for n, label in
          [(1, "direct"), (2, "direct"), (3, "not"), (4, "indirect"), (5, "direct"), (6, "not"),
           (7, "direct"), (8, "not"), (9, "indirect")]}


def test_event_sheet_lists_the_relevant_articles_the_runs_will_see_once(raw_dir):
    rows = sr.event_sheet_rows(["s1.json", "s2.json"], LABELS, "Tata Motors", aliases=None,
                               raw_dir=raw_dir, k=5)
    # s1 top 5: 1-5, s2 top 5: 4-8; relevant ones: 1, 2, 4, 5, 7 (4 and 5 appear in both searches)
    assert sorted(row["url"] for row in rows) == [f"https://x.com/{n}" for n in (1, 2, 4, 5, 7)]


def test_event_sheet_has_an_empty_event_column_and_no_labels(raw_dir):
    rows = sr.event_sheet_rows(["s1.json"], LABELS, "Tata Motors", aliases=None, raw_dir=raw_dir, k=5)
    assert list(rows[0].keys()) == sr.EVENT_COLUMNS
    assert all(row["event"] == "" for row in rows)
    assert "label" not in sr.EVENT_COLUMNS


def test_event_sheet_stops_when_an_article_has_no_label(raw_dir):
    labels = dict(LABELS)
    del labels["https://x.com/2"]
    with pytest.raises(sr.SummaryRunError):
        sr.event_sheet_rows(["s1.json"], labels, "Tata Motors", aliases=None, raw_dir=raw_dir, k=5)


def test_load_events_reads_one_event_per_article(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text("row,event,url\n1,iveco,https://www.x.com/1/\n2, Iveco ,https://x.com/2\n", encoding="utf-8")
    assert sr.load_events(path) == {"https://x.com/1": "iveco", "https://x.com/2": "iveco"}


def test_load_events_stops_on_an_empty_event(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text("row,event,url\n1,,https://x.com/1\n", encoding="utf-8")
    with pytest.raises(sr.SummaryRunError):
        sr.load_events(path)
