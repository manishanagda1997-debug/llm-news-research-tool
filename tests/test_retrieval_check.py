"""Tests for Proposal 4: does adding the company's other names to the NewsAPI search
bring in relevant articles that the plain search misses?

No live API calls here: a fake fetch function stands in for NewsAPI.
"""

import csv

import pytest

from src import retrieval_check as rc


def _article(n, title=None):
    return {
        "source": {"id": None, "name": f"Source {n}"},
        "author": "A",
        "title": title or f"Title {n}",
        "description": f"Description {n}",
        "url": f"https://news.example.com/story-{n}",
        "urlToImage": None,
        "publishedAt": "2026-10-20T10:00:00Z",
        "content": f"Content {n} " * 30,
    }


def _response(articles, aliases_added, from_date="2026-10-16"):
    return {
        "status": "ok",
        "totalResults": len(articles),
        "articles": articles,
        "request": {
            "query": "Tata Motors",
            "search_query": "x",
            "aliases_added": aliases_added,
            "from_date": from_date,
            "sort_by": "relevancy",
            "page_size": 30,
            "language": "en",
        },
    }


class FakeFetch:
    """Records every call and returns prepared responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, query, **options):
        self.calls.append({"query": query, **options})
        return self.responses.pop(0)


ALIASES = {"identities": {}, "search_groups": {}}


# ---------- 1. the two searches ----------

def test_run_pair_makes_two_searches_with_identical_settings():
    fake = FakeFetch([_response([_article(1)], []), _response([_article(2)], ["TMPV"])])
    rc.run_pair("Tata Motors", ALIASES, fetch=fake)
    assert len(fake.calls) == 2
    first, second = fake.calls
    assert first["aliases"] is None
    assert second["aliases"] is ALIASES
    for key in ("page_size", "days_back", "sort_by", "exact_phrase", "title_only"):
        assert first[key] == second[key]


def test_run_pair_uses_the_same_settings_as_the_app_by_default():
    fake = FakeFetch([_response([], []), _response([], ["TMPV"])])
    rc.run_pair("Tata Motors", ALIASES, fetch=fake)
    assert fake.calls[0]["page_size"] == rc.PAGE_SIZE == 30
    assert fake.calls[0]["days_back"] == rc.DAYS_BACK == 7
    assert fake.calls[0]["sort_by"] == "relevancy"


def test_run_pair_returns_both_responses_by_name():
    plain, wide = _response([_article(1)], []), _response([_article(2)], ["TMPV"])
    pair = rc.run_pair("Tata Motors", ALIASES, fetch=FakeFetch([plain, wide]))
    assert pair["without_aliases"] is plain
    assert pair["with_aliases"] is wide


def test_run_pair_stops_if_the_two_searches_cover_different_dates():
    # e.g. the clock passed midnight (UTC) between the two calls
    fake = FakeFetch([_response([], [], "2026-10-16"), _response([], ["TMPV"], "2026-10-17")])
    with pytest.raises(rc.RetrievalCheckError):
        rc.run_pair("Tata Motors", ALIASES, fetch=fake)


def test_run_pair_stops_if_no_alias_names_were_added():
    # without extra names both searches are the same, so the comparison means nothing
    fake = FakeFetch([_response([], []), _response([], [])])
    with pytest.raises(rc.RetrievalCheckError):
        rc.run_pair("Tata Motors", ALIASES, fetch=fake)


# ---------- 2. the blind labelling sheet ----------

def test_sheet_holds_only_articles_not_labelled_before():
    pair = {
        "without_aliases": _response([_article(1), _article(2)], []),
        "with_aliases": _response([_article(2), _article(3)], ["TMPV"]),
    }
    already = {"https://news.example.com/story-1"}
    rows = rc.labelling_rows(pair, already)
    assert sorted(r["url"] for r in rows) == ["https://news.example.com/story-2", "https://news.example.com/story-3"]


def test_sheet_matches_already_labelled_urls_after_normalizing():
    pair = {
        "without_aliases": _response([_article(1)], []),
        "with_aliases": _response([], ["TMPV"]),
    }
    already = {"https://www.NEWS.example.com/story-1/?utm_source=x"}
    already = {rc.normalize_url(u) for u in already}
    assert rc.labelling_rows(pair, already) == []


def test_sheet_lists_an_article_found_by_both_searches_once():
    pair = {
        "without_aliases": _response([_article(5)], []),
        "with_aliases": _response([_article(5)], ["TMPV"]),
    }
    assert len(rc.labelling_rows(pair, set())) == 1


def test_sheet_does_not_reveal_which_search_found_an_article():
    pair = {
        "without_aliases": _response([_article(1)], []),
        "with_aliases": _response([_article(2)], ["TMPV"]),
    }
    rows = rc.labelling_rows(pair, set())
    assert list(rows[0].keys()) == rc.SHEET_COLUMNS
    for column in rc.SHEET_COLUMNS:
        assert "search" not in column and "alias" not in column and "found" not in column


def test_sheet_has_empty_label_columns_and_numbered_rows():
    pair = {
        "without_aliases": _response([_article(1), _article(2)], []),
        "with_aliases": _response([_article(3)], ["TMPV"]),
    }
    rows = rc.labelling_rows(pair, set())
    assert [r["row"] for r in rows] == [1, 2, 3]
    assert all(r["label"] == "" and r["unsure"] == "" and r["note"] == "" for r in rows)
    assert all(len(r["content_first_200_chars"]) <= 200 for r in rows)


def test_sheet_order_is_shuffled_but_repeatable():
    many = [_article(n) for n in range(20)]
    pair = {"without_aliases": _response(many[:10], []), "with_aliases": _response(many[10:], ["TMPV"])}
    first = [r["url"] for r in rc.labelling_rows(pair, set())]
    second = [r["url"] for r in rc.labelling_rows(pair, set())]
    found_order = [a["url"] for a in many]
    assert first == second          # same seed, same order
    assert first != found_order     # the plain-search articles are not all at the top


def test_sheet_skips_removed_articles():
    removed = _article(9, title="[Removed]")
    pair = {"without_aliases": _response([removed, _article(1)], []), "with_aliases": _response([], ["TMPV"])}
    assert [r["url"] for r in rc.labelling_rows(pair, set())] == ["https://news.example.com/story-1"]


def test_write_sheet_creates_a_csv_and_never_overwrites(tmp_path):
    rows = [{c: "" for c in rc.SHEET_COLUMNS} | {"row": 1, "url": "u"}]
    first = rc.write_sheet(rows, tmp_path, "dev_tata_motors_retrieval")
    second = rc.write_sheet(rows, tmp_path, "dev_tata_motors_retrieval")
    assert first != second
    with open(first, newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == rc.SHEET_COLUMNS


# ---------- 3. the scores after labelling ----------

LABELS = {
    "https://news.example.com/story-1": "direct",
    "https://news.example.com/story-2": "indirect",
    "https://news.example.com/story-3": "direct",
    "https://news.example.com/story-4": "not",
}


def _scored_pair():
    return {
        "without_aliases": _response([_article(1), _article(2), _article(4)], []),
        "with_aliases": _response([_article(1), _article(3), _article(4)], ["TMPV"]),
    }


def test_scores_count_relevant_articles_each_search_found():
    result = rc.retrieval_scores(_scored_pair(), LABELS)
    assert result["pool_relevant"] == 3                    # stories 1, 2, 3
    assert result["without_aliases"]["returned"] == 3
    assert result["without_aliases"]["relevant_found"] == 2
    assert result["with_aliases"]["relevant_found"] == 2
    assert result["without_aliases"]["pool_recall"] == pytest.approx(2 / 3)
    assert result["without_aliases"]["candidate_precision"] == pytest.approx(2 / 3)


def test_scores_show_relevant_articles_found_by_only_one_search():
    result = rc.retrieval_scores(_scored_pair(), LABELS)
    assert result["without_aliases"]["only_here"] == ["https://news.example.com/story-2"]
    assert result["with_aliases"]["only_here"] == ["https://news.example.com/story-3"]


def test_scores_strict_counts_direct_only():
    result = rc.retrieval_scores(_scored_pair(), LABELS, strict=True)
    assert result["pool_relevant"] == 2                    # stories 1 and 3
    assert result["without_aliases"]["relevant_found"] == 1
    assert result["with_aliases"]["relevant_found"] == 2


def test_scores_stop_when_an_article_has_no_label():
    labels = dict(LABELS)
    del labels["https://news.example.com/story-3"]
    with pytest.raises(rc.RetrievalCheckError):
        rc.retrieval_scores(_scored_pair(), labels)
