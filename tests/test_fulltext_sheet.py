"""
Tests for two steps before full-text labelling (no network).

1. Paragraph check: cleaning must keep the article's real paragraphs, in
   their original order. Matching character counts alone do not prove that.
2. Labelling files: a reading page (full text, to read in the browser) and a
   small CSV for the labels, joined by a stable article id. The page shows
   title, publisher, date, extraction status and the cleaned text; it never
   shows old labels, tool decisions or which search found the article.
"""

import csv

from src import fulltext as ft

ORIGINAL = "\n".join([
    "Tata Motors sold 50,000 cars this Navratri, up 10% from last year, the company said on Monday in Mumbai.",
    "Also Read: Unrelated story",
    "Demand was strongest for electric and CNG models, with waiting periods of up to six weeks in large cities.",
    "Subscribe",
    "The company expects the festive season to lift full-year sales above the previous record set in 2025-26.",
])


# ---------- 1. paragraph check ----------

def test_paragraph_check_passes_when_only_junk_is_removed():
    cleaned = "\n".join(p for p in ORIGINAL.split("\n") if p not in ("Also Read: Unrelated story", "Subscribe"))
    result = ft.paragraph_check(ORIGINAL, cleaned)
    assert result["order_kept"] is True
    assert result["missing_long"] == []
    assert result["missing"] == ["Also Read: Unrelated story", "Subscribe"]
    assert result["kept"] == 3


def test_paragraph_check_finds_a_lost_real_paragraph():
    paragraphs = ORIGINAL.split("\n")
    cleaned = "\n".join([paragraphs[0], paragraphs[4]])      # the demand paragraph is lost
    result = ft.paragraph_check(ORIGINAL, cleaned)
    assert result["missing_long"] == [paragraphs[2]]


def test_paragraph_check_finds_changed_order():
    paragraphs = ORIGINAL.split("\n")
    cleaned = "\n".join([paragraphs[4], paragraphs[0], paragraphs[2]])
    assert ft.paragraph_check(ORIGINAL, cleaned)["order_kept"] is False


def test_paragraph_check_finds_text_that_was_not_in_the_original():
    cleaned = ORIGINAL + "\nA sentence that appeared from nowhere and was never part of the page."
    assert ft.paragraph_check(ORIGINAL, cleaned)["added"] == [
        "A sentence that appeared from nowhere and was never part of the page."]


# ---------- 2. labelling files ----------

META = [
    {"url": f"https://news.example.com/{n}", "title": f"Title {n}", "source": f"Paper {n}",
     "published": f"2026-10-2{n}T10:00:00Z"} for n in range(1, 7)
]
RESULTS = {
    m["url"]: {"url": m["url"], "status": "ok", "text": f"Paragraph one of {n}.\nParagraph two of {n}.",
               "html_cleaning": "done", "content_quality": "ok"}
    for n, m in enumerate(META, start=1)
}
RESULTS["https://news.example.com/6"]["html_cleaning"] = "unavailable"


def test_article_id_is_stable_and_based_on_the_web_address():
    first = ft.article_id("https://www.news.example.com/1/?utm_source=x")
    assert first == ft.article_id("https://news.example.com/1")
    assert first != ft.article_id("https://news.example.com/2")
    assert first.startswith("A") and len(first) == 11


def test_sheet_rows_join_csv_and_page_by_article_id():
    csv_rows, page_items = ft.labelling_items(META, RESULTS)
    assert [r["article_id"] for r in csv_rows] == [p["article_id"] for p in page_items]
    assert [r["row"] for r in csv_rows] == list(range(1, 7)) == [p["row"] for p in page_items]
    for row in csv_rows:
        assert row["article_id"] == ft.article_id(row["url"])


def test_csv_has_only_the_agreed_columns_and_empty_labels():
    csv_rows, _ = ft.labelling_items(META, RESULTS)
    assert list(csv_rows[0].keys()) == ft.LABEL_COLUMNS == ["row", "article_id", "title", "url", "label", "unsure", "note"]
    assert all(r["label"] == r["unsure"] == r["note"] == "" for r in csv_rows)


def test_rows_are_shuffled_but_repeatable():
    first, _ = ft.labelling_items(META, RESULTS)
    second, _ = ft.labelling_items(META, RESULTS)
    assert [r["url"] for r in first] == [r["url"] for r in second]
    assert [r["url"] for r in first] != [m["url"] for m in META]


def test_only_articles_with_full_text_are_included():
    results = dict(RESULTS)
    results["https://news.example.com/3"] = {"url": META[2]["url"], "status": "blocked_by_robots", "text": ""}
    csv_rows, _ = ft.labelling_items(META, results)
    assert "https://news.example.com/3" not in [r["url"] for r in csv_rows] and len(csv_rows) == 5


def test_page_shows_metadata_status_and_text():
    _, items = ft.labelling_items(META, RESULTS)
    item = [i for i in items if i["url"] == "https://news.example.com/1"][0]
    assert (item["title"], item["source"], item["published"]) == ("Title 1", "Paper 1", "2026-10-21T10:00:00Z")
    assert item["status"] == "complete" and item["paragraphs"] == ["Paragraph one of 1.", "Paragraph two of 1."]
    old = [i for i in items if i["url"] == "https://news.example.com/6"][0]
    assert old["status"] == "older text, page cleaning unavailable"


def test_page_html_contains_every_article_and_no_label_information(tmp_path):
    csv_rows, items = ft.labelling_items(META, RESULTS)
    csv_path, page_path = ft.write_labelling_files(csv_rows, items, tmp_path / "eval", tmp_path / "fulltext", "20261024")
    page = page_path.read_text(encoding="utf-8")
    for item in items:
        assert f'id="{item["article_id"]}"' in page and item["title"] in page
    for word in ("direct", "indirect", "found_by", "kept", "missed", "prediction"):
        assert word not in page.lower()
    assert page_path.parent == tmp_path / "fulltext"
    with open(csv_path, newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == ft.LABEL_COLUMNS


def test_page_text_is_escaped():
    meta = [{"url": "https://x.com/1", "title": "<script>alert(1)</script>", "source": "S", "published": ""}]
    results = {"https://x.com/1": {"url": "https://x.com/1", "status": "ok", "text": "a < b & c",
                                   "html_cleaning": "done", "content_quality": "ok"}}
    page = ft.labelling_page(ft.labelling_items(meta, results)[1])
    assert "<script>" not in page and "&lt;script&gt;" in page and "a &lt; b &amp; c" in page


# ---------- fixes found by the paragraph check ----------

REAL = ("Tata Motors shares rose 3% on Monday after the company reported higher sales in the festive season, "
        "analysts said in a note to clients.")


def test_a_label_inside_a_line_removes_the_whole_line_with_its_headline():
    html = (f"<html><body><article><p>{REAL}</p>"
            "<p><strong>Also read:</strong><a href='/x'>HDFC Bank shares drop 15% in 3 months since Q1 results</a></p>"
            "</article></body></html>")
    cleaned, removed = ft.strip_page_junk(html)
    assert "HDFC Bank shares drop" not in cleaned and REAL in cleaned
    assert removed == ["Also read:HDFC Bank shares drop 15% in 3 months since Q1 results"]


def test_a_label_on_its_own_also_removes_the_link_right_after_it():
    html = (f"<html><body><article><p>{REAL}</p>"
            "<div><strong>ALSO READ:</strong></div><div><a href='/x'>Polycab India's F&amp;O exit: NSE sets deadline</a></div>"
            f"<p>{REAL.replace('Monday', 'Tuesday')}</p></article></body></html>")
    cleaned, removed = ft.strip_page_junk(html)
    assert "Polycab" not in cleaned and REAL in cleaned and REAL.replace("Monday", "Tuesday") in cleaned


def test_a_small_element_inside_a_sentence_is_never_removed_by_its_name():
    html = ('<html><body><article><p>Shares of <a class="related-stock" href="/s">Tata Motors</a> rose 3% '
            'after the results were announced on Monday, with volumes well above the monthly average.</p>'
            '</article></body></html>')
    cleaned, removed = ft.strip_page_junk(html)
    assert "Tata Motors</a> rose 3%" in cleaned and removed == []


def test_paragraph_check_shows_the_words_cut_from_a_changed_paragraph():
    before = ("Shetti flags that a small candle was formed on the daily chart. ALSO READ | Nifty set to log 3rd "
              "longest correction The analyst says the market signals a Doji pattern with heightened volatility.")
    after = ("Shetti flags that a small candle was formed on the daily chart. "
             "The analyst says the market signals a Doji pattern with heightened volatility.")
    result = ft.paragraph_check(before, after)
    assert result["missing_long"] == [] and result["added"] == []
    assert len(result["changed"]) == 1
    assert result["changed"][0]["cut"] == "ALSO READ | Nifty set to log 3rd longest correction"
