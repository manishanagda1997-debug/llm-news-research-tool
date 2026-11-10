"""
Tests for the full-text check (no network: fake fetchers stand in for the web).

NewsAPI's free plan gives only the first 200 characters of each article.
The full-text step fetches the article page itself, only when the site's
robots.txt allows it, takes out the main text, and keeps it in a local cache
(never in Git: the text belongs to the publisher).

The check answers, for the labelled development articles:
    - how many full texts can be fetched (and why the others cannot)
    - for the relevant articles the filter missed: does the full text name
      the company (or its other names)? If yes, full text can fix the miss.
"""

import pytest

from src import fulltext as ft

PAGE = "<html><body><article>" + "Tata Motors sold 50,000 cars this Navratri. " * 30 + "</article></body></html>"


class FakeWeb:
    """robots.txt answers and pages by URL; records every request."""

    def __init__(self, robots=None, pages=None):
        self.robots = robots or {}
        self.pages = pages or {}
        self.requests = []

    def __call__(self, url, timeout=10):
        self.requests.append(url)
        if url.endswith("/robots.txt"):
            host = url.rsplit("/robots.txt", 1)[0]
            if host not in self.robots:
                return 404, ""
            answer = self.robots[host]
            if isinstance(answer, Exception):
                raise answer
            return 200, answer
        answer = self.pages.get(url, (404, ""))
        if isinstance(answer, Exception):
            raise answer
        return answer


def _extract(html):
    """A fake text extractor: everything between the article tags."""
    if "<article>" not in html:
        return ""
    return html.split("<article>")[1].split("</article>")[0].strip()


# ---------- robots.txt ----------

def test_robots_allows_when_the_rules_allow():
    web = FakeWeb(robots={"https://news.example.com": "User-agent: *\nDisallow: /private/\n"})
    robots = ft.RobotsCheck(web)
    assert robots.allows("https://news.example.com/2026/tata-motors.html") is True
    assert robots.allows("https://news.example.com/private/page.html") is False


def test_robots_missing_file_means_allowed():
    robots = ft.RobotsCheck(FakeWeb())
    assert robots.allows("https://news.example.com/a.html") is True


def test_robots_that_cannot_be_read_means_not_allowed():
    web = FakeWeb(robots={"https://news.example.com": ConnectionError("blocked")})
    assert ft.RobotsCheck(web).allows("https://news.example.com/a.html") is False


def test_robots_file_is_read_once_per_site():
    web = FakeWeb(robots={"https://news.example.com": "User-agent: *\nAllow: /\n"})
    robots = ft.RobotsCheck(web)
    robots.allows("https://news.example.com/a.html")
    robots.allows("https://news.example.com/b.html")
    assert web.requests.count("https://news.example.com/robots.txt") == 1


def test_robots_rules_for_our_agent_name_are_followed():
    rules = f"User-agent: {ft.USER_AGENT_NAME}\nDisallow: /\n\nUser-agent: *\nAllow: /\n"
    robots = ft.RobotsCheck(FakeWeb(robots={"https://news.example.com": rules}))
    assert robots.allows("https://news.example.com/a.html") is False


# ---------- fetching one article ----------

def _fetcher(web, tmp_path):
    return ft.FullTextFetcher(http_get=web, extract=_extract, cache_dir=tmp_path, delay_seconds=0)


def test_full_text_is_fetched_when_allowed(tmp_path):
    url = "https://news.example.com/navratri.html"
    web = FakeWeb(pages={url: (200, PAGE)})
    result = _fetcher(web, tmp_path).fetch(url)
    assert result["status"] == "ok"
    assert "50,000 cars" in result["text"] and result["chars"] == len(result["text"])


@pytest.mark.parametrize("page, status", [
    ((403, "Forbidden"), "http_error"),
    ((200, "<html><body>Subscribe to read</body></html>"), "too_short"),
    ((200, "<html><body><article>Short teaser only.</article></body></html>"), "too_short"),
    (TimeoutError("slow"), "error"),
])
def test_problems_are_named(tmp_path, page, status):
    url = "https://news.example.com/a.html"
    assert _fetcher(FakeWeb(pages={url: page}), tmp_path).fetch(url)["status"] == status


def test_a_page_blocked_by_robots_is_never_requested(tmp_path):
    url = "https://news.example.com/a.html"
    web = FakeWeb(robots={"https://news.example.com": "User-agent: *\nDisallow: /\n"}, pages={url: (200, PAGE)})
    result = _fetcher(web, tmp_path).fetch(url)
    assert result["status"] == "blocked_by_robots" and url not in web.requests


def test_a_cached_page_is_not_fetched_again(tmp_path):
    url = "https://news.example.com/navratri.html"
    web = FakeWeb(pages={url: (200, PAGE)})
    _fetcher(web, tmp_path).fetch(url)
    again = FakeWeb()
    result = _fetcher(again, tmp_path).fetch(url)
    assert result["status"] == "ok" and again.requests == []


def test_failures_are_cached_too_so_a_site_is_not_asked_twice(tmp_path):
    url = "https://news.example.com/a.html"
    _fetcher(FakeWeb(pages={url: (403, "")}), tmp_path).fetch(url)
    again = FakeWeb()
    assert _fetcher(again, tmp_path).fetch(url)["status"] == "http_error" and again.requests == []


def test_the_cache_folder_is_ignored_by_git():
    from pathlib import Path
    gitignore = (Path(__file__).resolve().parent.parent / ".gitignore").read_text(encoding="utf-8")
    assert "data/fulltext/" in gitignore


# ---------- the feasibility report ----------

ARTICLES = [
    {"url": "https://a.com/1", "title": "Navratri car sales fall", "description": "Registrations fell.", "content": ""},
    {"url": "https://a.com/2", "title": "Stellantis India plans", "description": "Faster development.", "content": ""},
    {"url": "https://a.com/3", "title": "Tata Motors raises Iveco offer", "description": "Offer raised.", "content": ""},
    {"url": "https://a.com/4", "title": "Crude oil rises", "description": "Brent at $114.", "content": ""},
]
LABELS = {"https://a.com/1": "indirect", "https://a.com/2": "indirect", "https://a.com/3": "direct",
          "https://a.com/4": "not"}
RESULTS = {
    "https://a.com/1": {"status": "ok", "text": "Tata Motors sold 50,000 cars, up 10%.", "chars": 40},
    "https://a.com/2": {"status": "too_short", "text": "", "chars": 0},
    "https://a.com/3": {"status": "ok", "text": "Tata Motors raised its offer.", "chars": 30},
    "https://a.com/4": {"status": "blocked_by_robots", "text": "", "chars": 0},
}


def test_report_counts_fetch_results_overall_and_for_relevant_articles():
    report = ft.feasibility_report(ARTICLES, LABELS, RESULTS, missed=set(), company="Tata Motors", aliases=None)
    assert report["status_counts"] == {"ok": 2, "too_short": 1, "blocked_by_robots": 1}
    assert report["relevant"] == {"total": 3, "fetched": 2}


def test_report_checks_whether_full_text_names_the_company_for_missed_articles():
    missed = {"https://a.com/1", "https://a.com/2"}
    report = ft.feasibility_report(ARTICLES, LABELS, RESULTS, missed=missed, company="Tata Motors", aliases=None)
    assert report["missed"] == {"total": 2, "fetched": 1, "names_company_in_full_text": 1}
    assert report["missed_details"] == [
        {"url": "https://a.com/1", "title": "Navratri car sales fall", "status": "ok", "names_company": True},
        {"url": "https://a.com/2", "title": "Stellantis India plans", "status": "too_short", "names_company": False},
    ]


# ---------- cleaning page junk at the HTML level ----------

ARTICLE_HTML = """<html><body>
<nav>Home | Markets | Auto</nav>
<article>
<p>Mahindra has started making its new seeding machine, the MaxSeeder, for sowing rice, wheat and cotton.</p>
<p class="also-read"><a href="/x">Also Read: To ride the EV wave, M&amp;M takes premium path, Tata Motors bets on scale</a></p>
<p>Made at Mahindra's exclusive farm machinery plant in Pithampur, the machine works with smaller tractors.</p>
<div class="related-stories"><h3>Related stories</h3><ul><li>Tata Motors asks suppliers to ramp up capacity</li></ul></div>
<p>Read more: Tata Motors Q2 preview</p>
<p>The company plans to sell 5,000 units in the first year.</p>
</article>
</body></html>"""


def test_related_story_elements_are_removed_from_the_page():
    cleaned, removed = ft.strip_page_junk(ARTICLE_HTML)
    assert "Also Read" not in cleaned and "Related stories" not in cleaned and "Read more" not in cleaned
    assert any("To ride the EV wave" in text for text in removed)
    assert any("ramp up capacity" in text for text in removed)


def test_real_paragraphs_are_kept():
    cleaned, _ = ft.strip_page_junk(ARTICLE_HTML)
    for sentence in ("seeding machine, the MaxSeeder", "Made at Mahindra's exclusive farm machinery plant",
                     "sell 5,000 units in the first year"):
        assert sentence in cleaned


@pytest.mark.parametrize("paragraph", [
    "<p>Analysts also read the results as a sign of recovery.</p>",          # "also read" inside a real sentence
    "<p>Readers can read more about the plan in the annual report.</p>",
    '<p class="story-text">Tata Motors raised its offer.</p>',
])
def test_ordinary_text_with_similar_words_is_kept(paragraph):
    html = f"<html><body><article>{paragraph}</article></body></html>"
    cleaned, removed = ft.strip_page_junk(html)
    assert removed == []


def test_text_rule_removes_an_also_read_line_left_by_the_extractor():
    text = "First paragraph about sales.\nAlso Read: Tata Motors bets on scale\nSecond paragraph."
    cleaned, removed = ft.strip_text_junk(text)
    assert cleaned == "First paragraph about sales.\nSecond paragraph."
    assert removed == ["Also Read: Tata Motors bets on scale"]


def test_text_rule_keeps_lines_that_only_contain_the_words():
    text = "Analysts also read the numbers closely.\nRead more carefully, the report shows growth."
    assert ft.strip_text_junk(text) == (text, [])


# ---------- the record kept for every article ----------

def test_fetch_keeps_extraction_details(tmp_path):
    url = "https://news.example.com/maxseeder.html"
    page = ARTICLE_HTML.replace("</article>", "<p>" + "More detail about the machine. " * 20 + "</p></article>")
    web = FakeWeb(pages={url: (200, page)})
    result = _fetcher(web, tmp_path).fetch(url)
    assert result["status"] == "ok"
    assert result["extraction_method"] == ft.EXTRACTION_METHOD
    assert result["boilerplate_removed"] is True and result["removed_texts"]
    assert result["original_length"] >= result["cleaned_length"] == result["chars"]
    assert result["content_quality"] == "ok"
    assert "Also Read" not in result["text"]


def test_the_page_code_is_kept_in_the_cache_for_later_cleaning(tmp_path):
    url = "https://news.example.com/navratri.html"
    _fetcher(FakeWeb(pages={url: (200, PAGE)}), tmp_path).fetch(url)
    assert ft.FullTextFetcher(cache_dir=tmp_path).cached_html(url) == PAGE


def test_an_old_cache_entry_without_page_code_is_fetched_again(tmp_path):
    url = "https://news.example.com/navratri.html"
    fetcher = _fetcher(FakeWeb(pages={url: (200, PAGE)}), tmp_path)
    fetcher._save({"url": url, "status": "ok", "text": "old text", "chars": 8, "http_status": 200})
    web = FakeWeb(pages={url: (200, PAGE)})
    result = _fetcher(web, tmp_path).fetch(url)
    assert url in web.requests and result["extraction_method"] == ft.EXTRACTION_METHOD


# ---------- edge cases ----------

def test_a_link_inside_a_real_sentence_is_kept():
    html = ('<html><body><article><p>Tata Motors said, as <a href="/x">reported earlier</a>, that sales rose 10%.</p>'
            '<p class="also-read"><a href="/y">Also Read: Unrelated story</a></p></article></body></html>')
    cleaned, removed = ft.strip_page_junk(html)
    assert "reported earlier" in cleaned and "sales rose 10%" in cleaned
    assert removed == ["Also Read: Unrelated story"]


def test_removing_a_nested_box_keeps_its_parent_section():
    html = ('<html><body><article><div class="article-body"><p>Real first paragraph about Tata Motors.</p>'
            '<div class="related-news"><p>Some other story</p></div>'
            '<p>Real second paragraph about exports.</p></div></article></body></html>')
    cleaned, removed = ft.strip_page_junk(html)
    assert "Real first paragraph" in cleaned and "Real second paragraph" in cleaned
    assert removed == ["Some other story"]


def test_a_wrapper_holding_most_of_the_article_is_never_removed():
    # a class name such as "post-with-related" on the main wrapper must not delete the article
    paragraphs = "".join(f"<p>Real paragraph number {n} about Tata Motors sales and exports.</p>" for n in range(8))
    html = f'<html><body><div class="post-with-related">{paragraphs}<div class="related"><p>Other story</p></div></div></body></html>'
    cleaned, removed = ft.strip_page_junk(html)
    assert all(f"Real paragraph number {n}" in cleaned for n in range(8))
    assert removed == ["Other story"]


def test_every_box_on_a_page_is_removed_and_recorded():
    html = ('<html><body><article><p>Real text.</p>'
            '<aside class="related-stories"><p>First box story</p></aside>'
            '<p>More real text.</p>'
            '<div class="read-more"><p>Second box story</p></div></article></body></html>')
    cleaned, removed = ft.strip_page_junk(html)
    assert "First box story" not in cleaned and "Second box story" not in cleaned
    assert sorted(removed) == ["First box story", "Second box story"]


def test_old_text_is_kept_when_the_page_cannot_be_fetched_again(tmp_path):
    url = "https://news.example.com/navratri.html"
    old_text = "Tata Motors sold 50,000 cars this Navratri. " * 20
    _fetcher(FakeWeb(), tmp_path)._save({"url": url, "status": "ok", "text": old_text,
                                          "chars": len(old_text), "http_status": 200})
    result = _fetcher(FakeWeb(pages={url: (403, "")}), tmp_path).fetch(url)
    assert result["status"] == "ok" and result["text"] == old_text
    assert result["html_cleaning"] == "unavailable"


# ---------- a real paragraph inside a junk-named box is kept ----------

LONG_PARAGRAPH = ("New Delhi: The chairperson of Tata Sons, the chief executive of Jaguar Land Rover and the managing "
                  "director of Tata Motors are among the top executives set to join the Iveco board once the deal "
                  "closes, the company said in a filing on Monday.")
MENU = "".join(f'<li><a href="/m{n}">Menu link number {n} about markets and money</a></li>' for n in range(40))


def test_a_real_paragraph_inside_a_box_named_like_junk_is_kept():
    # some sites wrap the article itself in an "expand to read more" box
    html = (f'<html><body><nav><ul>{MENU}</ul></nav>'
            f'<div class="taboola-readmore"><div class="storyContent"><p>{LONG_PARAGRAPH}</p></div></div>'
            f'</body></html>')
    cleaned, removed = ft.strip_page_junk(html)
    assert LONG_PARAGRAPH in cleaned and removed == []


def test_a_box_of_long_link_teasers_is_still_removed():
    teaser = ("Tata Motors asks suppliers to ramp up capacity for 100,000 passenger vehicles a month from FY28 "
              "as demand for electric and CNG models keeps rising across the country this festive season")
    html = (f'<html><body><nav><ul>{MENU}</ul></nav><article><p>{LONG_PARAGRAPH}</p>'
            f'<div class="related-stories"><p><a href="/t">{teaser}</a></p></div></article></body></html>')
    cleaned, removed = ft.strip_page_junk(html)
    assert LONG_PARAGRAPH in cleaned and teaser not in cleaned
    assert removed == [teaser]


# ---------- re-cleaning from the saved page code, without the web ----------

def test_a_result_from_an_older_cleaning_version_is_cleaned_again_from_the_saved_page(tmp_path):
    url = "https://news.example.com/navratri.html"
    fetcher = _fetcher(FakeWeb(pages={url: (200, PAGE)}), tmp_path)
    first = fetcher.fetch(url)
    fetcher._save({**first, "extraction_method": "an older cleaning version", "text": "old"})
    offline = FakeWeb()
    result = _fetcher(offline, tmp_path).fetch(url)
    assert offline.requests == []                         # no web request
    assert result["extraction_method"] == ft.EXTRACTION_METHOD and "50,000 cars" in result["text"]


def test_old_usable_text_is_kept_when_cleaning_again_leaves_too_little(tmp_path):
    url = "https://news.example.com/short.html"
    old_text = "Tata Motors sold 50,000 cars this Navratri. " * 20
    fetcher = _fetcher(FakeWeb(), tmp_path)
    fetcher._save({"url": url, "status": "ok", "text": old_text, "chars": len(old_text), "http_status": 200,
                   "extraction_method": "an older cleaning version"},
                  html="<html><body><article>Too short now.</article></body></html>")
    result = _fetcher(FakeWeb(), tmp_path).fetch(url)
    assert result["status"] == "ok" and result["text"] == old_text and result["html_cleaning"] == "unavailable"
