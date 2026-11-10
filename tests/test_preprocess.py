"""Checks for the cleaning rules in src/preprocess.py. Run with:  pytest"""

from src.news_fetcher import build_search_query
from src.preprocess import articles_to_text, clean_articles, clean_text


def make_article(title="Infosys wins deal", description="Infosys signed a deal.",
                 url="https://example.com/a", content=None, source="Example News"):
    return {
        "source": {"id": None, "name": source},
        "author": "Reporter",
        "title": title,
        "description": description,
        "url": url,
        "urlToImage": None,
        "publishedAt": "2026-10-10T08:00:00Z",
        "content": content,
    }


def test_clean_text_removes_html_codes_and_tail():
    raw = "Profit rose <b>12%</b> &amp; shares jumped… [+2345 chars]"
    assert clean_text(raw) == "Profit rose 12% & shares jumped"


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_removed_articles_are_dropped():
    raw = [make_article(title="[Removed]", description="[Removed]", url="https://removed.com")]
    articles, stats = clean_articles(raw)
    assert articles == []
    assert stats["removed_by_source"] == 1


def test_missing_description_is_dropped():
    raw = [make_article(description=None)]
    articles, stats = clean_articles(raw)
    assert articles == []
    assert stats["missing_fields"] == 1


def test_duplicates_are_dropped():
    raw = [
        make_article(url="https://site-a.com/x"),
        make_article(url="https://site-a.com/x"),                                  # same link
        make_article(title="Infosys Wins Deal - Other Site", url="https://b.com/y"),  # same headline
    ]
    articles, stats = clean_articles(raw)
    assert len(articles) == 1
    assert stats["duplicate_url"] == 1
    assert stats["duplicate_title"] == 1


def test_limit_and_numbering():
    raw = [make_article(title=f"Story {n}", url=f"https://e.com/{n}") for n in range(15)]
    articles, stats = clean_articles(raw, max_articles=10)
    assert [a["id"] for a in articles] == list(range(1, 11))
    assert stats["kept"] == 10
    assert stats["over_limit"] == 5


def test_prompt_text_has_citation_numbers():
    articles, _ = clean_articles([make_article(), make_article(title="Second", url="https://e.com/2")])
    text = articles_to_text(articles)
    assert text.startswith("[1] Infosys wins deal")
    assert "[2] Second" in text


def test_near_duplicate_titles_are_dropped():
    raw = [
        make_article(title="Jaguar unveils $130,500 electric vehicle after rebrand", url="https://a.com/1"),
        make_article(title="Jaguar unveils $130,000 electric vehicle after rebrand", url="https://b.com/2"),
    ]
    articles, stats = clean_articles(raw)
    assert len(articles) == 1
    assert stats["near_duplicate_title"] == 1


def test_articles_without_query_words_are_dropped():
    raw = [
        make_article(title="Tata Motors raises Iveco offer", url="https://a.com/1"),
        make_article(title="AirConsole partners with Zeekr", description="Gaming deal with Geely.", url="https://b.com/2"),
    ]
    articles, stats = clean_articles(raw, query="Tata Motors")
    assert [a["title"] for a in articles] == ["Tata Motors raises Iveco offer"]
    assert stats["query_not_in_visible_text"] == 1


def test_plain_multi_word_query_becomes_phrase():
    assert build_search_query("Tata Motors") == '"Tata Motors"'
    assert build_search_query("Infosys") == "Infosys"
    assert build_search_query("Tata AND Iveco") == "Tata AND Iveco"
