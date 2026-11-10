"""
Tests for the extra cleaning rules agreed after the code review:
search syntax inside cleaning, bad article data, URL and title
duplicates with an audit log, input checks and unique file names.

Run with:  pytest
"""

import pytest

import src.news_fetcher as news_fetcher
import src.preprocess as preprocess
from src.preprocess import clean_articles, save_processed


def make_article(title="Tata Motors wins deal", description="Tata Motors signed a deal.",
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


# ---------- Search syntax inside cleaning ----------

def test_or_query_keeps_articles_with_either_word():
    raw = [
        make_article(title="Google cloud grows", description="Google reported growth.", url="https://a.com/1"),
        make_article(title="Microsoft hires", description="Microsoft added staff.", url="https://b.com/2"),
    ]
    articles, stats = clean_articles(raw, query="Microsoft OR Google")
    assert len(articles) == 2
    assert stats["relevance_check"] == "applied"


def test_excluded_word_removes_article():
    raw = [
        make_article(title="Tata Motors profit rises", url="https://a.com/1"),
        make_article(title="Tata Motors unveils Jaguar EV", url="https://b.com/2"),
    ]
    articles, stats = clean_articles(raw, query="Tata Motors -Jaguar")
    assert [a["title"] for a in articles] == ["Tata Motors profit rises"]
    assert stats["query_not_in_visible_text"] == 1


def test_unsupported_query_skips_only_the_relevance_check():
    raw = [
        make_article(title="Something else entirely", description="No match here.", url="https://a.com/1"),
        make_article(description=None, url="https://b.com/2"),  # still dropped: missing field
    ]
    articles, stats = clean_articles(raw, query="Tata AND (EV OR hybrid)")
    assert len(articles) == 1
    assert stats["relevance_check"] == "skipped"
    assert stats["relevance_note"]
    assert stats["missing_fields"] == 1


# ---------- Bad article data never crashes cleaning ----------

@pytest.mark.parametrize(
    "bad_field",
    [{"url": 12345}, {"source": "Reuters"}, {"content": 456}, {"title": ["not", "text"]}],
)
def test_bad_article_data_is_skipped_and_counted(bad_field):
    bad = make_article(url="https://bad.com/x")
    bad.update(bad_field)
    good = make_article(title="Tata Motors second story", url="https://good.com/y")
    articles, stats = clean_articles([bad, good])
    assert [a["title"] for a in articles] == ["Tata Motors second story"]
    assert stats["invalid_data"] == 1


def test_article_that_is_not_a_dict_is_skipped():
    articles, stats = clean_articles(["just a string", make_article()])
    assert len(articles) == 1
    assert stats["invalid_data"] == 1


# ---------- Duplicates ----------

def test_normalize_url_removes_tracking_and_www():
    assert preprocess.normalize_url("https://www.Example.com/news/a/?utm_source=x&id=5") == preprocess.normalize_url(
        "https://example.com/news/a?id=5"
    )


def test_normalize_url_keeps_real_parameters():
    assert preprocess.normalize_url("https://example.com/news?id=5") != preprocess.normalize_url("https://example.com/news?id=6")


def test_same_link_with_tracking_code_is_a_url_duplicate():
    raw = [
        make_article(title="Tata Motors story one", url="https://example.com/a"),
        make_article(title="Different headline", url="https://www.example.com/a?utm_source=twitter"),
    ]
    articles, stats = clean_articles(raw)
    assert len(articles) == 1
    assert stats["duplicate_url"] == 1


def test_duplicate_log_records_both_articles_and_score():
    raw = [
        make_article(title="Jaguar unveils $130,500 electric vehicle after rebrand", url="https://a.com/1", source="Site A"),
        make_article(title="Jaguar unveils $130,000 electric vehicle after rebrand", url="https://b.com/2", source="Site B"),
    ]
    _, stats = clean_articles(raw)
    entry = stats["duplicate_log"][0]
    assert entry["type"] == "near_duplicate_title"
    assert {entry["kept"]["source"], entry["removed"]["source"]} == {"Site A", "Site B"}
    assert entry["kept"]["url"] and entry["removed"]["url"]
    assert 0.9 <= entry["score"] <= 1.0


def test_richer_copy_is_kept_in_the_first_slot():
    raw = [
        make_article(title="Tata Motors wins deal", description="Short.", url="https://a.com/1"),
        make_article(title="Tata Motors wins deal", description="Tata Motors signed a large export deal with a European buyer.",
                     url="https://b.com/2"),
    ]
    articles, _ = clean_articles(raw)
    assert len(articles) == 1
    assert articles[0]["id"] == 1
    assert articles[0]["description"].startswith("Tata Motors signed a large export deal")


# ---------- Input checks ----------

def test_clean_articles_rejects_bad_arguments():
    with pytest.raises(ValueError):
        clean_articles([], max_articles=0)
    with pytest.raises(ValueError):
        clean_articles("not a list")


@pytest.fixture
def no_network(monkeypatch):
    """Give fetch_news a fake key and make sure it never calls NewsAPI."""
    import dataclasses

    monkeypatch.setattr(news_fetcher, "settings", dataclasses.replace(news_fetcher.settings, newsapi_key="test-key"))

    def fail(*args, **kwargs):
        raise AssertionError("NewsAPI must not be called when the input is invalid")

    monkeypatch.setattr(news_fetcher.NewsApiClient, "get_everything", fail)


@pytest.mark.parametrize(
    "kwargs",
    [{"days_back": 0}, {"days_back": 30}, {"page_size": "abc"}, {"page_size": 0}, {"page_size": 101}],
)
def test_fetch_news_rejects_bad_settings_before_calling_newsapi(no_network, kwargs):
    with pytest.raises(news_fetcher.NewsFetchError):
        news_fetcher.fetch_news("Infosys", save=False, **kwargs)


# ---------- File names never overwrite ----------

def test_two_raw_saves_in_a_row_make_two_files(tmp_path):
    response = {"request": {"query": "Tata Motors"}, "articles": []}
    first = news_fetcher.save_raw(response, folder=tmp_path)
    second = news_fetcher.save_raw(response, folder=tmp_path)
    assert first != second
    assert len(list(tmp_path.iterdir())) == 2


def test_two_processed_saves_in_a_row_make_two_files(tmp_path):
    first = save_processed([], {}, "Tata Motors", folder=tmp_path)
    second = save_processed([], {}, "Tata Motors", folder=tmp_path)
    assert first != second
    assert len(list(tmp_path.iterdir())) == 2


def test_same_time_twice_still_makes_two_files(tmp_path, monkeypatch):
    """Copy what Windows does: the clock gives the same time to both saves."""
    import datetime as real_datetime

    class FrozenClock(real_datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 17, 20, 44, 52, 13163, tzinfo=tz)

    monkeypatch.setattr(news_fetcher, "datetime", FrozenClock)
    response = {"request": {"query": "Tata Motors"}, "articles": []}
    first = news_fetcher.save_raw(response, folder=tmp_path)
    second = news_fetcher.save_raw(response, folder=tmp_path)
    assert first != second
    assert second.name.endswith("_2.json")
