"""
Tests for fetching a fixed publication window (fake NewsAPI client, no requests).

The fresh evaluation needs exact dates, not "days back": from
2026-10-24T00:00:00Z to 2026-10-30T23:59:59Z. NewsAPI's Everything endpoint
takes ISO 8601 date-times in its `from` and `to` parameters. The times are
checked before any request is used, and the request record keeps them (and
never the API key).
"""

import dataclasses

import pytest

import src.news_fetcher as news_fetcher


@pytest.fixture
def sent(monkeypatch):
    calls = {}

    def fake_get_everything(self, **kwargs):
        calls.update(kwargs)
        return {"status": "ok", "totalResults": 0, "articles": []}

    monkeypatch.setattr(news_fetcher, "settings",
                        dataclasses.replace(news_fetcher.settings, newsapi_key="SECRET-TEST-KEY"))
    monkeypatch.setattr(news_fetcher.NewsApiClient, "get_everything", fake_get_everything)
    return calls


def test_exact_from_and_to_times_are_sent_in_the_form_the_client_accepts(sent):
    """The newsapi client takes YYYY-MM-DDTHH:MM:SS with no zone and treats it as UTC; a trailing Z is refused."""
    from newsapi.utils import stringify_date_param
    news_fetcher.fetch_news("Tata Motors", from_time="2026-10-24T00:00:00Z", to_time="2026-10-30T23:59:59Z",
                            save=False)
    assert sent["from_param"] == "2026-10-24T00:00:00" and sent["to"] == "2026-10-30T23:59:59"
    assert stringify_date_param(sent["from_param"]) and stringify_date_param(sent["to"])


def test_the_request_record_keeps_the_window_and_no_key(sent):
    response = news_fetcher.fetch_news("Tata Motors", from_time="2026-10-24T00:00:00Z",
                                       to_time="2026-10-30T23:59:59Z", save=False)
    request = response["request"]
    assert request["from_date"] == "2026-10-24T00:00:00Z" and request["to_date"] == "2026-10-30T23:59:59Z"
    assert "SECRET-TEST-KEY" not in str(response)


def test_without_a_window_days_back_works_as_before(sent):
    response = news_fetcher.fetch_news("Tata Motors", days_back=7, save=False)
    assert "to" not in sent and response["request"]["to_date"] is None


@pytest.mark.parametrize("from_time, to_time", [
    ("2026-10-24", "2026-10-30T23:59:59Z"),                       # not a full date-time
    ("2026-10-24T00:00:00Z", "2026-10-30 23:59:59"),              # no T and no Z
    ("2026-10-30T23:59:59Z", "2026-10-24T00:00:00Z"),             # reversed
    ("2026-10-24T00:00:00Z", None),                               # only one end
])
def test_a_bad_window_is_refused_before_any_request(sent, from_time, to_time):
    with pytest.raises(news_fetcher.NewsFetchError):
        news_fetcher.fetch_news("Tata Motors", from_time=from_time, to_time=to_time, save=False)
    assert sent == {}



def test_the_newsapi_request_has_a_time_limit_and_a_clear_message_when_it_is_reached(monkeypatch):
    """The newsapi client (the pinned 0.2.7) sends every search with timeout=30, and fetch_news turns a timeout
    into a plain message, so a slow answer can never leave the app waiting without end."""
    import dataclasses
    import inspect

    import pytest
    import requests
    from newsapi import NewsApiClient

    from src import news_fetcher as nf
    assert "timeout=30" in inspect.getsource(NewsApiClient.get_everything)

    class SlowClient:
        def __init__(self, api_key):
            pass

        def get_everything(self, **options):
            raise requests.exceptions.Timeout("slow")
    monkeypatch.setattr(nf, "NewsApiClient", SlowClient)
    monkeypatch.setattr(nf, "settings", dataclasses.replace(nf.settings, newsapi_key="test-key"))
    with pytest.raises(nf.NewsFetchError, match="took too long"):
        nf.fetch_news("Tata Motors", page_size=5, days_back=7, save=False)
