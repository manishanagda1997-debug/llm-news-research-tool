"""
Tests for langchain_config.py, the project's LangChain configuration file
(fake model class and fake NewsAPI fetch, no keys, no network).

The file is the entry point the project brief asks for. It holds no logic of
its own:
    get_llm(provider)            the LangChain chat model, from src/llm.py
    get_news_articles(query)     NewsAPI fetch (src/news_fetcher.py) and the
                                 project's cleaning step (src/preprocess.py)
Importing the file reads no settings and no .env file, builds no model and
makes no request: the project's modules load only inside the functions.
A failed fetch raises NewsFetchError; a search with no results returns an
empty list. Cleaned articles keep every field later stages need, and the
cleaning record (counts per rule and every duplicate decision) comes back
with them.
"""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ---------- importing has no side effects ----------

def test_importing_reads_no_settings_and_loads_no_client():
    code = ("import sys, langchain_config; "
            "print([m for m in ('src.config', 'dotenv', 'newsapi', 'langchain_google_genai', "
            "'src.news_fetcher', 'src.llm') if m in sys.modules])")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_the_file_offers_the_brief_functions():
    import langchain_config
    assert callable(langchain_config.get_llm) and callable(langchain_config.get_news_articles)


# ---------- get_llm ----------

class FakeGemini:
    def __init__(self, **kwargs):
        self.kwargs, self.prompts = kwargs, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return prompt


def test_get_llm_returns_the_langchain_model_without_calling_it(monkeypatch):
    from src import llm as llm_module
    monkeypatch.setattr(llm_module, "_gemini_class", lambda: FakeGemini)
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(
        llm_module.settings, google_api_key="TEST-KEY", gemini_model="gemini-test-model"))
    import langchain_config
    model = langchain_config.get_llm("gemini")
    assert isinstance(model, FakeGemini) and model.kwargs["model"] == "gemini-test-model"
    assert model.prompts == []


def test_get_llm_refuses_an_unknown_provider():
    import langchain_config
    with pytest.raises(ValueError):
        langchain_config.get_llm("someone-else")


# ---------- get_news_articles ----------

def _raw(n, title, description, url=None, source="Reuters"):
    return {"source": {"id": None, "name": source}, "author": "A", "title": title, "description": description,
            "content": f"{description} More text {n}.", "url": url or f"https://news.com/{n}",
            "urlToImage": None, "publishedAt": f"2026-10-2{n}T09:00:00Z"}


class FakeFetch:
    def __init__(self, articles=None, error=None):
        self.articles, self.error, self.calls = articles or [], error, []

    def __call__(self, query, **options):
        self.calls.append((query, options))
        if self.error:
            raise self.error
        return {"status": "ok", "totalResults": len(self.articles), "articles": self.articles,
                "request": {"query": query, "from_date": "2026-10-20"}}


ARTICLES = [
    _raw(1, "Tata Motors raises Iveco offer to 14.40 euro", "Tata Motors raised its offer for Iveco."),
    _raw(2, "Tata Motors raises Iveco offer to 14.40 euro", "The same story on another site.",
         url="https://other.com/2", source="ET"),
    _raw(3, "Crude oil rises above $114", "Brent rose on supply fears."),
    _raw(4, "Jaguar Land Rover opens a Delhi showroom", "JLR expands its retail network in India.",
         source="Mint"),
]


def test_get_news_articles_calls_the_fetch_once_and_nothing_else():
    import langchain_config
    fetch = FakeFetch(ARTICLES)
    langchain_config.get_news_articles("Tata Motors", fetch=fetch, aliases=None, days_back=7)
    assert fetch.calls == [("Tata Motors", {"days_back": 7})]


def test_get_news_articles_returns_cleaned_articles_with_their_metadata():
    import langchain_config
    result = langchain_config.get_news_articles("Tata Motors", fetch=FakeFetch(ARTICLES), aliases=None)
    titles = [a["title"] for a in result["articles"]]
    assert titles == ["Tata Motors raises Iveco offer to 14.40 euro"]          # copy and crude story removed
    first = result["articles"][0]
    assert first["id"] == 1 and first["source"] == "Reuters"
    assert first["url"] == "https://news.com/1" and first["published_at"] == "2026-10-21T09:00:00Z"
    assert first["description"] == "Tata Motors raised its offer for Iveco." and first["mention_level"] == "title"


def test_get_news_articles_returns_the_cleaning_record_and_the_request():
    import langchain_config
    result = langchain_config.get_news_articles("Tata Motors", fetch=FakeFetch(ARTICLES), aliases=None)
    stats = result["stats"]
    assert stats["received"] == 4 and stats["kept"] == 1
    assert stats["duplicate_title"] == 1 and stats["query_not_in_visible_text"] == 2
    assert len(stats["duplicate_log"]) == 1
    assert result["request"] == {"query": "Tata Motors", "from_date": "2026-10-20"}


def test_aliases_find_subsidiary_news():
    import langchain_config
    aliases = json.loads((ROOT / "data" / "company_aliases.json").read_text(encoding="utf-8"))
    result = langchain_config.get_news_articles("Tata Motors", fetch=FakeFetch(ARTICLES), aliases=aliases)
    assert "Jaguar Land Rover opens a Delhi showroom" in [a["title"] for a in result["articles"]]


def test_by_default_the_verified_alias_file_is_used():
    import langchain_config
    result = langchain_config.get_news_articles("Tata Motors", fetch=FakeFetch(ARTICLES))
    assert "Jaguar Land Rover opens a Delhi showroom" in [a["title"] for a in result["articles"]]


def test_a_search_with_no_results_returns_an_empty_list():
    import langchain_config
    result = langchain_config.get_news_articles("Tata Motors", fetch=FakeFetch([]), aliases=None)
    assert result["articles"] == [] and result["stats"]["received"] == 0


def test_a_failed_fetch_raises_instead_of_returning_an_empty_list():
    import langchain_config
    from src.news_fetcher import NewsFetchError
    with pytest.raises(NewsFetchError):
        langchain_config.get_news_articles("Tata Motors", fetch=FakeFetch(error=NewsFetchError("limit used up")),
                                           aliases=None)


def test_the_article_limit_is_passed_to_the_cleaning_step():
    import langchain_config
    topics = ["truck sales", "Iveco offer", "EV plant", "Q2 results", "Nexon prices", "bus orders", "export plans"]
    many = [_raw(n, f"Tata Motors {topic} update", f"Tata Motors {topic}.") for n, topic in enumerate(topics, start=1)]
    result = langchain_config.get_news_articles("Tata Motors", fetch=FakeFetch(many), aliases=None, max_articles=3)
    assert len(result["articles"]) == 3 and result["stats"]["over_limit"] == 4
