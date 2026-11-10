"""
LangChain configuration for the News Research Tool.

The project brief asks for this file. It is the entry point to the model and
the news, and it holds no logic of its own:

    get_llm(provider)          the LangChain chat model (src/llm.py)
    get_news_articles(query)   the NewsAPI fetch (src/news_fetcher.py) and the
                               cleaning step (src/preprocess.py)

The brief's example uses LLMChain, which LangChain 1.x no longer includes.
Chains here are built in the current way: prompt | model.

Importing this file reads no settings and no .env file, builds no model and
makes no request: the project's modules load only inside the functions.
"""

DEFAULT_ALIASES = "default"


def get_llm(provider: str | None = None):
    """The LangChain chat model for the provider (LLM_PROVIDER in .env when None). It is built, not called."""
    from src.llm import build_chat_model
    return build_chat_model(provider)


def get_news_articles(query: str, max_articles: int = 10, fetch=None, aliases=DEFAULT_ALIASES,
                      **fetch_options) -> dict:
    """
    Fetch news for the query and clean it with the project's rules.

    fetch         - the fetch function (src.news_fetcher.fetch_news when None)
    aliases       - the verified alias file by default; None for the query words only
    fetch_options - passed to the fetch unchanged (for example days_back, from_time, to_time)

    Returns {"articles": cleaned articles with id, title, source, url,
    published_at, description, content and mention_level; "stats": the
    cleaning record, with counts per rule and every duplicate decision;
    "request": the request settings}. A failed fetch raises NewsFetchError;
    a search with no results returns an empty list.
    """
    from src.config import DATA_DIR
    from src.preprocess import clean_articles
    from src.query import load_aliases
    if fetch is None:
        from src.news_fetcher import fetch_news as fetch
    if aliases == DEFAULT_ALIASES:
        aliases = load_aliases(DATA_DIR / "company_aliases.json")
    response = fetch(query, **fetch_options)
    articles, stats = clean_articles(response.get("articles", []), max_articles=max_articles, query=query,
                                     aliases=aliases, remove_junk=True, ranking="mention", drop_listings="both")
    return {"articles": articles, "stats": stats, "request": response.get("request", {})}
