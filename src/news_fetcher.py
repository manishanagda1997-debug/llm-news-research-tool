"""
Step 6 - Ingestion: get news articles from NewsAPI.

Main function: fetch_news(query, ...)
    1. checks the query and the settings (before any request is used)
    2. calls the NewsAPI /v2/everything endpoint
    3. turns any API problem into one clear error message
    4. saves the raw response in data/raw/ for later study

Free plan limits to keep in mind:
    - 100 requests per day
    - articles have a 24-hour delay
    - search goes back about 1 month only
"""

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException

from src.config import RAW_DIR, settings
from src.query import QueryError, build_search_plan, build_search_query  # noqa: F401
from src.utils import unique_path

MAX_QUERY_LENGTH = 500        # NewsAPI limit for the q parameter
MAX_DAYS_BACK = 29            # free plan searches up to about 1 month back
VALID_SORT_OPTIONS = ("relevancy", "popularity", "publishedAt")

# NewsAPI error codes and the message shown to the user
ERROR_MESSAGES = {
    "apiKeyInvalid": "The NewsAPI key is not valid. Check NEWSAPI_KEY in the .env file.",
    "apiKeyMissing": "The NewsAPI key is missing. Add NEWSAPI_KEY to the .env file.",
    "apiKeyDisabled": "The NewsAPI key has been disabled. Create a new key on newsapi.org.",
    "apiKeyExhausted": "The NewsAPI key has no requests left.",
    "rateLimited": "The daily NewsAPI limit (100 requests) is used up. Try again tomorrow.",
    "parameterInvalid": "One of the search settings is not accepted by NewsAPI.",
    "parametersMissing": "The search is missing a required setting.",
    "maximumResultsReached": "No more results can be fetched on the free plan.",
}


class NewsFetchError(Exception):
    """Raised when news cannot be fetched. The message is safe to show to the user."""


def _is_whole_number(value) -> bool:
    # True for 5, False for "5", 5.0 or True (bool counts as a number in Python)
    return isinstance(value, int) and not isinstance(value, bool)


def _check_inputs(query: str, page_size, days_back, sort_by: str) -> None:
    """Stop bad settings before a request is used. Every message is safe to show."""
    if not isinstance(query, str) or not query.strip():
        raise NewsFetchError("Please type a search query.")
    if len(query.strip()) > MAX_QUERY_LENGTH:
        raise NewsFetchError(f"The query is too long. Keep it under {MAX_QUERY_LENGTH} characters.")
    if not _is_whole_number(page_size) or not 1 <= page_size <= 100:
        raise NewsFetchError("Number of articles to fetch must be a whole number from 1 to 100.")
    if not _is_whole_number(days_back) or not 1 <= days_back <= MAX_DAYS_BACK:
        raise NewsFetchError(f"Days back must be a whole number from 1 to {MAX_DAYS_BACK} on the free plan.")
    if sort_by not in VALID_SORT_OPTIONS:
        raise NewsFetchError(f"Sort must be one of: {', '.join(VALID_SORT_OPTIONS)}.")


def fetch_news(
    query: str,
    page_size: int = 30,
    days_back: int | None = None,
    sort_by: str | None = None,
    title_only: bool = False,
    exact_phrase: bool = True,
    aliases: dict | None = None,
    save: bool = True,
    part: str = "all",
    from_time: str | None = None,
    to_time: str | None = None,
) -> dict:
    """
    Fetch articles about `query` from NewsAPI.

    page_size  - how many articles to ask for (more than we need, because
                 cleaning removes some of them later)
    days_back  - how many days of news to search
    sort_by    - relevancy, popularity or publishedAt (newest first)
    title_only - search the query in article titles only (fewer but more
                 relevant results, e.g. "Apple" the company, not the fruit)
    exact_phrase - for plain words: True sends "Tata Motors" as one phrase,
                 False sends Tata AND Motors (all words, any order)
    aliases    - loaded alias file (src.query.load_aliases) to also search for
                 verified company names such as TMPV, or None
    save       - save the raw response in data/raw/
    part       - which alias names to search for: "all", "own" or
                 "subsidiaries" (see src.query.build_search_plan)
    from_time, to_time - an exact publication window in UTC, both given as
                 "YYYY-MM-DDTHH:MM:SSZ" (for the fresh evaluation); when
                 given, days_back is not used

    Returns the NewsAPI response as a dict, with an extra "request" part
    that records the request and its time.
    """
    if not settings.newsapi_key:
        raise NewsFetchError(ERROR_MESSAGES["apiKeyMissing"])

    # "is None" instead of "or": a wrong value such as 0 must be rejected, not replaced
    window = _check_window(from_time, to_time)
    if days_back is None:
        days_back = settings.default_days_back
    if sort_by is None:
        sort_by = settings.default_sort_by
    _check_inputs(query, page_size, days_back, sort_by)
    query = query.strip()

    try:
        plan = build_search_plan(query, exact_phrase=exact_phrase, aliases=aliases, part=part)
    except QueryError as error:
        raise NewsFetchError(str(error)) from error
    search_query = plan["search_query"]
    if not search_query:
        raise NewsFetchError(plan["note"] or "Nothing to search for.")

    from_date = window[0] if window else (datetime.now(timezone.utc) - timedelta(days=days_back)).date().isoformat()
    to_date = window[1] if window else None
    # the newsapi client refuses a trailing Z; it treats every time without a zone as UTC
    sent_from = from_date.removesuffix("Z")
    dates = {"from_param": sent_from, **({"to": to_date.removesuffix("Z")} if to_date else {})}

    client = NewsApiClient(api_key=settings.newsapi_key)
    search = {"qintitle": search_query} if title_only else {"q": search_query}

    try:
        response = client.get_everything(
            **search,
            **dates,
            language=settings.language,
            sort_by=sort_by,
            page_size=page_size,
        )
    except NewsAPIException as error:
        details = error.get_exception() or {}
        code = details.get("code", "")
        message = ERROR_MESSAGES.get(code, f"NewsAPI error: {details.get('message', 'unknown problem')}")
        raise NewsFetchError(message) from error
    except requests.exceptions.Timeout as error:
        raise NewsFetchError("NewsAPI took too long to answer. Please try again.") from error
    except requests.exceptions.ConnectionError as error:
        raise NewsFetchError("Could not reach NewsAPI. Check the internet connection.") from error
    except requests.exceptions.RequestException as error:
        raise NewsFetchError(f"Network problem while calling NewsAPI: {error}") from error

    response["request"] = {
        "query": query,
        "search_query": search_query,
        "title_only": title_only,
        "exact_phrase": exact_phrase,
        "aliases_added": plan["aliases_added"],
        "search_part": part,
        "search_note": plan["note"],
        "from_date": from_date,
        "to_date": to_date,
        "sort_by": sort_by,
        "page_size": page_size,
        "language": settings.language,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if save:
        save_raw(response)

    return response


WINDOW_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _check_window(from_time: str | None, to_time: str | None) -> tuple[str, str] | None:
    """Both ends or neither; each a full UTC date-time; the start before the end."""
    if from_time is None and to_time is None:
        return None
    if from_time is None or to_time is None:
        raise NewsFetchError("Give both ends of the publication window, or neither.")
    try:
        start = datetime.strptime(from_time, WINDOW_FORMAT)
        end = datetime.strptime(to_time, WINDOW_FORMAT)
    except (TypeError, ValueError) as error:
        raise NewsFetchError("Window times must look like 2026-10-24T00:00:00Z.") from error
    if start >= end:
        raise NewsFetchError("The window must start before it ends.")
    return from_time, to_time


def _slug(text: str, max_length: int = 40) -> str:
    """Turn a query into a safe file name part, e.g. 'Tata Motors EV' -> 'tata-motors-ev'."""
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text[:max_length] or "query"


def save_raw(response: dict, folder: Path = RAW_DIR) -> Path:
    """Save one NewsAPI response as a JSON file and return its path."""
    request = response.get("request", {})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = unique_path(folder, f"{stamp}_{_slug(request.get('query', 'query'))}")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(response, file, ensure_ascii=False, indent=2)
    return path


def load_raw(path: Path) -> dict:
    """Read a saved raw response back from data/raw/."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    # Quick manual check:  python -m src.news_fetcher "Tata Motors"
    import sys

    from src.preprocess import clean_articles

    user_query = " ".join(sys.argv[1:]) or "Infosys"
    try:
        raw = fetch_news(user_query)
    except NewsFetchError as problem:
        print(f"[FAIL] {problem}")
        sys.exit(1)

    articles, stats = clean_articles(raw["articles"], max_articles=settings.default_num_articles, query=user_query)
    log = stats.pop("duplicate_log")
    print(f"Query: {user_query}   (sent to NewsAPI as: {raw['request']['search_query']})")
    print(f"Total matches on NewsAPI: {raw.get('totalResults', 0)}")
    print(f"Cleaning: {stats}")
    for entry in log:
        print(f"  {entry['type']} (score {entry['score']}): kept '{entry['kept']['title'][:50]}', "
              f"removed '{entry['removed']['title'][:50]}'")
    print()
    for article in articles:
        print(f"[{article['id']}] {article['title']}")
        print(f"     {article['source']} | {article['published_at'][:10]}")
