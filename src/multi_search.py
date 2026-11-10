"""
Separate NewsAPI searches for one company, each with its own article slots.

    A  the company name only                  "Tata Motors"
    B  the subsidiaries' names                ("Jaguar Land Rover" OR JLR OR ...)
    C  the company and its own names          ("Tata Motors" OR TMPV OR ... OR TMCV)

One search cannot push another search's articles out of the results,
because each has its own slots. The results are merged afterwards; exact
copies (same cleaned web address) are kept once, and the cleaning step
removes copies with the same or a nearly identical title.
"""

from src.news_fetcher import fetch_news
from src.preprocess import normalize_url
from src.query import build_search_plan

PAGE_SIZE = 30
DAYS_BACK = 7
SORT_BY = "relevancy"
SEARCH_NAMES = ("A", "B", "C")
PARTS = {"A": None, "B": "subsidiaries", "C": "own"}   # None: no alias names


class MultiSearchError(Exception):
    pass


def run_searches(company: str, aliases: dict | None, names=SEARCH_NAMES, fetch=fetch_news) -> dict:
    """Make the chosen searches with identical settings. A search with nothing to look for is None."""
    settings = {"page_size": PAGE_SIZE, "days_back": DAYS_BACK, "sort_by": SORT_BY,
                "exact_phrase": True, "title_only": False, "save": False}
    responses = {}
    for name in names:
        part = PARTS[name]
        if part is None:
            responses[name] = fetch(company, aliases=None, **settings)
        elif not build_search_plan(company, aliases=aliases, part=part)["search_query"]:
            responses[name] = None
        else:
            responses[name] = fetch(company, aliases=aliases, part=part, **settings)

    dates = {response["request"]["from_date"] for response in responses.values() if response}
    if len(dates) > 1:
        raise MultiSearchError("The searches cover different dates (the day changed between them). "
                               "Run them again.")
    return responses


def merge(responses: dict, names) -> list[dict]:
    """Articles of the chosen searches, in the order given, each once, with the searches that found it."""
    merged: dict[str, dict] = {}
    for name in names:
        response = responses.get(name)
        if not response:
            continue
        for article in response["articles"]:
            key = normalize_url(article.get("url") or "")
            if key in merged:
                if name not in merged[key]["found_by"]:
                    merged[key]["found_by"].append(name)
            else:
                merged[key] = {**article, "found_by": [name]}
    return list(merged.values())


def budget(responses: dict, names) -> dict:
    """Requests made and article slots asked for by the chosen searches."""
    used = [responses[name] for name in names if responses.get(name)]
    return {"requests": len(used), "slots": sum(response["request"]["page_size"] for response in used)}
