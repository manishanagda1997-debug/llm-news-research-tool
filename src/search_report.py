"""
Reports for the separate searches: what each configuration finds, at what
cost, how many slots sale listings take, and which articles are new.

Configurations (after merging and removing exact copies):
    A     the company name only
    A+B   plus the subsidiaries' search
    C     the company and its own names
    C+B   plus the subsidiaries' search

Each configuration reports its budget (requests and article slots), so that
a gain from a bigger budget is not mistaken for a gain from better terms.
"""

from difflib import SequenceMatcher

from src.evaluate import is_relevant
from src.listings import listing_signals
from src.multi_search import budget, merge
from src.preprocess import NEAR_DUPLICATE_MIN_LENGTH, NEAR_DUPLICATE_RATIO, _title_key, normalize_url

CONFIGURATIONS = {"A": ["A"], "A+B": ["A", "B"], "C": ["C"], "C+B": ["C", "B"]}


class SearchReportError(Exception):
    pass


def _is_listing(article: dict) -> bool:
    signals = listing_signals(article)
    return signals["pattern"] or signals["site"]


def configuration_scores(responses: dict, labels: dict[str, str], is_listing=_is_listing,
                         strict: bool = False, configurations: dict = CONFIGURATIONS) -> dict:
    every = merge(responses, [name for name in responses if responses[name]])
    missing = [a["url"] for a in every if normalize_url(a["url"]) not in labels]
    if missing:
        raise SearchReportError(f"{len(missing)} article(s) have no label.")
    relevant = {normalize_url(a["url"]) for a in every if is_relevant(labels[normalize_url(a["url"])], strict=strict)}

    result = {"pool_relevant": len(relevant), "configurations": {}, "added_by": {}}
    for config, names in configurations.items():
        articles = merge(responses, names)
        found = [normalize_url(a["url"]) for a in articles if normalize_url(a["url"]) in relevant]
        cost = budget(responses, names)
        result["configurations"][config] = {
            **cost,
            "returned": len(articles),
            "relevant": len(found),
            "pool_recall": len(found) / len(relevant) if relevant else 0.0,
            "precision": len(found) / len(articles) if articles else 0.0,
            "relevant_per_request": len(found) / cost["requests"] if cost["requests"] else 0.0,
            "listings": sum(1 for a in articles if is_listing(a)),
        }
        if len(names) > 1:
            first = {normalize_url(a["url"]) for a in merge(responses, names[:1])}
            result["added_by"][config] = {
                name: sorted(normalize_url(a["url"]) for a in merge(responses, [name])
                             if normalize_url(a["url"]) in relevant and normalize_url(a["url"]) not in first)
                for name in names[1:]
            }
    return result


def listing_check_scores(articles: list[dict], labels: dict[str, str], signals=listing_signals) -> dict:
    """For each check: articles it rejects and relevant articles it wrongly rejects; plus overlap."""
    found = [(signals(a), is_relevant(labels[normalize_url(a["url"])])) for a in articles]

    def count(test):
        hits = [relevant for sig, relevant in found if test(sig)]
        return {"rejected": len(hits), "relevant_rejected": sum(hits)}

    return {
        "pattern": count(lambda s: s["pattern"]),
        "site": count(lambda s: s["site"]),
        "both": sum(1 for sig, _ in found if sig["pattern"] and sig["site"]),
        "either": count(lambda s: s["pattern"] or s["site"]),
    }


def _seen_before(article: dict, urls: set[str], titles: set[str]) -> bool:
    if normalize_url(article.get("url") or "") in urls:
        return True
    key = _title_key(article.get("title") or "")
    if key in titles:
        return True
    if len(key) >= NEAR_DUPLICATE_MIN_LENGTH:
        return any(len(old) >= NEAR_DUPLICATE_MIN_LENGTH and
                   SequenceMatcher(None, key, old).ratio() >= NEAR_DUPLICATE_RATIO for old in titles)
    return False


def unseen(articles: list[dict], seen_articles: list[dict]) -> list[dict]:
    """Articles that are not a copy of any article seen before (same rules as cleaning)."""
    urls = {normalize_url(a.get("url") or "") for a in seen_articles}
    titles = {_title_key(a.get("title") or "") for a in seen_articles}
    return [a for a in articles if not _seen_before(a, urls, titles)]
