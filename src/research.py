"""
The analyst pipeline (Stage 5): a company in, ranked news and a summary out. The Streamlit app (Stage 6) only
shows what these steps return; every rule lives here and is tested without a screen.

Steps, in order:
    search       searches A (the company name) and B (its subsidiaries' names from the alias file) for the last
                 `days` days, as in the fresh evaluation. The result shows the window, the count per search and
                 the latest article date. A NewsAPI failure is an error, never an empty list shown as a success.
    provisional  the O1M5+S1 order from the NewsAPI data alone, at once. Stage 1 keeps or drops articles using the
                 full text, so this list is labelled provisional: it is not the tested result.
    enrich       the full text of each page (robots.txt respected, saved in data/fulltext/), within one time limit
                 for the whole step: no new page starts after ENRICH_SECONDS. A page not fetched in time keeps
                 its short text, as any page that cannot be fetched, and is listed by title.
    final        Stage 1 with the full text, then O1M5+S1: the default, labelled as such. O1M5 is O1 with one change:
                 an article whose title does not name the company but whose text names it at least 5 times
                 joins the title group (src/ranking_variants.py, chosen in the second development cycle).
    summary      the chosen summary design on the selected articles, on request. A saved summary is reused only
                 for the same company scope, the same articles in the same order with the same evidence, the
                 same model and settings, and the same design, configuration and prompt versions. The status is
                 always one of: ready, ready with removed points, model unavailable (with the provider's wait
                 when it gives one), failed checks.
    review       optional and off by default: the relevance model's label and reason beside each article. The
                 order does not change. An article without a model answer keeps the evaluated fallback.
    apply        on request: the O3 order (title protection), labelled experimental, with every article kept;
                 back_to_default returns to O1M5+S1 at once.

Messages never show keys, organisation ids, raw provider text or file paths.

Run (NewsAPI and the publishers' pages only, no model):
    python -m src.research "Tata Motors"
"""

import hashlib
import json
import re
import time

from src import ranking as rk

SEARCHES = ("A", "B")                                 # the configuration of the fresh evaluation
PARTS = {"A": None, "B": "subsidiaries"}
PAGE_SIZE = 30
SORT_BY = "relevancy"
DAYS = 7
K = 5                                                  # the size tested in Stage 4; k = 10 waits for its own test
ENRICH_SECONDS = 90
PROVISIONAL_LABEL = "Provisional: full text still loading. This is not the tested ranking."
DEFAULT_ORDER = "O1M5"                         # src/ranking_variants.py: O1, with the 5-mention rule
DEFAULT_ORDERING = f"{DEFAULT_ORDER}+S1"
DEFAULT_LABEL = f"Default ranking (Stage 1 with full text, then {DEFAULT_ORDERING})"
EXPERIMENTAL_LABEL = "Experimental: model-reviewed order (O3). Not the tested default."


class ResearchError(Exception):
    pass


# ---------- safe messages ----------

def safe_text(text: str) -> str:
    """A message for the screen: keys, organisation ids and file paths hidden."""
    from src.llm import _hide_keys
    text = _hide_keys(str(text))
    text = re.sub(r"org_[A-Za-z0-9]+", "org_[hidden]", text)
    text = re.sub(r"[A-Za-z]:\\[^\s'\"]+|/(?:home|Users|sessions|tmp|mnt)/[^\s'\"]+", "[path hidden]", text)
    return text[:500]


# ---------- search ----------

def search(company: str, aliases: dict | None, days: int = DAYS, fetch=None) -> dict:
    from src.multi_search import merge
    from src.news_fetcher import NewsFetchError, fetch_news
    from src.query import build_search_plan
    fetch = fetch or fetch_news
    settings = {"page_size": PAGE_SIZE, "days_back": days, "sort_by": SORT_BY, "exact_phrase": True,
                "title_only": False, "save": False}
    responses, notes = {}, []
    try:
        for name in SEARCHES:
            part = PARTS[name]
            if part is not None and not build_search_plan(company, aliases=aliases, part=part)["search_query"]:
                responses[name] = None
                notes.append("No subsidiary names are known for this company, so only the company name is searched.")
                continue
            if part is None:                       # search A: the company name alone, as fetch_news's default
                responses[name] = fetch(company, aliases=None, **settings)
            else:
                responses[name] = fetch(company, aliases=aliases, part=part, **settings)
    except NewsFetchError as error:
        return {"status": "failed", "company": company, "error": safe_text(error), "articles": [],
                "counts": {}, "window": None, "latest_published": None, "notes": notes}
    articles = merge(responses, SEARCHES)
    dates = [a.get("publishedAt") for a in articles if a.get("publishedAt")]
    first = next(r for r in responses.values() if r)
    return {"status": "ok", "company": company, "articles": articles,
            "counts": {name: (len(r["articles"]) if r else None) for name, r in responses.items()},
            "window": {"from": first["request"].get("from_date"), "days": days},
            "latest_published": max(dates) if dates else None, "notes": notes}


def _need(found: dict) -> None:
    if found.get("status") != "ok":
        raise ResearchError(found.get("error") or "The search did not return articles.")


# ---------- the lists ----------

def default_order(candidates: list[dict], company: str, aliases: dict | None) -> list[dict]:
    """The default order: O1M5 on the candidates Stage 1 keeps (in retrieval order), with Stage 1's own matching
    rule for the company, so the names counted are the ones Stage 1 matches."""
    from src.query import parse_query
    from src.ranking_variants import order_variant
    return order_variant(candidates, DEFAULT_ORDER, parse_query(company, aliases=aliases))


def _ranked(found: dict, texts: dict, company: str, aliases: dict | None, k: int) -> dict:
    from src.ranking_eval import stage1_candidates
    candidates = stage1_candidates(found["articles"], texts, company, aliases)
    ordered = default_order(candidates, company, aliases)
    selected = rk.select(ordered, k, "S1")["selected"]
    return {"company": company, "k": k, "ordering": DEFAULT_ORDERING, "ranked": ordered, "selected": selected,
            "additional": [c for c in ordered if c not in selected]}


def provisional(found: dict, company: str, aliases: dict | None, k: int = K) -> dict:
    _need(found)
    return {**_ranked(found, {}, company, aliases, k), "state": "provisional", "label": PROVISIONAL_LABEL}


def enrich(found: dict, pages, budget: float = ENRICH_SECONDS, clock=time.monotonic) -> dict:
    """Full text for each article; no new page starts after `budget` seconds."""
    _need(found)
    started, texts, status, late = clock(), {}, {}, []
    for article in found["articles"]:
        url = article["url"]
        if clock() - started >= budget:
            status[url] = "not_fetched_in_time"
            late.append(url)
            continue
        result = pages.fetch(url)
        status[url] = result.get("status")
        if result.get("status") == "ok" and result.get("text"):
            texts[url] = result["text"]
    titles = {a["url"]: a.get("title") or "" for a in found["articles"]}
    return {"texts": texts, "status": status, "not_in_time": late,
            "not_in_time_titles": [titles[url] for url in late], "seconds": round(clock() - started, 1)}


def final(found: dict, enriched: dict, company: str, aliases: dict | None, k: int = K) -> dict:
    _need(found)
    return {**_ranked(found, enriched["texts"], company, aliases, k), "state": "final", "label": DEFAULT_LABEL,
            "not_in_time_titles": enriched.get("not_in_time_titles", [])}


def list_changes(first: dict, last: dict) -> dict:
    """What the full text changes: articles added to or removed from the list."""
    before = {c["key"]: c["title"] for c in first["ranked"]}
    after = {c["key"]: c["title"] for c in last["ranked"]}
    return {"added": [after[k] for k in after if k not in before],
            "removed": [before[k] for k in before if k not in after]}


# ---------- the summary ----------

def summary_key(articles: list[dict], llm, design: str, company: str, aliases: dict | None) -> str:
    """Everything a summary depends on: the scope, the articles in order with their evidence, the model and its
    settings, the design, and the configuration and prompt versions."""
    from src.relevance_check import company_scope
    from src.summary_v2 import CORRECTION, SUMMARY_VERSION
    from src.summarizer import PROMPT_VERSION, fingerprint
    scope = hashlib.sha256(f"{company}\n{company_scope(company, aliases)}".encode("utf-8")).hexdigest()
    settings = f"{SUMMARY_VERSION}-{CORRECTION}-{design}-p{PROMPT_VERSION}-{getattr(llm, 'cache_tag', '')}-{scope}"
    return fingerprint(articles, getattr(llm, "model_name", "unknown"), settings, len(articles))


def _headlines(selected: list[dict]) -> list[dict]:
    return [{"title": c["title"], "source": c.get("source", ""), "url": c.get("url", "")} for c in selected]


def summarize_selection(result: dict, company: str, aliases: dict | None, llm, design: str, cache=None) -> dict:
    from src.summary_v2 import CORRECTION, Recorder, passage_articles, run_design, wait_seconds
    articles = passage_articles(result["selected"], aliases, company)
    key = summary_key(articles, llm, design, company, aliases)
    saved = cache.load(key) if cache else None
    if saved is not None:
        return _summary_status(saved, "saved", result["selected"])
    recorder = Recorder(llm)
    outcome = run_design(design, articles, recorder, CORRECTION)
    out = _summary_status(outcome, "fresh", result["selected"])
    if recorder.failures:
        out.update(summary_status="model_unavailable", failure=recorder.failures[-1],
                   wait_seconds=recorder.waits[-1]["wait_seconds"] if recorder.waits else None,
                   message=safe_text("The model is not available right now (" + recorder.failures[-1] + ")."))
    elif cache and outcome["status"] in ("ok", "partial"):
        cache.save(key, {k_: v for k_, v in outcome.items() if not k_.startswith("_")}, llm.model_name, design,
                   len(articles))
    return out


def _summary_status(outcome: dict, origin: str, selected: list[dict]) -> dict:
    status = {"ok": "ready", "partial": "ready_with_removed_points"}.get(outcome.get("status"), "failed_checks")
    return {"summary_status": status, "origin": origin, "summary": outcome.get("summary"),
            "removed": outcome.get("removed", []), "message": safe_text(outcome.get("message") or ""),
            "headlines": _headlines(selected), "wait_seconds": None}


# ---------- model review and apply updates ----------

def review(result: dict, company: str, aliases: dict | None, llm, cache=None, batch_size: int | None = None) -> dict:
    """The model's label and reason for each article; the order and the selection stay as they are."""
    from src.ranking_eval import query_items
    from src.relevance_check import BATCH_SIZE, classify, company_scope
    items = query_items(result["ranked"], company, aliases)
    answers = classify(items, llm, company, batch_size=batch_size or BATCH_SIZE, cache=cache,
                       scope=company_scope(company, aliases))
    ranked = [dict(c, check=answers[c["key"]]) for c in result["ranked"]]
    by_key = {c["key"]: c for c in ranked}
    answered = sum(1 for c in ranked if c["check"]["source"] == "model")
    return {**result, "ranked": ranked, "selected": [by_key[c["key"]] for c in result["selected"]],
            "additional": [by_key[c["key"]] for c in result["additional"]], "reviewed": True,
            "review_counts": {"answered": answered, "unanswered": len(ranked) - answered}}


def apply_updates(result: dict) -> dict:
    if not result.get("reviewed"):
        raise ResearchError("Run the model review before applying its order.")
    default_order = result.get("default_order") or [c["key"] for c in result["ranked"]]
    ordered = rk.order(result["ranked"], "O3")
    selected = rk.select(ordered, result["k"], "S1")["selected"]
    return {**result, "ranked": ordered, "selected": selected, "additional": [c for c in ordered if c not in selected],
            "ordering": "O3+S1", "label": EXPERIMENTAL_LABEL, "default_order": default_order}


def back_to_default(result: dict) -> dict:
    """The exact default order saved when the updates were applied (O1 alone would keep ties in the O3 order)."""
    by_key = {c["key"]: c for c in result["ranked"]}
    ordered = [by_key[key] for key in result.get("default_order") or by_key]
    selected = rk.select(ordered, result["k"], "S1")["selected"]
    return {**result, "ranked": ordered, "selected": selected, "additional": [c for c in ordered if c not in selected],
            "ordering": DEFAULT_ORDERING, "label": DEFAULT_LABEL}


def _show(result: dict) -> None:
    print(f"\n{result['label']}")
    for number, c in enumerate(result["ranked"], start=1):
        mark = "*" if c in result["selected"] else " "
        print(f" {mark}{number:>3}. [{c.get('mention_level') or '-':<11}] {c['title'][:80]}  ({c.get('source', '')})")


def main(company: str) -> None:
    from src.config import DATA_DIR
    from src.fulltext import FullTextFetcher
    from src.query import load_aliases
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    found = search(company, aliases)
    if found["status"] != "ok":
        print(f"[STOP] {found['error']}")
        return
    print(f"{company}: searched from {found['window']['from']} ({found['window']['days']} days); "
          f"articles per search {found['counts']}; latest article {found['latest_published']}")
    for note in found["notes"]:
        print(f"Note: {note}")
    first = provisional(found, company, aliases)
    _show(first)
    print(f"\nLoading full text (at most {ENRICH_SECONDS} s) ...")
    from src.safe_web import safe_http_get
    enriched = enrich(found, FullTextFetcher(http_get=safe_http_get))      # public addresses only
    last = final(found, enriched, company, aliases)
    _show(last)
    changes = list_changes(first, last)
    print(f"\nFull text added {len(changes['added'])} and removed {len(changes['removed'])} article(s); "
          f"{len(enriched['not_in_time'])} page(s) not fetched in time ({enriched['seconds']} s). * = selected for the summary")


if __name__ == "__main__":
    import sys
    main(" ".join(sys.argv[1:]) or "Tata Motors")
