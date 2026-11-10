"""
The summary evaluation run on development data (Tata Motors only).

run      - for each saved search and each k: clean and rank the articles with
           the frozen method, take the first k, summarise, and save the result
           with its fingerprint. A saved result with the same fingerprint is
           reused, so running again makes no new model calls.
judging  - one blind sheet: every combined point from every run, shuffled,
           without k or the search; a separate key file links each row back
           to its run. Article briefs are judged once per article.
coverage - per run: share of `direct` articles cited by a combined point,
           `not` articles cited, and share of events covered (events are
           grouped by hand from the labels, before any summary is read).

events   - before any summary is read: a sheet of every article labelled
           relevant under guide version 3 (direct or indirect), each once,
           shuffled, with an empty `event` column to group reports of the
           same development by hand. Event recall in the ranking evaluation
           needs an event for every relevant article, not only the ones a
           summary run picks.

Run:
    python -m src.summary_run events     (no model calls)
    python -m src.summary_run run        (up to 12 model calls, plus retries)
"""

import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from src.config import DATA_DIR, EVAL_DIR, RAW_DIR
from src.evaluate import DEV_LABELS, is_relevant, load_label_set
from src.preprocess import clean_articles, normalize_url
from src.query import load_aliases
from src.summarizer import PROMPT_VERSION, SummaryCache, build_prompt, fingerprint, summarize, used_ids
from src.utils import unique_path

KS = (5, 10, 15)
CLEANING = {"remove_junk": True, "ranking": "mention", "drop_listings": "off"}
RUN_SEARCHES = [
    "20261014_181801_tata-motors.json",
    "20261014_182548_tata-motors.json",
    "20261023_101707_tata-motors.json",
    "20261023_161430_tata-motors.json",     # the plain "Tata Motors" search of the retrieval check
]
SUMMARY_DIR = EVAL_DIR / "summaries"        # kept with the evaluation files, as evidence
SEED = 42
JUDGING_COLUMNS = ["row", "item", "kind", "text", "evidence", "judgement", "contradicted", "note"]
KINDS = {"key_points": "company point", "industry_context": "industry point"}
EVENT_COLUMNS = ["row", "event", "title", "source", "published", "description", "content_first_200_chars", "url",
                 "notes"]                    # notes: the reason for a grouping decision that is not obvious
EVENTS_FILE = "events_dev_tata_motors.csv"
EVENT_LABEL_FILE = "v3_dev_tata_motors.csv"     # guide version 3 labels


class SummaryRunError(Exception):
    pass


# ---------- the run ----------

def select_articles(raw_path, query: str, aliases: dict | None, k: int) -> list[dict]:
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))["articles"]
    articles, _ = clean_articles(raw, max_articles=k, query=query, aliases=aliases, **CLEANING)
    return articles


def run_summaries(searches, ks, llm, cache: SummaryCache, query: str, aliases: dict | None,
                  raw_dir: Path = RAW_DIR, prompt_version: str = PROMPT_VERSION) -> list[dict]:
    runs = []
    for search in searches:
        for k in ks:
            articles = select_articles(Path(raw_dir) / search, query, aliases, k)
            key = fingerprint(articles, model=llm.model_name, prompt_version=prompt_version, k=k)
            result = cache.load(key)
            if result is None:
                result = summarize(articles, llm)
                if result["status"] in ("ok", "partial"):
                    cache.save(key, result, model=llm.model_name, prompt_version=prompt_version, k=k)
            runs.append({"search": search, "k": k, "fingerprint": key, "model": llm.model_name,
                         "prompt_version": prompt_version, "articles": articles, "result": result,
                         "prompt_chars": len(build_prompt(articles))})
    return runs


# ---------- the blind judging sheet ----------

def _evidence(articles: list[dict], ids) -> str:
    by_id = {article["id"]: article for article in articles}
    return "\n".join(
        f"[{i}] {by_id[i]['title']} — {by_id[i]['description']} {by_id[i].get('content') or ''}".strip()
        for i in ids if i in by_id
    )


def judging_rows(runs: list[dict], seed: int = SEED) -> tuple[list[dict], dict]:
    items = []
    for run in runs:
        summary = run["result"].get("summary")
        if not summary:
            continue
        place = {"search": run["search"], "k": run["k"], "fingerprint": run["fingerprint"]}
        for section, kind in KINDS.items():
            for index, point in enumerate(summary[section]):
                items.append(({"kind": kind, "text": point["text"],
                               "evidence": _evidence(run["articles"], point["citations"])},
                              {**place, "section": section, "index": index}))

    # briefs: once per article, taken from the largest k that has a shown brief for it
    briefs_done = set()
    for run in sorted(runs, key=lambda r: -r["k"]):
        summary = run["result"].get("summary")
        if not summary:
            continue
        by_id = {article["id"]: article for article in run["articles"]}
        for brief in summary["article_briefs"]:
            article = by_id.get(brief["id"])
            if article is None or brief.get("hidden"):
                continue
            url = normalize_url(article["url"])
            if url in briefs_done:
                continue
            briefs_done.add(url)
            items.append(({"kind": "article brief", "text": brief["brief"],
                           "evidence": _evidence(run["articles"], [brief["id"]])},
                          {"search": run["search"], "k": run["k"], "fingerprint": run["fingerprint"],
                           "section": "article_briefs", "article_url": article["url"]}))

    random.Random(seed).shuffle(items)
    rows, key = [], {}
    for number, (row, place) in enumerate(items, start=1):
        item = f"J{number:03d}"              # numbered after shuffling, so the code gives no hint
        rows.append({"row": number, "item": item, **row, "judgement": "", "contradicted": "", "note": ""})
        key[item] = place
    return [{column: row[column] for column in JUDGING_COLUMNS} for row in rows], key


def write_judging_files(rows: list[dict], key: dict, folder: Path, stamp: str) -> tuple[Path, Path]:
    folder = Path(folder)
    sheet = unique_path(folder, f"summary_judging_{stamp}", suffix=".csv")
    with open(sheet, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=JUDGING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    key_file = unique_path(folder, f"summary_judging_key_{stamp}", suffix=".json")
    key_file.write_text(json.dumps(key, indent=2), encoding="utf-8")
    return sheet, key_file


# ---------- coverage ----------

def citation_coverage(articles: list[dict], summary: dict | None, labels: dict[str, str]) -> dict:
    labels = {normalize_url(url): label for url, label in labels.items()}
    cited = used_ids(summary) if summary else set()
    label_of = {article["id"]: labels.get(normalize_url(article["url"])) for article in articles}
    direct = [i for i, label in label_of.items() if label == "direct"]
    direct_cited = [i for i in direct if i in cited]
    return {
        "direct_given": len(direct),
        "direct_cited": len(direct_cited),
        "direct_coverage": len(direct_cited) / len(direct) if direct else 0.0,
        "not_cited": sum(1 for i in cited if label_of.get(i) == "not"),
    }


def event_coverage(articles: list[dict], summary: dict | None, events: dict[str, str]) -> dict:
    events = {normalize_url(url): event for url, event in events.items()}
    cited = used_ids(summary) if summary else set()
    given, covered = set(), set()
    for article in articles:
        event = events.get(normalize_url(article["url"]))
        if event:
            given.add(event)
            if article["id"] in cited:
                covered.add(event)
    return {"events_given": len(given), "events_covered": len(covered),
            "event_coverage": len(covered) / len(given) if given else 0.0}


# ---------- the event sheet ----------

def event_sheet_rows(searches, labels: dict[str, str], query: str, aliases: dict | None,
                     raw_dir: Path = RAW_DIR, k: int = max(KS), seed: int = SEED) -> list[dict]:
    """Relevant articles in the first k of each search, each once, shuffled, with empty event and notes columns."""
    labels = {normalize_url(url): label for url, label in labels.items()}
    found = {}
    for search in searches:
        for article in select_articles(Path(raw_dir) / search, query, aliases, k):
            key = normalize_url(article["url"])
            if key not in labels:
                raise SummaryRunError(f"No label for {article['url']} ({search}).")
            if is_relevant(labels[key]) and key not in found:
                found[key] = article
    articles = list(found.values())
    random.Random(seed).shuffle(articles)
    return [{"row": number, "event": "", "title": a["title"], "source": a["source"],
             "published": a["published_at"], "description": a["description"],
             "content_first_200_chars": (a.get("content") or "")[:200], "url": a["url"], "notes": ""}
            for number, a in enumerate(articles, start=1)]


def relevant_event_rows(articles: list[dict], labels: dict[str, str], seed: int = SEED) -> list[dict]:
    """Every relevant article (raw NewsAPI form), each once, shuffled, with empty event and notes columns."""
    labels = {normalize_url(url): label for url, label in labels.items()}
    found = {}
    for article in articles:
        key = normalize_url(article.get("url") or "")
        if key not in labels:
            raise SummaryRunError(f"No label for {article.get('url')}.")
        if is_relevant(labels[key]) and key not in found:
            found[key] = article
    chosen = [found[key] for key in sorted(found)]          # a fixed start, so the shuffle repeats
    random.Random(seed).shuffle(chosen)
    return [{"row": number, "event": "", "title": a.get("title") or "",
             "source": (a.get("source") or {}).get("name") or "", "published": a.get("publishedAt") or "",
             "description": a.get("description") or "", "content_first_200_chars": (a.get("content") or "")[:200],
             "url": normalize_url(a.get("url") or ""), "notes": ""}
            for number, a in enumerate(chosen, start=1)]


def load_events(path) -> dict[str, str]:
    """{tidy link: event name}. Every row needs an event; names are compared in lower case."""
    events = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            event = (row.get("event") or "").strip().lower()
            if not event:
                raise SummaryRunError(f"Row {row.get('row')}: the event is empty.")
            events[normalize_url(row["url"])] = event
    return events


# ---------- running it ----------

def _events_command() -> None:
    from src.fulltext_eval import _inputs
    path = EVAL_DIR / EVENTS_FILE
    if path.exists():
        raise SummaryRunError(f"{path.name} already exists; it is not overwritten.")
    articles, labels, _ = _inputs()                          # the labelled dev articles, guide version 3
    rows = relevant_event_rows(articles, labels)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} relevant articles to group into events  ->  {path}")


def _run_command() -> None:
    from src.llm import make_llm
    llm = make_llm()
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    runs = run_summaries(RUN_SEARCHES, KS, llm, SummaryCache(SUMMARY_DIR), "Tata Motors", aliases)
    stamp = datetime.now().strftime("%Y%m%d")
    runs_file = unique_path(EVAL_DIR, f"summary_runs_{stamp}")
    runs_file.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
    rows, key = judging_rows(runs)
    sheet, key_file = write_judging_files(rows, key, EVAL_DIR, stamp)
    print(f"{'Search':<38}{'k':>3}{'articles':>9}{'status':>13}{'attempts':>9}{'origin':>8}{'removed':>8}")
    for run in runs:
        result = run["result"]
        print(f"{run['search']:<38}{run['k']:>3}{len(run['articles']):>9}{result['status']:>13}"
              f"{result['attempts']:>9}{result['origin']:>8}{len(result['removed']):>8}")
    print(f"\nRuns saved:     {runs_file}\nJudging sheet:  {sheet}  ({len(rows)} rows)\n"
          f"Judging key:    {key_file}  (do not open before judging is finished)")


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        _run_command()
    elif sys.argv[1:] == ["events"]:
        _events_command()
    else:
        print(__doc__)
