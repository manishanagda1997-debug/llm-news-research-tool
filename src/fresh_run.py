"""
The fresh evaluation run: searches A, B and C on a new day, labelled blind,
scored once with the frozen method.

    A  the company name only                ("Tata Motors")
    B  the subsidiaries' names              ("Jaguar Land Rover" OR JLR OR ...)
    C  the company and its own names        ("Tata Motors" OR TMPV OR ... OR TMCV)

fetch  - makes the three searches, saves each raw response and a record.
         The record lists the raw files that existed before the run: only
         articles in those files count as "seen" later.
         Writes a blind sheet of every article without a label yet: shuffled,
         with no sign of which search found it or what the filter does.
score  - after labelling: each configuration (A, A+B, C, C+B) on the whole
         pool and on unseen articles only, the two listing checks, and the
         relevance filter with each listing mode.

Run:
    python -m src.fresh_run fetch                       (3 NewsAPI requests)
    python -m src.fresh_run score <record file> <sheet file>
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path, PureWindowsPath

from src.config import DATA_DIR, EVAL_DIR, RAW_DIR
from src.evaluate import confusion, is_relevant, load_label_set, scores
from src.listings import LISTING_MODES
from src.multi_search import SEARCH_NAMES, merge, run_searches
from src.news_fetcher import save_raw
from src.preprocess import clean_articles, clean_text, normalize_url
from src.query import load_aliases
from src.retrieval_check import SEED, _usable, labelled_urls, score_label_files, write_sheet
from src.search_report import CONFIGURATIONS, configuration_scores, listing_check_scores, unseen
from src.utils import unique_path


class FreshRunError(Exception):
    pass


# ---------- what counts as seen ----------

def existing_raw_files(raw_dir: Path = RAW_DIR) -> list[str]:
    return sorted(path.name for path in Path(raw_dir).glob("*.json"))


def seen_articles(files: list[str], raw_dir: Path = RAW_DIR) -> list[dict]:
    articles = []
    for name in files:
        articles += json.loads((Path(raw_dir) / name).read_text(encoding="utf-8")).get("articles", [])
    return articles


def fresh_record(query: str, raw_paths: dict, seen_files: list[str]) -> dict:
    return {
        "query": query,
        "raw_files": {name: PureWindowsPath(str(path)).name if path else None for name, path in raw_paths.items()},
        "seen_files": list(seen_files),
    }


# ---------- the blind sheet ----------

def sheet_rows(responses: dict, labelled: set[str], seed: int = SEED) -> list[dict]:
    """Every article without a label yet, once, shuffled, with empty label columns."""
    labelled = {normalize_url(url) for url in labelled}
    present = [name for name in SEARCH_NAMES if responses.get(name)]
    articles = [a for a in merge(responses, present)
                if _usable(a) and normalize_url(a["url"]) not in labelled]
    random.Random(seed).shuffle(articles)
    return [
        {"row": number, "label": "", "unsure": "", "note": "",
         "title": clean_text(a.get("title")),
         "source": (a.get("source") or {}).get("name") or "",
         "published": a.get("publishedAt") or "",
         "description": clean_text(a.get("description")),
         "content_first_200_chars": clean_text(a.get("content"))[:200],
         "url": a["url"]}
        for number, a in enumerate(articles, start=1)
    ]


def write_fresh_sheet(rows: list[dict], folder: Path, stamp: str, company_slug: str = "tata_motors") -> Path:
    return write_sheet(rows, Path(folder), f"dev_{company_slug}_fresh_{stamp}")


# ---------- scoring ----------

def _usable_only(responses: dict) -> dict:
    return {name: ({**response, "articles": [a for a in response["articles"] if _usable(a)]} if response else None)
            for name, response in responses.items()}


def new_only(responses: dict, seen: list[dict]) -> dict:
    """The same searches with every article seen before removed. The budget stays the same."""
    return {name: ({**response, "articles": unseen(response["articles"], seen)} if response else None)
            for name, response in responses.items()}


def filter_result(articles: list[dict], labels: dict[str, str], query: str, aliases: dict | None,
                  drop_listings: str, strict: bool = False) -> dict:
    """The relevance filter on one merged pool. A dropped copy counts as found when its copy is kept."""
    kept, stats = clean_articles(articles, max_articles=max(len(articles), 1), query=query,
                                 aliases=aliases, remove_junk=True, drop_listings=drop_listings)
    covered = {normalize_url(a["url"]) for a in kept}
    for entry in stats["duplicate_log"]:
        if normalize_url(entry["kept"]["url"]) in covered:
            covered.add(normalize_url(entry["removed"]["url"]))
    relevant = {normalize_url(a["url"]): is_relevant(labels[normalize_url(a["url"])], strict=strict)
                for a in articles}
    counts = confusion(relevant, covered)
    return {**counts, **scores(counts), "kept": len(kept), "sale_listing": stats["sale_listing"]}


def _report(responses: dict, labels: dict, query: str, aliases: dict | None,
            configurations: dict, strict: bool) -> dict:
    result = configuration_scores(responses, labels, strict=strict, configurations=configurations)
    every = merge(responses, [name for name in responses if responses[name]])
    result["listing"] = listing_check_scores(every, labels)
    result["filter"] = {
        config: {mode: filter_result(merge(responses, names), labels, query, aliases, mode, strict)
                 for mode in LISTING_MODES}
        for config, names in configurations.items()
    }
    return result


def score(responses: dict, labels: dict[str, str], seen: list[dict], query: str, aliases: dict | None,
          configurations: dict = CONFIGURATIONS, strict: bool = False) -> dict:
    labels = {normalize_url(url): label for url, label in labels.items()}
    whole = _usable_only(responses)
    fresh = new_only(whole, seen)
    present = [name for name in whole if whole[name]]
    missing = [a["url"] for a in merge(whole, present) if normalize_url(a["url"]) not in labels]
    if missing:
        raise FreshRunError(f"{len(missing)} article(s) have no label, for example {missing[0]}.")
    return {
        "counts": {"whole": len(merge(whole, present)), "new_only": len(merge(fresh, present))},
        "whole": _report(whole, labels, query, aliases, configurations, strict),
        "new_only": _report(fresh, labels, query, aliases, configurations, strict),
    }


# ---------- running it ----------

def _fetch_command(query: str = "Tata Motors") -> None:
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    seen_files = existing_raw_files()                    # frozen before anything new is saved
    responses = run_searches(query, aliases)
    raw_paths = {name: (save_raw(response) if response else None) for name, response in responses.items()}
    stamp = datetime.now().strftime("%Y%m%d")
    record_path = unique_path(EVAL_DIR, f"fresh_record_{stamp}")
    record_path.write_text(json.dumps(fresh_record(query, raw_paths, seen_files), indent=2), encoding="utf-8")
    rows = sheet_rows(responses, labelled_urls())
    sheet = write_fresh_sheet(rows, EVAL_DIR, stamp)
    for name, response in responses.items():
        if response:
            print(f"Search {name}: {len(response['articles']):>3} articles  "
                  f"(NewsAPI total {response['totalResults']})  {response['request']['search_query']}")
        else:
            print(f"Search {name}: skipped (nothing to search for)")
    print(f"\nRecord: {record_path}\nArticles to label: {len(rows)}  ->  {sheet}")


def _print_report(title: str, report: dict, count: int) -> None:
    print(f"\n=== {title}: {count} articles, {report['pool_relevant']} relevant ===\n")
    print(f"{'Config':<7}{'requests':>9}{'slots':>7}{'returned':>9}{'relevant':>9}{'pool recall':>12}"
          f"{'precision':>10}{'rel/request':>12}{'listings':>9}")
    for config, row in report["configurations"].items():
        print(f"{config:<7}{row['requests']:>9}{row['slots']:>7}{row['returned']:>9}{row['relevant']:>9}"
              f"{row['pool_recall']:>12.3f}{row['precision']:>10.3f}{row['relevant_per_request']:>12.2f}"
              f"{row['listings']:>9}")
    listing = report["listing"]
    print(f"\nListing checks: pattern rejects {listing['pattern']['rejected']} "
          f"({listing['pattern']['relevant_rejected']} relevant), site rejects {listing['site']['rejected']} "
          f"({listing['site']['relevant_rejected']} relevant), both {listing['both']}, "
          f"either {listing['either']['rejected']} ({listing['either']['relevant_rejected']} relevant)")
    print(f"\nRelevance filter (with each listing mode):")
    print(f"{'Config':<7}{'listings':<10}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}{'Precision':>11}{'Recall':>8}{'F1':>7}")
    for config, modes in report["filter"].items():
        for mode, row in modes.items():
            print(f"{config:<7}{mode:<10}{row['tp']:>4}{row['fp']:>4}{row['fn']:>4}{row['tn']:>4}"
                  f"{row['precision']:>11.3f}{row['recall']:>8.3f}{row['f1']:>7.3f}")


def _score_command(record_path: str, sheet_path: str) -> None:
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    responses = {name: (json.loads((RAW_DIR / file).read_text(encoding="utf-8")) if file else None)
                 for name, file in record["raw_files"].items()}
    files = score_label_files() + [Path(sheet_path)]
    labels = load_label_set(files)
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    seen = seen_articles(record["seen_files"])
    print(f"Labels: {', '.join(path.name for path in files)}")
    print(f"Seen before the run: {len(record['seen_files'])} raw files")
    for strict, name in ((False, "BROAD"), (True, "STRICT")):
        result = score(responses, labels, seen, record["query"], aliases, strict=strict)
        _print_report(f"{name}, whole pool", result["whole"], result["counts"]["whole"])
        _print_report(f"{name}, new articles only", result["new_only"], result["counts"]["new_only"])


if __name__ == "__main__":
    command = sys.argv[1:]
    if command == ["fetch"]:
        _fetch_command()
    elif len(command) == 3 and command[0] == "score":
        _score_command(command[1], command[2])
    else:
        print(__doc__)
