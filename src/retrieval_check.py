"""
Does adding the company's other names to the NewsAPI search find relevant
articles that the plain search misses?

Two searches are made one after the other with the same settings:
    without_aliases   "Tata Motors"
    with_aliases      "Tata Motors" OR TMPV OR ... OR JLR OR (Jaguar AND (car OR ...))

Articles not labelled before go into a new CSV sheet for blind labelling:
the sheet does not say which search found an article, and the rows are
shuffled, so the label cannot be affected by that knowledge.

After labelling, each search is scored against the pool of relevant
articles found by either search.

Run:
    python -m src.retrieval_check fetch            (2 NewsAPI requests)
    python -m src.retrieval_check score <pair file>
"""

import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path, PureWindowsPath

from src.config import DATA_DIR, EVAL_DIR, RAW_DIR
from src.evaluate import DEV_LABELS_V2, LABEL_VERSION, is_relevant, load_label_set
from src.news_fetcher import fetch_news, save_raw
from src.preprocess import clean_text, normalize_url
from src.query import load_aliases
from src.utils import unique_path

PAGE_SIZE = 30          # the same number of articles the app asks for
DAYS_BACK = 7           # the app's default search window
SORT_BY = "relevancy"   # the app's default order
SEED = 42               # fixed seed: the shuffled order is the same every run
SEARCHES = ("without_aliases", "with_aliases")
SHEET_COLUMNS = ["row", "label", "unsure", "note", "title", "source", "published",
                 "description", "content_first_200_chars", "url"]


class RetrievalCheckError(Exception):
    pass


# ---------- 1. the two searches ----------

def run_pair(query: str, aliases: dict, fetch=fetch_news) -> dict:
    """Make both searches with identical settings. `fetch` can be replaced in tests."""
    settings = {"page_size": PAGE_SIZE, "days_back": DAYS_BACK, "sort_by": SORT_BY,
                "exact_phrase": True, "title_only": False, "save": False}
    without_aliases = fetch(query, aliases=None, **settings)
    with_aliases = fetch(query, aliases=aliases, **settings)

    if without_aliases["request"]["from_date"] != with_aliases["request"]["from_date"]:
        raise RetrievalCheckError("The two searches cover different dates (the day changed between them). "
                                  "Run the check again.")
    if not with_aliases["request"].get("aliases_added"):
        raise RetrievalCheckError(f"No other names are known for {query!r}, so both searches are the same.")
    return {"without_aliases": without_aliases, "with_aliases": with_aliases}


# ---------- 2. the blind labelling sheet ----------

def _usable(article: dict) -> bool:
    title, url = clean_text(article.get("title")), article.get("url")
    return bool(title) and bool(url) and title != "[Removed]"


def labelling_rows(pair: dict, already_labelled: set[str], seed: int = SEED) -> list[dict]:
    """New articles from both searches, each once, shuffled, with empty label columns."""
    found = {}
    for name in SEARCHES:
        for article in pair[name]["articles"]:
            if _usable(article):
                key = normalize_url(article["url"])
                if key not in already_labelled and key not in found:
                    found[key] = article

    articles = list(found.values())
    random.Random(seed).shuffle(articles)
    return [
        {
            "row": number,
            "label": "",
            "unsure": "",
            "note": "",
            "title": clean_text(article.get("title")),
            "source": (article.get("source") or {}).get("name") or "",
            "published": article.get("publishedAt") or "",
            "description": clean_text(article.get("description")),
            "content_first_200_chars": clean_text(article.get("content"))[:200],
            "url": article["url"],
        }
        for number, article in enumerate(articles, start=1)
    ]


def write_sheet(rows: list[dict], folder: Path, stem: str) -> Path:
    path = unique_path(Path(folder), stem, suffix=".csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def labelled_urls(folder: Path = EVAL_DIR) -> set[str]:
    """URLs already in a development sheet (labelled or not)."""
    urls = set()
    for path in sorted(Path(folder).glob("dev_*.csv")):
        with open(path, newline="", encoding="utf-8") as handle:
            urls |= {normalize_url(row["url"]) for row in csv.DictReader(handle) if row.get("url")}
    return urls


# ---------- 3. the scores after labelling ----------

def retrieval_scores(pair: dict, labels: dict[str, str], strict: bool = False) -> dict:
    found = {}
    for name in SEARCHES:
        urls = {normalize_url(a["url"]) for a in pair[name]["articles"] if _usable(a)}
        missing = urls - set(labels)
        if missing:
            raise RetrievalCheckError(f"{len(missing)} article(s) from the {name} search have no label.")
        found[name] = urls

    relevant = {name: {u for u in urls if is_relevant(labels[u], strict=strict)} for name, urls in found.items()}
    pool = relevant["without_aliases"] | relevant["with_aliases"]
    result = {"pool_relevant": len(pool)}
    for name, other in (SEARCHES, SEARCHES[::-1]):
        result[name] = {
            "returned": len(found[name]),
            "relevant_found": len(relevant[name]),
            "pool_recall": len(relevant[name]) / len(pool) if pool else 0.0,
            "candidate_precision": len(relevant[name]) / len(found[name]) if found[name] else 0.0,
            "only_here": sorted(relevant[name] - relevant[other]),
        }
    return result


# ---------- files ----------

def score_label_files(folder: Path = EVAL_DIR) -> list[Path]:
    """Version 2 labels: the relabelled development file and the retrieval sheets (never the version 1 files)."""
    folder = Path(folder)
    main = [folder / name for name in DEV_LABELS_V2]
    missing = [path.name for path in main if not path.exists()]
    if missing:
        raise RetrievalCheckError(f"Missing version 2 label file(s): {', '.join(missing)}. "
                                  "Run python -m src.label_review finalize first.")
    return main + sorted(folder.glob("dev_*_retrieval_*.csv"))


def _file_name(stored: str) -> str:
    # PureWindowsPath understands both \ and / as separators
    return PureWindowsPath(stored).name


def raw_path(stored: str, raw_dir: Path = RAW_DIR) -> Path:
    """The saved raw file, found by its name in the raw folder (works on any computer)."""
    return Path(raw_dir) / _file_name(stored)


def pair_record(query: str, raw_paths: dict[str, str]) -> dict:
    return {"query": query, "raw_files": {name: _file_name(str(path)) for name, path in raw_paths.items()}}


# ---------- running it ----------

def _fetch_command(query: str = "Tata Motors") -> None:
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    pair = run_pair(query, aliases)
    raw_paths = {name: str(save_raw(pair[name])) for name in SEARCHES}
    stamp = datetime.now().strftime("%Y%m%d")
    record = unique_path(EVAL_DIR, f"retrieval_pair_{stamp}")
    record.write_text(json.dumps(pair_record(query, raw_paths), indent=2), encoding="utf-8")
    rows = labelling_rows(pair, labelled_urls())
    sheet = write_sheet(rows, EVAL_DIR, f"dev_tata_motors_retrieval_{stamp}")
    for name in SEARCHES:
        print(f"{name:<16} returned {len(pair[name]['articles'])} articles  "
              f"search: {pair[name]['request']['search_query']}")
    print(f"\nPair record:  {record}\nNew articles to label: {len(rows)}  ->  {sheet}")


def _score_command(record_path: str) -> None:
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    pair = {name: json.loads(raw_path(path).read_text(encoding="utf-8")) for name, path in record["raw_files"].items()}
    files = score_label_files()
    labels = load_label_set(files)
    print(f"Labels: guide version {LABEL_VERSION} ({', '.join(path.name for path in files)})")
    titles = {normalize_url(a["url"]): clean_text(a.get("title"))
              for name in SEARCHES for a in pair[name]["articles"] if a.get("url")}
    for strict, title in ((False, "BROAD (direct + indirect)"), (True, "STRICT (direct only)")):
        result = retrieval_scores(pair, labels, strict=strict)
        print(f"\n=== {title}: {result['pool_relevant']} relevant articles found by either search ===\n")
        print(f"{'Search':<18}{'returned':>9}{'relevant':>10}{'pool recall':>13}{'precision':>11}{'only here':>11}")
        for name in SEARCHES:
            r = result[name]
            print(f"{name:<18}{r['returned']:>9}{r['relevant_found']:>10}{r['pool_recall']:>13.3f}"
                  f"{r['candidate_precision']:>11.3f}{len(r['only_here']):>11}")
        for name in SEARCHES:
            for url in result[name]["only_here"]:
                print(f"   only {name}: {titles.get(url, url)}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "fetch":
        _fetch_command()
    elif len(sys.argv) == 3 and sys.argv[1] == "score":
        _score_command(sys.argv[2])
    else:
        print(__doc__)
