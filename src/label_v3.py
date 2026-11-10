"""
Version 2 -> version 3 labels (labelling guide version 3: full text).

compare  - joins the two label sets by article, reports agreement, Cohen's
           kappa and the moves (for example not -> indirect), and writes a
           sheet of the changed articles, each to get a reason from the guide
finalize - one version 3 label file for every labelled development article:
              full text   the articles labelled on their full text
              short text  articles without full text keep their version 2
                          label, with the reason written in `basis`: page
                          blocked by robots.txt, a sale listing (never
                          fetched), or a copy of a fetched article (the
                          duplicate rule keeps one copy; a copy must have the
                          same label as the article it copies)
           every changed label needs a written reason, and an article without
           full text and without one of these reasons stops the run

Run:
    python -m src.label_v3 compare  <version 3 label sheet>
    python -m src.label_v3 finalize <version 3 label sheet> <differences sheet>
"""

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.config import EVAL_DIR
from src.evaluate import VALID_LABELS, load_label_set
from src.fulltext import FullTextFetcher, article_id
from src.label_review import cohen_kappa
from src.listings import listing_signals
from src.preprocess import NEAR_DUPLICATE_MIN_LENGTH, NEAR_DUPLICATE_RATIO, _title_key, normalize_url
from src.retrieval_check import score_label_files
from src.utils import unique_path

DIFFERENCE_COLUMNS = ["article_id", "title", "url", "label_v2", "label_v3", "reason"]
FINAL_COLUMNS = ["article_id", "title", "url", "label", "unsure", "note", "basis", "label_v2", "changed",
                 "reason", "guide_version"]
GUIDE_VERSION = "3"
BASIS_FULL = "full text"
BASIS_BLOCKED = "short text (page blocked by robots.txt)"
BASIS_LISTING = "short text (sale listing, not fetched)"
BASIS_COPY = "short text (copy of a fetched article)"


class LabelV3Error(Exception):
    pass


def _tidy(mapping: dict) -> dict:
    return {normalize_url(url): value for url, value in mapping.items()}


# ---------- compare ----------

def compare_versions(v2: dict[str, str], v3: dict[str, str]) -> dict:
    v2, v3 = _tidy(v2), _tidy(v3)
    missing = [url for url in v3 if url not in v2]
    if missing:
        raise LabelV3Error(f"{len(missing)} article(s) have no version 2 label, for example {missing[0]}.")
    urls = sorted(v3)
    changes = [(url, v2[url], v3[url]) for url in urls if v2[url] != v3[url]]
    return {
        "articles": len(urls),
        "agreement": (len(urls) - len(changes)) / len(urls) if urls else 0.0,
        "kappa": cohen_kappa([v2[u] for u in urls], [v3[u] for u in urls]) if urls else 0.0,
        "moves": dict(Counter((before, after) for _, before, after in changes)),
        "changes": changes,
    }


def difference_rows(v2: dict[str, str], v3: dict[str, str], titles: dict[str, str]) -> list[dict]:
    titles = _tidy(titles)
    return [{"article_id": article_id(url), "title": titles.get(url, ""), "url": url,
             "label_v2": before, "label_v3": after, "reason": ""}
            for url, before, after in compare_versions(v2, v3)["changes"]]


# ---------- finalize ----------

def find_copies(others: dict[str, str], fetched: dict[str, str]) -> dict[str, str]:
    """{article: fetched article it copies}, by the duplicate rules of cleaning (same link, same or nearly the same title)."""
    from difflib import SequenceMatcher
    fetched_keys = {normalize_url(url): (url, _title_key(title)) for url, title in fetched.items()}
    copies = {}
    for url, title in others.items():
        key = _title_key(title)
        for fetched_key, (fetched_url, fetched_title) in fetched_keys.items():
            same_link = normalize_url(url) == fetched_key
            same_title = key == fetched_title
            near = (len(key) >= NEAR_DUPLICATE_MIN_LENGTH and len(fetched_title) >= NEAR_DUPLICATE_MIN_LENGTH
                    and SequenceMatcher(None, key, fetched_title).ratio() >= NEAR_DUPLICATE_RATIO)
            if same_link or same_title or near:
                copies[url] = fetched_url
                break
    return copies


def final_rows(v2: dict[str, str], v3_rows: list[dict], titles: dict[str, str], blocked: set[str],
               reasons: dict[str, str], listings: set[str] | None = None,
               copies: dict[str, str] | None = None) -> list[dict]:
    v2, titles, reasons = _tidy(v2), _tidy(titles), _tidy(reasons)
    blocked = {normalize_url(url) for url in blocked}
    listings = {normalize_url(url) for url in (listings or set())}
    copies = {normalize_url(url): normalize_url(twin) for url, twin in (copies or {}).items()}
    v3 = {normalize_url(row["url"]): row for row in v3_rows}
    unknown = [url for url in v3 if url not in v2]
    if unknown:
        raise LabelV3Error(f"{len(unknown)} article(s) have no version 2 label, for example {unknown[0]}.")

    rows = []
    for url in sorted(v2):
        row = {"article_id": article_id(url), "title": titles.get(url, ""), "url": url,
               "label_v2": v2[url], "guide_version": GUIDE_VERSION}
        if url in v3:
            label = v3[url]["label"].strip().lower()
            if label not in VALID_LABELS:
                raise LabelV3Error(f"{url}: label '{label}' must be one of {VALID_LABELS}.")
            changed = label != v2[url]
            reason = (reasons.get(url) or "").strip()
            if changed and not reason:
                raise LabelV3Error(f"{url}: the label changed from {v2[url]} to {label}; a reason is needed.")
            row |= {"label": label, "unsure": (v3[url].get("unsure") or "").strip(),
                    "note": (v3[url].get("note") or "").strip(), "basis": BASIS_FULL,
                    "changed": "yes" if changed else "", "reason": reason if changed else ""}
        else:
            if url in blocked:
                basis = BASIS_BLOCKED
            elif url in listings:
                basis = BASIS_LISTING
            elif url in copies:
                basis = BASIS_COPY
                twin = copies[url]
                twin_label = v3[twin]["label"].strip().lower() if twin in v3 else v2.get(twin)
                if twin_label != v2[url]:
                    raise LabelV3Error(f"{url} is a copy of {twin} but has label {v2[url]}, not {twin_label}.")
            else:
                raise LabelV3Error(f"{url} has no full text and no known reason (blocked, listing or copy).")
            row |= {"label": v2[url], "unsure": "", "note": "", "basis": basis, "changed": "", "reason": ""}
        rows.append({column: row[column] for column in FINAL_COLUMNS})
    return rows


def write_final(rows: list[dict], folder: Path, stem: str) -> Path:
    """Writes (or replaces) the final file; Git keeps every earlier version."""
    path = Path(folder) / f"{stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------- running it ----------

def _read(path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _v2_files() -> list[Path]:
    return score_label_files() + sorted(EVAL_DIR.glob("dev_*_fresh_*.csv"))


def _titles_and_raw_urls() -> tuple[dict[str, str], dict[str, str]]:
    titles, raw = {}, {}
    for path in _v2_files():
        for row in _read(path):
            key = normalize_url(row["url"])
            titles.setdefault(key, row.get("title", ""))
            raw.setdefault(key, row["url"].strip())
    return titles, raw


def _v3_labels(path) -> dict[str, str]:
    labels = {}
    for row in _read(path):
        label = (row.get("label") or "").strip().lower()
        if label not in VALID_LABELS:
            raise LabelV3Error(f"Row {row.get('row')}: label '{row.get('label')}' must be one of {VALID_LABELS}.")
        labels[row["url"].strip()] = label
    return labels


def _compare_command(v3_path: str) -> None:
    v2, v3 = load_label_set(_v2_files()), _v3_labels(v3_path)
    titles, _ = _titles_and_raw_urls()
    result = compare_versions(v2, v3)
    print(f"Articles labelled on full text: {result['articles']}")
    print(f"Same label as version 2: {result['agreement']:.3f}")
    print(f"Cohen's kappa: {result['kappa']:.3f}")
    for (before, after), count in sorted(result["moves"].items()):
        print(f"   {before:>8} -> {after:<8} {count}")
    rows = difference_rows(v2, v3, titles)
    path = unique_path(EVAL_DIR, f"differences_v2_v3_{datetime.now().strftime('%Y%m%d')}", suffix=".csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIFFERENCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} changed labels: write a reason for each  ->  {path}")


def _final_command(v3_path: str, differences_path: str) -> None:
    v2 = load_label_set(_v2_files())
    titles, raw = _titles_and_raw_urls()
    reasons = {row["url"]: row.get("reason", "") for row in _read(differences_path)}
    v3_rows = _read(v3_path)
    in_v3 = {normalize_url(row["url"]) for row in v3_rows}
    from src.config import RAW_DIR
    raw_articles = {}
    for path in sorted(RAW_DIR.glob("*.json")):
        for article in json.loads(path.read_text(encoding="utf-8")).get("articles", []):
            if article.get("url"):
                raw_articles.setdefault(normalize_url(article["url"]), article)
    fetcher, blocked, listings = FullTextFetcher(), set(), set()
    for key in v2:
        if key in in_v3:
            continue
        cached = fetcher._cache_path(raw.get(key, key))
        if cached.exists() and json.loads(cached.read_text(encoding="utf-8")).get("status") == "blocked_by_robots":
            blocked.add(key)
            continue
        signals = listing_signals(raw_articles.get(key, {"url": raw.get(key, key), "title": titles.get(key, "")}))
        if signals["pattern"] or signals["site"]:
            listings.add(key)
    others = {key: titles.get(key, "") for key in v2 if key not in in_v3 and key not in blocked | listings}
    fetched = {normalize_url(row["url"]): row.get("title", "") for row in v3_rows}
    copies = find_copies(others, fetched)
    rows = final_rows(v2, v3_rows, titles, blocked, reasons, listings=listings, copies=copies)
    path = write_final(rows, EVAL_DIR, "v3_dev_tata_motors")
    counts = Counter(row["basis"] for row in rows)
    print(f"Version 3 labels: {len(rows)} articles, {sum(r['changed'] == 'yes' for r in rows)} changed  ->  {path}")
    for basis, count in counts.items():
        print(f"   {basis}: {count}")


if __name__ == "__main__":
    command = sys.argv[1:]
    if len(command) == 2 and command[0] == "compare":
        _compare_command(command[1])
    elif len(command) == 3 and command[0] == "finalize":
        _final_command(command[1], command[2])
    else:
        print(__doc__)
