"""
Blind relabelling of the development set under labelling guide version 2.

Steps:
    1. sheet     every development article once, shuffled, with empty label
                 columns (the version 1 labels and notes are not shown)
    2. compare   agreement between version 1 and the blind version 2 labels
                 (share of equal labels and Cohen's kappa), and a sheet of the
                 articles whose label differs, for a final decision with a reason
    3. finalize  the version 2 label file: blind labels, with the decision used
                 where the two versions differ; the version 1 files stay unchanged

Run:
    python -m src.label_review sheet
    python -m src.label_review compare <blind sheet>
    python -m src.label_review finalize <blind sheet> <differences sheet>
"""

import csv
import random
import sys
from collections import Counter
from pathlib import Path

from src.config import EVAL_DIR
from src.evaluate import DEV_LABELS_V1, VALID_LABELS, load_labels
from src.preprocess import normalize_url
from src.retrieval_check import SEED, SHEET_COLUMNS, write_sheet
from src.utils import unique_path

DIFFERENCE_COLUMNS = ["title", "url", "label_v1", "label_v2_blind", "final_label", "reason", "unsure"]
V2_COLUMNS = SHEET_COLUMNS + ["label_v1", "changed", "reason", "rule_version"]
RULE_VERSION = "2"


class LabelReviewError(Exception):
    pass


def _read(path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


# ---------- 1. the blind sheet ----------

def recheck_rows(paths, seed: int = SEED) -> list[dict]:
    """Every article in the given label files once, shuffled, without the old label, note or file names."""
    found = {}
    for path in paths:
        for row in _read(path):
            key = normalize_url(row["url"])
            if key not in found:
                found[key] = row
    rows = list(found.values())
    random.Random(seed).shuffle(rows)
    return [
        {column: "" for column in SHEET_COLUMNS}
        | {column: row.get(column, "") for column in
           ("title", "source", "published", "description", "content_first_200_chars", "url")}
        | {"row": number}
        for number, row in enumerate(rows, start=1)
    ]


# ---------- 2. agreement ----------

def cohen_kappa(first: list[str], second: list[str]) -> float:
    """
    Agreement corrected for chance.
    observed = share of items with the same label
    chance   = sum over labels of (share in first) x (share in second)
    kappa    = (observed - chance) / (1 - chance)
    """
    total = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / total
    count_first, count_second = Counter(first), Counter(second)
    chance = sum(count_first[label] * count_second[label] for label in count_first) / total ** 2
    if chance == 1:                    # one label everywhere: kappa is not defined
        return 1.0 if observed == 1 else 0.0
    return (observed - chance) / (1 - chance)


def compare_labels(old: dict[str, str], new: dict[str, str]) -> dict:
    if set(old) != set(new):
        raise LabelReviewError("The two label sets do not cover the same articles.")
    urls = sorted(old)
    changes = [(url, old[url], new[url]) for url in urls if old[url] != new[url]]
    return {
        "articles": len(urls),
        "agreement": (len(urls) - len(changes)) / len(urls),
        "kappa": cohen_kappa([old[u] for u in urls], [new[u] for u in urls]),
        "changes": changes,
        "moves": dict(Counter((before, after) for _, before, after in changes)),
    }


def difference_rows(old: dict[str, str], new: dict[str, str], titles: dict[str, str]) -> list[dict]:
    return [
        {"title": titles.get(url, ""), "url": url, "label_v1": before, "label_v2_blind": after,
         "final_label": "", "reason": "", "unsure": ""}
        for url, before, after in compare_labels(old, new)["changes"]
    ]


# ---------- 3. the version 2 labels ----------

def final_labels(blind: dict[str, str], decisions: dict[str, str], changed: set[str] | None = None) -> dict[str, str]:
    """Blind labels, replaced by the decision where one is given. Every changed article needs a decision."""
    missing = [url for url in (changed or set()) if not (decisions.get(url) or "").strip()]
    if missing:
        raise LabelReviewError(f"{len(missing)} changed article(s) have no final label.")
    result = {}
    for url, label in blind.items():
        decision = (decisions.get(url) or "").strip().lower()
        result[url] = decision or label
        if result[url] not in VALID_LABELS:
            raise LabelReviewError(f"Label '{result[url]}' for {url} must be one of {VALID_LABELS}.")
    return result


# ---------- running it ----------

def _v1_labels() -> dict[str, str]:
    labels = {}
    for name in DEV_LABELS_V1:
        labels |= {normalize_url(url): label for url, label in load_labels(EVAL_DIR / name).items()}
    return labels


def _blind_labels(path) -> dict[str, str]:
    return {normalize_url(url): label for url, label in load_labels(path).items()}


def _sheet_command() -> None:
    rows = recheck_rows([EVAL_DIR / name for name in DEV_LABELS_V1])
    path = write_sheet(rows, EVAL_DIR, "recheck_v2_dev_tata_motors")
    print(f"{len(rows)} articles to label  ->  {path}")


def _compare_command(blind_path: str) -> None:
    old, new = _v1_labels(), _blind_labels(blind_path)
    result = compare_labels(old, new)
    print(f"Articles: {result['articles']}")
    print(f"Same label in both versions: {result['agreement']:.3f}")
    print(f"Cohen's kappa: {result['kappa']:.3f}")
    for (before, after), count in sorted(result["moves"].items()):
        print(f"   {before:>8} -> {after:<8} {count}")
    titles = {normalize_url(row["url"]): row["title"] for row in _read(blind_path)}
    rows = difference_rows(old, new, titles)
    path = unique_path(EVAL_DIR, "differences_v1_v2_dev_tata_motors", suffix=".csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIFFERENCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} differences to decide  ->  {path}")


def _finalize_command(blind_path: str, differences_path: str) -> None:
    old, blind = _v1_labels(), _blind_labels(blind_path)
    differences = {normalize_url(row["url"]): row for row in _read(differences_path)}
    decisions = {url: row["final_label"] for url, row in differences.items()}
    labels = final_labels(blind, decisions, changed=set(differences))
    rows = []
    for row in _read(blind_path):
        url = normalize_url(row["url"])
        decision = differences.get(url, {})
        rows.append({column: row.get(column, "") for column in SHEET_COLUMNS}
                    | {"label": labels[url],
                       "unsure": (decision.get("unsure") or row.get("unsure") or "").strip(),
                       "label_v1": old[url],
                       "changed": "yes" if labels[url] != old[url] else "",
                       "reason": decision.get("reason", ""),
                       "rule_version": RULE_VERSION})
    path = unique_path(EVAL_DIR, "v2_dev_tata_motors", suffix=".csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=V2_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Version 2 labels: {len(rows)} articles, {sum(r['changed'] == 'yes' for r in rows)} changed  ->  {path}")


if __name__ == "__main__":
    command = sys.argv[1:]
    if command == ["sheet"]:
        _sheet_command()
    elif len(command) == 2 and command[0] == "compare":
        _compare_command(command[1])
    elif len(command) == 3 and command[0] == "finalize":
        _finalize_command(command[1], command[2])
    else:
        print(__doc__)
