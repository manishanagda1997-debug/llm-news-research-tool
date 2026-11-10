"""
Measure the relevance rules against hand-made labels.

Labels (one per article, in a CSV file in data/eval/):
    direct    the article is mainly about the company
    indirect  not mainly about the company, but useful to its analyst
    not       not useful
"Relevant" means direct or indirect.

Two kinds of numbers:
    - the filter itself, before the article limit:
      true/false positives and negatives, precision, recall, F1
    - the articles a user would actually see: precision@k and recall@k

Recall here is candidate-set recall: it counts only the articles NewsAPI
returned for the search, not every article published that week.

A relevant article dropped only because it is a duplicate of a kept
article counts as found: its story is in the results.

Run:  python -m src.evaluate
"""

import csv
import json
from pathlib import Path

from src.config import DATA_DIR, EVAL_DIR, RAW_DIR
from src.preprocess import RANKINGS, articles_to_text, clean_articles, normalize_url
from src.query import load_aliases

VALID_LABELS = ("direct", "indirect", "not")


def load_labels(path) -> dict[str, str]:
    """Read a label CSV into {url: label}. Stops on any unknown label."""
    labels = {}
    with open(path, encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            label = (row.get("label") or "").strip().lower()
            if label not in VALID_LABELS:
                raise ValueError(f"Row {row.get('row')}: label '{row.get('label')}' must be one of {VALID_LABELS}.")
            labels[row["url"].strip()] = label
    return labels


def is_relevant(label: str, strict: bool = False) -> bool:
    """Broad relevance: direct or indirect. Strict relevance: direct only."""
    return label == "direct" if strict else label in ("direct", "indirect")


def confusion(relevant: dict[str, bool], kept: set[str]) -> dict[str, int]:
    """Count true/false positives/negatives. `relevant` maps each item to True/False."""
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for item, is_rel in relevant.items():
        if item in kept:
            counts["tp" if is_rel else "fp"] += 1
        else:
            counts["fn" if is_rel else "tn"] += 1
    return counts


def _divide(top: float, bottom: float) -> float:
    return top / bottom if bottom else 0.0


def scores(counts: dict[str, int]) -> dict[str, float]:
    precision = _divide(counts["tp"], counts["tp"] + counts["fp"])
    recall = _divide(counts["tp"], counts["tp"] + counts["fn"])
    f1 = _divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def precision_at_k(ranked: list[str], relevant: dict[str, bool], k: int) -> float:
    """Share of the shown items (the first k, or fewer if fewer exist) that are relevant.
    When fewer than k items exist, it divides by the number actually shown."""
    shown = ranked[:k]
    return _divide(sum(relevant.get(item, False) for item in shown), len(shown))


def recall_at_k(ranked: list[str], relevant: dict[str, bool], k: int) -> float:
    """Share of all relevant items that appear in the first k shown items."""
    return _divide(sum(relevant.get(item, False) for item in ranked[:k]), sum(relevant.values()))


def evaluate_file(raw_path, labels_path, query: str, k: int = 10, **clean_options) -> dict:
    """
    Run clean_articles on a saved NewsAPI file and compare with the labels.
    clean_options are passed to clean_articles (aliases, remove_junk, exact_phrase).
    """
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))["articles"]
    labels = load_labels(labels_path)
    relevant = {url: is_relevant(label) for url, label in labels.items()}
    titles = {article["url"]: article["title"] for article in raw}

    articles, stats = clean_articles(raw, max_articles=max(len(raw), 1), query=query, **clean_options)
    ranked = [article["url"] for article in articles]
    covered = set(ranked)
    for entry in stats["duplicate_log"]:
        if entry["kept"]["url"] in covered:
            covered.add(entry["removed"]["url"])

    counts = confusion(relevant, covered)
    return {
        "counts": counts,
        "scores": scores(counts),
        f"precision@{k}": precision_at_k(ranked, relevant, k),
        f"recall@{k}": recall_at_k(ranked, relevant, k),
        "kept": len(ranked),
        "relevant_total": sum(relevant.values()),
        "wrongly_kept": [titles.get(url, url) for url in covered if not relevant.get(url, False) and url in relevant],
        "missed": [titles.get(url, url) for url, rel in relevant.items() if rel and url not in covered],
        "junk_text_removed": stats["junk_text_removed"],
        "duplicate_log": stats["duplicate_log"],
    }


class LabelConflictError(ValueError):
    pass


def load_label_set(paths) -> dict[str, str]:
    """Labels from several CSV files, keyed by tidy link. Stops if two files give one article different labels."""
    labels: dict[str, str] = {}
    for path in paths:
        for url, label in load_labels(path).items():
            key = normalize_url(url)
            if key in labels and labels[key] != label:
                raise LabelConflictError(f"{url}: '{labels[key]}' in an earlier file, '{label}' in {Path(path).name}.")
            labels[key] = label
    return labels


def load_all_labels(paths) -> dict[str, str]:
    """Labels from several CSV files, keyed by tidy link (one label per article)."""
    labels: dict[str, str] = {}
    for path in paths:
        for url, label in load_labels(path).items():
            labels[normalize_url(url)] = label
    return labels


def run_fetch(raw_path, labels: dict[str, str], query: str, strict: bool = False, **clean_options) -> dict:
    """
    One saved NewsAPI search, cleaned with the given options.
    Returns the shown order, the covered articles and the labels that apply.
    """
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))["articles"]
    articles, stats = clean_articles(raw, max_articles=max(len(raw), 1), query=query, **clean_options)
    ranked = [normalize_url(article["url"]) for article in articles]
    covered = set(ranked)
    for entry in stats["duplicate_log"]:
        if normalize_url(entry["kept"]["url"]) in covered:
            covered.add(normalize_url(entry["removed"]["url"]))
    in_fetch = {normalize_url(article["url"]) for article in raw}
    missing = in_fetch - set(labels)
    if missing:
        raise ValueError(f"{len(missing)} article(s) in {Path(raw_path).name} have no label.")
    return {
        "articles": articles,
        "ranked": ranked,
        "covered": covered,
        "relevant": {url: is_relevant(labels[url], strict=strict) for url in in_fetch},
        "titles": {normalize_url(article["url"]): article["title"] for article in raw},
        "junk_text_removed": stats["junk_text_removed"],
    }


def pooled_filter_scores(runs: list[dict]) -> dict:
    """Filter scores over all unique labelled articles in the runs (each article counted once)."""
    relevant, covered, titles = {}, set(), {}
    for run in runs:
        relevant.update(run["relevant"])
        covered |= run["covered"]
        titles.update(run["titles"])
    counts = confusion(relevant, covered)
    return {
        "counts": counts,
        "scores": scores(counts),
        "articles": len(relevant),
        "relevant_total": sum(relevant.values()),
        "wrongly_kept": sorted(titles[url] for url in covered if url in relevant and not relevant[url]),
        "missed": sorted(titles[url] for url, rel in relevant.items() if rel and url not in covered),
    }


def top_k_scores(runs: list[dict], k: int) -> dict:
    """Precision@k and Recall@k for each search, and their average."""
    per_run = [
        {"precision": precision_at_k(run["ranked"], run["relevant"], k),
         "recall": recall_at_k(run["ranked"], run["relevant"], k),
         "prompt_chars": len(articles_to_text(run["articles"][:k]))}
        for run in runs
    ]
    average = {key: sum(item[key] for item in per_run) / len(per_run) for key in per_run[0]}
    return {"per_search": per_run, "average": average}


VERSIONS = {
    "Baseline": {"remove_junk": False, "alias_file": None},
    "+ Fix A (junk)": {"remove_junk": True, "alias_file": None},
    "+ Fix B (aliases)": {"remove_junk": False, "alias_file": "tmpv_tmcv"},
    "+ A + B": {"remove_junk": True, "alias_file": "tmpv_tmcv"},
    "+ A + B + JLR": {"remove_junk": True, "alias_file": "full"},
}


def _alias_variants() -> dict:
    """The full alias file, and the earlier version without Jaguar Land Rover (to measure each step)."""
    full = load_aliases(DATA_DIR / "company_aliases.json")
    without_jlr = json.loads(json.dumps(full))
    without_jlr["identities"].pop("jaguar_land_rover", None)
    for group in without_jlr["search_groups"].values():
        group["identities"] = [i for i in group["identities"] if i != "jaguar_land_rover"]
    return {None: None, "tmpv_tmcv": without_jlr, "full": full}


DEV_SEARCHES = [
    ("14 Oct 18:18 (search without quotes)", "20261014_181801_tata-motors.json"),
    ("14 Oct 18:25", "20261014_182548_tata-motors.json"),
    ("23 Oct 10:17", "20261023_101707_tata-motors.json"),
]
DEV_LABELS_V1 = ["dev_tata_motors_20261014.csv", "dev_tata_motors_20261023.csv"]
DEV_LABELS_V2 = ["v2_dev_tata_motors.csv"]
LABEL_VERSION = "2"                 # the labelling guide version used for all scores
DEV_LABELS = DEV_LABELS_V2


def _runs(query: str, labels: dict, aliases: dict | None, strict: bool = False, **options) -> list[dict]:
    return [run_fetch(RAW_DIR / name, labels, query, strict=strict, aliases=aliases, **options) for _, name in DEV_SEARCHES]


def main(query: str = "Tata Motors") -> None:
    labels = load_label_set([EVAL_DIR / name for name in DEV_LABELS])
    print(f"Labels: guide version {LABEL_VERSION} ({', '.join(DEV_LABELS)})")
    variants = _alias_variants()

    for strict, title in ((False, "BROAD relevance (direct + indirect)"), (True, "STRICT relevance (direct only)")):
        print(f"\n=== {title}: development set, Tata Motors, 3 saved searches, each article counted once ===\n")
        print(f"{'Version':<20}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}{'Precision':>11}{'Recall':>9}{'F1':>7}")
        results = {}
        for name, version in VERSIONS.items():
            runs = _runs(query, labels, variants[version["alias_file"]], strict=strict, remove_junk=version["remove_junk"])
            results[name] = pooled_filter_scores(runs)
            c, sc = results[name]["counts"], results[name]["scores"]
            print(f"{name:<20}{c['tp']:>4}{c['fp']:>4}{c['fn']:>4}{c['tn']:>4}"
                  f"{sc['precision']:>11.3f}{sc['recall']:>9.3f}{sc['f1']:>7.3f}")
        best = results["+ A + B + JLR"]
        print(f"\n+ A + B + JLR on {best['articles']} labelled articles ({best['relevant_total']} relevant):")
        for title_text in best["wrongly_kept"]:
            print(f"   wrongly kept: {title_text}")
        for title_text in best["missed"]:
            print(f"   missed:       {title_text}")

    print("\n=== Top-k (A + B + JLR, broad relevance), per search ===")
    print("P@k divides by the number of articles actually shown when fewer than k exist.\n")
    print(f"{'Ranking':<9}{'k':>3}  {'Search':<38}{'shown':>6}{'P@k':>7}{'R@k':>7}{'chars':>7}")
    for ranking in RANKINGS:
        runs = _runs(query, labels, variants["full"], remove_junk=True, ranking=ranking)
        for k in (10, 15, 20):
            result = top_k_scores(runs, k)
            for (label, _), run, item in zip(DEV_SEARCHES, runs, result["per_search"]):
                print(f"{ranking:<9}{k:>3}  {label:<38}{min(k, len(run['ranked'])):>6}"
                      f"{item['precision']:>7.3f}{item['recall']:>7.3f}{item['prompt_chars']:>7.0f}")
            avg = result["average"]
            print(f"{ranking:<9}{k:>3}  {'AVERAGE':<38}{'':>6}{avg['precision']:>7.3f}{avg['recall']:>7.3f}"
                  f"{avg['prompt_chars']:>7.0f}")


if __name__ == "__main__":
    main()
