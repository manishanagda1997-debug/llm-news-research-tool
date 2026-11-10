"""
The fresh evaluation of the ranking configurations: one new week of news,
labelled blind, scored once with the frozen method.

The run has a fixed id (RUN_ID) and its own folder, data/eval/<RUN_ID>/.
Each stage is recorded in state.json, so a stopped run resumes where it
stopped, reuses its first fetch and never mixes two fetches.

Stages, in order:
    fetch    - searches A and B for the publication window WINDOW_FROM to
               WINDOW_TO (UTC), refused before FETCH_NOT_BEFORE (1 November,
               06:00 UTC), because the free NewsAPI plan shows articles a day
               late; the 6 hours after that delay are a margin. The number of
               articles per day of the window is saved and printed, so an
               empty last day is visible. Request settings are saved without
               keys or secrets.
    overlap  - articles with the same tidy link as a development article, and
               articles published outside the window, are excluded and
               logged. Near-identical titles are not excluded automatically:
               they go to overlap_review.csv, where `same` means the same
               article or a repost (not just the same news event) and
               `different` keeps the article.
    sheet    - full text for the kept articles (robots.txt respected, cached
               in data/fulltext/, never in Git), then a blind label sheet
               (labels.csv) and a reading page (in data/fulltext/): shuffled,
               with no search, filter or model output shown.
    events   - after the labels: an event sheet (events.csv) of the relevant
               articles, as for the development data.
    protocol - after the development record (python -m src.ranking_eval --check):
               fresh_protocol.json fixes, before the freeze, the fingerprints
               of labels.csv and events.csv, the eligible set and the
               development winner (development evidence only), the model and
               its settings, the six configurations, the S2 threshold, the
               acceptance rule, the collection budget and the package
               versions. It records the source commit; the freeze tag marks
               the commit that adds the protocol.
    collect  - after the freeze tag: asks the model for the candidates without
               a saved answer, at most MAX_COLLECTION_PASSES passes in all.
               Each pass is recorded before its first request, so a restart or
               a crash cannot add passes. Saved answers are never asked again.
               Every pass keeps its failures by kind. Only counts are printed.
    score    - once only, after every answer is in or the passes are used.
               Refused unless the frozen files equal the freeze tag and the
               labels and events equal the protocol. Reads saved answers only
               (no model calls); a missing answer falls back to the O1 place.
               The runtime condition uses the first pass only, and only when
               it is live and answers at least MIN_MODEL_COVERAGE of the
               candidates. The development winner becomes the default only if
               it passes on the fresh run; other configurations are
               experimental evidence. Saves fresh_ranking_record.json.

Run, in order:
    python -m src.fresh_eval fetch      (2 NewsAPI requests; from 1 November)
    python -m src.fresh_eval overlap
    python -m src.fresh_eval sheet      (after overlap_review.csv is decided, if it has rows)
    python -m src.fresh_eval events     (after labels.csv is filled)
    python -m src.ranking_eval --check gpt-oss-120b   (the development record)
    python -m src.fresh_eval protocol   (then commit, then: git tag freeze-fresh-eval)
    python -m src.fresh_eval collect    (model calls; again later if answers are missing)
    python -m src.fresh_eval score      (once)
"""

import csv
import json
import random
import re
import subprocess
import time
import sys
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from importlib import metadata
from pathlib import Path

from src.config import DATA_DIR, EVAL_DIR, RAW_DIR, ROOT_DIR
from src.evaluate import VALID_LABELS, is_relevant
from src.fulltext import article_id
from src.preprocess import NEAR_DUPLICATE_MIN_LENGTH, NEAR_DUPLICATE_RATIO, _title_key, normalize_url

RUN_ID = "fresh_2026-10-24_30"
WINDOW_FROM = "2026-10-24T00:00:00Z"
WINDOW_TO = "2026-10-30T23:59:59Z"
FETCH_NOT_BEFORE = datetime(2026, 11, 1, 6, 0, tzinfo=timezone.utc)   # the one-day delay, plus 6 hours of margin
SEARCHES = ("A", "B")
RUN_DIR = EVAL_DIR / RUN_ID
STATE_FILE = "state.json"
REVIEW_FILE = "overlap_review.csv"
LABEL_FILE = "labels.csv"
EVENTS_FILE = "events.csv"
RECORD_FILE = "fresh_ranking_record.json"
PROGRESS_SEED = 7                   # the order of the progress list, apart from the row order of the sheet
STATUS_TEASER = "short NewsAPI text only (the page holds only a teaser)"
STATUS_ANOTHER_STORY = "short NewsAPI text only (the page holds another story)"
STATUS_NOT_FETCHED = "short NewsAPI text only (the page could not be fetched)"
RUN_NOTES = ["The progress list of the first sheet command shows the titles in search order (search A, "
             "then search B). The label sheet and the reading page are shuffled."]
DEV_RECORD = EVAL_DIR / "ranking_dev_record.json"
FREEZE_TAG = "freeze-fresh-eval"
FROZEN_FOLDERS = ("src/", "tests/")
PROTOCOL_FILE = "fresh_protocol.json"
MAX_COLLECTION_PASSES = 3           # passes of `collect`; the count is saved, so a restart cannot reset it
MIN_MODEL_COVERAGE = 0.9            # share of model answers the first pass needs for its runtime to count
FROZEN_FILES = ("requirements.txt", "pytest.ini", "data/company_aliases.json", "data/eval/LABELLING_GUIDE.md",
                "data/eval/SUMMARY_JUDGING_GUIDE.md", "data/eval/ranking_dev_record.json",
                f"data/eval/{RUN_ID}/labels.csv", f"data/eval/{RUN_ID}/events.csv",
                f"data/eval/{RUN_ID}/{PROTOCOL_FILE}")
PACKAGES = ("streamlit", "langchain", "langchain-core", "langchain-google-genai", "langchain-groq", "groq",
            "newsapi-python", "trafilatura", "pandas", "python-dotenv")
SECRET_FIELD = re.compile(r"key|token|auth|secret|password", re.I)
REVIEW_COLUMNS = ["row", "decision", "fresh_title", "dev_title", "fresh_source", "dev_source",
                  "fresh_published", "dev_published", "fresh_url", "dev_url"]
DECISIONS = ("same", "different")


class FreshEvalError(Exception):
    pass


# ---------- the run's state ----------

def load_state(run_dir) -> dict:
    path = run_dir / STATE_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"run_id": RUN_ID, "stages": {}}


def save_state(run_dir, state: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_request(request: dict) -> dict:
    """Request settings without keys, tokens or other secrets."""
    return {name: value for name, value in request.items() if not SECRET_FIELD.search(name)}


# ---------- fetch ----------

def fetch_stage(fetch, now: datetime, run_dir, aliases: dict | None, save, company: str = "Tata Motors",
                seen_files: list[str] | None = None) -> dict:
    state = load_state(run_dir)
    done = state["stages"].get("fetch")
    if done:
        if done.get("done"):
            return state                                     # resume: the first fetch is reused
        if done.get("raw_files"):
            raise FreshEvalError("The run folder holds an unfinished fetch. Inspect it before going on; "
                                 "a second fetch would mix two sets of results.")
        # no response is saved yet (the first request failed): nothing can mix, so the fetch starts again
    if now < FETCH_NOT_BEFORE:
        raise FreshEvalError(f"Too early: the window ends {WINDOW_TO} and the free plan shows articles a day "
                             f"late. Fetch on or after {FETCH_NOT_BEFORE:%d %B %Y %H:%M} (UTC).")
    if seen_files is None:
        seen_files = sorted(path.name for path in RAW_DIR.glob("*.json"))
    stage = {"done": False, "raw_files": {}, "requests": {}, "seen_files": seen_files}
    state["stages"]["fetch"] = stage
    save_state(run_dir, state)
    settings = {"page_size": 30, "sort_by": "relevancy", "exact_phrase": True, "title_only": False,
                "save": False, "from_time": WINDOW_FROM, "to_time": WINDOW_TO}
    for name in SEARCHES:
        if name == "A":
            response = fetch(company, aliases=None, **settings)
        else:
            response = fetch(company, aliases=aliases, part="subsidiaries", **settings)
        response["request"] = safe_request(response.get("request", {}))
        stage["raw_files"][name] = save(response).name
        stage["requests"][name] = response["request"]
        save_state(run_dir, state)
    stage["done"] = True
    stage["fetched_at"] = now.isoformat(timespec="seconds")
    save_state(run_dir, state)
    return state


# ---------- overlap with development data ----------

def _published(article: dict) -> datetime | None:
    text = (article.get("publishedAt") or "").strip()
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def day_counts(articles: list[dict]) -> dict[str, int]:
    """Articles per day of the window (UTC), every day listed, empty days as 0."""
    start, end = _window()
    counts, day = {}, start
    while day.date() <= end.date():
        counts[day.date().isoformat()] = 0
        day += timedelta(days=1)
    for article in articles:
        published = _published(article)
        if published is not None and published.date().isoformat() in counts:
            counts[published.date().isoformat()] += 1
    return counts


def _window() -> tuple[datetime, datetime]:
    return (datetime.fromisoformat(WINDOW_FROM.replace("Z", "+00:00")),
            datetime.fromisoformat(WINDOW_TO.replace("Z", "+00:00")))


def _near_title(a: str, b: str) -> bool:
    a, b = _title_key(a or ""), _title_key(b or "")
    if not a or not b:
        return False
    return a == b or (min(len(a), len(b)) >= NEAR_DUPLICATE_MIN_LENGTH
                      and SequenceMatcher(None, a, b).ratio() >= NEAR_DUPLICATE_RATIO)


def _entry(article: dict, reason: str) -> dict:
    return {"url": article.get("url"), "title": article.get("title"), "published": article.get("publishedAt"),
            "reason": reason}


def overlap(fresh: list[dict], dev: list[dict]) -> dict:
    start, end = _window()
    dev_links = {normalize_url(a.get("url") or "") for a in dev}
    kept, excluded, review = [], [], []
    for article in fresh:
        published = _published(article)
        if normalize_url(article.get("url") or "") in dev_links:
            excluded.append(_entry(article, "development_link"))
        elif published is None:
            excluded.append(_entry(article, "no_date"))
        elif not start <= published <= end:
            excluded.append(_entry(article, "outside_window"))
        else:
            kept.append(article)
            match = next((d for d in dev if _near_title(article.get("title"), d.get("title"))), None)
            if match:
                review.append({"fresh_url": article["url"], "dev_url": match["url"],
                               "fresh_title": article.get("title"), "dev_title": match.get("title"),
                               "fresh_source": (article.get("source") or {}).get("name", ""),
                               "dev_source": (match.get("source") or {}).get("name", ""),
                               "fresh_published": article.get("publishedAt"),
                               "dev_published": match.get("publishedAt")})
    return {"kept": kept, "excluded": excluded, "review": review}


def apply_review(result: dict, decisions: list[dict]) -> dict:
    """`same` (the same article or a repost) excludes; `different` keeps. Every reviewed pair needs one."""
    given = {normalize_url(row.get("fresh_url") or ""): (row.get("decision") or "").strip().lower()
             for row in decisions}
    confirmed = set()
    for pair in result["review"]:
        key = normalize_url(pair["fresh_url"])
        decision = given.get(key)
        if decision not in DECISIONS:
            raise FreshEvalError(f"Overlap review: '{pair['fresh_title']}' needs a decision: same or different "
                                 f"(found: {decision!r}).")
        if decision == "same":
            confirmed.add(key)
    kept = [a for a in result["kept"] if normalize_url(a["url"]) not in confirmed]
    excluded = result["excluded"] + [_entry(a, "confirmed_repost") for a in result["kept"]
                                     if normalize_url(a["url"]) in confirmed]
    return {"kept": kept, "excluded": excluded, "review": result["review"]}


# ---------- validation before scoring ----------

def validation_problems(candidates: list[dict], labels: dict[str, str], events: dict[str, str],
                        article_ids: set[str], threshold, dev_links: set[str]) -> list[str]:
    from src.ranking_eval import THRESHOLDS
    problems = []
    dev = {normalize_url(link) for link in dev_links}
    for c in candidates:
        key = c["key"]
        if key not in article_ids:
            problems.append(f"{key}: candidate not in the run's articles")
        label = labels.get(key)
        if label is None:
            problems.append(f"{key}: no label")
        elif label not in VALID_LABELS:
            problems.append(f"{key}: invalid label {label!r}")
        elif is_relevant(label) and key not in events:
            problems.append(f"{key}: relevant but no event")
        if normalize_url(c.get("url") or "") in dev:
            problems.append(f"{key}: a development article is still in the fresh set")
    problems += [f"{key}: labelled but not in the run's articles" for key in labels if key not in article_ids]
    if threshold is None:
        problems.append("frozen threshold missing (data/eval/ranking_dev_record.json)")
    elif threshold not in THRESHOLDS:
        problems.append(f"frozen threshold {threshold} is not one of {THRESHOLDS}")
    return problems


def experiment_problems(dev_record: dict, current: dict) -> list[str]:
    """The fresh run must use exactly the provider, model and settings of the development record."""
    from src.experiment import record_problems
    return record_problems(dev_record.get("experiment"), current)


# ---------- the freeze ----------

def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise FreshEvalError(f"git {' '.join(args)} failed: {error}") from error


def _frozen(path: str) -> bool:
    path = path.strip().replace("\\", "/")
    if not path or "__pycache__" in path or path.endswith(".pyc"):
        return False
    return path.startswith(FROZEN_FOLDERS) or path in FROZEN_FILES


def freeze_check(run_git=git) -> dict:
    """The frozen implementation must equal the tag; new data files are allowed."""
    try:
        tag_commit = run_git("rev-parse", "--verify", f"{FREEZE_TAG}^{{commit}}")
    except FreshEvalError:
        return {"ok": False, "problems": [f"the tag {FREEZE_TAG} does not exist"], "head": None, "tag_commit": None}
    head = run_git("rev-parse", "HEAD")
    changed = run_git("diff", "--name-only", FREEZE_TAG).splitlines()
    untracked = run_git("ls-files", "--others", "--exclude-standard").splitlines()
    problems = sorted({path.strip() for path in changed + untracked if _frozen(path)})
    return {"ok": not problems, "problems": problems, "head": head, "tag_commit": tag_commit}


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions



# ---------- the eligible set and the default (development evidence only) ----------

def eligible_set(dev_record: dict, compare_choice: str | None) -> dict:
    """
    The baseline is always eligible. Another configuration is eligible when it passes the acceptance rule on at
    least two of the three development runs; a configuration that uses the model also needs to be the model
    comparison's choice. The development winner: the eligible configuration with the most development runs
    passed, then the simpler one; the baseline when no other configuration is eligible.
    """
    from src import ranking as rk
    passed = {}
    for report in (dev_record.get("reports") or {}).values():
        for name, verdict in (report.get("acceptance") or {}).items():
            passed[name] = passed.get(name, 0) + (1 if verdict.get("passed") else 0)
    eligible = [rk.BASELINE]
    for name in rk.SIMPLICITY:
        if name == rk.BASELINE or passed.get(name, 0) < 2:
            continue
        if not name.startswith("O1") and name != compare_choice:
            continue
        eligible.append(name)
    others = sorted((n for n in eligible if n != rk.BASELINE), key=lambda n: (-passed[n], rk.SIMPLICITY.index(n)))
    return {"eligible": eligible, "dev_winner": others[0] if others else rk.BASELINE, "dev_runs_passed": passed}


def default_configuration(dev_winner: str, fresh_report: dict) -> str:
    """The fresh run confirms or rejects the development winner; it never promotes another configuration."""
    from src import ranking as rk
    if dev_winner == rk.BASELINE:
        return rk.BASELINE
    verdict = (fresh_report.get("acceptance") or {}).get(dev_winner) or {}
    return dev_winner if verdict.get("passed") else rk.BASELINE


# ---------- collecting the model answers ----------

def collection_complete(collect: dict) -> bool:
    """Every answer is in after a finished pass, or the budget of passes is used."""
    passes = collect.get("passes", [])
    done = [p for p in passes if p.get("status") == "done"]
    if done and done[-1].get("missing_after") == 0:
        return True
    return len(passes) >= MAX_COLLECTION_PASSES


def start_pass(run_dir, now: datetime, cached_before: int) -> int:
    """Records a new pass before any request, so a crash still uses up that pass."""
    state = load_state(run_dir)
    collect = state["stages"].setdefault("collect", {"cached_before": cached_before, "passes": []})
    if collection_complete(collect):
        raise FreshEvalError("The collection is finished: every answer is in, or all "
                             f"{MAX_COLLECTION_PASSES} passes are used. Run the score.")
    number = len(collect["passes"]) + 1
    collect["passes"].append({"pass": number, "status": "started", "started": now.isoformat(timespec="seconds")})
    save_state(run_dir, state)
    return number


def finish_pass(run_dir, number: int, record: dict) -> dict:
    state = load_state(run_dir)
    entry = state["stages"]["collect"]["passes"][number - 1]
    entry.update(record, status="done", finished=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    save_state(run_dir, state)
    return state


def pass_record(asked: list[str], answers: dict[str, dict], missing_after: int, seconds: float,
                stats: dict | None) -> dict:
    failed = {key: answers[key].get("kind") or "unknown" for key in asked if answers[key].get("source") != "model"}
    failures = {}
    for kind in failed.values():
        failures[kind] = failures.get(kind, 0) + 1
    return {"asked": len(asked), "answered": len(asked) - len(failed), "missing_after": missing_after,
            "failures": failures, "failed_keys": failed, "seconds": round(seconds, 1) if seconds is not None else None,
            "tokens": (stats or {}).get("tokens")}


def first_pass_runtime(collect: dict, candidates: int) -> dict:
    """
    Only the first pass measures the live runtime. It counts for the runtime condition only when it finished,
    started with no saved answers, and answered at least MIN_MODEL_COVERAGE of the candidates.
    """
    passes = collect.get("passes", [])
    first = passes[0] if passes else {}
    kind = "live" if not collect.get("cached_before") else "mixed"
    finished = first.get("status") == "done"
    coverage = ((candidates - first["missing_after"]) / candidates
                if finished and candidates and first.get("missing_after") is not None else None)
    counts = finished and kind == "live" and coverage is not None and coverage >= MIN_MODEL_COVERAGE
    reason = ("counts" if counts else "the first pass did not finish" if not finished
              else "the first pass started with saved answers" if kind != "live"
              else f"first-pass coverage below {MIN_MODEL_COVERAGE:.0%}")
    return {"kind": kind, "status": first.get("status"), "seconds": first.get("seconds"), "coverage": coverage,
            "seconds_for_rule": first.get("seconds") if counts else None, "reason": reason}


def final_answers(keys: list[str], saved: dict[str, dict], collect: dict) -> dict[str, dict]:
    """Saved model answers; every other article falls back, with the last failure recorded for it."""
    from src.relevance_check import _fallback
    last_failure = {}
    for entry in collect.get("passes", []):
        last_failure.update(entry.get("failed_keys") or {})
    return {key: saved[key] if key in saved else
            _fallback("no model answer after the collection passes", last_failure.get(key, "unknown"))
            for key in keys}


# ---------- the frozen protocol ----------

def file_fingerprint(path) -> str:
    """SHA-256 of the text with Windows line endings made plain, so a Git checkout gives the same value."""
    import hashlib
    text = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(text).hexdigest()


def protocol_problems(protocol: dict, labels_path, events_path) -> list[str]:
    problems = []
    if protocol.get("labels_sha256") != file_fingerprint(labels_path):
        problems.append(f"{Path(labels_path).name} differs from the frozen protocol")
    if protocol.get("events_sha256") != file_fingerprint(events_path):
        problems.append(f"{Path(events_path).name} differs from the frozen protocol")
    return problems


def protocol_record(source_commit: str, dev_record: dict, eligibility: dict, compare_choice: str | None,
                    llm_identity: dict, labels_sha256: str, events_sha256: str, dev_record_sha256: str,
                    created: str) -> dict:
    from src import ranking as rk
    from src.ranking_eval import KS, PRIMARY_K
    configurations = list(rk.SIMPLICITY)
    return {
        "run_id": RUN_ID, "created": created,
        "source_commit": source_commit,
        "freeze_tag": FREEZE_TAG,
        "commits": "source_commit is the code before this file; the freeze tag marks the commit that adds it",
        "labels_sha256": labels_sha256, "events_sha256": events_sha256, "dev_record_sha256": dev_record_sha256,
        "threshold": (dev_record.get("threshold") or {}).get("chosen"),
        "experiment": dev_record.get("experiment"), "llm": llm_identity,
        "compare_choice": compare_choice,
        "eligible": eligibility["eligible"], "dev_winner": eligibility["dev_winner"],
        "dev_runs_passed": eligibility.get("dev_runs_passed", {}),
        "configurations": configurations,
        "experimental": [name for name in configurations if name not in eligibility["eligible"]],
        "ks": list(KS), "primary_k": PRIMARY_K,
        "acceptance": {"rule": "src/ranking.py acceptance(), the same for every configuration",
                       "max_model_seconds": rk.MAX_MODEL_SECONDS,
                       "runtime_from": "the first collection pass only",
                       "min_first_pass_coverage": MIN_MODEL_COVERAGE,
                       "fallback": "an article without a model answer keeps its O1 place"},
        "default_rule": "The development winner becomes the default only if it passes the acceptance rule on "
                        "the fresh run; otherwise the default is the baseline. Configurations outside the "
                        "eligible set are reported as experimental evidence only.",
        "collection": {"max_passes": MAX_COLLECTION_PASSES,
                       "asks": "only articles without a saved model answer",
                       "saved_answers": "never asked again",
                       "cache_identity": "company, model, prompt version, title, passage, company scope and "
                                         "provider settings (src/relevance_check.py, _key)"},
        "packages": package_versions(),
    }

# ---------- commands ----------

def _raw_articles(state: dict) -> list[dict]:
    from src.ranking_eval import merge_run
    files = state["stages"]["fetch"]["raw_files"]
    return merge_run([json.loads((RAW_DIR / files[name]).read_text(encoding="utf-8")) for name in SEARCHES])


def _dev_articles(state: dict) -> list[dict]:
    articles = []
    for name in state["stages"]["fetch"]["seen_files"]:
        articles += json.loads((RAW_DIR / name).read_text(encoding="utf-8")).get("articles", [])
    return articles


def _need(state: dict, stage: str) -> dict:
    if not state["stages"].get(stage, {}).get("done"):
        raise FreshEvalError(f"Run the '{stage}' stage first.")
    return state["stages"][stage]


def _fetch_command() -> None:
    from src.news_fetcher import fetch_news, save_raw
    from src.query import load_aliases
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    state = fetch_stage(fetch_news, datetime.now(timezone.utc), RUN_DIR, aliases, save_raw)
    stage = state["stages"]["fetch"]
    for name, file in stage["raw_files"].items():
        articles = json.loads((RAW_DIR / file).read_text(encoding="utf-8"))["articles"]
        counts = day_counts(articles)
        stage.setdefault("day_counts", {})[name] = counts
        print(f"Search {name}: {len(articles)} articles -> data/raw/{file}")
        print("   per day (UTC): " + ", ".join(f"{day[5:]} {n}" for day, n in counts.items()))
        if not counts[max(counts)]:
            print(f"   NOTE: no article on the last day ({max(counts)}); check before going on.")
    save_state(RUN_DIR, state)


def _overlap_command() -> None:
    state = load_state(RUN_DIR)
    _need(state, "fetch")
    result = overlap(_raw_articles(state), _dev_articles(state))
    state["stages"]["overlap"] = {"done": True, "kept": [a["url"] for a in result["kept"]],
                                  "excluded": result["excluded"], "review": result["review"]}
    save_state(RUN_DIR, state)
    if result["review"] and not (RUN_DIR / REVIEW_FILE).exists():
        with open(RUN_DIR / REVIEW_FILE, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerows({"row": n, "decision": "", **pair} for n, pair in enumerate(result["review"], start=1))
    print(f"Kept {len(result['kept'])}, excluded {len(result['excluded'])}: "
          f"{sorted({e['reason'] for e in result['excluded']})}")
    if result["review"]:
        print(f"{len(result['review'])} near-identical title(s) to review -> {RUN_DIR / REVIEW_FILE}\n"
              f"Write `same` only for the same article or a repost, otherwise `different`.")


def _kept_articles(state: dict) -> list[dict]:
    stage = _need(state, "overlap")
    result = {"kept": [a for a in _raw_articles(state) if a["url"] in set(stage["kept"])],
              "excluded": stage["excluded"], "review": stage["review"]}
    decisions = []
    if result["review"]:
        with open(RUN_DIR / REVIEW_FILE, encoding="utf-8-sig", newline="") as handle:
            decisions = list(csv.DictReader(handle))
    return apply_review(result, decisions)["kept"]


def progress_order(articles: list[dict]) -> list[dict]:
    """A shuffled copy for the progress list, so it shows nothing about the searches; the fetch order is kept."""
    shown = list(articles)
    random.Random(PROGRESS_SEED).shuffle(shown)
    return shown


def label_sheet_is_empty(path) -> bool:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return not any((row.get("label") or "").strip() for row in csv.DictReader(handle))


def build_sheet(articles: list[dict], results: dict[str, dict], company: str, aliases: dict | None) -> dict:
    """Rows, reading-page items and the evidence of each article: the text shown is the text the tool uses."""
    from src.fulltext import SEED, _paragraphs, article_id
    from src.fulltext_eval import usable_full_texts
    fetched = {a["url"]: results[a["url"]].get("text", "") for a in articles
               if results[a["url"]].get("status") == "ok"}
    texts, fallbacks = usable_full_texts(articles, fetched, company, aliases)
    ordered = list(articles)
    random.Random(SEED).shuffle(ordered)
    rows, items, evidence = [], [], {}
    for number, a in enumerate(ordered, start=1):
        key, result = article_id(a["url"]), results[a["url"]]
        if a["url"] in texts:
            text = texts[a["url"]]
            status = "complete" if result.get("html_cleaning") == "done" else "older text, page cleaning unavailable"
            evidence[key] = "full text"
        else:
            text = "\n".join([a.get("description") or "", (a.get("content") or "")[:200]])
            if a["url"] not in fetched:
                status, evidence[key] = STATUS_NOT_FETCHED, "short text: page not fetched"
            elif "teaser" in fallbacks.get(a["url"], []):
                status, evidence[key] = STATUS_TEASER, "short text: teaser"
            else:
                status, evidence[key] = STATUS_ANOTHER_STORY, "short text: the page holds another story"
        rows.append({"row": number, "article_id": key, "title": a["title"], "url": a["url"],
                     "label": "", "unsure": "", "note": ""})
        items.append({"row": number, "article_id": key, "url": a["url"], "title": a["title"],
                      "source": (a.get("source") or {}).get("name", ""), "published": a.get("publishedAt", ""),
                      "status": status, "paragraphs": _paragraphs(text)})
    return {"rows": rows, "items": items, "evidence": evidence}


def _sheet_command(company: str = "Tata Motors") -> None:
    from src.fulltext import FULLTEXT_DIR, LABEL_COLUMNS, FullTextFetcher, labelling_page
    from src.query import load_aliases
    if (RUN_DIR / LABEL_FILE).exists() and not label_sheet_is_empty(RUN_DIR / LABEL_FILE):
        raise FreshEvalError(f"{LABEL_FILE} already holds labels; it is not overwritten.")
    state = load_state(RUN_DIR)
    articles = _kept_articles(state)
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    fetcher = FullTextFetcher()
    results = {}
    for number, article in enumerate(progress_order(articles), start=1):
        results[article["url"]] = fetcher.fetch(article["url"])
        print(f"{number:>3}/{len(articles)}  {results[article['url']]['status']:<18} {article['title'][:70]}")
    sheet = build_sheet(articles, results, company, aliases)
    with open(RUN_DIR / LABEL_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(sheet["rows"])
    page = FULLTEXT_DIR / f"labelling_view_{RUN_ID}.html"           # publisher text: never in Git
    page.write_text(labelling_page(sheet["items"]), encoding="utf-8")
    state["stages"]["sheet"] = {"done": True, "article_ids": [r["article_id"] for r in sheet["rows"]],
                                "evidence": sheet["evidence"], "notes": RUN_NOTES}
    save_state(RUN_DIR, state)
    counts = {}
    for kind in sheet["evidence"].values():
        counts[kind] = counts.get(kind, 0) + 1
    print(f"\n{len(sheet['rows'])} articles to label -> {RUN_DIR / LABEL_FILE}\nReading page: {page}")
    print("Evidence: " + ", ".join(f"{kind} {n}" for kind, n in sorted(counts.items())))


def _read_labels() -> dict[str, str]:
    with open(RUN_DIR / LABEL_FILE, encoding="utf-8-sig", newline="") as handle:
        return {row["article_id"]: (row.get("label") or "").strip().lower() for row in csv.DictReader(handle)}


def _events_command() -> None:
    from src.fulltext import article_id
    from src.summary_run import EVENT_COLUMNS, relevant_event_rows
    state = load_state(RUN_DIR)
    _need(state, "sheet")
    labels = _read_labels()
    bad = [key for key, label in labels.items() if label not in VALID_LABELS]
    if bad:
        raise FreshEvalError(f"Missing or invalid labels in {LABEL_FILE}: {', '.join(bad)}")
    articles = _kept_articles(state)
    by_url = {a["url"]: labels[article_id(a["url"])] for a in articles}
    if (RUN_DIR / EVENTS_FILE).exists():
        raise FreshEvalError(f"{EVENTS_FILE} already exists; it is not overwritten.")
    rows = relevant_event_rows(articles, by_url)
    with open(RUN_DIR / EVENTS_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    state["stages"]["events"] = {"done": True}
    save_state(RUN_DIR, state)
    print(f"{len(rows)} relevant articles to group into events -> {RUN_DIR / EVENTS_FILE}")


def _candidates(state: dict, company: str, aliases: dict | None) -> list[dict]:
    """Stage 1 on the run's kept articles, with the saved full texts, as the app sees them."""
    from src.fulltext import FullTextFetcher
    from src.ranking_eval import stage1_candidates
    articles = _kept_articles(state)
    fetcher, full_texts = FullTextFetcher(), {}
    for a in articles:
        cached = fetcher._cache_path(a["url"])
        if cached.exists():
            result = json.loads(cached.read_text(encoding="utf-8"))
            if result.get("status") == "ok":
                full_texts[a["url"]] = result["text"]
    return stage1_candidates(articles, full_texts, company, aliases)


def llm_from_experiment(experiment: dict):
    """The exact model setting of a record: provider, model and generation settings (no .env defaults)."""
    from src import llm as llm_module
    return llm_module.make_llm(experiment["provider"], model=experiment["model"],
                               generation=experiment["generation"])


def _load_protocol() -> dict:
    path = RUN_DIR / PROTOCOL_FILE
    if not path.exists():
        raise FreshEvalError(f"{PROTOCOL_FILE} is missing. Run: python -m src.fresh_eval protocol")
    return json.loads(path.read_text(encoding="utf-8"))


def _frozen_and_unchanged() -> tuple[dict, dict]:
    """The freeze tag and the protocol are checked before any answer or result is read."""
    freeze = freeze_check()
    if not freeze["ok"]:
        raise FreshEvalError("The frozen experiment has changed since the tag: " + ", ".join(freeze["problems"]))
    protocol = _load_protocol()
    problems = protocol_problems(protocol, RUN_DIR / LABEL_FILE, RUN_DIR / EVENTS_FILE)
    if problems:
        raise FreshEvalError("; ".join(problems))
    return freeze, protocol


def _protocol_command(company: str = "Tata Motors") -> None:
    from src.experiment import build_record, record_problems
    from src.ranking_eval import ranking_settings
    from src.summary_run import load_events
    path = RUN_DIR / PROTOCOL_FILE
    if path.exists():
        raise FreshEvalError(f"{PROTOCOL_FILE} already exists; it is not overwritten.")
    if not DEV_RECORD.exists():
        raise FreshEvalError("The development record is missing. Run: python -m src.ranking_eval --check")
    dev_record = json.loads(DEV_RECORD.read_text(encoding="utf-8"))
    if not dev_record.get("experiment"):
        raise FreshEvalError("The development record has no model run. Run: python -m src.ranking_eval --check")
    state = load_state(RUN_DIR)
    _need(state, "sheet")
    _need(state, "events")
    bad = [key for key, label in _read_labels().items() if label not in VALID_LABELS]
    if bad:
        raise FreshEvalError(f"Missing or invalid labels in {LABEL_FILE}: {', '.join(bad)}")
    load_events(RUN_DIR / EVENTS_FILE)                               # every row needs an event
    changed = [line[3:].strip() for line in git("status", "--porcelain").splitlines()
               if _frozen(line[3:].strip()) and not line[3:].strip().endswith(PROTOCOL_FILE)]
    if changed:
        raise FreshEvalError("Commit these files first, so the source commit holds them: " + ", ".join(changed))
    compare = json.loads((EVAL_DIR / "model_compare" / "report.json").read_text(encoding="utf-8"))
    llm = llm_from_experiment(dev_record["experiment"])
    threshold = (dev_record.get("threshold") or {}).get("chosen")
    problems = record_problems(dev_record["experiment"], build_record(llm, ranking_settings(threshold), None))
    if problems:
        raise FreshEvalError("The model settings now differ from the development record:\n  "
                             + "\n  ".join(problems))
    eligibility = eligible_set(dev_record, compare.get("choice"))
    protocol = protocol_record(
        source_commit=git("rev-parse", "HEAD"), dev_record=dev_record, eligibility=eligibility,
        compare_choice=compare.get("choice"),
        llm_identity={"provider": llm.provider, "model": llm.model_name, "cache_tag": llm.cache_tag},
        labels_sha256=file_fingerprint(RUN_DIR / LABEL_FILE), events_sha256=file_fingerprint(RUN_DIR / EVENTS_FILE),
        dev_record_sha256=file_fingerprint(DEV_RECORD), created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved: {path}")
    print(f"Eligible (development only): {protocol['eligible']}; development winner: {protocol['dev_winner']}")
    print(f"Experimental (evidence only): {protocol['experimental']}")
    print(f"Model: {llm.provider} {llm.model_name}; S2 threshold {threshold}; source commit {protocol['source_commit']}")
    print(f"Next: commit {PROTOCOL_FILE}, then: git tag {FREEZE_TAG}")


def _collect_command(company: str = "Tata Motors") -> None:
    from src.experiment import build_record, record_problems
    from src.query import load_aliases
    from src.ranking_eval import CHECK_CACHE, cache_hits, query_items, ranking_settings
    from src.relevance_check import WORKERS, CheckCache, classify, company_scope
    freeze, protocol = _frozen_and_unchanged()
    state = load_state(RUN_DIR)
    _need(state, "events")
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    items = query_items(_candidates(state, company, aliases), company, aliases)
    scope, cache = company_scope(company, aliases), CheckCache(CHECK_CACHE)
    llm = llm_from_experiment(protocol["experiment"])
    problems = record_problems(protocol["experiment"], build_record(llm, ranking_settings(protocol["threshold"]),
                                                                    freeze["head"]))
    if llm.model_name != protocol["llm"]["model"] or llm.cache_tag != protocol["llm"]["cache_tag"]:
        problems.append("the model or its provider settings differ from the protocol")
    if problems:
        raise FreshEvalError("The model settings differ from the frozen protocol:\n  " + "\n  ".join(problems))
    saved = set(cache_hits(items, llm, company, scope, cache))
    collect = state["stages"].get("collect")
    number = start_pass(RUN_DIR, datetime.now(timezone.utc),
                        cached_before=len(saved) if collect is None else collect["cached_before"])
    missing = [item for item in items if item["id"] not in saved]
    print(f"Pass {number} of {MAX_COLLECTION_PASSES}: asking {len(missing)} of {len(items)} candidates "
          f"({len(saved)} already answered)", flush=True)
    started = time.perf_counter()
    answers = classify(missing, llm, company, cache=cache, scope=scope, workers=WORKERS,
                       progress=lambda done, total, seconds: print(f"   batch {done}/{total}", flush=True))
    seconds = time.perf_counter() - started
    missing_after = len(items) - len(cache_hits(items, llm, company, scope, cache))
    stats = getattr(llm, "stats", None) or {}
    record = pass_record([item["id"] for item in missing], answers, missing_after, seconds, stats)
    record.update(provider=llm.provider, model=llm.model_name,
                  llm_stats={name: value for name, value in stats.items() if name != "per_call"})
    finish_pass(RUN_DIR, number, record)
    print(f"Pass {number}: answered {record['answered']} of {record['asked']} asked; still missing {missing_after}"
          + (f"; failures {record['failures']}" if record["failures"] else ""))
    if missing_after == 0:
        print("Every answer is in. Next: python -m src.fresh_eval score")
    elif number >= MAX_COLLECTION_PASSES:
        print("The budget of passes is used. The score counts the missing answers as fallbacks. "
              "Next: python -m src.fresh_eval score")
    else:
        print("Run this command again later; it asks only for the missing answers.")


def _score_command(company: str = "Tata Motors") -> None:
    from src.fulltext import article_id
    from src.query import load_aliases
    from src.ranking_eval import CHECK_CACHE, evaluate_run, query_items
    from src.relevance_check import CheckCache, _key, company_scope
    from src.summary_run import load_events
    if (RUN_DIR / RECORD_FILE).exists():
        raise FreshEvalError("The fresh run has already been scored; it is scored once only.")
    freeze, protocol = _frozen_and_unchanged()
    state = load_state(RUN_DIR)
    sheet = _need(state, "sheet")
    _need(state, "events")
    collect = state["stages"].get("collect")
    if not collect or not collection_complete(collect):
        used = len((collect or {}).get("passes", []))
        raise FreshEvalError(f"The collection is not finished ({used} of {MAX_COLLECTION_PASSES} passes used). "
                             "Run: python -m src.fresh_eval collect")
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    candidates = _candidates(state, company, aliases)
    labels = _read_labels()
    events = {article_id(url): event for url, event in load_events(RUN_DIR / EVENTS_FILE).items()}
    problems = validation_problems(candidates, labels, events, set(sheet["article_ids"]), protocol["threshold"],
                                   {a.get("url") or "" for a in _dev_articles(state)})
    if problems:
        raise FreshEvalError("The fresh run cannot be scored:\n  " + "\n  ".join(problems))
    items = query_items(candidates, company, aliases)
    scope, cache = company_scope(company, aliases), CheckCache(CHECK_CACHE)
    saved = {}
    for item in items:                                   # saved answers only: the score asks the model nothing
        value = cache.get(_key(item, company, protocol["llm"]["model"], scope, protocol["llm"]["cache_tag"]))
        if value is not None and value.get("source") == "model":
            saved[item["id"]] = value
    answers = final_answers([item["id"] for item in items], saved, collect)
    for c in candidates:
        c["check"] = answers[c["key"]]
    runtime = first_pass_runtime(collect, len(items))
    report = evaluate_run(candidates, labels, events, threshold=protocol["threshold"],
                          ks=tuple(protocol["ks"]), primary_k=protocol["primary_k"],
                          seconds=runtime["seconds_for_rule"])
    default = default_configuration(protocol["dev_winner"], report)
    failures = {}
    for entry in collect["passes"]:
        for kind, n in (entry.get("failures") or {}).items():
            failures[kind] = failures.get(kind, 0) + n
    record = {"run_id": RUN_ID, "window": [WINDOW_FROM, WINDOW_TO], "freeze": freeze,
              "protocol_sha256": file_fingerprint(RUN_DIR / PROTOCOL_FILE),
              "default": default, "dev_winner": protocol["dev_winner"], "eligible": protocol["eligible"],
              "experimental": protocol["experimental"],
              "answers": {"candidates": len(items), "model_answers": len(saved), "fallbacks": len(items) - len(saved),
                          "saved_before_collection": collect.get("cached_before", 0),
                          "collected_by_pass": [entry.get("answered") for entry in collect["passes"]]},
              "reliability": {"passes": collect["passes"], "failures_all_passes": failures},
              "first_pass_runtime": runtime,
              "evidence": sheet.get("evidence", {}), "notes": sheet.get("notes", []),
              "package_versions": package_versions(),
              "report": {**report, "configurations": {c: {str(k): m for k, m in by_k.items()}
                                                      for c, by_k in report["configurations"].items()}}}
    (RUN_DIR / RECORD_FILE).write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str),
                                       encoding="utf-8")
    print(f"Fresh run {RUN_ID}: {report['candidates']} candidates, {report['relevant_candidates']} relevant; "
          f"model answers {len(saved)} of {len(items)}")
    print(f"First pass: {runtime['seconds']} s, coverage {runtime['coverage']}, {runtime['kind']} "
          f"-> runtime condition: {runtime['reason']}")
    for config, verdict in report["acceptance"].items():
        failed = [c for c, ok in verdict["conditions"].items() if not ok]
        e = verdict["events"]
        role = "eligible" if config in protocol["eligible"] else "experimental"
        print(f"   {config} ({role}): {'PASS' if verdict['passed'] else 'fail'}  gained {e['gained']} "
              f"lost {e['lost']}  direct gained {e['direct_gained']} lost {e['direct_lost']}"
              + (f"  failed: {failed}" if failed else ""))
    print(f"Development winner: {protocol['dev_winner']} -> default: {default}")
    print(f"Saved: {RUN_DIR / RECORD_FILE}  (commit {freeze['head']}, tag {freeze['tag_commit']})")


COMMANDS = {"fetch": _fetch_command, "overlap": _overlap_command, "sheet": _sheet_command,
            "protocol": _protocol_command, "collect": _collect_command,
            "events": _events_command, "score": _score_command}

if __name__ == "__main__":
    command = COMMANDS.get(sys.argv[1] if len(sys.argv) > 1 else "")
    if command is None:
        print(__doc__)
    else:
        try:
            command()
        except FreshEvalError as error:
            print(f"[STOP] {error}")
            sys.exit(1)
