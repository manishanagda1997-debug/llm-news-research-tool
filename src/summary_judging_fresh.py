"""
Judging and scoring of the fresh-week summary (SUMMARY_JUDGING_GUIDE_PART6.md): the frozen summary design on the
week of 24 to 30 October, made once by python -m src.summary_fresh. The Part 4 sheet code is reused; nothing
is carried over from an earlier sheet. The development locks and scores are read, never written.

sheet   every item of the summary, shuffled, in summary_v2_fresh_judging.csv; one row per relevant event among
        the selected articles in summary_v2_fresh_events.csv (from the fresh week's event sheet); one row in
        summary_v2_fresh_industry.csv; the key (repairs only there); the reading page, industry evidence first
lock    every mark present and valid; then the fingerprints of the three sheets, the key, the guides (Parts 1 to
        3, 4 and 6), the freeze record, the fresh record, the saved passages, the fresh labels and events, the
        Part 4 development score (read by the score), and the code that builds the sheets and the score
        (CODE_FILES)
score   refuses when any of those changed after the lock; then the Part 6 pass rule

Run:
    python -m src.summary_judging_fresh sheet
    python -m src.summary_judging_fresh lock
    python -m src.summary_judging_fresh score
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src import summary_judging as sj
from src import summary_judging_p4 as p4
from src.config import EVAL_DIR

SCORING_VERSION = "1"
SEED = 46
MAX_CONTRADICTED = 0.05

ITEMS_FILE = "summary_v2_fresh_judging.csv"
EVENTS_FILE = "summary_v2_fresh_events.csv"
INDUSTRY_FILE = "summary_v2_fresh_industry.csv"
KEY_FILE = "summary_v2_fresh_judging_key.json"
LOCK_FILE = "summary_v2_fresh_judging_lock.json"
SCORE_FILE = "summary_v2_fresh_judging_score.json"
PAGE_FILE = "summary_v2_fresh_judging.html"              # in data/fulltext/, outside Git
GUIDE_FILES = (sj.GUIDE_FILE, p4.GUIDE_FILE, "SUMMARY_JUDGING_GUIDE_PART6.md")
CODE_FILES = ("src/summary_judging_fresh.py", "src/summary_judging_p4.py", "src/summary_judging.py",
              "src/summary_fresh.py", "src/summary_v2.py", "src/fresh_eval.py", "src/evaluate.py",
              "src/summary_run.py", "src/preprocess.py")


class JudgingError(sj.JudgingError):
    pass


# ---------- the sheets ----------

def build_sheets(record: dict, passages: dict, events: dict, labels: dict, seed: int = SEED) -> dict:
    """The Part 4 sheets for the one fresh summary, with no carry-over."""
    if len(record["records"]) != 1:
        raise JudgingError("The fresh record must hold exactly one summary.")
    if not sj._produced(record):
        raise JudgingError("The fresh record has no summary: there is no summary to judge.")
    return p4.build_sheets(record, passages, events, labels, {}, seed=seed)


def no_summary_result(record: dict) -> dict:
    """When the run gives no summary, the result is a failed check and nothing is judged."""
    return {"checks": {"summary_produced": False}, "passes": False,
            "failures": record["records"][0]["result"].get("failures", [])}


# ---------- the lock ----------

def _code_files() -> list[Path]:
    from src.config import ROOT_DIR
    return [ROOT_DIR / name for name in CODE_FILES]


def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in _code_files():
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def lock_record(items, events, industry, key, inputs: dict) -> dict:
    found = p4.problems(items, events, industry, key)
    if found:
        raise JudgingError("The sheets are not complete:\n" + "\n".join(found))
    return {"items_fingerprint": sj.sheet_fingerprint(items), "events_fingerprint": sj.sheet_fingerprint(events),
            "industry_fingerprint": sj.sheet_fingerprint(industry), "key_fingerprint": sj._hash(key),
            "code_fingerprint": _code_fingerprint(), "inputs": inputs, "scoring_version": SCORING_VERSION,
            "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def lock_problems(lock: dict, items, events, industry, key, inputs: dict) -> list[str]:
    """Everything that changed after the lock; empty when the score may go ahead."""
    found = [f"the {name} sheet changed after the lock" for name, rows in
             (("items", items), ("events", events), ("industry", industry))
             if lock[f"{name}_fingerprint"] != sj.sheet_fingerprint(rows)]
    if lock["key_fingerprint"] != sj._hash(key):
        found.append("the key changed after the lock")
    if lock["code_fingerprint"] != _code_fingerprint():
        found.append("the judging code changed after the lock")
    found += [f"{name} changed after the lock" for name in sorted(set(lock["inputs"]) | set(inputs))
              if lock["inputs"].get(name) != inputs.get(name)]
    return found


# ---------- the score ----------

def contradiction_ok(contradicted: int, total: int) -> bool:
    """Contradicted items / all judged items, unrounded, at most 5%."""
    return total > 0 and contradicted / total <= MAX_CONTRADICTED


def score(items, events, industry, key, record, development: dict, run: str) -> dict:
    """The Part 6 pass rule. The Part 4 score gives the counts; its tolerance check is reported only."""
    base = p4.score(items, events, industry, key, record, [], development, [run])
    contradicted = [r for r in items if sj._value(r["contradicted"]) == "yes"]
    central = [r for r in contradicted if sj._value(r["central"]) == "yes"]
    checks = {
        "summary_produced": base["checks"]["every_development_run_has_one_summary"],
        "event_coverage_at_least_80_percent": base["checks"]["event_coverage"],
        "contradicted_at_most_5_percent": contradiction_ok(len(contradicted), len(items)),
        "no_central_contradiction": not central,
        "no_unsupported_industry_point": base["checks"]["no_unsupported_industry_point"],
        "industry_section_matches_evidence": base["checks"]["empty_section_only_without_evidence"]
        and base["checks"]["industry_points_only_with_evidence"],
    }
    kb = base["key_points_and_briefs"]
    dev = f"{development['supported']}/{development['total']} ({development['supported'] / development['total']:.1%})"
    return {"total": base["total"], "per_run": base["per_run"],
            "contradicted": {"count": len(contradicted), "total": len(items),
                             "share": len(contradicted) / len(items) if items else None},
            "central_contradicted": len(central),
            "key_points_and_briefs": {"supported": kb["supported"], "total": kb["total"], "share": kb["share"],
                                      "development": dev},
            "reported_only": {"within_development_tolerance":
                              base["checks"]["key_points_and_briefs_within_tolerance"],
                              "note": "one summary gives few items; no pass level and no statistical comparison"},
            "checks": checks, "passes": all(checks.values()),
            "rule": {"min_event_coverage": p4.MIN_EVENT_COVERAGE, "max_contradicted": MAX_CONTRADICTED,
                     "contradicted": "contradicted items / all judged items, unrounded"}}


# ---------- files and commands ----------

def _record_and_inputs():
    from src.fresh_eval import file_fingerprint
    from src.summary_freeze import EVAL_DIR as FREEZE_DIR, FREEZE_FILE
    from src.summary_fresh import FRESH_FILE, FRESH_RUN, fresh_paths
    paths = fresh_paths()
    record_path = paths["record"]
    if not record_path.exists():
        raise JudgingError(f"{FRESH_FILE} does not exist: run python -m src.summary_fresh first.")
    inputs = {"guides": [file_fingerprint(EVAL_DIR / g) for g in GUIDE_FILES],
              "freeze": file_fingerprint(FREEZE_DIR / FREEZE_FILE), "record": file_fingerprint(record_path),
              "passages": file_fingerprint(paths["passages"]),
              "development_score": file_fingerprint(EVAL_DIR / p4.SCORE_FILE),
              "labels": file_fingerprint(EVAL_DIR / FRESH_RUN / "labels.csv"),
              "events_data": file_fingerprint(EVAL_DIR / FRESH_RUN / "events.csv")}
    return json.loads(record_path.read_text(encoding="utf-8")), inputs


def passages_from_record(record: dict, saved: list[dict]) -> dict:
    """{article key: passage} from the passages saved by the run: their keys, IDs and order must equal the record's
    selection, and each must match its fingerprint in the record. Nothing is rebuilt."""
    from src.summary_fresh import passage_entry
    articles = record["records"][0]["articles"]
    prints = record.get("passage_fingerprints") or []
    if [(p["id"], p["key"]) for p in saved] != [(a["id"], a["key"]) for a in articles] or \
            [(p["id"], p["key"]) for p in prints] != [(a["id"], a["key"]) for a in articles]:
        raise JudgingError("The saved passages are not the record's selection (IDs or order differ).")
    for entry, expected in zip(saved, prints):
        if passage_entry(entry)["sha256"] != expected["sha256"] or entry.get("sha256") != expected["sha256"]:
            raise JudgingError(f"The saved passage of {entry['key']} differs from the one the model received.")
    return {p["key"]: p["description"] for p in saved}


def _passages(record: dict) -> dict:
    from src.summary_fresh import fresh_paths
    path = fresh_paths()["passages"]
    if not path.exists():
        raise JudgingError(f"{path} does not exist: the evidence the model received is missing.")
    return passages_from_record(record, json.loads(path.read_text(encoding="utf-8")))


def _sheet_command() -> None:
    from src.evaluate import load_label_set
    from src.fulltext import FULLTEXT_DIR
    from src.summary_run import load_events
    from src.summary_fresh import FRESH_RUN
    record, _ = _record_and_inputs()
    if not sj._produced(record):
        print(json.dumps(no_summary_result(record), indent=1))
        raise JudgingError("The fresh run gives no summary: the check 'summary produced' fails; nothing to judge.")
    sheets = build_sheets(record, _passages(record), load_events(EVAL_DIR / FRESH_RUN / "events.csv"),
                          load_label_set([EVAL_DIR / FRESH_RUN / "labels.csv"]))
    files = {EVAL_DIR / ITEMS_FILE: sj._csv_text(sheets["items"], sj.ITEM_COLUMNS),
             EVAL_DIR / EVENTS_FILE: sj._csv_text(sheets["events"], sj.EVENT_COLUMNS),
             EVAL_DIR / INDUSTRY_FILE: sj._csv_text(sheets["industry"], p4.INDUSTRY_COLUMNS),
             EVAL_DIR / KEY_FILE: json.dumps(sheets["key"], indent=1),
             Path(FULLTEXT_DIR) / PAGE_FILE: sheets["page"]}
    for path in files:
        if path.exists():
            raise JudgingError(f"{path.name} already exists; it is not overwritten.")
    for path, text in files.items():
        sj._write_new(path, text)
    print(f"Items: {len(sheets['items'])} rows in data/eval/{ITEMS_FILE} (all judged from scratch)")
    print(f"Events: {len(sheets['events'])} rows in data/eval/{EVENTS_FILE}")
    print(f"Industry evidence: 1 row in data/eval/{INDUSTRY_FILE}")
    print(f"Reading page: {Path(FULLTEXT_DIR) / PAGE_FILE}  (open it in your browser)")
    print(f"Key: data/eval/{KEY_FILE}  (do not open it until the sheets are judged and locked)")


def _sheets():
    return tuple(sj._read_csv(EVAL_DIR / name) for name in (ITEMS_FILE, EVENTS_FILE, INDUSTRY_FILE))


def _lock_command() -> None:
    if (EVAL_DIR / LOCK_FILE).exists():
        raise JudgingError(f"{LOCK_FILE} already exists: the sheets are locked.")
    _, inputs = _record_and_inputs()
    key = sj._load_json(EVAL_DIR / KEY_FILE)
    lock = lock_record(*_sheets(), key, inputs)
    sj._write_new(EVAL_DIR / LOCK_FILE, json.dumps(lock, indent=1))
    print(f"Locked: data/eval/{LOCK_FILE}")


def _score_command() -> None:
    from src.summary_fresh import FRESH_RUN
    lock = sj._load_json(EVAL_DIR / LOCK_FILE)
    record, inputs = _record_and_inputs()
    items, events, industry = _sheets()
    key = sj._load_json(EVAL_DIR / KEY_FILE)
    found = lock_problems(lock, items, events, industry, key, inputs)
    if found:
        raise JudgingError("The fresh score cannot be made:\n" + "\n".join(found))
    development = sj._load_json(EVAL_DIR / p4.SCORE_FILE)["key_points_and_briefs"]
    result = {"lock": lock, **score(items, events, industry, key, record,
                                    {"supported": development["supported"], "total": development["total"]},
                                    FRESH_RUN)}
    sj._write_new(EVAL_DIR / SCORE_FILE, json.dumps(result, indent=1))
    t, kb, c = result["total"], result["key_points_and_briefs"], result["contradicted"]
    print(f"Industry: produced {t['industry']['produced']}, supported {t['industry']['supported']}, partly "
          f"{t['industry']['partly_supported']}, not {t['industry']['not_supported']} | events "
          f"{t['events']['covered']}/{t['events']['total']}")
    print(f"Contradicted: {c['count']}/{c['total']}  central: {result['central_contradicted']}")
    print(f"Key points and briefs (reported only): {kb['supported']}/{kb['total']} "
          f"({kb['share'] * 100:.1f}%); development {kb['development']}")
    for check, ok in result["checks"].items():
        print(f"  {'pass' if ok else 'FAIL'}  {check}")
    print(f"Passes: {result['passes']}")
    print(f"Saved: data/eval/{SCORE_FILE}")


COMMANDS = {"sheet": _sheet_command, "lock": _lock_command, "score": _score_command}

if __name__ == "__main__":
    try:
        if len(sys.argv) == 2 and sys.argv[1] in COMMANDS:
            COMMANDS[sys.argv[1]]()
        else:
            print(__doc__)
    except sj.JudgingError as error:
        print(f"[STOP] {error}")
        sys.exit(1)
