"""
Judging of the prompt version 5 rerun (SUMMARY_JUDGING_GUIDE_PART5.md): the direct design on the three
development runs, judged and scored with Part 4 (src/summary_judging_p4.py and SUMMARY_JUDGING_GUIDE_PART4.md)
unchanged. The locked comparison (Part 3) and the locked Part 4 check are read, never written.

What differs from Part 4:
    industry evidence  not judged again: the selected articles are the same, so each run keeps its locked Part 4
                       decision (carried in the key by run)
    carried items      an item keeps a locked judgement only when its section, exact text and cited articles match
                       a direct item of the locked Part 3 or Part 4 sheet; when the two give different judgements,
                       it is judged again
    files              summary_v2_p5_*; the reading page has the items and the event coverage only

sheet <record>  the items sheet, the event sheet, the key and the reading page
lock            both earlier locks still hold; every mark present and valid; then the fingerprints of both sheets,
                the key, the guides, the record and the judging code (this module, src/summary_judging_p4.py and
                src/summary_judging.py) are saved
score           refuses without a lock, or when a sheet, the key or the judging code changed after it; then the
                Part 4 pass rule, in total and per run

Run:
    python -m src.summary_judging_p5 sheet summary_v2_dev_k5_c7_p5.json
    python -m src.summary_judging_p5 lock
    python -m src.summary_judging_p5 score
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
SEED = 45

ITEMS_FILE = "summary_v2_p5_judging.csv"
EVENTS_FILE = "summary_v2_p5_events.csv"
KEY_FILE = "summary_v2_p5_judging_key.json"
LOCK_FILE = "summary_v2_p5_judging_lock.json"
SCORE_FILE = "summary_v2_p5_judging_score.json"
PAGE_FILE = "summary_v2_p5_judging.html"                 # in data/fulltext/, outside Git
GUIDE_FILES = (p4.GUIDE_FILE, "SUMMARY_JUDGING_GUIDE_PART5.md")


class JudgingError(sj.JudgingError):
    pass


# ---------- the locked judgements of Part 3 and Part 4 ----------

def locked_sources(p3_items, p3_key, p4_items, p4_key) -> tuple[dict, list[dict]]:
    """{signature: "p3:<row>" or "p4:<row>"} for the direct items of both locked sheets, and one combined sheet
    with those row names. A signature with different judgements (within or across the two) is left out."""
    sheet, found, clash = [], {}, set()
    for name, items, key in (("p3", p3_items, p3_key), ("p4", p4_items, p4_key)):
        rows = sj._rows_by_number(items)
        for number, entry in key["items"].items():
            if entry["design"] != p4.DESIGN:
                continue
            row = dict(rows[number], row=f"{name}:{number}")
            sheet.append(row)
            sig = p4.signature(entry["section"], row["text"], row["evidence"])
            if sig in found:
                earlier = next(r for r in sheet if r["row"] == found[sig])
                if sj._value(earlier["judgement"]) != sj._value(row["judgement"]):
                    clash.add(sig)
                continue
            found[sig] = row["row"]
    return {sig: name for sig, name in found.items() if sig not in clash}, sheet


def carried_evidence(p4_industry: list[dict], p4_key: dict) -> dict[str, dict]:
    """{run: {"industry_evidence", "note", "p4_row"}} from the locked Part 4 industry sheet."""
    return {p4_key["summaries"][row["summary"]]["run"]: {"industry_evidence": sj._value(row["industry_evidence"]),
                                                        "note": row["note"], "p4_row": str(row["row"]).strip()}
            for row in p4_industry}


# ---------- the sheets ----------

def build_sheets(record: dict, passages: dict, events: dict, labels: dict, locked: dict, evidence: dict,
                 seed: int = SEED) -> dict:
    sheets = p4.build_sheets(record, passages, events, labels, locked, seed=seed)
    runs = {entry["run"] for entry in sj._produced(record)}
    if not runs <= set(evidence):
        raise JudgingError(f"No locked Part 4 evidence decision for {sorted(runs - set(evidence))}.")
    key = dict(sheets["key"])
    key.pop("industry")
    key["industry_carried"] = {run: evidence[run] for run in sorted(runs)}
    return {"items": sheets["items"], "events": sheets["events"], "key": key, "page": _page(sheets["page"])}


def _page(page: str) -> str:
    """The Part 4 page without its industry-evidence section (that decision is carried, not judged again)."""
    head = page[:page.index("<h1>1. Industry evidence</h1>")].replace("Part 4", "Part 5")
    body = page[page.index("<h1>2. Items</h1>"):]
    return head + body.replace("<h1>2. Items</h1>", "<h1>1. Items</h1>").replace(
        "<h1>3. Event coverage</h1>", "<h1>2. Event coverage</h1>")


def industry_rows(key: dict) -> list[dict]:
    """The carried evidence as rows under this check's letters, in the form the Part 4 score reads."""
    by_run = key["industry_carried"]
    return [{"row": n, "summary": letter, "industry_evidence": by_run[entry["run"]]["industry_evidence"],
             "note": by_run[entry["run"]]["note"]}
            for n, (letter, entry) in enumerate(sorted(key["summaries"].items()), start=1)]


def problems(items, events, key) -> list[str]:
    return sj.item_problems(items, key["items"]) + sj.event_problems(events, key["events"])


# ---------- the lock ----------

def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(p4.__file__), Path(sj.__file__)):
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def lock_record(items, events, key, guide_texts: list[str], record_text: str, record_file: str = "") -> dict:
    found = problems(items, events, key)
    if found:
        raise JudgingError("The sheets are not complete:\n" + "\n".join(found))
    return {"items_fingerprint": sj.sheet_fingerprint(items), "events_fingerprint": sj.sheet_fingerprint(events),
            "key_fingerprint": sj._hash(key),
            "guide_fingerprints": [sj._hash(t.replace("\r\n", "\n")) for t in guide_texts],
            "record_fingerprint": sj._hash(record_text.replace("\r\n", "\n")),
            "code_fingerprint": _code_fingerprint(), "scoring_version": SCORING_VERSION, "record_file": record_file,
            "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def check_lock(lock: dict, items, events, key) -> None:
    if lock["items_fingerprint"] != sj.sheet_fingerprint(items) or \
            lock["events_fingerprint"] != sj.sheet_fingerprint(events):
        raise JudgingError("A sheet changed after the lock. The locked judgements are the ones scored.")
    if lock["key_fingerprint"] != sj._hash(key):
        raise JudgingError("The key changed after the lock.")
    if lock["code_fingerprint"] != _code_fingerprint():
        raise JudgingError("The judging code changed after the lock.")


# ---------- the score ----------

def score(items, events, key, record, locked_sheet, baseline, expected_runs) -> dict:
    """The Part 4 score and pass rule, unchanged, with the carried evidence decisions."""
    return {**p4.score(items, events, industry_rows(key), key, record, locked_sheet, baseline, expected_runs),
            "industry_evidence": "carried from the locked Part 4 sheet"}


# ---------- files and commands ----------

def _earlier_locks():
    """Part 3 and Part 4, each checked against its own lock before anything is taken from it."""
    p3_items, p3_key = p4._locked_comparison()
    p4_lock = sj._load_json(EVAL_DIR / p4.LOCK_FILE)
    p4_items, p4_events, p4_industry = p4._sheets()
    p4_key = sj._load_json(EVAL_DIR / p4.KEY_FILE)
    p4.check_lock(p4_lock, p4_items, p4_events, p4_industry, p4_key)
    return p3_items, p3_key, p4_items, p4_key, p4_industry


def _sheet_command(record_name: str) -> None:
    from src.evaluate import load_label_set
    from src.fulltext import FULLTEXT_DIR
    from src.fulltext_eval import CHECKED_LABEL_FILE
    from src.summary_run import EVENTS_FILE as DEV_EVENTS, load_events
    p3_items, p3_key, p4_items, p4_key, p4_industry = _earlier_locks()
    locked, _ = locked_sources(p3_items, p3_key, p4_items, p4_key)
    record = json.loads((EVAL_DIR / record_name).read_text(encoding="utf-8"))
    sheets = build_sheets(record, sj._passages(record), load_events(EVAL_DIR / DEV_EVENTS),
                          load_label_set([EVAL_DIR / CHECKED_LABEL_FILE]), locked,
                          carried_evidence(p4_industry, p4_key))
    sheets["key"]["record_file"] = record_name
    files = {EVAL_DIR / ITEMS_FILE: sj._csv_text(sheets["items"], sj.ITEM_COLUMNS),
             EVAL_DIR / EVENTS_FILE: sj._csv_text(sheets["events"], sj.EVENT_COLUMNS),
             EVAL_DIR / KEY_FILE: json.dumps(sheets["key"], indent=1),
             Path(FULLTEXT_DIR) / PAGE_FILE: sheets["page"]}
    for path in files:
        if path.exists():
            raise JudgingError(f"{path.name} already exists; it is not overwritten.")
    for path, text in files.items():
        sj._write_new(path, text)
    print(f"Items to judge: {len(sheets['items'])} rows in data/eval/{ITEMS_FILE} "
          f"({len(sheets['key']['carried'])} unchanged items keep their locked judgement)")
    print(f"Events: {len(sheets['events'])} rows in data/eval/{EVENTS_FILE}")
    print("Industry evidence: carried from the locked Part 4 sheet (not judged again)")
    print(f"Reading page: {Path(FULLTEXT_DIR) / PAGE_FILE}  (open it in your browser)")
    print(f"Key: data/eval/{KEY_FILE}  (do not open it until the sheets are judged and locked)")


def _sheets():
    return sj._read_csv(EVAL_DIR / ITEMS_FILE), sj._read_csv(EVAL_DIR / EVENTS_FILE)


def _lock_command() -> None:
    _earlier_locks()
    key = sj._load_json(EVAL_DIR / KEY_FILE)
    if (EVAL_DIR / LOCK_FILE).exists():
        raise JudgingError(f"{LOCK_FILE} already exists: the sheets are locked.")
    items, events = _sheets()
    lock = lock_record(items, events, key, [(EVAL_DIR / g).read_text(encoding="utf-8") for g in GUIDE_FILES],
                       (EVAL_DIR / key["record_file"]).read_text(encoding="utf-8"), key["record_file"])
    sj._write_new(EVAL_DIR / LOCK_FILE, json.dumps(lock, indent=1))
    print(f"Locked: data/eval/{LOCK_FILE}")


def _score_command() -> None:
    from src.ranking_eval import DEV_RUNS
    lock = sj._load_json(EVAL_DIR / LOCK_FILE)
    items, events = _sheets()
    key = sj._load_json(EVAL_DIR / KEY_FILE)
    check_lock(lock, items, events, key)
    p3_items, p3_key, p4_items, p4_key, _ = _earlier_locks()
    _, locked_sheet = locked_sources(p3_items, p3_key, p4_items, p4_key)
    record = json.loads((EVAL_DIR / key["record_file"]).read_text(encoding="utf-8"))
    result = {"lock": lock, "record_file": key["record_file"],
              **score(items, events, key, record, locked_sheet, p4.locked_baseline(p3_items, p3_key),
                      list(DEV_RUNS))}
    sj._write_new(EVAL_DIR / SCORE_FILE, json.dumps(result, indent=1))
    _print(result)
    print(f"Saved: data/eval/{SCORE_FILE}")


def _print(result: dict) -> None:
    for name, part in [("total", result["total"])] + list(result["per_run"].items()):
        i, c, ev = part["industry"], part["core"], part["events"]
        print(f"{name:<11} industry produced {i['produced']}: supported {i['supported']}, partly "
              f"{i['partly_supported']}, not {i['not_supported']} | key points and briefs {c['supported_count']}/"
              f"{c['total']} supported, not {c['not_supported']} | events {ev['covered']}/{ev['total']}"
              + (f" | evidence {part['industry_evidence']}, section {'empty' if part['industry_empty'] else 'has points'}"
                 if name != "total" else ""))
    kb = result["key_points_and_briefs"]
    print(f"Key points and briefs: {kb['supported']}/{kb['total']} = {kb['share'] * 100:.1f}%; locked "
          f"{kb['locked_supported']}/{kb['locked_total']} = {kb['locked_share'] * 100:.1f}%; threshold "
          f"{kb['threshold']!r} (compared unrounded)")
    for check, ok in result["checks"].items():
        print(f"  {'pass' if ok else 'FAIL'}  {check}")
    print(f"Passes: {result['passes']}")


COMMANDS = {"lock": _lock_command, "score": _score_command}

if __name__ == "__main__":
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "sheet":
            _sheet_command(sys.argv[2])
        elif len(sys.argv) == 2 and sys.argv[1] in COMMANDS:
            COMMANDS[sys.argv[1]]()
        else:
            print(__doc__)
    except sj.JudgingError as error:
        print(f"[STOP] {error}")
        sys.exit(1)
