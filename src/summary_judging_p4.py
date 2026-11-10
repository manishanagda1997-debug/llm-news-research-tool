"""
Judging of the prompt version 4 rerun (SUMMARY_JUDGING_GUIDE_PART4.md): the direct design on the three
development runs, after the industry-context rule. The locked comparison (src/summary_judging.py, its sheets,
key, lock and score) is read, never written.

sheet <record>  from the version 4 record: every item that is new or changed, shuffled, in
                summary_v2_p4_judging.csv; an item with exactly the same section, text and cited articles as a
                direct item of the locked sheet is not shown and keeps its locked judgement; one row per relevant
                event of each summary in summary_v2_p4_events.csv; one row per summary in
                summary_v2_p4_industry.csv (does any selected article report on the wider industry?); the key;
                and the reading page (industry evidence first, then the items, then each summary for the events).
                Repairs are marked in the key only.
lock            the locked comparison still unchanged; every mark present and valid; then the fingerprints of
                the three sheets, the key, the guide, the record and the judging code (this module and
                src/summary_judging.py) are saved
score           refuses without a lock, or when a sheet, the key or the judging code changed after it; then the
                pass rule of Part 4, in total and per run

Run:
    python -m src.summary_judging_p4 sheet summary_v2_dev_k5_c7_p4.json
    python -m src.summary_judging_p4 lock
    python -m src.summary_judging_p4 score
"""

import hashlib
import html
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from src import summary_judging as sj
from src.config import EVAL_DIR

SCORING_VERSION = "1"
SEED = 44
DESIGN = "direct"

ITEMS_FILE = "summary_v2_p4_judging.csv"
EVENTS_FILE = "summary_v2_p4_events.csv"
INDUSTRY_FILE = "summary_v2_p4_industry.csv"
KEY_FILE = "summary_v2_p4_judging_key.json"
LOCK_FILE = "summary_v2_p4_judging_lock.json"
SCORE_FILE = "summary_v2_p4_judging_score.json"
PAGE_FILE = "summary_v2_p4_judging.html"                 # in data/fulltext/, outside Git
GUIDE_FILE = "SUMMARY_JUDGING_GUIDE_PART4.md"

INDUSTRY_COLUMNS = ["row", "summary", "industry_evidence", "note"]
CORE = ("key_points", "article_briefs")               # the sections checked for a drop
TOLERANCE = 0.05                                      # percentage points below the locked supported share
MIN_EVENT_COVERAGE = sj.MIN_EVENT_COVERAGE


class JudgingError(sj.JudgingError):
    pass


def signature(section: str, text: str, evidence: str) -> tuple:
    """Exactly the same section, text and cited articles: nothing is trimmed or normalised."""
    return (section, text, evidence)


# ---------- the locked judgements ----------

def locked_items(items: list[dict], key: dict, design: str = DESIGN) -> dict[tuple, str]:
    """{signature: locked row number} for the items of one design. A signature found twice with different
    judgements is left out, so it is judged again."""
    rows, found, clash = sj._rows_by_number(items), {}, set()
    for number, entry in key["items"].items():
        if entry["design"] != design:
            continue
        row = rows[number]
        sig = signature(entry["section"], row["text"], row["evidence"])
        if sig in found and sj._value(rows[found[sig]]["judgement"]) != sj._value(row["judgement"]):
            clash.add(sig)
        found.setdefault(sig, number)
    return {sig: number for sig, number in found.items() if sig not in clash}


def locked_baseline(items: list[dict], key: dict, design: str = DESIGN) -> dict:
    """The locked supported share of the key points and briefs of one design, as a numerator and denominator."""
    rows = sj._rows_by_number(items)
    mine = [rows[n] for n, e in key["items"].items() if e["design"] == design and e["section"] in CORE]
    supported = sum(1 for r in mine if sj._value(r["judgement"]) == "supported")
    return {"supported": supported, "total": len(mine)}


# ---------- building the sheets ----------

def build_sheets(record: dict, passages: dict, events: dict, labels: dict, locked: dict, seed: int = SEED) -> dict:
    produced = sj._produced(record)
    designs = {r["design"] for r in record["records"]}
    if designs != {DESIGN}:
        raise JudgingError(f"The record holds {sorted(designs)}; the version 4 check judges {DESIGN} only.")
    rng = random.Random(seed)
    new, carried = [], []
    for entry in produced:
        for item in sj._items_of(entry):
            evidence = "; ".join(item["evidence"])
            base = {"run": entry["run"], "section": item["section"], "index": item["index"], "ids": item["ids"],
                    "repaired": item["repaired"]}
            sig = signature(item["section"], item["text"], evidence)
            if sig in locked:
                carried.append({**base, "locked_row": locked[sig]})
            else:
                new.append((base, item, evidence))
    rng.shuffle(new)
    items, item_key = [], {}
    for number, (base, item, evidence) in enumerate(new, start=1):
        row = {"row": number, "kind": sj.KINDS[item["section"]], "text": item["text"], "evidence": evidence,
               "judgement": "", "contradicted": "", "central": "", "note": ""}
        items.append(row)
        item_key[str(number)] = {"design": DESIGN, **base, "fingerprint": sj._hash(row["kind"], row["text"], evidence)}

    shared = sj.build_sheets(record, passages, events, labels, seed=seed)     # the event rows and the letters
    letters = sorted(shared["key"]["summaries"])
    industry = [{"row": n, "summary": letter, "industry_evidence": "", "note": ""}
                for n, letter in enumerate(letters, start=1)]
    industry_key = {str(row["row"]): {"letter": row["summary"], "fingerprint": sj._hash(row["summary"])}
                    for row in industry}
    by_run = {entry["run"]: entry for entry in produced}
    shown = [(letter, by_run[shared["key"]["summaries"][letter]["run"]]) for letter in letters]
    key = {"record_fingerprint": sj._hash(record), "seed": seed, "items": item_key, "carried": carried,
           "summaries": shared["key"]["summaries"], "events": shared["key"]["events"], "industry": industry_key}
    titles = {a["key"]: a["title"] for entry in produced for a in entry["articles"]}
    return {"items": items, "events": shared["events"], "industry": industry, "key": key,
            "page": reading_page(items, shown, passages, titles)}


def reading_page(items: list[dict], summaries: list[tuple[str, dict]], passages: dict, titles: dict) -> str:
    """Industry evidence (the articles of each summary, without the summary), the items with their evidence,
    then each summary's points for the event check. No design, no run, no repair."""
    e = html.escape
    parts = ["<!doctype html><meta charset='utf-8'><title>Summary judging, Part 4</title>",
             "<style>body{font-family:sans-serif;max-width:900px;margin:auto;padding:16px;line-height:1.5}"
             ".item{border-top:2px solid #444;margin-top:24px}.ev{background:#f4f4f4;padding:8px;margin:6px 0}"
             "</style>", "<h1>1. Industry evidence</h1>"]
    for letter, entry in summaries:
        parts.append(f"<div class='item'><h2>Summary {letter}: the selected articles</h2>")
        for article in entry["articles"]:
            parts.append(f"<div class='ev'><b>{e(article['key'])}: {e(article['title'])}</b>"
                         f"<p>{e(passages.get(article['key'], '(no text)'))}</p></div>")
        parts.append("</div>")
    parts.append("<h1>2. Items</h1>")
    for row in items:
        parts.append(f"<div class='item'><h2>Row {row['row']}: {e(row['kind'])}</h2><p><b>{e(row['text'])}</b></p>")
        for article in row["evidence"].split("; "):
            parts.append(f"<div class='ev'><b>{e(article)}: {e(titles.get(article, ''))}</b>"
                         f"<p>{e(passages.get(article, '(no text)'))}</p></div>")
        parts.append("</div>")
    parts.append("<h1>3. Event coverage</h1>")
    for letter, entry in summaries:
        summary, keys = entry["result"]["summary"], sj._keys(entry)
        all_titles = {a["key"]: a["title"] for a in entry["articles"]}
        parts.append(f"<div class='item'><h2>Summary {letter}</h2>")
        for section, name in (("key_points", "Company points"), ("industry_context", "Industry points")):
            parts.append(f"<h3>{name}</h3><ul>")
            for point in summary.get(section, []):
                cited = "; ".join(e(all_titles[keys[c]]) for c in point["citations"])
                parts.append(f"<li>{e(point['text'])}<br><small>Cites: {cited}</small></li>")
            if not summary.get(section):
                parts.append("<li><i>(none)</i></li>")
            parts.append("</ul>")
        parts.append("</div>")
    return "".join(parts)


# ---------- checking the marks ----------

def industry_problems(rows: list[dict], industry_key: dict) -> list[str]:
    found, by_number = [], sj._rows_by_number(rows)
    if set(by_number) != set(industry_key) or len(by_number) != len(rows):
        found.append("industry: rows do not match the sheet made by the tool")
    for number in sorted(set(by_number) & set(industry_key), key=int):
        row, where = by_number[number], f"industry row {number}"
        if sj._hash(row["summary"]) != industry_key[number]["fingerprint"]:
            found.append(f"{where}: summary changed")
            continue
        if sj._value(row["industry_evidence"]) not in sj.YES_NO:
            found.append(f"{where}: industry_evidence must be yes or no")
        if not sj._value(row.get("note")):
            found.append(f"{where}: the note names the article and the fact, or the reason for no")
    return found


def problems(items, events, industry, key) -> list[str]:
    return sj.item_problems(items, key["items"]) + sj.event_problems(events, key["events"]) + \
        industry_problems(industry, key["industry"])


# ---------- the lock ----------

def _code_fingerprint() -> str:
    """This module and the helpers it uses from src/summary_judging.py."""
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(sj.__file__)):
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def lock_record(items, events, industry, key, guide_text: str, record_text: str, record_file: str = "") -> dict:
    found = problems(items, events, industry, key)
    if found:
        raise JudgingError("The sheets are not complete:\n" + "\n".join(found))
    return {"items_fingerprint": sj.sheet_fingerprint(items), "events_fingerprint": sj.sheet_fingerprint(events),
            "industry_fingerprint": sj.sheet_fingerprint(industry), "key_fingerprint": sj._hash(key),
            "guide_fingerprint": sj._hash(guide_text.replace("\r\n", "\n")),
            "record_fingerprint": sj._hash(record_text.replace("\r\n", "\n")),
            "code_fingerprint": _code_fingerprint(), "scoring_version": SCORING_VERSION, "record_file": record_file,
            "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def check_lock(lock: dict, items, events, industry, key) -> None:
    if (lock["items_fingerprint"] != sj.sheet_fingerprint(items)
            or lock["events_fingerprint"] != sj.sheet_fingerprint(events)
            or lock["industry_fingerprint"] != sj.sheet_fingerprint(industry)):
        raise JudgingError("A sheet changed after the lock. The locked judgements are the ones scored.")
    if lock["key_fingerprint"] != sj._hash(key):
        raise JudgingError("The key changed after the lock.")
    if lock["code_fingerprint"] != _code_fingerprint():
        raise JudgingError("The judging code changed after the lock.")


# ---------- the score ----------

def _counts(rows: list[dict]) -> dict:
    judgements = [sj._value(r["judgement"]) for r in rows]
    supported = judgements.count("supported")
    return {"total": len(rows), **{j: judgements.count(j) for j in sj.JUDGEMENTS},
            "supported_share": sj._share(supported, len(rows)),
            "contradicted": sum(1 for r in rows if sj._value(r["contradicted"]) == "yes")}


def score(items, events, industry, key, record, locked_sheet: list[dict], baseline: dict,
          expected_runs: list[str]) -> dict:
    rows, event_rows = sj._rows_by_number(items), sj._rows_by_number(events)
    locked_rows = sj._rows_by_number(locked_sheet)
    judged = [dict(rows[n], **{k: e[k] for k in ("run", "section", "repaired")}, origin="new")
              for n, e in key["items"].items()]
    judged += [dict(locked_rows[str(c["locked_row"])], run=c["run"], section=c["section"], repaired=c["repaired"],
                    origin="carried") for c in key["carried"]]
    letters = key["summaries"]
    evidence = {letters[r["summary"]]["run"]: sj._value(r["industry_evidence"]) for r in industry}
    produced = {r["run"]: r for r in sj._produced(record)}
    runs = sorted(produced)

    def part(selected_runs):
        mine = [r for r in judged if r["run"] in selected_runs]
        core = [r for r in mine if r["section"] in CORE]
        marked = [(n, e) for n, e in key["events"].items() if letters[e["letter"]]["run"] in selected_runs]
        covered = sum(1 for n, _ in marked if sj._value(event_rows[n]["covered"]) == "yes")
        return {"industry": {"produced": sum(len(produced[r]["result"]["summary"].get("industry_context", []))
                                             for r in selected_runs),
                             **_counts([r for r in mine if r["section"] == "industry_context"])},
                "sections": {s: _counts([r for r in mine if r["section"] == s]) for s in CORE},
                "core": {**_counts(core), "supported_count": sum(1 for r in core
                                                                 if sj._value(r["judgement"]) == "supported")},
                "events": {"covered": covered, "total": len(marked), "coverage": sj._share(covered, len(marked))},
                "carried": sum(1 for r in mine if r["origin"] == "carried"),
                "repaired": _counts([r for r in mine if r["repaired"]])}

    total = part(runs)
    per_run = {run: {**part([run]), "industry_evidence": evidence.get(run),
                     "industry_empty": not produced[run]["result"]["summary"].get("industry_context")}
               for run in runs}
    locked_share = baseline["supported"] / baseline["total"]
    threshold = locked_share - TOLERANCE
    core = total["core"]
    core_share = core["supported_count"] / core["total"] if core["total"] else 0.0
    empty_wrong = [run for run in runs if per_run[run]["industry_empty"] and per_run[run]["industry_evidence"] == "yes"]
    points_without_evidence = [run for run in runs if not per_run[run]["industry_empty"]
                               and per_run[run]["industry_evidence"] == "no"]
    all_empty_justified = all(per_run[r]["industry_empty"] and per_run[r]["industry_evidence"] == "no" for r in runs)
    record_runs = [r["run"] for r in record["records"]]
    runs_complete = (sorted(record_runs) == sorted(expected_runs) and sorted(runs) == sorted(expected_runs)
                     and all(r["design"] == DESIGN for r in record["records"]))
    checks = {
        "no_unsupported_industry_point": total["industry"]["not_supported"] == 0,
        "empty_section_only_without_evidence": not empty_wrong,
        "industry_points_only_with_evidence": not points_without_evidence,
        "a_supported_industry_point_or_all_empty_without_evidence": total["industry"]["supported"] >= 1
        or all_empty_justified,
        "no_unsupported_key_point_or_brief": core["not_supported"] == 0,
        "key_points_and_briefs_within_tolerance": core_share >= threshold,
        "event_coverage": total["events"]["coverage"] is not None
        and total["events"]["coverage"] >= MIN_EVENT_COVERAGE,
        "every_development_run_has_one_summary": runs_complete,
    }
    return {"total": total, "per_run": per_run,
            "key_points_and_briefs": {"supported": core["supported_count"], "total": core["total"],
                                      "share": core_share, "locked_supported": baseline["supported"],
                                      "locked_total": baseline["total"], "locked_share": locked_share,
                                      "threshold": threshold},
            "empty_section_with_evidence": empty_wrong, "points_without_evidence": points_without_evidence,
            "expected_runs": sorted(expected_runs), "checks": checks, "passes": all(checks.values()),
            "rule": {"tolerance": TOLERANCE, "min_event_coverage": MIN_EVENT_COVERAGE,
                     "threshold": "locked supported share minus the tolerance, unrounded; compared unrounded"}}


# ---------- files and commands ----------

def _locked_comparison():
    """The locked comparison, checked against its own lock before anything is taken from it."""
    lock = sj._load_json(EVAL_DIR / sj.LOCK_FILE)
    items, events = sj._read_csv(EVAL_DIR / sj.ITEMS_FILE), sj._read_csv(EVAL_DIR / sj.EVENTS_FILE)
    sj.check_lock(lock, items, events)
    return items, sj._load_json(EVAL_DIR / sj.KEY_FILE)


def _sheet_command(record_name: str) -> None:
    from src.evaluate import load_label_set
    from src.fulltext import FULLTEXT_DIR
    from src.fulltext_eval import CHECKED_LABEL_FILE
    from src.summary_run import EVENTS_FILE as DEV_EVENTS, load_events
    locked_sheet, locked_key = _locked_comparison()
    record = json.loads((EVAL_DIR / record_name).read_text(encoding="utf-8"))
    sheets = build_sheets(record, sj._passages(record), load_events(EVAL_DIR / DEV_EVENTS),
                          load_label_set([EVAL_DIR / CHECKED_LABEL_FILE]), locked_items(locked_sheet, locked_key))
    sheets["key"]["record_file"] = record_name
    files = {EVAL_DIR / ITEMS_FILE: sj._csv_text(sheets["items"], sj.ITEM_COLUMNS),
             EVAL_DIR / EVENTS_FILE: sj._csv_text(sheets["events"], sj.EVENT_COLUMNS),
             EVAL_DIR / INDUSTRY_FILE: sj._csv_text(sheets["industry"], INDUSTRY_COLUMNS),
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
    print(f"Industry evidence: {len(sheets['industry'])} rows in data/eval/{INDUSTRY_FILE}")
    print(f"Reading page: {Path(FULLTEXT_DIR) / PAGE_FILE}  (open it in your browser)")
    print(f"Key: data/eval/{KEY_FILE}  (do not open it until the sheets are judged and locked)")


def _sheets():
    return (sj._read_csv(EVAL_DIR / ITEMS_FILE), sj._read_csv(EVAL_DIR / EVENTS_FILE),
            sj._read_csv(EVAL_DIR / INDUSTRY_FILE))


def _lock_command() -> None:
    _locked_comparison()
    key = sj._load_json(EVAL_DIR / KEY_FILE)
    if (EVAL_DIR / LOCK_FILE).exists():
        raise JudgingError(f"{LOCK_FILE} already exists: the sheets are locked.")
    items, events, industry = _sheets()
    lock = lock_record(items, events, industry, key, (EVAL_DIR / GUIDE_FILE).read_text(encoding="utf-8"),
                       (EVAL_DIR / key["record_file"]).read_text(encoding="utf-8"), key["record_file"])
    sj._write_new(EVAL_DIR / LOCK_FILE, json.dumps(lock, indent=1))
    print(f"Locked: data/eval/{LOCK_FILE}")


def _pct(share) -> str:
    return "-" if share is None else f"{share * 100:.1f}%"


def _score_command() -> None:
    lock = sj._load_json(EVAL_DIR / LOCK_FILE)
    from src.ranking_eval import DEV_RUNS
    items, events, industry = _sheets()
    key = sj._load_json(EVAL_DIR / KEY_FILE)
    check_lock(lock, items, events, industry, key)
    locked_sheet, locked_key = _locked_comparison()
    record = json.loads((EVAL_DIR / key["record_file"]).read_text(encoding="utf-8"))
    result = {"lock": lock, "record_file": key["record_file"],
              **score(items, events, industry, key, record, locked_sheet, locked_baseline(locked_sheet, locked_key),
                      list(DEV_RUNS))}
    sj._write_new(EVAL_DIR / SCORE_FILE, json.dumps(result, indent=1))
    for name, part in [("total", result["total"])] + list(result["per_run"].items()):
        i, c, ev = part["industry"], part["core"], part["events"]
        print(f"{name:<11} industry produced {i['produced']}: supported {i['supported']}, partly "
              f"{i['partly_supported']}, not {i['not_supported']} | key points and briefs {c['supported_count']}/"
              f"{c['total']} supported, not {c['not_supported']} | events {ev['covered']}/{ev['total']}"
              + (f" | evidence {part['industry_evidence']}, section {'empty' if part['industry_empty'] else 'has points'}"
                 if name != "total" else ""))
    kb = result["key_points_and_briefs"]
    print(f"Key points and briefs: {kb['supported']}/{kb['total']} = {_pct(kb['share'])}; locked "
          f"{kb['locked_supported']}/{kb['locked_total']} = {_pct(kb['locked_share'])}; threshold "
          f"{kb['threshold']:.4f}")
    for check, ok in result["checks"].items():
        print(f"  {'pass' if ok else 'FAIL'}  {check}")
    print(f"Passes: {result['passes']}")
    print(f"Saved: data/eval/{SCORE_FILE}")


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
