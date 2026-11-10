"""
Checks for a filled event sheet (events_dev_tata_motors.csv, or events.csv
of a fresh run). The check lists cases for the labeller to look at. It never
changes a label: every decision stays with the labeller.

Findings, most important first:
    missing_event        - an empty event, or a name that is not lower case
                           words joined by "_"
    possible_typo        - two different names that are almost the same
    possible_duplicate   - very similar headlines under different event names
    possible_wrong_merge - two articles under one event name that share no
                           content word in their title and description
    single_article       - an event with one article; optional review, because
                           some developments get only one report

Run:
    python -m src.event_check                      (the development sheet)
    python -m src.event_check <path to sheet>
"""

import csv
import re
import sys
from difflib import SequenceMatcher
from itertools import combinations

KINDS = ["missing_event", "possible_typo", "possible_duplicate", "possible_wrong_merge", "single_article"]
NAME = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")
TYPO_RATIO = 0.85            # names at least this alike are probably one name typed twice
TITLE_RATIO = 0.85           # headlines at least this alike are probably the same news
WORD_SHARE = 0.6             # or this share of shared content words in title and description


def _event(row: dict) -> str:
    return (row.get("event") or "").strip()


def _text(row: dict) -> str:
    return f"{row.get('title') or ''} {row.get('description') or ''}"


def _words(row: dict) -> set[str]:
    from src.fulltext import _content_words
    return set(_content_words(_text(row)))


def _similar_news(a: dict, b: dict) -> bool:
    from src.preprocess import _title_key
    ta, tb = _title_key(a.get("title") or ""), _title_key(b.get("title") or "")
    if ta and tb and SequenceMatcher(None, ta, tb).ratio() >= TITLE_RATIO:
        return True
    wa, wb = _words(a), _words(b)
    return bool(wa | wb) and len(wa & wb) / len(wa | wb) >= WORD_SHARE


def _finding(kind: str, rows: list[dict], events: list[str], detail: str) -> dict:
    return {"kind": kind, "rows": [str(r.get("row")) for r in rows], "events": events, "detail": detail}


def check_events(rows: list[dict]) -> list[dict]:
    findings = []
    valid = [r for r in rows if NAME.match(_event(r))]
    for row in rows:
        if not NAME.match(_event(row)):
            findings.append(_finding("missing_event", [row], [_event(row)],
                                     "empty, or not lower case words joined by _"))

    groups: dict[str, list[dict]] = {}
    for row in valid:
        groups.setdefault(_event(row), []).append(row)
    names = sorted(groups)

    for a, b in combinations(names, 2):
        if SequenceMatcher(None, a, b).ratio() >= TYPO_RATIO:
            findings.append(_finding("possible_typo", groups[a][:1] + groups[b][:1], [a, b],
                                     "two names that are almost the same"))

    duplicates: dict[tuple[str, str], list[dict]] = {}            # one finding per pair of event names
    for a, b in combinations(valid, 2):
        if _event(a) != _event(b) and _similar_news(a, b):
            pair = duplicates.setdefault(tuple(sorted((_event(a), _event(b)))), [])
            pair += [r for r in (a, b) if r not in pair]
    for (first, second), involved in duplicates.items():
        findings.append(_finding("possible_duplicate", involved, [first, second],
                                 "very similar headlines under different event names"))

    for name in names:
        for a, b in combinations(groups[name], 2):
            if not _words(a) & _words(b):
                findings.append(_finding("possible_wrong_merge", [a, b], [name],
                                         "one event name, but no shared word in title and description"))

    for name in names:
        if len(groups[name]) == 1:
            findings.append(_finding("single_article", groups[name], [name],
                                     "optional review: some developments get only one report"))
    return sorted(findings, key=lambda f: KINDS.index(f["kind"]))     # sorted() keeps the order inside a kind


def main(path=None) -> None:
    from src.config import EVAL_DIR
    from src.summary_run import EVENTS_FILE
    path = path or EVAL_DIR / EVENTS_FILE
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    findings = check_events(rows)
    events = {_event(r) for r in rows if NAME.match(_event(r))}
    print(f"{len(rows)} articles, {len(events)} events, {len(findings)} findings to look at\n")
    for kind in KINDS:
        found = [f for f in findings if f["kind"] == kind]
        if not found:
            continue
        print(f"== {kind} ({len(found)})")
        by_row = {str(r.get("row")): r for r in rows}
        for f in found:
            print(f"   {f['detail']}  |  events: {', '.join(e or '(empty)' for e in f['events'])}")
            for number in f["rows"]:
                print(f"      row {number}: {(by_row[number].get('title') or '')[:90]}")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
