"""
Blind judging for the comparison of the two summary designs (SUMMARY_JUDGING_GUIDE.md, Part 3).

sheet <record>  from a comparison record (src/summary_v2.py, run): every item of every produced summary
                (company points, industry points, article briefs), shuffled, in summary_v2_judging.csv; one
                row per relevant event of each summary, under a random letter, in summary_v2_events.csv; the
                key that links rows and letters to design and run; and the reading page with the evidence.
                The evidence column holds article IDs only: publisher text is shown only on the reading page,
                in data/fulltext/, which Git ignores. Nothing shown names a design, its settings or a run.
lock            every judgement present and valid (an empty field is never read as "no"); then the
                fingerprints of both sheets, the guide, the record and this code are saved, before anything is
                un-blinded
score           refuses without a lock or when a sheet changed after it; then un-blinds. Per design: produced
                summaries, support, contradictions, central contradictions, event coverage (marked by hand,
                and by citation as a separate measure), calls; the acceptance rule and the choice order
rejudge         after the lock: 20% of the items, in a new order with new row numbers, without the first
                judgements, for the consistency check; it reads the lock and the items sheet, never the key
rejudge-score   the share of matching judgements and the three-by-three table; the first judgements stay the
                ones used for the results

Run:
    python -m src.summary_judging sheet summary_v2_dev_k5_p3_run2.json
    python -m src.summary_judging lock
    python -m src.summary_judging score
    python -m src.summary_judging rejudge
    python -m src.summary_judging rejudge-score
"""

import csv
import hashlib
import html
import io
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import EVAL_DIR

SCORING_VERSION = "1"
SEED = 42
REJUDGE_SEED = 43
REJUDGE_SHARE = 0.2

ITEMS_FILE = "summary_v2_judging.csv"
EVENTS_FILE = "summary_v2_events.csv"
KEY_FILE = "summary_v2_judging_key.json"
LOCK_FILE = "summary_v2_judging_lock.json"
SCORE_FILE = "summary_v2_judging_score.json"
PAGE_FILE = "summary_v2_judging.html"                 # in data/fulltext/, outside Git
REJUDGE_FILE = "summary_v2_rejudge.csv"
REJUDGE_KEY_FILE = "summary_v2_rejudge_key.json"
REJUDGE_PAGE_FILE = "summary_v2_rejudge.html"
REJUDGE_SCORE_FILE = "summary_v2_rejudge_score.json"
GUIDE_FILE = "SUMMARY_JUDGING_GUIDE.md"

ITEM_COLUMNS = ["row", "kind", "text", "evidence", "judgement", "contradicted", "central", "note"]
EVENT_COLUMNS = ["row", "summary", "event", "articles", "covered", "note"]
KINDS = {"key_points": "company point", "industry_context": "industry point", "article_briefs": "article brief"}
JUDGEMENTS = ("supported", "partly_supported", "not_supported")
YES_NO = ("yes", "no")

# the acceptance rule (Stage 4.2 plan): every rule must hold
MIN_PRODUCED = 0.9
MIN_EVENT_COVERAGE = 0.8
MAX_CONTRADICTED = 0.05


class JudgingError(Exception):
    pass


def _value(text) -> str:
    return str(text or "").strip().lower()


def _hash(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()


# ---------- building the sheets ----------

def _produced(record: dict) -> list[dict]:
    return [r for r in record["records"] if r["result"]["status"] in ("ok", "partial") and r["result"]["summary"]]


def _keys(entry: dict) -> dict[int, str]:
    return {a["id"]: a["key"] for a in entry["articles"]}


def _items_of(entry: dict) -> list[dict]:
    """Every item of a produced summary. `repaired` marks an item whose text comes from a successful repair:
    it goes into the key only, so repaired items are scored on their own without the judge knowing."""
    summary, keys, found = entry["result"]["summary"], _keys(entry), []
    repaired = {(r["section"], r["after"]) for r in entry["result"].get("repairs") or [] if r.get("ok") and r.get("after")}
    for section in ("key_points", "industry_context"):
        for index, point in enumerate(summary.get(section, [])):
            found.append({"section": section, "index": index, "text": point["text"],
                          "ids": list(point["citations"]), "evidence": [keys[c] for c in point["citations"]],
                          "repaired": (section, point["text"]) in repaired})
    for index, brief in enumerate(summary.get("article_briefs", [])):
        if brief.get("hidden") or not brief.get("brief", "").strip():
            continue
        found.append({"section": "article_briefs", "index": index, "text": brief["brief"],
                      "ids": [brief["id"]], "evidence": [keys[brief["id"]]],
                      "repaired": ("article_briefs", brief["brief"]) in repaired})
    return found


def _relevant_events(entry: dict, events: dict, labels: dict) -> dict[str, list[int]]:
    """{event: [article ids in this summary]} for the articles labelled relevant (direct or indirect)."""
    from src.evaluate import is_relevant
    from src.preprocess import normalize_url
    found = {}
    for headline in entry["result"]["headlines"]:
        url = normalize_url(headline["url"]) if headline.get("url") else ""
        url = url if url in events or url in labels else headline.get("url", "")
        if is_relevant(labels.get(url, "")) and url in events:
            found.setdefault(events[url], []).append(headline["id"])
    return found


def build_sheets(record: dict, passages: dict, events: dict, labels: dict, seed: int = SEED) -> dict:
    """The items sheet, the event sheet, the key and the reading page. passages: {article key: text}."""
    rng = random.Random(seed)
    produced = _produced(record)

    pool = []
    for entry in produced:
        for item in _items_of(entry):
            pool.append((entry, item))
    rng.shuffle(pool)
    items, item_key = [], {}
    for number, (entry, item) in enumerate(pool, start=1):
        row = {"row": number, "kind": KINDS[item["section"]], "text": item["text"],
               "evidence": "; ".join(item["evidence"]), "judgement": "", "contradicted": "", "central": "", "note": ""}
        items.append(row)
        item_key[str(number)] = {"design": entry["design"], "run": entry["run"], "section": item["section"],
                                 "index": item["index"], "ids": item["ids"], "repaired": item["repaired"],
                                 "fingerprint": _hash(row["kind"], row["text"], row["evidence"])}

    order = list(produced)
    rng.shuffle(order)
    letters = [chr(ord("A") + i) for i in range(len(order))]
    summaries, event_rows, event_key, shown = {}, [], {}, []
    for letter, entry in zip(letters, order):
        summaries[letter] = {"design": entry["design"], "run": entry["run"]}
        titles = {h["id"]: h["title"] for h in entry["result"]["headlines"]}
        relevant = _relevant_events(entry, events, labels)
        for event in sorted(relevant):
            number = len(event_rows) + 1
            row = {"row": number, "summary": letter, "event": event,
                   "articles": " | ".join(titles[i] for i in relevant[event]), "covered": "", "note": ""}
            event_rows.append(row)
            event_key[str(number)] = {"letter": letter, "event": event, "ids": relevant[event],
                                      "fingerprint": _hash(letter, event, row["articles"])}
        shown.append((letter, entry))

    key = {"record_fingerprint": _hash(record), "seed": seed, "items": item_key, "summaries": summaries,
           "events": event_key}
    titles_by_key = {a["key"]: a["title"] for entry in produced for a in entry["articles"]}
    return {"items": items, "events": event_rows, "key": key,
            "page": reading_page(items, shown, passages, titles_by_key)}


def reading_page(items: list[dict], summaries: list[tuple[str, dict]], passages: dict, titles: dict) -> str:
    """Items with their evidence text, then each summary's points for the event check. No design, no run."""
    e = html.escape
    parts = ["<!doctype html><meta charset='utf-8'><title>Summary judging</title>",
             "<style>body{font-family:sans-serif;max-width:900px;margin:auto;padding:16px;line-height:1.5}"
             ".item{border-top:2px solid #444;margin-top:24px}.ev{background:#f4f4f4;padding:8px;margin:6px 0}"
             "</style>", "<h1>Items</h1>"]
    for row in items:
        parts.append(f"<div class='item'><h2>Row {row['row']}: {e(row['kind'])}</h2><p><b>{e(row['text'])}</b></p>")
        for article in row["evidence"].split("; "):
            parts.append(f"<div class='ev'><b>{e(article)}: {e(titles.get(article, ''))}</b>"
                         f"<p>{e(passages.get(article, '(no text)'))}</p></div>")
        parts.append("</div>")
    parts.append("<h1>Event coverage</h1>")
    for letter, entry in sorted(summaries, key=lambda pair: pair[0]):
        summary, keys = entry["result"]["summary"], _keys(entry)
        all_titles = {a["key"]: a["title"] for a in entry["articles"]}
        parts.append(f"<div class='item'><h2>Summary {letter}</h2>")
        for section, name in (("key_points", "Company points"), ("industry_context", "Industry points")):
            parts.append(f"<h3>{name}</h3><ul>")
            for point in summary.get(section, []):
                cited = "; ".join(e(all_titles[keys[c]]) for c in point["citations"])
                parts.append(f"<li>{e(point['text'])}<br><small>Cites: {cited}</small></li>")
            parts.append("</ul>")
        parts.append("</div>")
    return "".join(parts)


# ---------- checking the marks ----------

def _rows_by_number(rows: list[dict]) -> dict[str, dict]:
    return {str(row["row"]).strip(): row for row in rows}


def item_problems(items: list[dict], item_key: dict) -> list[str]:
    found, rows = [], _rows_by_number(items)
    if set(rows) != set(item_key) or len(rows) != len(items):
        missing, extra = sorted(set(item_key) - set(rows), key=int), sorted(set(rows) - set(item_key), key=int)
        found.append(f"items: rows do not match the sheet made by the tool (missing {missing}, extra {extra})")
    for number in sorted(set(rows) & set(item_key), key=int):
        row, where = rows[number], f"items row {number}"
        if _hash(row["kind"], row["text"], row["evidence"]) != item_key[number]["fingerprint"]:
            found.append(f"{where}: kind, text or evidence changed")
            continue
        judgement, contradicted, central = _value(row["judgement"]), _value(row["contradicted"]), _value(row["central"])
        if judgement not in JUDGEMENTS:
            found.append(f"{where}: judgement must be one of {', '.join(JUDGEMENTS)}")
        if contradicted not in YES_NO:
            found.append(f"{where}: contradicted must be yes or no")
        elif contradicted == "yes":
            if central not in YES_NO:
                found.append(f"{where}: central must be yes or no when contradicted is yes")
            if judgement == "supported":
                found.append(f"{where}: a contradicted item is never supported")
            if not _value(row.get("note")):
                found.append(f"{where}: the note names the contradicted claim")
        elif central:
            found.append(f"{where}: central stays empty when contradicted is no")
        if judgement == "partly_supported" and not _value(row.get("note")):
            found.append(f"{where}: the note names the unsupported detail")
    return found


def event_problems(events: list[dict], event_key: dict) -> list[str]:
    found, rows = [], _rows_by_number(events)
    if set(rows) != set(event_key) or len(rows) != len(events):
        missing, extra = sorted(set(event_key) - set(rows), key=int), sorted(set(rows) - set(event_key), key=int)
        found.append(f"events: rows do not match the sheet made by the tool (missing {missing}, extra {extra})")
    for number in sorted(set(rows) & set(event_key), key=int):
        row, where = rows[number], f"events row {number}"
        if _hash(row["summary"], row["event"], row["articles"]) != event_key[number]["fingerprint"]:
            found.append(f"{where}: summary, event or articles changed")
            continue
        if _value(row["covered"]) not in YES_NO:
            found.append(f"{where}: covered must be yes or no")
    return found


def problems(items: list[dict], events: list[dict], key: dict) -> list[str]:
    return item_problems(items, key["items"]) + event_problems(events, key["events"])


# ---------- the lock ----------

def sheet_fingerprint(rows: list[dict]) -> str:
    """Every column of every row, trimmed and in lower case, in row order: spacing or a spreadsheet's quoting
    does not change it; any change of a mark does."""
    normal = sorted(([str(row["row"]).strip()] + [_value(row[c]) for c in sorted(row) if c != "row"]
                     for row in rows), key=lambda r: int(r[0]))
    return _hash(normal)


def _code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def lock_record(items, events, key, guide_text: str, record_text: str, record_file: str = "") -> dict:
    found = problems(items, events, key)
    if found:
        raise JudgingError("The sheets are not complete:\n" + "\n".join(found))
    return {"items_fingerprint": sheet_fingerprint(items), "events_fingerprint": sheet_fingerprint(events),
            "guide_fingerprint": _hash(guide_text.replace("\r\n", "\n")),
            "record_fingerprint": _hash(record_text.replace("\r\n", "\n")),
            "code_fingerprint": _code_fingerprint(), "scoring_version": SCORING_VERSION,
            "record_file": record_file,                    # so the rejudge never needs the key
            "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def check_lock(lock: dict, items, events) -> None:
    if lock["items_fingerprint"] != sheet_fingerprint(items) or lock["events_fingerprint"] != sheet_fingerprint(events):
        raise JudgingError("A sheet changed after the lock. The locked judgements are the ones scored; "
                           "a correction goes into a separate, recorded adjudication.")


# ---------- the score ----------

def _share(count, total):
    return round(count / total, 4) if total else None


def score(items, events, key, record) -> dict:
    designs = sorted({r["design"] for r in record["records"]})
    rows, event_rows = _rows_by_number(items), _rows_by_number(events)
    out = {}
    for design in designs:
        runs = [r for r in record["records"] if r["design"] == design]
        produced = [r for r in _produced(record) if r["design"] == design]
        mine = [rows[n] for n, entry in key["items"].items() if entry["design"] == design]
        judgements = [_value(r["judgement"]) for r in mine]
        contradicted = [r for r in mine if _value(r["contradicted"]) == "yes"]
        letters = {letter for letter, s in key["summaries"].items() if s["design"] == design}
        marked = [(n, entry) for n, entry in key["events"].items() if entry["letter"] in letters]
        covered = sum(1 for n, _ in marked if _value(event_rows[n]["covered"]) == "yes")
        cited = 0
        for n, entry in marked:
            summary_entry = next(r for r in produced if key["summaries"][entry["letter"]]["run"] == r["run"])
            used = {c for s in ("key_points", "industry_context")
                    for p in summary_entry["result"]["summary"].get(s, []) for c in p["citations"]}
            cited += bool(used & set(entry["ids"]))
        calls = [r["result"].get("usage", {}).get("calls", 0) for r in runs]
        fixed = [rows[n] for n, entry in key["items"].items() if entry["design"] == design and entry.get("repaired")]
        fixed_supported = sum(1 for r in fixed if _value(r["judgement"]) == "supported")
        out[design] = {
            "produced": {"count": len(produced), "total": len(runs), "share": _share(len(produced), len(runs))},
            "items": {"total": len(mine), **{j: judgements.count(j) for j in JUDGEMENTS},
                      "supported_share": _share(judgements.count("supported"), len(mine)),
                      "contradicted": len(contradicted), "contradicted_share": _share(len(contradicted), len(mine)),
                      "central_contradicted": sum(1 for r in contradicted if _value(r["central"]) == "yes")},
            "events": {"covered": covered, "total": len(marked), "coverage": _share(covered, len(marked)),
                       "cited": cited, "cited_share": _share(cited, len(marked))},
            "repaired_items": {"total": len(fixed), "supported": fixed_supported,
                               "supported_share": _share(fixed_supported, len(fixed)),
                               "contradicted": sum(1 for r in fixed if _value(r["contradicted"]) == "yes")},
            "calls": round(sum(calls) / len(calls), 2) if calls else None}
        d = out[design]
        d["checks"] = {"produced": (d["produced"]["share"] or 0) >= MIN_PRODUCED,
                       "event_coverage": d["events"]["coverage"] is not None
                       and d["events"]["coverage"] >= MIN_EVENT_COVERAGE,
                       "contradicted": d["items"]["contradicted_share"] is not None
                       and d["items"]["contradicted_share"] <= MAX_CONTRADICTED,
                       "no_central_contradiction": d["items"]["central_contradicted"] == 0}
        d["passes"] = all(d["checks"].values())
    return {"designs": out, "choice": choose(out),
            "rule": {"min_produced": MIN_PRODUCED, "min_event_coverage": MIN_EVENT_COVERAGE,
                     "max_contradicted": MAX_CONTRADICTED, "no_central_contradiction": True,
                     "choice": "higher supported share, then higher event coverage, then fewer calls"}}


def choose(designs: dict) -> str | None:
    passing = [name for name, d in designs.items() if d["passes"]]
    if not passing:
        return None
    return min(passing, key=lambda n: (-designs[n]["items"]["supported_share"], -designs[n]["events"]["coverage"],
                                       designs[n]["calls"]))


# ---------- the consistency check ----------

def rejudge_rows(items: list[dict], seed: int = REJUDGE_SEED, share: float = REJUDGE_SHARE):
    rng = random.Random(seed)
    chosen = rng.sample(items, math.ceil(len(items) * share))
    rng.shuffle(chosen)
    rows, rejudge_key = [], {}
    for number, row in enumerate(chosen, start=1):
        rows.append({"row": number, "kind": row["kind"], "text": row["text"], "evidence": row["evidence"],
                     "judgement": "", "contradicted": "", "central": "", "note": ""})
        rejudge_key[str(number)] = int(row["row"])
    return rows, rejudge_key


def agreement(first: list[dict], second: list[dict], rejudge_key: dict) -> dict:
    by_row = _rows_by_number(first)
    pairs = [(by_row[str(original)], row) for row in second
             for original in [rejudge_key[str(row["row"]).strip()]]]
    table = {a: {b: 0 for b in JUDGEMENTS} for a in JUDGEMENTS}
    for a, b in pairs:
        table[_value(a["judgement"])][_value(b["judgement"])] += 1
    same = sum(1 for a, b in pairs if _value(a["judgement"]) == _value(b["judgement"]))
    same_c = sum(1 for a, b in pairs if _value(a["contradicted"]) == _value(b["contradicted"]))
    return {"items": len(pairs), "judgement_agreement": _share(same, len(pairs)),
            "contradicted_agreement": _share(same_c, len(pairs)), "table": table,
            "note": "rows of the table: first judgement; columns: second judgement"}


# ---------- files ----------

def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_text(rows: list[dict], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise JudgingError(f"{path.name} already exists; it is not overwritten.")
    path.write_text(text, encoding="utf-8")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise JudgingError(f"{path.name} does not exist yet.")
    return json.loads(path.read_text(encoding="utf-8"))


def _passages(record: dict) -> dict:
    """The passages the model had, made again by the same code, checked against the record's article keys."""
    from src.summary_v2 import DEV_K, _dev_candidates, summary_articles
    runs, aliases = _dev_candidates()
    found = {}
    for run_name in sorted({r["run"] for r in record["records"]}):
        articles = summary_articles(runs[run_name], DEV_K, aliases)
        expected = next(r["articles"] for r in record["records"] if r["run"] == run_name)
        if [a["key"] for a in articles] != [a["key"] for a in expected]:
            raise JudgingError(f"{run_name}: the articles differ from the record; the evidence cannot be shown.")
        found.update({a["key"]: a["description"] for a in articles})
    return found


def _sheet_command(record_name: str) -> None:
    from src.config import DATA_DIR
    from src.evaluate import load_label_set
    from src.fulltext import FULLTEXT_DIR
    from src.fulltext_eval import CHECKED_LABEL_FILE
    from src.summary_run import EVENTS_FILE as DEV_EVENTS, load_events
    record = json.loads((EVAL_DIR / record_name).read_text(encoding="utf-8"))
    sheets = build_sheets(record, _passages(record), load_events(EVAL_DIR / DEV_EVENTS),
                          load_label_set([EVAL_DIR / CHECKED_LABEL_FILE]))
    sheets["key"]["record_file"] = record_name
    files = {EVAL_DIR / ITEMS_FILE: _csv_text(sheets["items"], ITEM_COLUMNS),
             EVAL_DIR / EVENTS_FILE: _csv_text(sheets["events"], EVENT_COLUMNS),
             EVAL_DIR / KEY_FILE: json.dumps(sheets["key"], indent=1),
             Path(FULLTEXT_DIR) / PAGE_FILE: sheets["page"]}
    for path in files:
        if path.exists():
            raise JudgingError(f"{path.name} already exists; it is not overwritten.")
    for path, text in files.items():
        _write_new(path, text)
    print(f"Items: {len(sheets['items'])} rows in data/eval/{ITEMS_FILE}")
    print(f"Events: {len(sheets['events'])} rows in data/eval/{EVENTS_FILE}")
    print(f"Reading page: {Path(FULLTEXT_DIR) / PAGE_FILE}  (open it in your browser)")
    print(f"Key: data/eval/{KEY_FILE}  (do not open it until both sheets are judged and locked)")


def _lock_command() -> None:
    key = _load_json(EVAL_DIR / KEY_FILE)
    if (EVAL_DIR / LOCK_FILE).exists():
        raise JudgingError(f"{LOCK_FILE} already exists: the sheets are locked.")
    lock = lock_record(_read_csv(EVAL_DIR / ITEMS_FILE), _read_csv(EVAL_DIR / EVENTS_FILE), key,
                       (EVAL_DIR / GUIDE_FILE).read_text(encoding="utf-8"),
                       (EVAL_DIR / key["record_file"]).read_text(encoding="utf-8"), key["record_file"])
    _write_new(EVAL_DIR / LOCK_FILE, json.dumps(lock, indent=1))
    print(f"Locked: data/eval/{LOCK_FILE}")


def _score_command() -> None:
    lock = _load_json(EVAL_DIR / LOCK_FILE)
    items, events = _read_csv(EVAL_DIR / ITEMS_FILE), _read_csv(EVAL_DIR / EVENTS_FILE)
    check_lock(lock, items, events)
    key = _load_json(EVAL_DIR / KEY_FILE)
    record = json.loads((EVAL_DIR / key["record_file"]).read_text(encoding="utf-8"))
    result = {"lock": lock, "record_file": key["record_file"], **score(items, events, key, record)}
    _write_new(EVAL_DIR / SCORE_FILE, json.dumps(result, indent=1))
    for design, d in result["designs"].items():
        print(f"{design:<12} produced {d['produced']['count']}/{d['produced']['total']}  "
              f"supported {d['items']['supported_share']}  contradicted {d['items']['contradicted']} "
              f"({d['items']['contradicted_share']})  central {d['items']['central_contradicted']}  "
              f"event coverage {d['events']['coverage']}  passes {d['passes']}")
    print(f"Choice: {result['choice']}")
    print(f"Saved: data/eval/{SCORE_FILE}")


def _rejudge_command() -> None:
    from src.fulltext import FULLTEXT_DIR
    lock = _load_json(EVAL_DIR / LOCK_FILE)                # the first judgements are fixed first
    record = json.loads((EVAL_DIR / lock["record_file"]).read_text(encoding="utf-8"))   # never the key
    rows, rejudge_key = rejudge_rows(_read_csv(EVAL_DIR / ITEMS_FILE))
    titles = {a["key"]: a["title"] for r in record["records"] for a in r["articles"]}
    _write_new(EVAL_DIR / REJUDGE_FILE, _csv_text(rows, ITEM_COLUMNS))
    _write_new(EVAL_DIR / REJUDGE_KEY_FILE, json.dumps(rejudge_key, indent=1))
    _write_new(Path(FULLTEXT_DIR) / REJUDGE_PAGE_FILE, reading_page(rows, [], _passages(record), titles))
    print(f"Rejudge: {len(rows)} rows in data/eval/{REJUDGE_FILE}; page {Path(FULLTEXT_DIR) / REJUDGE_PAGE_FILE}")


def _rejudge_score_command() -> None:
    rejudge_key = _load_json(EVAL_DIR / REJUDGE_KEY_FILE)
    second = _read_csv(EVAL_DIR / REJUDGE_FILE)
    key = _load_json(EVAL_DIR / KEY_FILE)
    fake_key = {n: {"fingerprint": key["items"][str(original)]["fingerprint"]} for n, original in rejudge_key.items()}
    found = item_problems(second, fake_key)
    if found:
        raise JudgingError("The rejudge sheet is not complete:\n" + "\n".join(found))
    result = agreement(_read_csv(EVAL_DIR / ITEMS_FILE), second, rejudge_key)
    _write_new(EVAL_DIR / REJUDGE_SCORE_FILE, json.dumps(result, indent=1))
    print(f"Agreement: {result['judgement_agreement']} on {result['items']} items; saved data/eval/{REJUDGE_SCORE_FILE}")


COMMANDS = {"lock": _lock_command, "score": _score_command, "rejudge": _rejudge_command,
            "rejudge-score": _rejudge_score_command}

if __name__ == "__main__":
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "sheet":
            _sheet_command(sys.argv[2])
        elif len(sys.argv) == 2 and sys.argv[1] in COMMANDS:
            COMMANDS[sys.argv[1]]()
        else:
            print(__doc__)
    except JudgingError as error:
        print(f"[STOP] {error}")
        sys.exit(1)
