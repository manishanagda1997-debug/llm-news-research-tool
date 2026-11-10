"""
Stage 7, week 2, part 2: one summary run per week with the final frozen design, its blind judging, the lock and
the score. The rules are the ones fixed in each week's protocol (week2_protocol.json, SUMMARY_RULE) before its
fetch; this module only carries them out.

The run (once per week; every guard is checked before any model request):
    - the week's protocol equals its tag, and the labels, events and pairs equal the seal
    - round 3 is scored (summary_v2_r3_judging_score.json) and the final design is frozen: the summary freeze
      accepts the current code and settings for k = 5 (src/summary_freeze.py, as in week 1), and the frozen prompt
      version is the one round 3 chose
    - the run cut-off has not passed (10 November 2026, 22:00 IST)
    - no record, attempt marker or saved passages exist for this week
    - tokens: the round 3 log (round 2, round 3 and every week 2 summary run) inside the last 24 hours, plus the
      stated other use, plus 45,000 fit in the daily limit; every call goes through the round 3 allowance guard
    - the ranking test's result exists, and the summary's five articles are its O1M5+S1 top 5
    Then: the summary's five articles are the default's top 5 (O1M5+S1 on the candidates Stage 1 keeps); the
    attempt marker, before any model request and never removed; the passages, outside Git, with fingerprints in
    the record; one summary (direct design, no answer cache, no retry); the record. The relevance check is not run.

Judging (after the run): the industry evidence row first, from the five articles alone; then every item; then the
events; then the Part 7 pair rows for each sealed pair whose two events are among the five articles. Blind, with
the guides of Parts 2, 4, 6 and 7. The sheets are locked before the score; the judging cut-off is 11 November
2026, 05:30 IST: after it, the score records the summary as run but not judged, with no result claimed.

Commands, for each week (tata or reliance):
    python -m src.week2_summary tata run --other-tokens N   then commit the record and the attempt marker
    python -m src.week2_summary tata judge                  then judge the industry row first, then the rest
    python -m src.week2_summary tata lock
    python -m src.week2_summary tata score
"""

import argparse
import csv
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src import week2_eval as w2
from src.config import EVAL_DIR

RECORD_FILE = "week2_summary_run.json"
ATTEMPT_FILE = "week2_summary_attempt.json"
ITEMS_FILE = "summary_judging.csv"
EVENTS_FILE = "summary_events.csv"
INDUSTRY_FILE = "summary_industry.csv"
PAIRS_FILE = "summary_pairs.csv"
KEY_FILE = "summary_judging_key.json"
LOCK_FILE = "summary_judging_lock.json"
SCORE_FILE = "week2_summary_score.json"
K = 5
DESIGN = "direct"
JUDGING_SEED = 49
R3_SCORE = "summary_v2_r3_judging_score.json"
RUN_CUTOFF = datetime.fromisoformat(w2.RUN_CUTOFF_IST)
JUDGING_CUTOFF = datetime.fromisoformat(w2.JUDGING_CUTOFF_IST)
GUIDES = ("SUMMARY_JUDGING_GUIDE.md", "SUMMARY_JUDGING_GUIDE_PART4.md", "SUMMARY_JUDGING_GUIDE_PART6.md",
          "SUMMARY_JUDGING_GUIDE_PART7.md")
JUDGING_CODE = ("src/week2_summary.py", "src/week2_eval.py", "src/summary_judging.py", "src/summary_judging_p4.py",
                "src/transfer_eval.py", "src/summary_v2.py", "src/summary_run.py", "src/evaluate.py",
                "src/preprocess.py")
PAIR_COLUMNS = ["row", "summary", "pair", "outlook_stated", "kept_apart", "note"]


class Week2SummaryError(Exception):
    pass


def paths(w: dict, fulltext_dir=None) -> dict:
    from src.fulltext import FULLTEXT_DIR
    fulltext = Path(fulltext_dir or FULLTEXT_DIR)
    return {"record": w["run_dir"] / RECORD_FILE, "attempt": w["run_dir"] / ATTEMPT_FILE,
            "passages": fulltext / f"week2_passages_{w['run_id']}.json",
            "page": fulltext / f"week2_judging_{w['run_id']}.html"}


# ---------- the guards before the run ----------

def round3_prompt(eval_dir=None) -> str | None:
    path = Path(eval_dir or EVAL_DIR) / R3_SCORE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["choice"]["prompt"]


def freeze_found(chosen_prompt: str | None) -> list[str]:
    """The summary freeze as week 1 checks it, for k = 5, and its prompt version against round 3's choice."""
    from src import summary_freeze as sf
    if chosen_prompt is None:
        return [f"{R3_SCORE} does not exist: round 3 is not scored, so the final design is not decided"]
    try:
        record = sf.frozen_summary(K)
    except sf.FreezeError as error:
        return [str(error)]
    found = []
    if str(record["settings"].get("prompt_version")) != str(chosen_prompt):
        found.append(f"the frozen prompt version is {record['settings'].get('prompt_version')}, round 3 chose "
                     f"prompt {chosen_prompt}: the final design is not frozen yet")
    if record.get("design") != DESIGN:
        found.append(f"the frozen design is {record.get('design')}, not {DESIGN}")
    return found


def week_found(w: dict) -> list[str]:
    found = []
    try:
        protocol_text = w2._protocol_text(w)
        tag = w2._tag_text(w)
        if tag is None or tag.replace("\r\n", "\n") != protocol_text.replace("\r\n", "\n"):
            found.append(f"{w2.PROTOCOL_FILE} differs from the tag {w['tag']}")
    except w2.Week2Error as error:
        found.append(str(error))
    seal = w["run_dir"] / w2.SEAL_FILE
    if not seal.exists():
        found.append("the week is not sealed")
    else:
        changed = w2.seal_changes(json.loads(seal.read_text(encoding="utf-8")), w["run_dir"])
        if changed:
            found.append("changed after the seal: " + ", ".join(changed))
    return found


def run_problems(w: dict, now: datetime, chosen_prompt: str | None, freeze: list[str], budget: dict,
                 fulltext_dir=None) -> list[str]:
    found = week_found(w) + freeze
    if now >= RUN_CUTOFF:                       # "not started by 22:00": a start at 22:00 itself is refused
        found.append(f"the run cut-off ({w2.RUN_CUTOFF_IST}) has passed: the run is reported as not run")
    found += [f"{p.name} already exists: the run is made once" for name, p in paths(w, fulltext_dir).items()
              if name != "page" and p.exists()]
    if not budget["fits"]:
        found.append(f"the tokens do not fit: {budget['logged_tokens']} logged + {budget['other_tokens']} other + "
                     f"{budget['need']} > {budget['limit']}"
                     + (f"; it fits from {budget['earliest_ist']} IST" if budget.get("earliest_ist") else ""))
    return found


def log_runs() -> list:
    """The round 3 token log, with every week 2 summary run added."""
    from src import summary_round3 as r3
    records, markers, _ = r3._load_round()
    runs = r3._log_runs(records, markers)
    for name in w2.WEEKS:
        w = w2.week(name)
        marker, record = w["run_dir"] / ATTEMPT_FILE, w["run_dir"] / RECORD_FILE
        if marker.exists():
            m = json.loads(marker.read_text(encoding="utf-8"))
            if record.exists():
                runs.append(({"records": json.loads(record.read_text(encoding="utf-8"))["summary_record"]["records"]},
                             m["clock_start"]))
            else:
                runs.append((None, m["started_at"]))
    return runs


# ---------- the run ----------

def top5(w: dict, aliases: dict) -> tuple[list[dict], list[dict]]:
    """(candidates, the default's top 5) on the week's sealed sample, with no model."""
    from src import ranking as rk
    from src.research import default_order
    candidates = w2._candidates(w, w2.load_state(w["run_dir"]), aliases)
    ordered = default_order([dict(c, check=None) for c in candidates], w["company"], aliases)
    return candidates, rk.select(ordered, K, "S1")["selected"]


def _fingerprint(data: bytes) -> str:
    """The same value as fresh_eval.file_fingerprint: SHA-256 with Windows line endings made plain."""
    import hashlib
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def checked_ranking(w: dict, keys: list[str]) -> tuple[str, str | None]:
    """(problem, fingerprint): the ranking file is read once; its structure and its O1M5+S1 top 5 are checked,
    and the fingerprint is taken from those same bytes, so it describes exactly the file that was checked. The
    ranking test (no model) comes first: without its result the run is refused."""
    path = w["run_dir"] / w2.RANK_FILE
    if not path.exists():
        return (f"{w2.RANK_FILE} does not exist: run the ranking test first "
                f"(python -m src.week2_eval {w['name']} rank)"), None
    data = path.read_bytes()
    try:
        ranked = json.loads(data.decode("utf-8"))["orders"][w2.DEFAULT]
    except (ValueError, KeyError, TypeError) as error:
        return f"{w2.RANK_FILE} cannot be read as a ranking result ({type(error).__name__}): the run is refused", None
    if not isinstance(ranked, list):
        return f"{w2.RANK_FILE} holds no {w2.DEFAULT} order: the run is refused", None
    if ranked[:K] != keys:
        return f"the summary's articles {keys} are not the ranking test's top {K} {ranked[:K]}", None
    return "", _fingerprint(data)


def rank_mismatch(w: dict, keys: list[str]) -> str:
    return checked_ranking(w, keys)[0]


def ranking_fingerprint(w: dict) -> str | None:
    path = w["run_dir"] / w2.RANK_FILE
    return _fingerprint(path.read_bytes()) if path.exists() else None


def execute(w: dict, articles: list[dict], llm, chosen_prompt: str, other: int, budget: dict, source_commit: str,
            fulltext_dir=None, *, ranking_fp: str) -> dict:
    """The marker first, then the passages, then one summary, then the record. Nothing is retried."""
    from src import summary_round3 as r3
    from src import summary_v2 as sv
    from src.summary_fresh import passage_entry
    p = paths(w, fulltext_dir)
    passages = [passage_entry(a) for a in articles]
    attempt_id = uuid.uuid4().hex
    clock_start = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(p["attempt"], "x", encoding="utf-8") as handle:      # before any model request; never removed
        handle.write(json.dumps({"attempt_id": attempt_id, "source_commit": source_commit, "week": w["name"],
                                 "started_at": clock_start, "clock_start": clock_start, "prompt": chosen_prompt,
                                 "ranking_fingerprint": ranking_fp,
                                 "other_tokens": other, "budget": budget}, indent=1))
    p["passages"].parent.mkdir(parents=True, exist_ok=True)
    with open(p["passages"], "x", encoding="utf-8") as handle:     # the evidence for judging, outside Git
        handle.write(json.dumps({"attempt_id": attempt_id, "source_commit": source_commit, "passages": passages},
                                ensure_ascii=False, indent=1))
    guard = r3.AllowanceGuard(llm)
    result = sv._one(DESIGN, articles, guard, sv.NoCache(), K, sv.CORRECTION)
    messages = sv.take_messages(result)
    record = {"week": w["name"], "company": w["company"], "attempt_id": attempt_id, "source_commit": source_commit,
              "prompt": chosen_prompt, "design": DESIGN, "k": K, "clock_start": clock_start,
              "ranking_file": w2.RANK_FILE, "ranking_fingerprint": ranking_fp,
              "summary_record": {"records": [{"run": w["run_id"], "design": DESIGN, "k": K,
                                              "articles": [{"id": a["id"], "key": a["key"], "title": a["title"],
                                                            "url": a.get("url", "")} for a in articles],
                                              "result": result}]},
              "passage_fingerprints": [{"id": x["id"], "key": x["key"], "sha256": x["sha256"]} for x in passages],
              "tokens": sum(int(c.get("input", 0)) + int(c.get("output", 0))
                            for c in result.get("usage", {}).get("per_call", [])),
              "allowance": r3.ALLOWANCE, "allowance_refused": guard.refused, "allowance_exceeded": guard.exceeded,
              "relevance_check": "not run (experimental, not part of the default)", "cache": "none",
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(p["record"], "x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=1, default=str))
    return {**record, "_messages": messages}


def produced(record: dict) -> bool:
    from src import summary_judging as sj
    return bool(sj._produced(record["summary_record"]))


def _run_command(w: dict, other: int) -> None:
    from src import summary_round3 as r3
    from src import summary_v2 as sv
    from src.query import load_aliases
    if other < 0:
        raise Week2SummaryError("--other-tokens cannot be negative.")
    now = datetime.now(timezone.utc)
    chosen = round3_prompt()
    budget = r3.budget_check(r3.token_log(log_runs()), other, r3.ALLOWANCE, _daily_limit(), now)
    found = run_problems(w, now, chosen, freeze_found(chosen), budget)
    if found:
        raise Week2SummaryError("The run is refused (nothing was sent):\n" + "\n".join(found))
    aliases = load_aliases(w2.ALIAS_FILE)
    _, chosen_articles = top5(w, aliases)
    mismatch, ranking_fp = checked_ranking(w, [c["key"] for c in chosen_articles])   # one read: check and fingerprint
    if mismatch:
        raise Week2SummaryError("The run is refused (nothing was sent):\n" + mismatch)
    articles = sv.passage_articles(chosen_articles, aliases, w["company"])
    record = execute(w, articles, sv.summary_llm(), chosen, other, budget, w2._git("rev-parse", "HEAD"),
                     ranking_fp=ranking_fp)
    for message in record.pop("_messages"):
        print(f"    {message}")
    result = record["summary_record"]["records"][0]["result"]
    print(f"{w['company']}: summary {result['status']}, calls {result.get('usage', {}).get('calls', '-')}, "
          f"tokens {record['tokens']}")
    print(f"Saved: {paths(w)['record']} and {paths(w)['attempt']}  (commit both; do not open the record before "
          f"judging)")


def _daily_limit() -> int:
    from src import summary_round3 as r3
    return json.loads((EVAL_DIR / r3.PLAN_FILE).read_text(encoding="utf-8"))["daily_limit"]


# ---------- judging ----------

def attempt_found(w: dict, record: dict, fulltext_dir=None) -> list[str]:
    p = paths(w, fulltext_dir)
    found = []
    if not p["attempt"].exists():
        return ["the attempt marker is missing"]
    marker = json.loads(p["attempt"].read_text(encoding="utf-8"))
    if marker.get("attempt_id") != record.get("attempt_id") or marker.get("source_commit") != record.get("source_commit"):
        found.append("the attempt marker and the record come from different attempts")
    if not p["passages"].exists():
        found.append("the saved passages are missing")
    if not record.get("ranking_fingerprint") or marker.get("ranking_fingerprint") != record.get("ranking_fingerprint"):
        found.append("the record holds no ranking fingerprint, or the marker holds another one")
    elif ranking_fingerprint(w) != record["ranking_fingerprint"]:
        found.append(f"{w2.RANK_FILE} changed after the run: the summary's articles can no longer be traced to it")
    return found


def saved_passages(record: dict, saved: dict) -> dict:
    from src import summary_judging_fresh as sjf
    if saved.get("attempt_id") != record.get("attempt_id") or saved.get("source_commit") != record.get("source_commit"):
        raise Week2SummaryError("The saved passages come from another attempt than the record.")
    shaped = {"records": record["summary_record"]["records"], "passage_fingerprints": record["passage_fingerprints"]}
    try:
        return sjf.passages_from_record(shaped, saved.get("passages") or [])
    except sjf.JudgingError as error:
        raise Week2SummaryError(str(error)) from None


def labels_by_url(run_dir: Path) -> dict[str, str]:
    from src.preprocess import normalize_url
    with open(Path(run_dir) / w2.LABEL_FILE, encoding="utf-8-sig", newline="") as handle:
        return {normalize_url(row["url"]): (row.get("label") or "").strip().lower() for row in csv.DictReader(handle)}


def pair_rows(record: dict, events_by_url: dict[str, str], pairs: list[dict], letter: str = "A") -> list[dict]:
    """One row per sealed pair whose two events are both among the summary's five articles."""
    from src.preprocess import normalize_url
    present = {events_by_url.get(normalize_url(a.get("url") or "")) for a in record["summary_record"]["records"][0]["articles"]}
    rows = []
    for pair in pairs:
        first = (pair.get("expectation_event") or "").strip().lower()
        second = (pair.get("outcome_event") or "").strip().lower()
        if first and second and first in present and second in present:
            rows.append({"row": len(rows) + 1, "summary": letter, "pair": f"{first} -> {second}",
                         "outlook_stated": "", "kept_apart": "", "note": ""})
    return rows


def build(w: dict, record: dict, passages: dict) -> dict:
    from src import summary_judging as sj
    from src import summary_judging_p4 as p4
    from src.summary_run import load_events
    events_by_url = load_events(w["run_dir"] / w2.EVENTS_FILE)
    sheets = p4.build_sheets(record["summary_record"], passages, events_by_url, labels_by_url(w["run_dir"]), {},
                             seed=JUDGING_SEED)
    letter = sorted(sheets["key"]["summaries"])[0]
    pairs = pair_rows(record, events_by_url, w2._read_pairs(w["run_dir"]), letter)
    sheets["key"]["pairs"] = {str(r["row"]): {"pair": r["pair"], "fingerprint": sj._hash(r["summary"], r["pair"])}
                              for r in pairs}
    sheets["key"]["record_file"] = RECORD_FILE
    return {**sheets, "pairs": pairs}


def pair_problems(rows: list[dict], key: dict) -> list[str]:
    from src import summary_judging as sj
    found = []
    by_number = sj._rows_by_number(rows)
    if set(by_number) != set(key.get("pairs", {})):
        return ["pairs: the rows do not match the sheet made by the tool"]
    for number, row in by_number.items():
        if sj._hash(row["summary"], row["pair"]) != key["pairs"][number]["fingerprint"]:
            found.append(f"pairs row {number}: the summary or the pair changed")
            continue
        for column in ("outlook_stated", "kept_apart"):
            if sj._value(row[column]) not in sj.YES_NO:
                found.append(f"pairs row {number}: {column} must be yes or no")
        if "no" in (sj._value(row["outlook_stated"]), sj._value(row["kept_apart"])) and not sj._value(row.get("note")):
            found.append(f"pairs row {number}: every no needs a note")
    return found


def _load_record(w: dict) -> dict:
    path = paths(w)["record"]
    if not path.exists():
        raise Week2SummaryError("The run has not been made for this week.")
    return json.loads(path.read_text(encoding="utf-8"))


def _judge_command(w: dict) -> None:
    from src import summary_judging as sj
    from src import summary_judging_p4 as p4
    found = week_found(w)
    record = _load_record(w)
    found += attempt_found(w, record)
    if found:
        raise Week2SummaryError("Judging is refused:\n" + "\n".join(found))
    if not produced(record):
        raise Week2SummaryError("The run gives no summary: 'produced' fails and there is nothing to judge. Run the score.")
    passages = saved_passages(record, json.loads(paths(w)["passages"].read_text(encoding="utf-8")))
    sheets = build(w, record, passages)
    files = {w["run_dir"] / ITEMS_FILE: sj._csv_text(sheets["items"], sj.ITEM_COLUMNS),
             w["run_dir"] / EVENTS_FILE: sj._csv_text(sheets["events"], sj.EVENT_COLUMNS),
             w["run_dir"] / INDUSTRY_FILE: sj._csv_text(sheets["industry"], p4.INDUSTRY_COLUMNS),
             w["run_dir"] / PAIRS_FILE: sj._csv_text(sheets["pairs"], PAIR_COLUMNS),
             w["run_dir"] / KEY_FILE: json.dumps(sheets["key"], indent=1),
             paths(w)["page"]: sheets["page"]}
    for target in files:
        if target.exists():
            raise Week2SummaryError(f"{target.name} already exists; it is not overwritten.")
    for target, text in files.items():
        sj._write_new(target, text)
    print(f"Industry evidence: 1 row in {w['run_dir'] / INDUSTRY_FILE}  (judge it first, from the articles alone)")
    print(f"Items: {len(sheets['items'])} rows; events: {len(sheets['events'])} rows; pairs: {len(sheets['pairs'])} rows")
    print(f"Reading page: {paths(w)['page']}\nKey: {w['run_dir'] / KEY_FILE}  (do not open it before the lock)")


def _sheets(w: dict) -> tuple:
    from src import summary_judging as sj
    return tuple(sj._read_csv(w["run_dir"] / name) for name in (ITEMS_FILE, EVENTS_FILE, INDUSTRY_FILE, PAIRS_FILE))


def judging_inputs(w: dict) -> dict:
    from src.fresh_eval import file_fingerprint
    named = {"protocol": w["run_dir"] / w2.PROTOCOL_FILE, "seal": w["run_dir"] / w2.SEAL_FILE,
             "labels": w["run_dir"] / w2.LABEL_FILE, "events": w["run_dir"] / w2.EVENTS_FILE,
             "pairs": w["run_dir"] / w2.PAIRS_FILE, "record": paths(w)["record"], "attempt": paths(w)["attempt"],
             "passages": paths(w)["passages"], "freeze": EVAL_DIR / "summary_freeze.json",
             "ranking": w["run_dir"] / w2.RANK_FILE}
    return {"guides": [file_fingerprint(EVAL_DIR / g) for g in GUIDES],
            **{name: file_fingerprint(path) if path.exists() else None for name, path in named.items()}}


def code_fingerprint() -> str:
    from src import summary_prompt6 as p6
    return p6.code_fingerprint(JUDGING_CODE)


def lock_record(w: dict, sheets: tuple, key: dict) -> dict:
    from src import summary_judging as sj
    from src import summary_judging_p4 as p4
    items, events, industry, pairs = sheets
    found = p4.problems(items, events, industry, key) + pair_problems(pairs, key)
    if found:
        raise Week2SummaryError("The sheets are not complete:\n" + "\n".join(found))
    return {"sheets": [sj.sheet_fingerprint(s) for s in sheets], "key": sj._hash(key), "code": code_fingerprint(),
            "inputs": judging_inputs(w), "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def lock_changes(lock: dict, w: dict, sheets: tuple, key: dict) -> list[str]:
    from src import summary_judging as sj
    if len(lock.get("sheets") or []) != 4 or len(sheets) != 4:
        return [f"the lock must hold exactly 4 sheet fingerprints (items, events, industry, pairs); it holds "
                f"{len(lock.get('sheets') or [])}"]
    found = [f"the {n} sheet changed after the lock" for n, s, fp in
             zip(("items", "events", "industry", "pairs"), sheets, lock["sheets"]) if sj.sheet_fingerprint(s) != fp]
    if lock["key"] != sj._hash(key):
        found.append("the key changed after the lock")
    if lock["code"] != code_fingerprint():
        found.append("the judging code changed after the lock")
    now = judging_inputs(w)
    found += [f"{n} changed after the lock" for n in sorted(set(lock["inputs"]) | set(now))
              if lock["inputs"].get(n) != now.get(n)]
    return found


def _lock_command(w: dict, now: datetime | None = None) -> None:
    from src import summary_judging as sj
    if (w["run_dir"] / LOCK_FILE).exists():
        raise Week2SummaryError(f"{LOCK_FILE} already exists: the sheets are locked.")
    found = week_found(w) + attempt_found(w, _load_record(w))
    if found:
        raise Week2SummaryError("The lock is refused:\n" + "\n".join(found))
    lock = lock_record(w, _sheets(w), sj._load_json(w["run_dir"] / KEY_FILE))
    lock["after_judging_cutoff"] = (now or datetime.now(timezone.utc)) > JUDGING_CUTOFF
    sj._write_new(w["run_dir"] / LOCK_FILE, json.dumps(lock, indent=1))
    print(f"Locked: {w['run_dir'] / LOCK_FILE}")


# ---------- the score ----------

def pass_levels() -> dict:
    return {"pass_levels": {"summary": {**w2.SUMMARY_RULE["pass_levels"], "k": K}}}


def score_result(w: dict, record: dict, sheets: tuple | None, key: dict | None, lock: dict | None) -> dict:
    from src import summary_judging as sj
    from src import transfer_eval as te
    base = {"week": w["name"], "company": w["company"], "prompt": record["prompt"], "design": record["design"],
            "relevance_check": "not run"}
    if not produced(record):
        return {**base, "summary": te.no_summary_result(), "status": "run, no summary"}
    if lock is None or lock.get("after_judging_cutoff"):
        return {**base, "summary": {"status": "run, not judged", "reason": "the judgements are not locked by the "
                                    f"judging cut-off ({w2.JUDGING_CUTOFF_IST}); no result is claimed"},
                "status": "run, not judged"}
    items, events, industry, pairs = sheets
    judged = sj.score(items, events, key, record["summary_record"])["designs"][record["design"]]
    pair_values = [{"pair": r["pair"], "outlook_stated": sj._value(r["outlook_stated"]),
                    "kept_apart": sj._value(r["kept_apart"])} for r in pairs]
    return {**base, "status": "judged", "summary": te.summary_result(judged, pass_levels()),
            "industry": te.industry_report(items, industry, key, record["summary_record"]),
            "pairs": {"rows": pair_values, "level": "reported without a pass level"}}


def _score_command(w: dict) -> None:
    from src import summary_judging as sj
    path = w["run_dir"] / SCORE_FILE
    if path.exists():
        raise Week2SummaryError(f"{SCORE_FILE} already exists; it is not overwritten.")
    record = _load_record(w)
    found = week_found(w) + attempt_found(w, record)
    sheets = key = lock = None
    if produced(record) and (w["run_dir"] / LOCK_FILE).exists():
        sheets, key = _sheets(w), sj._load_json(w["run_dir"] / KEY_FILE)
        lock = sj._load_json(w["run_dir"] / LOCK_FILE)
        found += lock_changes(lock, w, sheets, key)
    if found:
        raise Week2SummaryError("The score is refused:\n" + "\n".join(found))
    result = {**score_result(w, record, sheets, key, lock),
              "scored": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    summary = result["summary"]
    print(f"{w['company']}: {result['status']}; passed {summary.get('passed')}; checks {summary.get('checks')}")
    print(f"Saved: {path}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.week2_summary")
    parser.add_argument("week", choices=list(w2.WEEKS))
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--other-tokens", type=int, required=True,
                     help="Groq tokens used outside this project's records in the last 24 hours (0 when none)")
    for name in ("judge", "lock", "score"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    w = w2.week(args.week)
    if args.command == "run":
        _run_command(w, args.other_tokens)
    else:
        {"judge": _judge_command, "lock": _lock_command, "score": _score_command}[args.command](w)


if __name__ == "__main__":
    try:
        main()
    except (Week2SummaryError, w2.Week2Error) as error:
        print(f"[STOP] {error}")
        sys.exit(1)
