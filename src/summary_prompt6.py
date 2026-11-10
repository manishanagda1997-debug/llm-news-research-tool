"""
Prompt version 6 (second development cycle, round 2): prompt version 4 with one new rule for an expectation and
the outcome of the same development, and one added sentence to the industry rule. It is tested against the locked
prompt version 4 runs on the three Tata Motors development days, with the same five articles per day and the same
settings, so only the prompt differs. The frozen files are not changed: the run sets the prompt for its own
requests only and records its exact text and fingerprint. Three outputs at temperature 1.0 are evidence about
these outputs, not a stable estimate of the prompt.

Judging (SUMMARY_JUDGING_GUIDE_PART7.md), in order:
    sheet 1   both versions hidden: the six summaries under shuffled letters; for each pair in a summary's
              articles, outlook_stated and kept_apart (yes or no); for each industry point, overreach (yes or no)
    sheet 2   prompt 6's new or changed items, judged with Parts 2 and 4, mixed with 10 locked prompt 4 items as
              unmarked decoys (their new judgements are only compared with the locked ones); an item identical to
              a locked item keeps its judgement; the industry evidence is carried, because the articles are the
              same; and the event coverage of prompt 6's summaries
The rule (plan, fixed before generation): prompt 6 replaces prompt 4 only if its pair score is not lower on either
pair and higher in total, and every guardrail holds: no unsupported industry point; no industry points without
evidence; days with an empty industry section despite evidence not more than prompt 4's; overreach count not higher than prompt 4's; no unsupported key point or brief;
no contradicted item; key points and briefs supported share at least 0.75, unrounded; event coverage on each day not lower than prompt 4's. A tie keeps prompt 4.

Commands, in order:
    python -m src.summary_prompt6 plan    then commit, then: git tag freeze-prompt6-plan
    python -m src.summary_prompt6 run     once; model calls (about 30,000 tokens)
    python -m src.summary_prompt6 sheet   then judge sheet 1 first, then sheet 2 and the events
    python -m src.summary_prompt6 lock
    python -m src.summary_prompt6 score
"""

import hashlib
import json
import random
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src import summary_judging as sj
from src import summary_judging_p4 as p4
from src import summary_judging_p5 as p5
from src.config import EVAL_DIR, ROOT_DIR

PROMPT_VERSION = "6"
PLAN_FILE = "summary_v2_p6_plan.json"
RECORD_FILE = "summary_v2_dev_k5_c7_p6.json"
ATTEMPT_FILE = "summary_v2_dev_k5_c7_p6_attempt.json"
PLAN_TAG = "freeze-prompt6-plan"
P4_RECORD = "summary_v2_dev_k5_c7_p4.json"
PAIRS_FILE = "summary_v2_p6_pairs.csv"
OVERREACH_FILE = "summary_v2_p6_overreach.csv"
ITEMS_FILE = "summary_v2_p6_judging.csv"
EVENTS_FILE = "summary_v2_p6_events.csv"
KEY_FILE = "summary_v2_p6_judging_key.json"
LOCK_FILE = "summary_v2_p6_judging_lock.json"
SCORE_FILE = "summary_v2_p6_judging_score.json"
PAGE1_FILE = "summary_v2_p6_sheet1.html"          # in data/fulltext/, outside Git
PAGE2_FILE = "summary_v2_p6_sheet2.html"          # in data/fulltext/, outside Git
GUIDE_FILE = "SUMMARY_JUDGING_GUIDE_PART7.md"
GUIDES = (sj.GUIDE_FILE, p4.GUIDE_FILE, GUIDE_FILE)
SEED_SHEET1, SEED_SHEET2, SEED_DECOYS = 61, 62, 63
DECOYS = 10
MIN_SHARE = 0.75                                   # prompt 4's 32 of 40, minus 5 percentage points
PAIRS = {"2026-10-24": ("tata_motors_cv_q2_preview", "tata_motors_q2_actual_results"),
         "2026-10-23": ("iveco_shareholder_acceptance", "iveco_shareholders_approve")}
PAIR_COLUMNS = ["row", "summary", "pair", "outlook_stated", "kept_apart", "note"]
OVERREACH_COLUMNS = ["row", "summary", "text", "evidence", "overreach", "note"]
CODE_FILES = ("src/summary_prompt6.py", "src/summary_judging.py", "src/summary_judging_p4.py",
              "src/summary_judging_p5.py", "src/summarizer.py", "src/summary_v2.py", "src/llm.py",
              "src/relevance_check.py", "src/summary_passage.py", "src/boilerplate.py", "src/ranking.py",
              "src/ranking_eval.py", "src/preprocess.py", "src/query.py", "src/fulltext.py", "src/fulltext_eval.py")

NEW_RULE = ("When one article reports what is expected or an earlier stage of a development (a forecast, a preview, "
            "a deadline, a stage of acceptance) and another article reports what then happens in the same "
            "development, give each its own point: state the expectation as an expectation, with who expects it, and "
            "the outcome as the outcome. Never present an expectation as a result, or a result as an expectation. "
            "Articles about different developments are not linked by this rule.")
INDUSTRY_SENTENCE = ('An "industry_context" point states only what the article says about the sector, the market or '
                     "other companies; it never extends the company's own results to them.")
RULE = {
    "pairs": {day: list(pair) for day, pair in PAIRS.items()},
    "no_pair": "2026-10-14 has no pair; it counts only in the guardrails",
    "pair_score": "per pair and version: outlook_stated + kept_apart, each yes = 1 (0 to 2)",
    "replace": "prompt 6 replaces prompt 4 only if its pair score is not lower on either pair and its total is higher, "
               "and every guardrail holds; a tie keeps prompt 4; with prompt 4 at 4 of 4, prompt 4 stays",
    "guardrails": ["no unsupported industry point", "no unsupported key point or brief",
                   "no day with industry points when the evidence decision is no",
                   "days with an empty industry section despite evidence not more than prompt 4's, counted from "
                   "its locked score",
                   "overreach count of prompt 6 not higher than prompt 4's (both judged hidden on sheet 1)",
                   "no contradicted item",
                   "key points and visible briefs judged supported / all key points and visible briefs, over prompt "
                   "6's three summaries, at least 0.75, unrounded; industry points are not in this share",
                   "event coverage on each day not lower than prompt 4's (same articles, same denominators; a tie is "
                   "not lower)"],
    "decoys": "10 locked prompt 4 items, fixed seed, unmarked on sheet 2; never used in the choice",
    "outputs": "one output per day at temperature 1.0: evidence about these outputs, not a stable estimate",
    "failed_day": "if any expected day has no usable summary (status ok or partial, with a summary) in either "
                  "version, prompt 4 remains selected, no sheet is judged, and the run is not repeated",
    "empty_days_counted": "the industry guardrail compares the number of days with an empty section despite "
                          "evidence, not which days; the same count can stand for different days",
    "recognition": "the judge has seen the prompt 4 summaries in Part 4, so the hiding reduces recognition bias but "
                   "is not perfect; each row is judged only against the articles shown, without the Part 4 sheets",
}


class Prompt6Error(Exception):
    pass


# ---------- the prompt ----------

def prompt_v6(v4: str) -> str:
    """Prompt 4 with the new rule after rule 3, the industry sentence added to rule 4, and the rules renumbered."""
    lines = v4.split("\n")
    numbered = [i for i, line in enumerate(lines) if line[:1].isdigit() and line[1:3] == ". "]
    if [lines[i][0] for i in numbered] != [str(n) for n in range(1, 10)]:
        raise Prompt6Error("Prompt 4 does not have rules 1 to 9 in order.")
    rules = [lines[i][3:] for i in numbered]
    if not rules[3].startswith('An "industry_context" point needs an article'):
        raise Prompt6Error("Rule 4 of prompt 4 is not the industry rule.")
    rules = rules[:3] + [NEW_RULE, rules[3] + " " + INDUSTRY_SENTENCE] + rules[4:]
    before, after = lines[:numbered[0]], lines[numbered[-1] + 1:]
    return "\n".join(before + [f"{n}. {text}" for n, text in enumerate(rules, start=1)] + after)


def prompt_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@contextmanager
def prompt_in_use(text: str):
    """Prompt 6 for the requests made inside the block only; prompt 4 again afterwards, whatever happens."""
    from src import summarizer as sm
    saved = sm.RULES
    sm.RULES = text
    try:
        yield
    finally:
        sm.RULES = saved


# ---------- the plan and the run ----------

def code_fingerprint(names=CODE_FILES) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8") + b"\0" + (ROOT_DIR / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def plan_record(prompt: str, p4_record_text: str, source_commit: str) -> dict:
    from src import summary_v2 as sv
    return {"prompt_version": PROMPT_VERSION, "prompt_text": prompt, "prompt_fingerprint": prompt_fingerprint(prompt),
            "new_rule": NEW_RULE, "industry_sentence": INDUSTRY_SENTENCE, "rule": RULE,
            "settings": {"design": "direct", "k": sv.DEV_K, "model_setting": sv.MODEL_SETTING,
                         "output_limit": sv.OUTPUT_LIMIT, "correction": sv.CORRECTION,
                         "configuration": sv.SUMMARY_VERSION, "pacer_version": sv.PACER_VERSION,
                         "cache": "none"},
            "baseline_record": P4_RECORD, "baseline_fingerprint": sj._hash(p4_record_text.replace("\r\n", "\n")),
            "seeds": {"sheet1": SEED_SHEET1, "sheet2": SEED_SHEET2, "decoys": SEED_DECOYS}, "decoys": DECOYS,
            "guide": GUIDE_FILE, "guide_fingerprint": sj._hash((EVAL_DIR / GUIDE_FILE).read_text(encoding="utf-8")
                                                               .replace("\r\n", "\n")),
            "code_files": list(CODE_FILES), "code_fingerprint": code_fingerprint(), "source_commit": source_commit,
            "tag": PLAN_TAG, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def run_problems(plan_text, tag_text, plan, p4_record_text, prompt, existing) -> list[str]:
    def same(a, b):
        return a is not None and b is not None and a.replace("\r\n", "\n") == b.replace("\r\n", "\n")
    found = [f"{name} already exists: the run is made once" for name in existing]
    if not same(plan_text, tag_text):
        found.append(f"{PLAN_FILE} is missing or differs from the tag {PLAN_TAG}")
    if plan is not None:
        if plan["prompt_fingerprint"] != prompt_fingerprint(prompt):
            found.append("the prompt differs from the plan")
        if plan["baseline_fingerprint"] != sj._hash(p4_record_text.replace("\r\n", "\n")):
            found.append(f"{P4_RECORD} differs from the plan")
        if plan["code_fingerprint"] != code_fingerprint():
            found.append("the code differs from the plan")
        guide = EVAL_DIR / GUIDE_FILE
        if "guide_fingerprint" in plan and (not guide.exists() or plan["guide_fingerprint"] != sj._hash(
                guide.read_text(encoding="utf-8").replace("\r\n", "\n"))):
            found.append(f"{GUIDE_FILE} differs from the plan")
    return found


def same_articles(baseline: dict, articles_by_run: dict) -> list[str]:
    """Each development day must give exactly the articles of the locked prompt 4 run, in the same order."""
    found = []
    for entry in baseline["records"]:
        keys = [a["key"] for a in articles_by_run.get(entry["run"], [])]
        if keys != [a["key"] for a in entry["articles"]]:
            found.append(f"{entry['run']}: the articles differ from the prompt 4 run")
    return found


def _git(*args) -> str:
    from src.fresh_eval import git
    return git(*args)


def _tag_text() -> str | None:
    try:
        return _git("show", f"{PLAN_TAG}:{(EVAL_DIR / PLAN_FILE).relative_to(ROOT_DIR).as_posix()}")
    except Exception:
        return None


def _v4_prompt() -> str:
    from src import summarizer as sm
    return sm.PROMPT_TEXTS["4"]


def _plan_command() -> None:
    path = EVAL_DIR / PLAN_FILE
    for done in (path, EVAL_DIR / RECORD_FILE, EVAL_DIR / ATTEMPT_FILE):
        if done.exists():
            raise Prompt6Error(f"{done.name} exists: the plan comes once, before the run.")
    record = plan_record(prompt_v6(_v4_prompt()), (EVAL_DIR / P4_RECORD).read_text(encoding="utf-8"),
                         _git("rev-parse", "HEAD"))
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=1))
    print(record["prompt_text"])
    print(f"\nSaved: {path}\nNext: commit it, then: git tag {PLAN_TAG}")


def _run_command() -> None:
    from src import summary_v2 as sv
    plan_path, record_path, attempt_path = EVAL_DIR / PLAN_FILE, EVAL_DIR / RECORD_FILE, EVAL_DIR / ATTEMPT_FILE
    plan_text = plan_path.read_text(encoding="utf-8") if plan_path.exists() else None
    plan = json.loads(plan_text) if plan_text else None
    p4_text = (EVAL_DIR / P4_RECORD).read_text(encoding="utf-8")
    prompt = prompt_v6(_v4_prompt())
    found = run_problems(plan_text, _tag_text(), plan, p4_text, prompt,
                         [p.name for p in (record_path, attempt_path) if p.exists()])
    runs, aliases = sv._dev_candidates()
    articles_by_run = {name: sv.summary_articles(candidates, sv.DEV_K, aliases) for name, candidates in runs.items()}
    found += same_articles(json.loads(p4_text), articles_by_run)
    if found:
        raise Prompt6Error("The run is refused:\n" + "\n".join(found))
    with open(attempt_path, "x", encoding="utf-8") as handle:       # before any model request; never removed
        handle.write(json.dumps({"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "prompt_fingerprint": prompt_fingerprint(prompt)}, indent=1))
    llm, records = sv.summary_llm(), []
    with prompt_in_use(prompt):
        for name, articles in articles_by_run.items():
            result = sv._one("direct", articles, llm, sv.NoCache(), sv.DEV_K)
            for message in sv.take_messages(result):
                print(f"    {message}")
            records.append(sv._record(name, "direct", articles, result))
            print(f"{name} direct {result['status']} calls {result.get('usage', {}).get('calls', '-')} "
                  f"failures {result.get('failures', [])} removed {len(result.get('removed', []))}")
    out = {"version": sv.SUMMARY_VERSION, "prompt_version": PROMPT_VERSION,
           "prompt_fingerprint": prompt_fingerprint(prompt), "number_check": sv.NUMBER_CHECK_VERSION,
           "pacer_version": sv.PACER_VERSION, "model_setting": sv.MODEL_SETTING, "model": llm.model_name,
           "generation": getattr(llm, "generation", None), "output_limit_sent": getattr(llm, "output_limit", None),
           "k": sv.DEV_K, "designs": ["direct"], "cache": "none", "records": records}
    with open(record_path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"Saved: {record_path}  (commit it with the attempt marker)")


# ---------- complete summaries ----------

def missing_summaries(records: dict[str, dict], expected_runs) -> list[str]:
    """Every expected day must have one usable summary in each version; anything else is listed."""
    found = []
    for version in ("4", "6"):
        record = records.get(version) or {"records": []}
        runs = [r["run"] for r in record["records"]]
        usable = {r["run"] for r in sj._produced(record)}
        for run in sorted(expected_runs):
            if runs.count(run) != 1:
                found.append(f"prompt {version}, {run}: {runs.count(run)} records instead of 1")
            elif run not in usable:
                found.append(f"prompt {version}, {run}: no usable summary")
        found += [f"prompt {version}, {run}: not an expected day" for run in sorted(set(runs) - set(expected_runs))]
    return found


def expected_pairs() -> list[tuple[str, str]]:
    """The planned pair rows: one per version and paired day, four in all."""
    return [(version, run) for version in ("4", "6") for run in sorted(PAIRS)]


def failed_result(missing: list[str]) -> dict:
    return {"choice": {"prompt": "4", "reason": "failed day", "missing": missing}, "rule": RULE,
            "scored": datetime.now(timezone.utc).isoformat(timespec="seconds")}


# ---------- sheet 1: pairs and overreach, both versions hidden ----------

def sheet1(records: dict[str, dict], passages: dict, seed: int = SEED_SHEET1) -> dict:
    """records: {"4": record, "6": record}. One letter per summary, shuffled; versions only in the key."""
    entries = [(version, entry) for version, record in records.items() for entry in sj._produced(record)]
    rng = random.Random(seed)
    rng.shuffle(entries)
    letters = {chr(ord("A") + n): {"version": v, "run": e["run"]} for n, (v, e) in enumerate(entries)}
    by_letter = {letter: entry for letter, (_, entry) in zip(letters, entries)}
    pair_rows, overreach_rows, pair_key, overreach_key = [], [], {}, {}
    for letter, entry in by_letter.items():
        if entry["run"] in PAIRS:
            number = len(pair_rows) + 1
            pair = " -> ".join(PAIRS[entry["run"]])
            pair_rows.append({"row": number, "summary": letter, "pair": pair, "outlook_stated": "",
                              "kept_apart": "", "note": ""})
            pair_key[str(number)] = {"letter": letter, "fingerprint": sj._hash(letter, pair)}
    points = [(letter, point) for letter, entry in by_letter.items()
              for point in entry["result"]["summary"].get("industry_context", [])]
    rng.shuffle(points)
    for letter, point in points:
        keys = sj._keys(by_letter[letter])
        number = len(overreach_rows) + 1
        evidence = "; ".join(keys[c] for c in point["citations"])
        overreach_rows.append({"row": number, "summary": letter, "text": point["text"], "evidence": evidence,
                               "overreach": "", "note": ""})
        overreach_key[str(number)] = {"letter": letter, "fingerprint": sj._hash(letter, point["text"], evidence)}
    planned = sorted(f"{v}:{r}" for v, r in expected_pairs())
    made = sorted(f"{letters[pair_key[n]['letter']]['version']}:{letters[pair_key[n]['letter']]['run']}" for n in pair_key)
    if made != planned:
        raise Prompt6Error(f"Sheet 1 needs the planned pair rows {planned}; it would have {made}.")
    key = {"letters": letters, "pairs": pair_key, "overreach": overreach_key, "seed": seed, "planned_pairs": planned}
    return {"pairs": pair_rows, "overreach": overreach_rows, "key": key, "page": _page1(by_letter, passages)}


def _page1(by_letter: dict, passages: dict) -> str:
    import html
    e = html.escape
    parts = ["<!doctype html><meta charset='utf-8'><title>Summary judging, Part 7, sheet 1</title>",
             "<style>body{font-family:sans-serif;max-width:900px;margin:auto;padding:16px;line-height:1.5}"
             ".item{border-top:2px solid #444;margin-top:24px}.ev{background:#f4f4f4;padding:8px;margin:6px 0}</style>"]
    for letter, entry in by_letter.items():
        summary, keys = entry["result"]["summary"], sj._keys(entry)
        parts.append(f"<div class='item'><h2>Summary {letter}</h2>")
        for section, name in (("key_points", "Company points"), ("industry_context", "Industry points")):
            parts.append(f"<h3>{name}</h3><ul>")
            for point in summary.get(section, []):
                parts.append(f"<li>{e(point['text'])}<br><small>Cites: "
                             f"{'; '.join(e(keys[c]) for c in point['citations'])}</small></li>")
            if not summary.get(section):
                parts.append("<li><i>(none)</i></li>")
            parts.append("</ul>")
        parts.append("<h3>Its articles</h3>")
        for article in entry["articles"]:
            parts.append(f"<div class='ev'><b>{e(article['key'])}: {e(article['title'])}</b>"
                         f"<p>{e(passages.get(article['key'], '(no text)'))}</p></div>")
        parts.append("</div>")
    return "".join(parts)


def sheet1_problems(pairs, overreach, key) -> list[str]:
    found = []
    planned = sorted(f"{v}:{r}" for v, r in expected_pairs())
    if key.get("planned_pairs") != planned:
        found.append("the key does not hold the planned pair rows")
    in_key = sorted(f"{key['letters'][e['letter']]['version']}:{key['letters'][e['letter']]['run']}"
                    for e in key["pairs"].values())
    if in_key != planned:
        found.append(f"pairs: the key has {in_key}, not the planned {planned}")
    for name, rows, columns in (("pairs", pairs, ("outlook_stated", "kept_apart")), ("overreach", overreach, ("overreach",))):
        by_number = sj._rows_by_number(rows)
        if set(by_number) != set(key[name]) or len(by_number) != len(rows):
            found.append(f"{name}: rows do not match the sheet made by the tool")
            continue
        for number, row in by_number.items():
            parts = (row["summary"], row["pair"]) if name == "pairs" else (row["summary"], row["text"], row["evidence"])
            if sj._hash(*parts) != key[name][number]["fingerprint"]:
                found.append(f"{name} row {number}: summary, pair or text changed")
                continue
            for column in columns:
                if sj._value(row[column]) not in sj.YES_NO:
                    found.append(f"{name} row {number}: {column} must be yes or no")
            if name == "overreach" and sj._value(row["overreach"]) == "yes" and not sj._value(row.get("note")):
                found.append(f"{name} row {number}: the note names the extended claim")
    return found


# ---------- sheet 2: prompt 6's items with decoys ----------

def add_decoys(sheets: dict, p4_items: list[dict], seed_decoys: int = SEED_DECOYS,
               seed: int = SEED_SHEET2, count: int = DECOYS) -> dict:
    """Mix `count` locked prompt 4 items into the items sheet as unmarked rows; renumber everything in a new
    shuffled order. Decoys are marked in the key only."""
    rows = sj._rows_by_number(p4_items)
    pool = sorted(rows, key=int)
    picked = random.Random(seed_decoys).sample(pool, count)
    combined = [(row, sheets["key"]["items"][str(row["row"])]) for row in sheets["items"]]
    combined += [({"row": f"p4:{n}", "kind": rows[n]["kind"], "text": rows[n]["text"], "evidence": rows[n]["evidence"]},
                  {"decoy": True, "locked_row": n, "locked_judgement": sj._value(rows[n]["judgement"]),
                   "locked_contradicted": sj._value(rows[n]["contradicted"])}) for n in picked]
    random.Random(seed).shuffle(combined)
    items, item_key = [], {}
    for number, (row, entry) in enumerate(combined, start=1):
        new = {"row": number, "kind": row["kind"], "text": row["text"], "evidence": row["evidence"],
               "judgement": "", "contradicted": "", "central": "", "note": ""}
        items.append(new)
        item_key[str(number)] = {**{k: v for k, v in entry.items() if k != "fingerprint"},
                                 "fingerprint": sj._hash(new["kind"], new["text"], new["evidence"])}
    key = dict(sheets["key"], items=item_key)
    return dict(sheets, items=items, key=key)


def scored_items(items: list[dict], key: dict) -> tuple[list[dict], dict]:
    """The rows and key without the decoys, renumbered, in the form the Part 4 score reads."""
    rows = sj._rows_by_number(items)
    kept = [(rows[n], entry) for n, entry in sorted(key["items"].items(), key=lambda p: int(p[0])) if not entry.get("decoy")]
    out_rows, out_key = [], {}
    for number, (row, entry) in enumerate(kept, start=1):
        out_rows.append(dict(row, row=number))
        out_key[str(number)] = entry
    return out_rows, dict(key, items=out_key)


def decoy_agreement(items: list[dict], key: dict) -> dict:
    rows = sj._rows_by_number(items)
    decoys = [(rows[n], e) for n, e in key["items"].items() if e.get("decoy")]
    same = sum(1 for row, e in decoys if sj._value(row["judgement"]) == e["locked_judgement"])
    return {"decoys": len(decoys), "same_judgement": same,
            "same_contradicted": sum(1 for row, e in decoys if sj._value(row["contradicted"]) == e["locked_contradicted"]),
            "note": "consistency only; decoys are never used in the choice"}


# ---------- the choice ----------

def pair_scores(pairs: list[dict], key: dict) -> dict:
    """{version: {run: 0 to 2}}."""
    scores = {}
    for row in pairs:
        letter = key["pairs"][str(row["row"]).strip()]["letter"]
        version, run = key["letters"][letter]["version"], key["letters"][letter]["run"]
        scores.setdefault(version, {})[run] = (sj._value(row["outlook_stated"]) == "yes") + \
            (sj._value(row["kept_apart"]) == "yes")
    return scores


def overreach_counts(overreach: list[dict], key: dict) -> dict:
    counts = {"4": 0, "6": 0}
    for row in overreach:
        version = key["letters"][key["overreach"][str(row["row"]).strip()]["letter"]]["version"]
        counts[version] += sj._value(row["overreach"]) == "yes"
    return counts


def choose(pairs: dict, overreach: dict, p6_score: dict, p4_coverage: dict, missing: list[str] | None = None,
           p4_empty: int | None = None) -> dict:
    """pairs: pair_scores; overreach: overreach_counts; p6_score: the Part 4 score of prompt 6 (decoys removed);
    p4_coverage: {run: (covered, total)} and p4_empty: the number of days with an empty industry section despite
    evidence, both from the locked prompt 4 score. A missing `missing` or `p4_empty` keeps prompt 4."""
    runs = sorted(PAIRS)
    not_lower = all(pairs["6"].get(r, 0) >= pairs["4"].get(r, 0) for r in runs)
    higher = sum(pairs["6"].get(r, 0) for r in runs) > sum(pairs["4"].get(r, 0) for r in runs)
    kb = p6_score["key_points_and_briefs"]
    coverage_ok = {}
    for run, (covered, total) in p4_coverage.items():
        mine = p6_score["per_run"].get(run, {}).get("events", {"covered": 0, "total": -1})
        coverage_ok[run] = mine["total"] == total and mine["covered"] >= covered       # same denominators
    planned = all(r in pairs.get(v, {}) for v, r in expected_pairs())
    guardrails = {
        "every_expected_day_has_a_usable_summary": missing == [] and planned
        and set(p6_score["per_run"]) == set(p4_coverage),
        "no_unsupported_industry_point": p6_score["total"]["industry"]["not_supported"] == 0,
        "no_industry_points_without_evidence": not p6_score["points_without_evidence"],
        "empty_sections_with_evidence_not_more_than_prompt4": p4_empty is not None
        and len(p6_score["empty_section_with_evidence"]) <= p4_empty,
        "overreach_not_higher": overreach["6"] <= overreach["4"],
        "no_unsupported_key_point_or_brief": p6_score["total"]["core"]["not_supported"] == 0,
        "no_contradicted_item": p6_score["total"]["core"]["contradicted"] == 0
        and p6_score["total"]["industry"]["contradicted"] == 0,
        "supported_share_at_least_0_75": kb["total"] > 0 and kb["supported"] / kb["total"] >= MIN_SHARE,
        "event_coverage_not_lower": all(coverage_ok.values()),
    }
    replace = not_lower and higher and all(guardrails.values())
    return {"pair_scores": pairs, "pair_not_lower_on_each": not_lower, "pair_total_higher": higher,
            "overreach": overreach, "coverage_per_day": coverage_ok, "guardrails": guardrails,
            "prompt": "6" if replace else "4"}


# ---------- the sheets, the lock and the score ----------

def _records():
    p4_record = json.loads((EVAL_DIR / P4_RECORD).read_text(encoding="utf-8"))
    path = EVAL_DIR / RECORD_FILE
    if not path.exists():
        raise Prompt6Error(f"{RECORD_FILE} does not exist: run first.")
    return p4_record, json.loads(path.read_text(encoding="utf-8"))


def _sheet_command() -> None:
    from src.evaluate import load_label_set
    from src.fulltext import FULLTEXT_DIR
    from src.fulltext_eval import CHECKED_LABEL_FILE
    from src.summary_run import EVENTS_FILE as DEV_EVENTS, load_events
    from src.ranking_eval import DEV_RUNS
    p4_record, p6_record = _records()
    missing = missing_summaries({"4": p4_record, "6": p6_record}, list(DEV_RUNS))
    if missing:
        result = failed_result(missing)
        sj._write_new(EVAL_DIR / SCORE_FILE, json.dumps(result, indent=1))
        print("A day has no usable summary:\n" + "\n".join(missing))
        print(f"Prompt 4 remains selected; no sheet is made; the run is not repeated.\nSaved: data/eval/{SCORE_FILE}")
        return
    passages = sj._passages(p6_record)
    one = sheet1({"4": p4_record, "6": p6_record}, passages)
    p3_items, p3_key, p4_items, p4_key, p4_industry = p5._earlier_locks()
    locked, _ = p5.locked_sources(p3_items, p3_key, p4_items, p4_key)
    two = p5.build_sheets(p6_record, passages, load_events(EVAL_DIR / DEV_EVENTS),
                          load_label_set([EVAL_DIR / CHECKED_LABEL_FILE]), locked,
                          p5.carried_evidence(p4_industry, p4_key), seed=SEED_SHEET2)
    two = add_decoys(two, p4_items)
    letters2 = sorted(two["key"]["summaries"])
    by_run = {e["run"]: e for e in sj._produced(p6_record)}
    shown = [(letter, by_run[two["key"]["summaries"][letter]["run"]]) for letter in letters2]
    titles = {a["key"]: a["title"] for e in p6_record["records"] + p4_record["records"] for a in e["articles"]}
    page2 = p5._page(p4.reading_page(two["items"], shown, passages, titles))
    key = {"sheet1": one["key"], "sheet2": two["key"], "record_file": RECORD_FILE, "baseline_file": P4_RECORD}
    files = {EVAL_DIR / PAIRS_FILE: sj._csv_text(one["pairs"], PAIR_COLUMNS),
             EVAL_DIR / OVERREACH_FILE: sj._csv_text(one["overreach"], OVERREACH_COLUMNS),
             EVAL_DIR / ITEMS_FILE: sj._csv_text(two["items"], sj.ITEM_COLUMNS),
             EVAL_DIR / EVENTS_FILE: sj._csv_text(two["events"], sj.EVENT_COLUMNS),
             EVAL_DIR / KEY_FILE: json.dumps(key, indent=1),
             Path(FULLTEXT_DIR) / PAGE1_FILE: one["page"], Path(FULLTEXT_DIR) / PAGE2_FILE: page2}
    for path in files:
        if path.exists():
            raise Prompt6Error(f"{path.name} already exists; it is not overwritten.")
    for path, text in files.items():
        sj._write_new(path, text)
    carried = len(two["key"]["carried"])
    print(f"Sheet 1, judge first: {len(one['pairs'])} pair rows in data/eval/{PAIRS_FILE}, "
          f"{len(one['overreach'])} industry rows in data/eval/{OVERREACH_FILE}; page {Path(FULLTEXT_DIR) / PAGE1_FILE}")
    print(f"Sheet 2: {len(two['items'])} item rows in data/eval/{ITEMS_FILE} ({carried} unchanged items keep "
          f"their locked judgement), {len(two['events'])} event rows in data/eval/{EVENTS_FILE}; "
          f"page {Path(FULLTEXT_DIR) / PAGE2_FILE}")
    print(f"Key: data/eval/{KEY_FILE}  (do not open it until every sheet is judged and locked)")


def _code_fp() -> str:
    return code_fingerprint(("src/summary_prompt6.py", "src/summary_judging.py", "src/summary_judging_p4.py",
                             "src/summary_judging_p5.py"))


def _all_sheets():
    return tuple(sj._read_csv(EVAL_DIR / name) for name in (PAIRS_FILE, OVERREACH_FILE, ITEMS_FILE, EVENTS_FILE))


def _inputs_fp() -> dict:
    from src.fresh_eval import file_fingerprint
    return {"guides": [file_fingerprint(EVAL_DIR / g) for g in GUIDES],
            **{name: file_fingerprint(EVAL_DIR / name) for name in (PLAN_FILE, RECORD_FILE, ATTEMPT_FILE, P4_RECORD,
                                                                   p4.SCORE_FILE)}}


def lock_record(sheets: tuple, key: dict, inputs: dict, code: str) -> dict:
    pairs, overreach, items, events = sheets
    found = sheet1_problems(pairs, overreach, key["sheet1"]) + \
        sj.item_problems(items, key["sheet2"]["items"]) + sj.event_problems(events, key["sheet2"]["events"])
    if found:
        raise Prompt6Error("The sheets are not complete:\n" + "\n".join(found))
    return {"sheets": [sj.sheet_fingerprint(s) for s in sheets], "key": sj._hash(key), "code": code,
            "inputs": inputs, "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def lock_changes(lock: dict, sheets: tuple, key: dict, inputs: dict, code: str) -> list[str]:
    names = ("pairs", "overreach", "items", "events")
    found = [f"the {n} sheet changed after the lock" for n, s, fp in zip(names, sheets, lock["sheets"])
             if sj.sheet_fingerprint(s) != fp]
    if lock["key"] != sj._hash(key):
        found.append("the key changed after the lock")
    if lock["code"] != code:
        found.append("the judging code changed after the lock")
    found += [f"{n} changed after the lock" for n in sorted(set(lock["inputs"]) | set(inputs))
              if lock["inputs"].get(n) != inputs.get(n)]
    return found


def _lock_command() -> None:
    if (EVAL_DIR / LOCK_FILE).exists():
        raise Prompt6Error(f"{LOCK_FILE} already exists: the sheets are locked.")
    lock = lock_record(_all_sheets(), sj._load_json(EVAL_DIR / KEY_FILE), _inputs_fp(), _code_fp())
    sj._write_new(EVAL_DIR / LOCK_FILE, json.dumps(lock, indent=1))
    print(f"Locked: data/eval/{LOCK_FILE}")


def _score_command() -> None:
    from src.ranking_eval import DEV_RUNS
    if (EVAL_DIR / SCORE_FILE).exists():
        raise Prompt6Error(f"{SCORE_FILE} already exists; it is not overwritten.")
    sheets, key = _all_sheets(), sj._load_json(EVAL_DIR / KEY_FILE)
    found = lock_changes(sj._load_json(EVAL_DIR / LOCK_FILE), sheets, key, _inputs_fp(), _code_fp())
    if found:
        raise Prompt6Error("The score cannot be made:\n" + "\n".join(found))
    pairs, overreach, items, events = sheets
    p4_record, p6_record = _records()
    p3_items, p3_key, p4_items, p4_key, _ = p5._earlier_locks()
    _, locked_sheet = p5.locked_sources(p3_items, p3_key, p4_items, p4_key)
    rows, score_key = scored_items(items, key["sheet2"])
    p6 = p5.score(rows, events, score_key, p6_record, locked_sheet, p4.locked_baseline(p3_items, p3_key), list(DEV_RUNS))
    p4_score = sj._load_json(EVAL_DIR / p4.SCORE_FILE)
    p4_cov = {run: (r["events"]["covered"], r["events"]["total"]) for run, r in p4_score["per_run"].items()}
    p4_empty = len(p4_score["empty_section_with_evidence"])
    missing = missing_summaries({"4": p4_record, "6": p6_record}, list(DEV_RUNS))
    result = {"choice": choose(pair_scores(pairs, key["sheet1"]), overreach_counts(overreach, key["sheet1"]), p6, p4_cov,
                               missing, p4_empty),
              "prompt6": {k: p6[k] for k in ("total", "per_run", "key_points_and_briefs")},
              "decoys": decoy_agreement(items, key["sheet2"]), "rule": RULE,
              "scored": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    sj._write_new(EVAL_DIR / SCORE_FILE, json.dumps(result, indent=1))
    c = result["choice"]
    print(f"Pair scores: {c['pair_scores']} | not lower on each: {c['pair_not_lower_on_each']} | total higher: "
          f"{c['pair_total_higher']}")
    print(f"Overreach: {c['overreach']} | guardrails: {c['guardrails']}")
    print(f"Decoys: {result['decoys']}")
    print(f"Prompt chosen: {c['prompt']}\nSaved: data/eval/{SCORE_FILE}")


COMMANDS = {"plan": _plan_command, "run": _run_command, "sheet": _sheet_command, "lock": _lock_command,
            "score": _score_command}

if __name__ == "__main__":
    try:
        if len(sys.argv) == 2 and sys.argv[1] in COMMANDS:
            COMMANDS[sys.argv[1]]()
        else:
            print(__doc__)
    except (Prompt6Error, sj.JudgingError) as error:
        print(f"[STOP] {error}")
        sys.exit(1)
