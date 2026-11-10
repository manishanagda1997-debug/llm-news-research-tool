"""
Round 3 of the second development cycle: prompt version 6 against prompt version 4 over repeated runs.

Question: does prompt 6 keep an expectation and its outcome apart better than prompt 4, without failing more often?
The answer covers only the three Tata Motors development days, their articles and the frozen settings; it does not
show that either prompt is more reliable in general.

Runs: three matched pairs, each one prompt 4 run and one prompt 6 run, in an order inside the pair fixed in the plan.
Every run covers the three development days with the same five articles per day as the locked prompt 4 run, and is
made once. Only these six runs count; the locked prompt 4 run and the round 2 prompt 6 run are context only.

Outcome of each run-day (in the record and in the score):
    usable          status ok
    partial         status partial: the remaining content is judged; removed items are recorded
    failed          no usable summary because of the model's answer (invalid JSON, or a check still failing after
                    the correction): an explicit 0 for the pair score and for every event of that day
    provider stop   no usable summary because a call was refused or failed (rate limit, daily limit, size, network):
                    the round stops; the pair is dropped for both prompts, and so is every later pair
    dropped         a run-day of a dropped pair: out of every denominator, for both prompts
A started run that has no record (an interrupted run) counts as a provider stop. Nothing is retried.

Tokens: each run has an allowance of 45,000 tokens, guarded by an estimate before every call and checked with the
real counts after it. Before a call, the run's real tokens so far plus that call's estimated worst case (the careful
input estimate, 3 characters a token, plus the 4,096-token output limit) must fit in 45,000, or the call is not sent
and the run ends as a provider stop ("run allowance"). After a call, if the run's real tokens are over 45,000, no
further call of the run is sent and the round stops (tokens already used cannot be taken back). Before a pair starts, the
tokens of every logged call inside the last 24 hours, plus the Groq use the user states outside this log, plus
2 x 45,000 must not exceed the daily limit recorded in the plan; otherwise the pair does not start and the earliest
time it fits is printed. The input estimate is careful (real text has about 4 characters a token) but not a proof;
the provider's own daily limit still refuses anything over it, and that refusal is a provider stop. The rolling
24 hours is never less strict than a daily limit that resets at a fixed time: every token of the current calendar
day falls inside the last 24 hours. The one assumption is that the provider's daily window is not longer than
24 hours. The log is every call of the round 2 run and of this
round's runs, timed from the run's clock; a started run without a record counts as 45,000 at its start. If the
second run of a started pair would not fit, it is not sent: the pair is dropped and the round stops.

Judging (Part 7 for the pairs and overreach, Parts 2 and 4 for items): every usable or partial summary under one
shuffled letter, prompt and run hidden. Sheet 1: pair rows for 23 and 24 October, an overreach row per industry
point. Sheet 2: the items of each prompt's full-item evidence plus 10 locked prompt 4 decoys, and the event rows of
every usable or partial summary. Full-item evidence: for each day, the first run in the plan's seeded order, among
the runs of kept pairs, with a usable or partial summary; none gives "unavailable".

Commands, in order:
    python -m src.summary_round3 plan --tpd 200000      then commit, then: git tag freeze-round3-plan
    python -m src.summary_round3 pair --other-tokens N  three times (N: Groq tokens used outside this project's
                                                        records in the last 24 hours; 0 when none); commit after each
    python -m src.summary_round3 sheet                  then judge sheet 1 first, then sheet 2
    python -m src.summary_round3 lock
    python -m src.summary_round3 score
"""

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import summary_judging as sj
from src import summary_judging_p4 as p4
from src import summary_judging_p5 as p5
from src import summary_prompt6 as p6
from src.config import EVAL_DIR, ROOT_DIR

PLAN_FILE = "summary_v2_r3_plan.json"
PLAN_TAG = "freeze-round3-plan"
STOP_FILE = "summary_v2_r3_stop.json"
PAIRS_FILE = "summary_v2_r3_pairs.csv"
OVERREACH_FILE = "summary_v2_r3_overreach.csv"
ITEMS_FILE = "summary_v2_r3_judging.csv"
EVENTS_FILE = "summary_v2_r3_events.csv"
KEY_FILE = "summary_v2_r3_judging_key.json"
LOCK_FILE = "summary_v2_r3_judging_lock.json"
SCORE_FILE = "summary_v2_r3_judging_score.json"
PAGE1_FILE = "summary_v2_r3_sheet1.html"            # in data/fulltext/, outside Git
PAGE2_FILE = "summary_v2_r3_sheet2.html"            # in data/fulltext/, outside Git
PROMPTS = ("4", "6")
PAIR_COUNT = 3
MIN_KEPT = 2
ALLOWANCE = 45_000
SEED_PLAN, SEED_SHEET1, SEED_SHEET2, SEED_DECOYS = 71, 72, 73, 74
DECOYS = 10
MIN_SHARE = 0.75
PROVIDER_KINDS = "any refusal or failure of a call reported by the model client (LLMUnavailable), and a call not " \
                 "sent because it would go over the run allowance"
OUTCOMES = ("usable", "partial", "failed", "provider_stop")
IST = timezone(timedelta(hours=5, minutes=30))
CODE_FILES = ("src/summary_round3.py",) + p6.CODE_FILES + (
    "src/model_compare.py", "src/evaluate.py", "src/summary_run.py", "src/config.py", "src/fresh_eval.py")
DATA_FILES = ("events_dev_tata_motors.csv", "v3_dev_tata_motors_checked.csv", "summary_v2_judging.csv",
              "summary_v2_judging_key.json", "summary_v2_judging_lock.json", "summary_v2_p4_judging.csv",
              "summary_v2_p4_industry.csv", "summary_v2_p4_judging_key.json", "summary_v2_p4_judging_lock.json",
              "summary_v2_p4_judging_score.json", "summary_v2_events.csv", "summary_v2_p4_events.csv")
PAIR_COLUMNS = p6.PAIR_COLUMNS
OVERREACH_COLUMNS = p6.OVERREACH_COLUMNS

RULE = {
    "scope": "only these three development days, these articles and these settings",
    "pairs": {day: list(pair) for day, pair in p6.PAIRS.items()},
    "outcomes": {"usable": "status ok", "partial": "status partial; the remaining content is judged",
                 "failed": "no usable summary because of the model's answer; 0 for the pair score and every event",
                 "provider_stop": "no usable summary because a call was refused or failed; the round stops",
                 "dropped": "a run-day of a dropped pair; out of every denominator for both prompts",
                 "interrupted": "a started run without a record counts as a provider stop"},
    "provider_stop": "the round stops at once with no retry; the pair is dropped for both prompts, and so is every "
                     "later pair; completed earlier pairs are kept; fewer than 2 kept pairs: no selection, prompt 4",
    "run_allowance": "an estimated guard, not a guaranteed maximum: before every call, the run's real tokens so far + "
                     "the call's estimated worst case (careful input estimate + the output limit) <= 45,000, or the "
                     "call is not sent and the run ends as a provider stop; after every call, real tokens over "
                     "45,000 stop the run's further calls and the round",
    "tokens": "before a pair: logged tokens inside the last 24 hours + stated other use + 2 x 45,000 <= the daily "
              "limit; if the second run of a started pair would not fit, it is not sent, the pair is dropped and "
              "the round stops",
    "full_item_evidence": "for each day, the first run in the seeded order, among runs of kept pairs, with a usable "
                          "or partial summary; none gives unavailable; nothing is substituted",
    "denominators": {"failed_run_days": "3 x kept pairs", "pair_scores": "2 x kept pairs",
                     "events_per_day": "the day's events x kept pairs", "overreach": "all industry points produced"},
    "replace": "prompt 6 replaces prompt 4 only if every condition holds; a tie keeps prompt 4",
    "conditions": ["at least 2 kept pairs",
                   "prompt 6 failed run-days not more than prompt 4's",
                   "prompt 6 total pair score higher than prompt 4's, and not lower on either paired day",
                   "prompt 6 overreach count over all runs not higher than prompt 4's",
                   "events covered per day over all runs not lower than prompt 4's (same denominators)",
                   "prompt 6 full-item evidence available for every day",
                   "no unsupported industry point", "no unsupported key point or brief", "no contradicted item",
                   "key points and briefs supported share at least 0.75, unrounded",
                   "no industry points without evidence",
                   "days with an empty industry section despite evidence not more than prompt 4's, over the days "
                   "where both prompts have evidence (a count, not which days)"],
    "guardrail_scope": "the full-item guardrails cover only the full-item evidence, named run by run for each day; "
                       "they do not show that every point of all six runs is supported",
    "decoys": "10 locked prompt 4 items, fixed seed, unmarked; consistency only, never in the choice",
    "daily_window": "the budget counts a rolling 24 hours; this is at least as strict as any daily limit whose window "
                    "is not longer than 24 hours, rolling or reset at a fixed time",
    "insufficient_data": "with fewer than 2 kept pairs no comparison is made: prompt 4 remains because it is the "
                         "frozen design, not because it won",
    "recognition": "outputs are new, so recognition of earlier summaries is less likely; the judge does not open "
                   "earlier sheets",
}


class Round3Error(Exception):
    pass


# ---------- names ----------

def run_id(pair: int, prompt: str) -> str:
    return f"pair{pair}_p{prompt}"


def record_file(pair: int, prompt: str) -> str:
    return f"summary_v2_r3_{run_id(pair, prompt)}.json"


def marker_file(pair: int, prompt: str) -> str:
    return f"summary_v2_r3_{run_id(pair, prompt)}_attempt.json"


def prompt_text(prompt: str) -> str:
    from src import summarizer as sm
    v4 = sm.PROMPT_TEXTS["4"]
    return v4 if prompt == "4" else p6.prompt_v6(v4)


# ---------- the plan ----------

def seeded_orders(seed: int = SEED_PLAN) -> dict:
    """The order inside each pair, and the order of the three runs for the full-item evidence of each prompt."""
    rng = random.Random(seed)
    pair_order = {}
    for pair in range(1, PAIR_COUNT + 1):
        order = list(PROMPTS)
        rng.shuffle(order)
        pair_order[str(pair)] = order
    full_item = {}
    for prompt in PROMPTS:
        runs = list(range(1, PAIR_COUNT + 1))
        rng.shuffle(runs)
        full_item[prompt] = runs
    return {"pair_order": pair_order, "full_item_order": full_item}


def plan_record(tpd: int, p4_record_text: str, source_commit: str) -> dict:
    from src import summary_v2 as sv
    if tpd < 2 * ALLOWANCE:
        raise Round3Error(f"A daily limit of {tpd} cannot hold one pair (2 x {ALLOWANCE}).")
    return {"round": 3, "rule": RULE, "daily_limit": tpd, "allowance_per_run": ALLOWANCE,
            "prompts": {p: {"fingerprint": p6.prompt_fingerprint(prompt_text(p))} for p in PROMPTS},
            "prompt6_text": prompt_text("6"),
            "orders": seeded_orders(),
            "settings": _settings(),
            "baseline_record": p6.P4_RECORD,
            "baseline_fingerprint": sj._hash(p4_record_text.replace("\r\n", "\n")),
            "guides": guide_fingerprints(),
            "seeds": {"plan": SEED_PLAN, "sheet1": SEED_SHEET1, "sheet2": SEED_SHEET2, "decoys": SEED_DECOYS},
            "decoys": DECOYS, "data_files": data_fingerprints(), "code_files": list(CODE_FILES), "code_fingerprint": p6.code_fingerprint(CODE_FILES),
            "source_commit": source_commit, "tag": PLAN_TAG,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}


GUIDES = (sj.GUIDE_FILE, p4.GUIDE_FILE, p6.GUIDE_FILE)


def guide_fingerprints() -> dict:
    """Parts 2, 4 and 7 of the judging guide: every guide the judging of this round follows."""
    return {name: sj._hash((EVAL_DIR / name).read_text(encoding="utf-8").replace("\r\n", "\n")) if
            (EVAL_DIR / name).exists() else None for name in GUIDES}


def data_fingerprints() -> dict:
    """The judging inputs: the event and label files and the locked earlier sheets the carry-over reads."""
    return {name: sj._hash((EVAL_DIR / name).read_text(encoding="utf-8").replace("\r\n", "\n")) if
            (EVAL_DIR / name).exists() else None for name in DATA_FILES}


def _settings() -> dict:
    from src import summary_v2 as sv
    from src.model_compare import CONFIGS
    config = CONFIGS[sv.MODEL_SETTING]
    return {"design": "direct", "k": sv.DEV_K, "model_setting": sv.MODEL_SETTING, "model": config["model"],
            "generation": {**config["generation"], "max_completion_tokens": sv.OUTPUT_LIMIT},
            "output_limit": sv.OUTPUT_LIMIT, "correction": sv.CORRECTION, "configuration": sv.SUMMARY_VERSION,
            "number_check": sv.NUMBER_CHECK_VERSION, "pacer_version": sv.PACER_VERSION, "cache": "none"}


def plan_problems(plan_text, tag_text, plan, p4_record_text) -> list[str]:
    def same(a, b):
        return a is not None and b is not None and a.replace("\r\n", "\n") == b.replace("\r\n", "\n")
    found = []
    if not same(plan_text, tag_text):
        found.append(f"{PLAN_FILE} is missing or differs from the tag {PLAN_TAG}")
    if plan is None:
        return found
    for prompt in PROMPTS:
        if plan["prompts"][prompt]["fingerprint"] != p6.prompt_fingerprint(prompt_text(prompt)):
            found.append(f"prompt {prompt} differs from the plan")
    if plan["baseline_fingerprint"] != sj._hash(p4_record_text.replace("\r\n", "\n")):
        found.append(f"{p6.P4_RECORD} differs from the plan")
    if plan["code_fingerprint"] != p6.code_fingerprint(CODE_FILES):
        found.append("the code differs from the plan")
    if plan["settings"] != _settings():
        found.append("the frozen settings differ from the plan")
    current = data_fingerprints()
    for name in DATA_FILES:
        if plan.get("data_files", {}).get(name) != current[name] or current[name] is None:
            found.append(f"{name} is missing or differs from the plan")
    guides = guide_fingerprints()
    for name in GUIDES:
        if plan.get("guides", {}).get(name) != guides[name] or guides[name] is None:
            found.append(f"{name} is missing or differs from the plan")
    return found


# ---------- the state of the round ----------

def outcome_problem(result: dict) -> str | None:
    """A saved outcome must follow from its evidence: a provider stop needs a recorded refusal or failure (with no
    usable summary, or a call refused by the run allowance); every other outcome must equal the classification."""
    outcome, failures = result.get("outcome"), result.get("failures") or []
    if outcome not in OUTCOMES:
        return "no valid outcome"
    if outcome == "provider_stop":
        if not failures:
            return "a provider stop without a recorded refusal or failure"
        if result.get("status") in ("ok", "partial") and "run_allowance" not in failures:
            return "a provider stop with a usable summary and no run-allowance refusal"
        return None
    if "run_allowance" in failures:
        return "a run-allowance refusal not marked as a provider stop"
    return None if outcome == classify(result) else "the outcome does not match the result"


def classify(result: dict) -> str:
    status = result.get("status")
    if status == "ok":
        return "usable"
    if status == "partial":
        return "partial"
    return "provider_stop" if result.get("failures") else "failed"


def round_state(records: dict, markers: set, stop: dict | None) -> dict:
    """records: {run_id: record}; markers: run ids with an attempt marker; stop: the stop file or None.
    Returns the kept pairs, the first stopped pair (or None), and the next pair to run (or None)."""
    stopped_at = None
    for pair in range(1, PAIR_COUNT + 1):
        ids = [run_id(pair, p) for p in PROMPTS]
        started = any(i in markers or i in records for i in ids)
        half = started and not all(i in records for i in ids)
        provider = any(e["result"].get("outcome") == "provider_stop" for i in ids if i in records
                       for e in records[i]["records"]) or any(records[i].get("allowance_exceeded") for i in ids
                                                              if i in records)
        interrupted = any(i in markers and i not in records for i in ids)
        refused = stop is not None and stop.get("pair") == pair
        if provider or interrupted or refused or half:
            stopped_at = pair
            break
        if not all(i in records for i in ids):
            break
    if stop is not None and stopped_at is None:
        stopped_at = stop.get("pair")
    kept = [pair for pair in range(1, PAIR_COUNT + 1)
            if (stopped_at is None or pair < stopped_at) and all(run_id(pair, p) in records for p in PROMPTS)]
    next_pair = None
    if stopped_at is None:
        next_pair = next((pair for pair in range(1, PAIR_COUNT + 1) if pair not in kept), None)
    return {"kept": kept, "stopped_at": stopped_at, "next": next_pair,
            "finished": stopped_at is not None or len(kept) == PAIR_COUNT}


# ---------- checking the saved runs ----------

def record_problems(name: str, record: dict, marker: dict | None, plan: dict, baseline: dict) -> list[str]:
    """One run's record against the plan: its identity, prompt, settings, days, outcomes and articles."""
    found, where = [], f"{name}:"
    pair, prompt = int(name[4]), name[-1]
    if record.get("run") != name or record.get("pair") != pair or record.get("prompt") != prompt:
        found.append(f"{where} the run, pair or prompt does not match its file")
    if record.get("prompt_fingerprint") != plan["prompts"][prompt]["fingerprint"]:
        found.append(f"{where} the prompt differs from the plan")
    found += marker_problems(name, record, marker, plan)
    settings = plan["settings"]
    wanted = {"model_setting": settings["model_setting"], "model": settings["model"],
              "generation": settings["generation"], "k": settings["k"], "version": settings["configuration"],
              "correction": settings["correction"], "number_check": settings["number_check"],
              "pacer_version": settings["pacer_version"], "cache": settings["cache"],
              "output_limit_sent": settings["output_limit"], "allowance": plan["allowance_per_run"]}
    for field, value in wanted.items():
        if record.get(field) != value:
            found.append(f"{where} {field} is {record.get(field)!r}, the plan says {value!r}")
    days = [e.get("run") for e in record.get("records", [])]
    expected = [e["run"] for e in baseline["records"]]
    outcomes = [e.get("result", {}).get("outcome") for e in record.get("records", [])]
    stop_at = outcomes.index("provider_stop") if "provider_stop" in outcomes else None
    if stop_at is not None and stop_at != len(days) - 1:
        found.append(f"{where} a day follows a provider stop")
    cut_short = stop_at is not None or bool(record.get("allowance_exceeded"))
    if days != (expected[:len(days)] if cut_short else expected) or not days:
        found.append(f"{where} the days are {days}, not {expected} in order, once each")
    articles = {e["run"]: [a["key"] for a in e["articles"]] for e in baseline["records"]}
    for entry in record.get("records", []):
        day, result = entry.get("run"), entry.get("result", {})
        problem = outcome_problem(result)
        if problem:
            found.append(f"{where} {day}: {problem}")
        if day in articles and [a.get("key") for a in entry.get("articles", [])] != articles[day]:
            found.append(f"{where} {day}: the articles differ from the plan's baseline")
    tokens = sum(n for _, n in call_entries(record, record.get("clock_start") or "2000-01-01T00:00:00+00:00"))
    if record.get("tokens") != tokens:
        found.append(f"{where} the saved token total {record.get('tokens')} is not the sum of its calls ({tokens})")
    if bool(record.get("allowance_exceeded")) != (tokens > plan["allowance_per_run"]):
        found.append(f"{where} allowance_exceeded does not match the real token total {tokens}")
    return found


def marker_problems(name: str, record: dict, marker: dict | None, plan: dict) -> list[str]:
    """The attempt marker: the same run, prompt and clock as the record, and the pair budget check it passed."""
    where = f"{name}:"
    if marker is None:
        return [f"{where} the attempt marker is missing"]
    found = []
    if marker.get("run") != name:
        found.append(f"{where} the attempt marker names another run")
    if marker.get("prompt_fingerprint") != record.get("prompt_fingerprint") \
            or marker.get("clock_start") != record.get("clock_start"):
        found.append(f"{where} the attempt marker does not match the record")
    other, budget = marker.get("other_tokens"), marker.get("budget") or {}
    if not isinstance(other, int) or other < 0:
        found.append(f"{where} the attempt marker has no valid stated other use")
    if budget.get("fits") is not True or budget.get("need") != 2 * plan["allowance_per_run"] \
            or budget.get("limit") != plan["daily_limit"] or budget.get("other_tokens") != other:
        found.append(f"{where} the attempt marker does not hold the pair budget check the plan requires")
    return found


def frozen_problems(plan_text, tag_text, plan, p4_text, records, markers) -> list[str]:
    found = plan_problems(plan_text, tag_text, plan, p4_text)
    if plan is None:
        return found
    baseline = json.loads(p4_text)
    for name in sorted(records):
        found += record_problems(name, records[name], markers.get(name), plan, baseline)
    return found


def _preflight():
    """The plan, the tag, the prompts, the baseline, the guide, the code and every saved run, checked together."""
    plan_path = EVAL_DIR / PLAN_FILE
    plan_text = plan_path.read_text(encoding="utf-8") if plan_path.exists() else None
    plan = json.loads(plan_text) if plan_text else None
    p4_text = (EVAL_DIR / p6.P4_RECORD).read_text(encoding="utf-8")
    records, markers, stop = _load_round()
    found = frozen_problems(plan_text, _tag_text(), plan, p4_text, records, markers)
    return plan, p4_text, records, markers, stop, found


# ---------- tokens ----------

def call_entries(record: dict, started_at: str) -> list[tuple[datetime, int]]:
    """(time, tokens) for every call of a record, timed from the run's clock start."""
    start = datetime.fromisoformat(started_at)
    found = []
    for entry in record["records"]:
        for call in entry["result"].get("usage", {}).get("per_call", []):
            found.append((start + timedelta(seconds=call.get("sent_at") or 0),
                          int(call.get("input", 0)) + int(call.get("output", 0))))
    return found


def token_log(runs: list[tuple[dict | None, str]]) -> list[tuple[datetime, int]]:
    """runs: (record or None, clock start). A started run without a record counts as the full allowance."""
    found = []
    for record, started_at in runs:
        if record is None:
            found.append((datetime.fromisoformat(started_at), ALLOWANCE))
        else:
            found += call_entries(record, started_at)
    return sorted(found)


def budget_check(log, other: int, need: int, limit: int, now: datetime) -> dict:
    """Whether `need` more tokens fit inside the rolling 24 hours; if not, the earliest time they fit."""
    window_start = now - timedelta(hours=24)
    inside = [(t, n) for t, n in log if t > window_start]
    used = sum(n for _, n in inside)
    fits = used + other + need <= limit
    earliest, never = None, other + need > limit
    if not fits and not never:
        over = used + other + need - limit
        for t, n in inside:
            over -= n
            if over <= 0:
                earliest = t + timedelta(hours=24)
                break
    return {"window_start_utc": window_start.isoformat(timespec="seconds"),
            "window_start_ist": window_start.astimezone(IST).isoformat(timespec="seconds"),
            "now_utc": now.isoformat(timespec="seconds"), "now_ist": now.astimezone(IST).isoformat(timespec="seconds"),
            "logged_tokens": used, "other_tokens": other, "need": need, "limit": limit, "fits": fits,
            "fits_later_from_the_log": not fits and not never,
            "earliest_utc": earliest.isoformat(timespec="seconds") if earliest else None,
            "earliest_ist": earliest.astimezone(IST).isoformat(timespec="seconds") if earliest else None}


# ---------- one run ----------

class AllowanceGuard:
    """Wraps the model for one run: a call is sent only when the run's tokens so far plus the call's worst case fit
    in the allowance. A refused call raises LLMUnavailable (kind "run_allowance"); the run then ends."""

    def __init__(self, llm, allowance: int = ALLOWANCE, output_limit: int | None = None):
        from src import summary_v2 as sv
        self.llm, self.allowance = llm, allowance
        self.output_limit = output_limit if output_limit is not None else sv.OUTPUT_LIMIT
        stats = getattr(llm, "stats", None)
        self.stats = stats if stats is not None else {"per_call": [], "requests": []}
        self.start = len(self.stats.get("per_call", []))
        self.refused, self.exceeded = False, False
        for name in ("model_name", "cache_tag", "provider", "generation", "output_limit_sent"):
            if hasattr(llm, name):
                setattr(self, name, getattr(llm, name))

    def used(self) -> int:
        return sum(int(c.get("input", 0)) + int(c.get("output", 0)) for c in self.stats.get("per_call", [])[self.start:])

    def worst(self, prompt: str) -> int:
        from src.llm import CHARS_PER_TOKEN
        return len(prompt) // CHARS_PER_TOKEN + 1 + self.output_limit

    def __call__(self, prompt: str) -> str:
        from src.summarizer import LLMUnavailable
        if self.refused or self.used() + self.worst(prompt) > self.allowance:
            self.refused = True
            error = LLMUnavailable(f"Not sent: the run has used {self.used()} tokens and this call could use up to "
                                   f"{self.worst(prompt)}; the run allowance is {self.allowance}.")
            error.kind = "run_allowance"
            raise error
        answer = self.llm(prompt)
        if self.used() > self.allowance:
            self.exceeded = True                    # the real count passed the allowance: no further call
            self.refused = True
        return answer

def run_diagnostics(recorder, attempt_log: list, per_call: list) -> list[dict]:
    """Every call with its classification and evidence: the raw answer, or an empty answer and the error."""
    found, answers = [], list(recorder.answers)
    for number, call in enumerate(per_call, start=1):
        answer = answers[number - 1] if number <= len(answers) else ""
        attempt = attempt_log[number - 1] if number <= len(attempt_log) else None
        problems = [] if attempt is None else attempt.get("structural", []) + attempt.get("numbers", []) + \
            attempt.get("sources", [])
        found.append({"call": number, "role": "summary attempt" if attempt is not None else "repair",
                      "attempt": attempt.get("attempt") if attempt else None,
                      "classification": "answer with check problems" if problems else "answer",
                      "problems": problems, "finish_reason": call.get("finish_reason"), "answer": answer, "error": ""})
    from src.summary_v2 import safe_message
    for kind, message in zip(recorder.failures, recorder.messages):
        found.append({"call": None, "role": "refused or failed call", "attempt": None,
                      "classification": "provider: " + kind, "problems": [], "finish_reason": None, "answer": "",
                      "error": safe_message(message)})
    return found


def one_day(articles: list[dict], llm) -> dict:
    """The frozen summary path for one day (as summary_v2._one with no cache), keeping every answer."""
    from src import summary_v2 as sv
    recorder = sv.Recorder(llm)
    stats = getattr(llm, "stats", None)
    stats = stats if stats is not None else {}
    before, requests_before = len(stats.get("per_call", [])), len(stats.get("requests", []))
    result = sv.run_design("direct", articles, recorder, sv.CORRECTION)
    usage = sv._usage(llm, before, requests_before)
    result.update(usage=usage, failures=recorder.failures, failure_waits=recorder.waits,
                  prompt_chars=recorder.prompt_chars,
                  bad_answers=sv.bad_answers(recorder.answers, stats.get("per_call", [])[before:]))
    result["_messages"] = list(recorder.messages)
    result["diagnostics"] = run_diagnostics(recorder, result.get("attempt_log", []), usage["per_call"])
    result["outcome"] = "provider_stop" if "run_allowance" in result.get("failures", []) else classify(result)
    return result


def _pair_command(other: int) -> None:
    from src import summary_v2 as sv
    if other < 0:
        raise Round3Error("--other-tokens cannot be negative.")
    plan, p4_text, records, markers, stop, found = _preflight()
    state = round_state(records, set(markers), stop)
    if state["finished"]:
        found.append(f"the round is finished (kept pairs {state['kept']}, stopped at {state['stopped_at']})")
    runs, aliases = sv._dev_candidates()
    articles_by_run = {name: sv.summary_articles(c, sv.DEV_K, aliases) for name, c in runs.items()}
    found += p6.same_articles(json.loads(p4_text), articles_by_run)
    if found:
        raise Round3Error("The pair is refused:\n" + "\n".join(found))
    pair = state["next"]
    log_runs = _log_runs(records, markers)
    check = budget_check(token_log(log_runs), other, 2 * ALLOWANCE, plan["daily_limit"], datetime.now(timezone.utc))
    print(f"Tokens in the last 24 hours: {check['logged_tokens']} logged + {other} other + {2 * ALLOWANCE} for the "
          f"pair; limit {check['limit']} (window from {check['window_start_ist']} IST)")
    if not check["fits"]:
        if check["earliest_utc"] is None:
            raise Round3Error(f"Pair {pair} does not fit: the stated other use ({other}) plus the pair "
                              f"({2 * ALLOWANCE}) is over the daily limit ({check['limit']}) even with nothing "
                              f"logged, so no time can be worked out from this information. Nothing was sent.")
        raise Round3Error(f"Pair {pair} does not fit yet; it fits from {check['earliest_ist']} IST "
                          f"({check['earliest_utc']} UTC). Nothing was sent.")
    llm = sv.summary_llm()
    clock_start = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for position, prompt in enumerate(plan["orders"]["pair_order"][str(pair)]):
        if position == 1:
            records, markers, _ = _load_round()
            second = budget_check(token_log(_log_runs(records, markers)), other, ALLOWANCE, plan["daily_limit"],
                                  datetime.now(timezone.utc))
            if not second["fits"]:
                sj._write_new(EVAL_DIR / STOP_FILE, json.dumps({"pair": pair, "reason": "second run would not fit",
                                                                "check": second}, indent=1))
                raise Round3Error(f"The second run of pair {pair} would not fit; it is not sent. Pair {pair} is "
                                  f"dropped and the round stops. Saved: data/eval/{STOP_FILE}")
        _one_run(pair, prompt, articles_by_run, llm, clock_start, other, check)
        records, markers, _ = _load_round()
        saved = records[run_id(pair, prompt)]
        if saved.get("allowance_exceeded") or any(e["result"]["outcome"] == "provider_stop" for e in saved["records"]):
            print(f"A provider stop or a run over its allowance in {run_id(pair, prompt)}: pair {pair} is dropped "
                  f"and the round stops. The other run of the pair is not sent.")
            return
    print(f"Pair {pair} complete. Commit its two records and two attempt markers.")


def _one_run(pair, prompt, articles_by_run, llm, clock_start, other, check) -> None:
    from src import summary_v2 as sv
    name = run_id(pair, prompt)
    text = prompt_text(prompt)
    with open(EVAL_DIR / marker_file(pair, prompt), "x", encoding="utf-8") as handle:
        handle.write(json.dumps({"run": name, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "clock_start": clock_start, "prompt_fingerprint": p6.prompt_fingerprint(text),
                                 "other_tokens": other, "budget": check}, indent=1))
    entries, guard = [], AllowanceGuard(llm)
    with p6.prompt_in_use(text):
        for day, articles in articles_by_run.items():
            result = one_day(articles, guard)
            for message in sv.take_messages(result):
                print(f"    {message}")
            entries.append(sv._record(day, "direct", articles, result))
            usage = result.get("usage", {})
            print(f"{name} {day} {result['outcome']} calls {usage.get('calls', '-')} "
                  f"tokens {usage.get('input_tokens', 0) + usage.get('output_tokens', 0)}")
            if result["outcome"] == "provider_stop" or guard.exceeded:
                break
    out = {"run": name, "pair": pair, "prompt": prompt, "prompt_fingerprint": p6.prompt_fingerprint(text),
           "clock_start": clock_start, "version": sv.SUMMARY_VERSION, "number_check": sv.NUMBER_CHECK_VERSION,
           "pacer_version": sv.PACER_VERSION, "model_setting": sv.MODEL_SETTING, "model": llm.model_name,
           "generation": getattr(llm, "generation", None), "output_limit_sent": getattr(llm, "output_limit", None),
           "k": sv.DEV_K, "cache": "none", "records": entries,
           "tokens": sum(n for _, n in call_entries({"records": entries}, clock_start)),
           "allowance": ALLOWANCE, "allowance_refused": guard.refused, "allowance_exceeded": guard.exceeded,
           "correction": sv.CORRECTION}
    sj._write_new(EVAL_DIR / record_file(pair, prompt), json.dumps(out, ensure_ascii=False, indent=1))
    print(f"Saved: data/eval/{record_file(pair, prompt)}  ({out['tokens']} tokens)")


def _git(*args) -> str:
    from src.fresh_eval import git
    return git(*args)


def _tag_text() -> str | None:
    try:
        return _git("show", f"{PLAN_TAG}:{(EVAL_DIR / PLAN_FILE).relative_to(ROOT_DIR).as_posix()}")
    except Exception:
        return None


def _load_round():
    records, markers = {}, {}
    for pair in range(1, PAIR_COUNT + 1):
        for prompt in PROMPTS:
            name = run_id(pair, prompt)
            if (EVAL_DIR / marker_file(pair, prompt)).exists():
                markers[name] = sj._load_json(EVAL_DIR / marker_file(pair, prompt))
            if (EVAL_DIR / record_file(pair, prompt)).exists():
                records[name] = sj._load_json(EVAL_DIR / record_file(pair, prompt))
    stop = sj._load_json(EVAL_DIR / STOP_FILE) if (EVAL_DIR / STOP_FILE).exists() else None
    return records, markers, stop


def _log_runs(records: dict, markers: dict) -> list:
    """The round 2 run and every started run of this round, for the token log."""
    runs = []
    r2_record, r2_marker = EVAL_DIR / p6.RECORD_FILE, EVAL_DIR / p6.ATTEMPT_FILE
    if r2_marker.exists():                       # a started round 2 run without a record counts as the allowance
        runs.append((sj._load_json(r2_record) if r2_record.exists() else None,
                     sj._load_json(r2_marker)["started_at"]))
    for name, marker in markers.items():
        # a saved run is timed call by call from its clock; a run without a record counts from its own start
        runs.append((records[name], marker["clock_start"]) if name in records else (None, marker["started_at"]))
    return runs


# ---------- the sheets ----------

def summaries(records: dict, kept: list[int]) -> list[tuple[str, str, dict]]:
    """(prompt, run id, day entry) for every usable or partial run-day of the kept pairs."""
    found = []
    for pair in kept:
        for prompt in PROMPTS:
            for entry in records[run_id(pair, prompt)]["records"]:
                if entry["result"]["outcome"] in ("usable", "partial") and entry["result"].get("summary"):
                    found.append((prompt, run_id(pair, prompt), entry))
    return found


def full_item_evidence(records: dict, kept: list[int], order: dict, days: list[str]) -> dict:
    """{prompt: {day: run id or None}}: the first run in the seeded order, among kept pairs, usable on that day."""
    chosen = {}
    for prompt in PROMPTS:
        chosen[prompt] = {}
        for day in days:
            chosen[prompt][day] = None
            for pair in order[prompt]:
                if pair not in kept:
                    continue
                entry = next((e for e in records[run_id(pair, prompt)]["records"] if e["run"] == day), None)
                if entry is not None and entry["result"]["outcome"] in ("usable", "partial") \
                        and entry["result"].get("summary"):
                    chosen[prompt][day] = run_id(pair, prompt)
                    break
    return chosen


def letters_for(shown: list, seed: int = SEED_SHEET2) -> dict:
    order = list(range(len(shown)))
    random.Random(seed).shuffle(order)
    names = [chr(ord("A") + n) if n < 26 else "A" + chr(ord("A") + n - 26) for n in range(len(shown))]
    return {names[n]: shown[i] for n, i in enumerate(order)}


def build_sheets(records, kept, order, days, passages, events, labels, locked, p4_items) -> dict:
    shown = summaries(records, kept)
    by_letter = letters_for(shown)
    letters = {letter: {"prompt": p, "run": r, "day": e["run"]} for letter, (p, r, e) in by_letter.items()}
    rng = random.Random(SEED_SHEET1)

    # sheet 1: pairs and overreach
    pair_rows, pair_key, over_rows, over_key = [], {}, [], {}
    for letter, (_, _, entry) in by_letter.items():
        if entry["run"] in p6.PAIRS:
            number = len(pair_rows) + 1
            pair = " -> ".join(p6.PAIRS[entry["run"]])
            pair_rows.append({"row": number, "summary": letter, "pair": pair, "outlook_stated": "", "kept_apart": "",
                              "note": ""})
            pair_key[str(number)] = {"letter": letter, "fingerprint": sj._hash(letter, pair)}
    points = [(letter, point) for letter, (_, _, entry) in by_letter.items()
              for point in entry["result"]["summary"].get("industry_context", [])]
    rng.shuffle(points)
    for letter, point in points:
        keys = sj._keys(by_letter[letter][2])
        number = len(over_rows) + 1
        evidence = "; ".join(keys[c] for c in point["citations"])
        over_rows.append({"row": number, "summary": letter, "text": point["text"], "evidence": evidence,
                          "overreach": "", "note": ""})
        over_key[str(number)] = {"letter": letter, "fingerprint": sj._hash(letter, point["text"], evidence)}

    # sheet 2: the items of the full-item evidence, then decoys; events of every shown summary
    evidence = full_item_evidence(records, kept, order, days)
    new, carried = [], []
    for prompt in PROMPTS:
        for day, chosen in evidence[prompt].items():
            if chosen is None:
                continue
            entry = next(e for e in records[chosen]["records"] if e["run"] == day)
            for item in sj._items_of(entry):
                ev = "; ".join(item["evidence"])
                base = {"prompt": prompt, "source_run": chosen, "run": day, "section": item["section"],
                        "index": item["index"], "ids": item["ids"], "repaired": item["repaired"]}
                sig = p4.signature(item["section"], item["text"], ev)
                if sig in locked:
                    carried.append({**base, "locked_row": locked[sig]})
                else:
                    new.append((base, item, ev))
    random.Random(SEED_SHEET2).shuffle(new)
    items, item_key = [], {}
    for number, (base, item, ev) in enumerate(new, start=1):
        row = {"row": number, "kind": sj.KINDS[item["section"]], "text": item["text"], "evidence": ev,
               "judgement": "", "contradicted": "", "central": "", "note": ""}
        items.append(row)
        item_key[str(number)] = {"design": "direct", **base, "fingerprint": sj._hash(row["kind"], row["text"], ev)}
    two = p6.add_decoys({"items": items, "key": {"items": item_key}}, p4_items, SEED_DECOYS, SEED_SHEET2, DECOYS)

    event_rows, event_key = [], {}
    for letter, (_, _, entry) in sorted(by_letter.items()):
        titles = {h["id"]: h["title"] for h in entry["result"]["headlines"]}
        relevant = sj._relevant_events(entry, events, labels)
        for event in sorted(relevant):
            number = len(event_rows) + 1
            row = {"row": number, "summary": letter, "event": event,
                   "articles": " | ".join(titles[i] for i in relevant[event]), "covered": "", "note": ""}
            event_rows.append(row)
            event_key[str(number)] = {"letter": letter, "event": event, "ids": relevant[event],
                                      "fingerprint": sj._hash(letter, event, row["articles"])}

    key = {"letters": letters, "pairs": pair_key, "overreach": over_key, "items": two["key"]["items"],
           "carried": carried, "events": event_key, "full_item_evidence": evidence, "kept": kept,
           "seeds": {"sheet1": SEED_SHEET1, "sheet2": SEED_SHEET2, "decoys": SEED_DECOYS}}
    titles = {a["key"]: a["title"] for _, _, e in shown for a in e["articles"]}
    page2 = sj.reading_page(two["items"], sorted((l, e) for l, (_, _, e) in by_letter.items()), passages, titles)
    page1 = p6._page1({l: e for l, (_, _, e) in sorted(by_letter.items())}, passages)
    return {"pairs": pair_rows, "overreach": over_rows, "items": two["items"], "events": event_rows, "key": key,
            "page1": page1, "page2": page2}


def sheet1_problems(pairs: list[dict], overreach: list[dict], key: dict) -> list[str]:
    found = []
    planned = sorted(l for l, v in key["letters"].items() if v["day"] in p6.PAIRS)
    if sorted(e["letter"] for e in key["pairs"].values()) != planned:
        found.append("pairs: the key does not hold one row for every shown summary of a paired day")
    for name, rows, columns in (("pairs", pairs, ("outlook_stated", "kept_apart")),
                                ("overreach", overreach, ("overreach",))):
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
            if name == "pairs" and "no" in (sj._value(row["outlook_stated"]), sj._value(row["kept_apart"])) \
                    and not sj._value(row.get("note")):
                found.append(f"pairs row {number}: every no needs a note")
            if name == "overreach" and sj._value(row["overreach"]) == "yes" and not sj._value(row.get("note")):
                found.append(f"overreach row {number}: the note names the extended claim")
    return found


def key_problems(key: dict, records: dict, markers: dict, stop, plan: dict, days: list[str]) -> list[str]:
    """The key against the validated saved runs: the kept pairs, the shown summaries and their letters, and the
    full-item evidence must all be what the runs and the plan give, not only what the key file says."""
    found = []
    state = round_state(records, set(markers), stop)
    if key.get("kept") != state["kept"]:
        found.append(f"the key's kept pairs {key.get('kept')} are not the saved runs' {state['kept']}")
        return found
    shown = sorted((p, r, e["run"]) for p, r, e in summaries(records, state["kept"]))
    in_key = sorted((v["prompt"], v["run"], v["day"]) for v in key.get("letters", {}).values())
    if in_key != shown:
        found.append("the key's summaries are not the usable summaries of the kept pairs")
    order = {p: plan["orders"]["full_item_order"][p] for p in PROMPTS}
    if key.get("full_item_evidence") != full_item_evidence(records, state["kept"], order, days):
        found.append("the key's full-item evidence is not the plan's seeded choice from the saved runs")
    by_letter = letters_for(summaries(records, state["kept"]))
    if {l: (p, r, e["run"]) for l, (p, r, e) in by_letter.items()} != \
            {l: (v["prompt"], v["run"], v["day"]) for l, v in key.get("letters", {}).items()}:
        found.append("the key's letters are not the seeded letters of the saved runs")
    return found


def sheet_problems(sheets: tuple, key: dict) -> list[str]:
    pairs, overreach, items, events = sheets
    return sheet1_problems(pairs, overreach, key) + sj.item_problems(items, key["items"]) + \
        sj.event_problems(events, key["events"])


# ---------- the score ----------

def day_events(baseline: dict, events: dict, labels: dict) -> dict:
    """{day: number of events}, from the locked prompt 4 run (the same articles on every run)."""
    return {e["run"]: len(sj._relevant_events(e, events, labels)) for e in baseline["records"]}


def outcome_table(records: dict, kept: list[int]) -> dict:
    """{prompt: {run id: {day: outcome}}} for the kept pairs."""
    return {p: {run_id(pair, p): {e["run"]: e["result"]["outcome"] for e in records[run_id(pair, p)]["records"]}
                for pair in kept} for p in PROMPTS}


def all_run_measures(pairs, overreach, events, key, outcomes, n_events: dict, kept, records) -> dict:
    letters = key["letters"]
    measure = {}
    pair_rows = {str(r["row"]).strip(): r for r in pairs}
    event_rows = {str(r["row"]).strip(): r for r in events}
    for prompt in PROMPTS:
        runs = outcomes[prompt]
        failed = sum(1 for days in runs.values() for o in days.values() if o == "failed")
        pair_score = {day: 0 for day in sorted(p6.PAIRS)}
        for number, entry in key["pairs"].items():
            info = letters[entry["letter"]]
            if info["prompt"] == prompt:
                row = pair_rows[number]
                pair_score[info["day"]] += (sj._value(row["outlook_stated"]) == "yes") + \
                    (sj._value(row["kept_apart"]) == "yes")
        covered = {day: 0 for day in n_events}
        for number, entry in key["events"].items():
            info = letters[entry["letter"]]
            if info["prompt"] == prompt and sj._value(event_rows[number]["covered"]) == "yes":
                covered[info["day"]] += 1
        over_rows = {str(r["row"]).strip(): r for r in overreach}
        over = sum(1 for n, e in key["overreach"].items() if letters[e["letter"]]["prompt"] == prompt
                   and sj._value(over_rows[n]["overreach"]) == "yes")
        points = sum(1 for e in key["overreach"].values() if letters[e["letter"]]["prompt"] == prompt)
        measure[prompt] = {"failed_run_days": failed, "run_days": 3 * len(kept),
                           "partial_run_days": sum(1 for d in runs.values() for o in d.values() if o == "partial"),
                           "pair_score": pair_score, "pair_total": sum(pair_score.values()),
                           "pair_scores_count": 2 * len(kept),
                           "events": {d: {"covered": covered[d], "total": n * len(kept)} for d, n in n_events.items()},
                           "overreach": over, "industry_points": points}
    return measure


def full_item_measures(items, key, locked_sheet, carried_evidence: dict, records) -> dict:
    rows = {str(r["row"]).strip(): r for r in items}
    locked_rows = sj._rows_by_number(locked_sheet)
    judged = [dict(rows[n], prompt=e["prompt"], run=e["run"], section=e["section"])
              for n, e in key["items"].items() if not e.get("decoy")]
    judged += [dict(locked_rows[str(c["locked_row"])], prompt=c["prompt"], run=c["run"], section=c["section"])
               for c in key["carried"]]
    out = {}
    for prompt in PROMPTS:
        mine = [r for r in judged if r["prompt"] == prompt]
        core = [r for r in mine if r["section"] in p4.CORE]
        industry = [r for r in mine if r["section"] == "industry_context"]
        chosen = key["full_item_evidence"][prompt]
        empty, without = [], []
        for day, source in chosen.items():
            if source is None:
                continue
            entry = next(e for e in records[source]["records"] if e["run"] == day)
            is_empty = not entry["result"]["summary"].get("industry_context")
            evidence = carried_evidence[day]["industry_evidence"]
            if is_empty and evidence == "yes":
                empty.append(day)
            if not is_empty and evidence == "no":
                without.append(day)
        out[prompt] = {"evidence": chosen, "unavailable_days": sorted(d for d, s in chosen.items() if s is None),
                       "core": {"total": len(core), "supported": sum(sj._value(r["judgement"]) == "supported"
                                                                     for r in core),
                                "not_supported": sum(sj._value(r["judgement"]) == "not_supported" for r in core),
                                "contradicted": sum(sj._value(r["contradicted"]) == "yes" for r in core)},
                       "industry": {"total": len(industry),
                                    "not_supported": sum(sj._value(r["judgement"]) == "not_supported"
                                                         for r in industry),
                                    "contradicted": sum(sj._value(r["contradicted"]) == "yes" for r in industry)},
                       "empty_with_evidence_days": empty, "points_without_evidence_days": without}
    return out


def choose(measures: dict, full: dict, kept: list[int]) -> dict:
    m4, m6, f4, f6 = measures["4"], measures["6"], full["4"], full["6"]
    days = sorted(p6.PAIRS)
    both = [d for d in f6["evidence"] if f6["evidence"][d] is not None and f4["evidence"][d] is not None]
    share = f6["core"]["supported"] / f6["core"]["total"] if f6["core"]["total"] else 0.0
    conditions = {
        "at_least_2_kept_pairs": len(kept) >= MIN_KEPT,
        "failures_not_more": m6["failed_run_days"] <= m4["failed_run_days"],
        "pair_total_higher": m6["pair_total"] > m4["pair_total"],
        "pair_not_lower_on_each_day": all(m6["pair_score"][d] >= m4["pair_score"][d] for d in days),
        "overreach_not_higher": m6["overreach"] <= m4["overreach"],
        "events_not_lower_each_day": all(m6["events"][d]["total"] == m4["events"][d]["total"]
                                         and m6["events"][d]["covered"] >= m4["events"][d]["covered"]
                                         for d in m4["events"]),
        "prompt6_evidence_every_day": not f6["unavailable_days"],
        "no_unsupported_industry_point": f6["industry"]["not_supported"] == 0,
        "no_unsupported_key_point_or_brief": f6["core"]["not_supported"] == 0,
        "no_contradicted_item": f6["core"]["contradicted"] == 0 and f6["industry"]["contradicted"] == 0,
        "supported_share_at_least_0_75": f6["core"]["total"] > 0 and share >= MIN_SHARE,
        "no_industry_points_without_evidence": not f6["points_without_evidence_days"],
        "empty_with_evidence_not_more": sum(d in both for d in f6["empty_with_evidence_days"])
        <= sum(d in both for d in f4["empty_with_evidence_days"]),
    }
    winner = all(conditions.values())
    return {"comparison_made": True, "conditions": conditions, "supported_share": share,
            "full_item_evidence": {"prompt4": f4["evidence"], "prompt6": f6["evidence"]},
            "days_with_both_evidence": both,
            "days_left_out_of_the_empty_section_comparison": sorted(set(f6["evidence"]) - set(both)),
            "conclusion": "prompt 6 meets every condition" if winner else
            "prompt 6 does not meet every condition; prompt 4 remains",
            "prompt": "6" if winner else "4"}


# ---------- commands ----------

def _plan_command(tpd: int) -> None:
    path = EVAL_DIR / PLAN_FILE
    if path.exists():
        raise Round3Error(f"{PLAN_FILE} exists: the plan comes once, before the runs.")
    record = plan_record(tpd, (EVAL_DIR / p6.P4_RECORD).read_text(encoding="utf-8"), _git("rev-parse", "HEAD"))
    sj._write_new(path, json.dumps(record, ensure_ascii=False, indent=1))
    print(f"Orders: {record['orders']}\nDaily limit: {tpd}; allowance {ALLOWANCE} a run")
    print(f"Saved: {path}\nNext: commit it, then: git tag {PLAN_TAG}")


def _inputs(records):
    from src.evaluate import load_label_set
    from src.fulltext_eval import CHECKED_LABEL_FILE
    from src.summary_run import EVENTS_FILE as DEV_EVENTS, load_events
    return load_events(EVAL_DIR / DEV_EVENTS), load_label_set([EVAL_DIR / CHECKED_LABEL_FILE])


def _sheet_command() -> None:
    from src.fulltext import FULLTEXT_DIR
    from src.ranking_eval import DEV_RUNS
    plan, _, records, markers, stop, found = _preflight()
    if found:
        raise Round3Error("The sheets are refused:\n" + "\n".join(found))
    state = round_state(records, set(markers), stop)
    if not state["finished"]:
        raise Round3Error(f"The round is not finished: kept pairs {state['kept']}, next pair {state['next']}.")
    if len(state["kept"]) < MIN_KEPT:
        result = insufficient_result(state, records)
        sj._write_new(EVAL_DIR / SCORE_FILE, json.dumps(result, indent=1))
        print(f"Fewer than {MIN_KEPT} kept pairs: no comparison is made; prompt 4 remains as the frozen design, "
              f"not as a winner. Saved: data/eval/{SCORE_FILE}")
        return
    events, labels = _inputs(records)
    passages = sj._passages(records[run_id(state["kept"][0], "4")])
    p3_items, p3_key, p4_items, p4_key, _ = p5._earlier_locks()
    locked, _ = p5.locked_sources(p3_items, p3_key, p4_items, p4_key)
    order = {p: plan["orders"]["full_item_order"][p] for p in PROMPTS}
    built = build_sheets(records, state["kept"], order, sorted(DEV_RUNS), passages, events, labels, locked, p4_items)
    built["key"].update(stopped_at=state["stopped_at"])
    files = {EVAL_DIR / PAIRS_FILE: sj._csv_text(built["pairs"], PAIR_COLUMNS),
             EVAL_DIR / OVERREACH_FILE: sj._csv_text(built["overreach"], OVERREACH_COLUMNS),
             EVAL_DIR / ITEMS_FILE: sj._csv_text(built["items"], sj.ITEM_COLUMNS),
             EVAL_DIR / EVENTS_FILE: sj._csv_text(built["events"], sj.EVENT_COLUMNS),
             EVAL_DIR / KEY_FILE: json.dumps(built["key"], indent=1),
             Path(FULLTEXT_DIR) / PAGE1_FILE: built["page1"], Path(FULLTEXT_DIR) / PAGE2_FILE: built["page2"]}
    for path in files:
        if path.exists():
            raise Round3Error(f"{path.name} already exists; it is not overwritten.")
    for path, text in files.items():
        sj._write_new(path, text)
    print(f"Sheet 1, judge first: {len(built['pairs'])} pair rows (data/eval/{PAIRS_FILE}), "
          f"{len(built['overreach'])} overreach rows (data/eval/{OVERREACH_FILE}); page {Path(FULLTEXT_DIR) / PAGE1_FILE}")
    print(f"Sheet 2: {len(built['items'])} item rows (data/eval/{ITEMS_FILE}; {len(built['key']['carried'])} unchanged "
          f"items keep their locked judgement), {len(built['events'])} event rows (data/eval/{EVENTS_FILE}); "
          f"page {Path(FULLTEXT_DIR) / PAGE2_FILE}")
    print(f"Key: data/eval/{KEY_FILE}  (do not open it until every sheet is judged and locked)")


def insufficient_result(state: dict, records: dict) -> dict:
    """The record when fewer than 2 pairs are kept: no comparison, with the evidence of why."""
    return {"comparison_made": False,
            "conclusion": "insufficient data: no comparison; prompt 4 remains as the frozen design, not as a winner",
            "choice": {"prompt": "4", "reason": f"only {len(state['kept'])} kept pairs; at least {MIN_KEPT} needed"},
            "kept": state["kept"], "stopped_at": state["stopped_at"],
            "runs": {name: {"outcomes": {e["run"]: e["result"].get("outcome") for e in r["records"]},
                            "tokens": r.get("tokens"), "allowance_exceeded": r.get("allowance_exceeded")}
                     for name, r in sorted(records.items())},
            "rule": RULE, "scored": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _sheets():
    return tuple(sj._read_csv(EVAL_DIR / n) for n in (PAIRS_FILE, OVERREACH_FILE, ITEMS_FILE, EVENTS_FILE))


def _code_fp() -> str:
    """The same code files as the plan, for the lock and the score."""
    return p6.code_fingerprint(CODE_FILES)


def _inputs_fp(records) -> dict:
    from src.fresh_eval import file_fingerprint
    names = [PLAN_FILE, p6.GUIDE_FILE, sj.GUIDE_FILE, p4.GUIDE_FILE, p6.P4_RECORD] + list(DATA_FILES) + \
        [record_file(int(n[4]), n[-1]) for n in sorted(records)] + ([STOP_FILE] if (EVAL_DIR / STOP_FILE).exists()
                                                                    else [])
    return {n: file_fingerprint(EVAL_DIR / n) for n in names}


def _lock_command() -> None:
    if (EVAL_DIR / LOCK_FILE).exists():
        raise Round3Error(f"{LOCK_FILE} already exists: the sheets are locked.")
    from src.ranking_eval import DEV_RUNS
    plan, _, records, markers, stop, frozen = _preflight()
    if frozen:
        raise Round3Error("The lock is refused:\n" + "\n".join(frozen))
    sheets, key = _sheets(), sj._load_json(EVAL_DIR / KEY_FILE)
    frozen = key_problems(key, records, markers, stop, plan, sorted(DEV_RUNS))
    if frozen:
        raise Round3Error("The lock is refused:\n" + "\n".join(frozen))
    found = sheet_problems(sheets, key)
    if found:
        raise Round3Error("The sheets are not complete:\n" + "\n".join(found))
    lock = {"sheets": [sj.sheet_fingerprint(s) for s in sheets], "key": sj._hash(key), "code": _code_fp(),
            "inputs": _inputs_fp(records), "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    sj._write_new(EVAL_DIR / LOCK_FILE, json.dumps(lock, indent=1))
    print(f"Locked: data/eval/{LOCK_FILE}")


def _score_command() -> None:
    if (EVAL_DIR / SCORE_FILE).exists():
        raise Round3Error(f"{SCORE_FILE} already exists; it is not overwritten.")
    from src.ranking_eval import DEV_RUNS
    plan, _, records, markers, stop, frozen = _preflight()
    if frozen:
        raise Round3Error("The score is refused:\n" + "\n".join(frozen))
    sheets, key = _sheets(), sj._load_json(EVAL_DIR / KEY_FILE)
    frozen = key_problems(key, records, markers, stop, plan, sorted(DEV_RUNS))
    if frozen:
        raise Round3Error("The score is refused:\n" + "\n".join(frozen))
    found = p6.lock_changes(sj._load_json(EVAL_DIR / LOCK_FILE), sheets, key, _inputs_fp(records), _code_fp())
    if found:
        raise Round3Error("The score cannot be made:\n" + "\n".join(found))
    pairs, overreach, items, events = sheets
    kept = key["kept"]
    ev, labels = _inputs(records)
    baseline = sj._load_json(EVAL_DIR / p6.P4_RECORD)
    n_events = day_events(baseline, ev, labels)
    measures = all_run_measures(pairs, overreach, events, key, outcome_table(records, kept), n_events, kept, records)
    p3_items, p3_key, p4_items, p4_key, p4_industry = p5._earlier_locks()
    _, locked_sheet = p5.locked_sources(p3_items, p3_key, p4_items, p4_key)
    full = full_item_measures(items, key, locked_sheet, p5.carried_evidence(p4_industry, p4_key), records)
    result = {"choice": choose(measures, full, kept), "all_runs": measures, "full_item": full,
              "outcomes": outcome_table(records, kept), "kept": kept, "stopped_at": key.get("stopped_at"),
              "decoys": p6.decoy_agreement(items, key), "rule": RULE,
              "scored": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    sj._write_new(EVAL_DIR / SCORE_FILE, json.dumps(result, indent=1))
    for prompt in PROMPTS:
        m = measures[prompt]
        print(f"Prompt {prompt}: failed {m['failed_run_days']}/{m['run_days']}, pairs {m['pair_score']} "
              f"(total {m['pair_total']}/{2 * m['pair_scores_count']}), overreach {m['overreach']}/"
              f"{m['industry_points']}, events {m['events']}")
    c = result["choice"]
    print(f"Full-item evidence: {c['full_item_evidence']}; left out of the empty-section comparison: "
          f"{c['days_left_out_of_the_empty_section_comparison']}")
    print(f"Conditions: {c['conditions']}\nDecoys: {result['decoys']}")
    print(f"Prompt chosen: {result['choice']['prompt']}\nSaved: data/eval/{SCORE_FILE}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.summary_round3")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--tpd", type=int, required=True, help="the daily token limit shown in the Groq console")
    pair = sub.add_parser("pair")
    pair.add_argument("--other-tokens", type=int, required=True,
                      help="Groq tokens used outside this project's records in the last 24 hours (0 when none)")
    for name in ("sheet", "lock", "score"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    if args.command == "plan":
        _plan_command(args.tpd)
    elif args.command == "pair":
        _pair_command(args.other_tokens)
    else:
        {"sheet": _sheet_command, "lock": _lock_command, "score": _score_command}[args.command]()


if __name__ == "__main__":
    try:
        main()
    except (Round3Error, p6.Prompt6Error, sj.JudgingError) as error:
        print(f"[STOP] {error}")
        sys.exit(1)
