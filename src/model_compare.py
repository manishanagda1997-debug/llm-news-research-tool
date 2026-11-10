"""
Model comparison on the development runs: Gemini and three Groq models, for the
relevance check that orders the ranking (O2, O3).

Configurations are fixed before any run (CONFIGS). A `low` reasoning version
of a Groq configuration is added only when its pilot estimate is over the
180-second limit; quality scores never decide which configurations run.

    pilot        one batch of 5 real candidates; tokens per article give an
                 estimate for a whole query, always labelled as an estimate
    query run    each dev run is one real query: its own candidates, no saved
                 answers, wall-clock seconds from start to the last answer or
                 fallback (waits, retries and splits included); on Groq, the
                 next query starts after a clear minute, as a new search would
    model        coverage, precision on the answered articles, model recall
                 (a relevant article the model fails on is not found), F1,
                 unsure share, fallbacks by kind
    end to end   the six ranking configurations and the unchanged acceptance
                 rule; a failed article stays in the ranking in the middle
                 group; wrong promotions are counted
    consistency  articles in more than one run: a different label on the same
                 input is an inconsistency; a different input is listed apart
    coverage     a run is valid for quality only when the model answers at
                 least 90% of the articles; every failure keeps its kind. A
                 second full run is allowed once, only when the first falls
                 under 90% mainly through provider failures (server, rate or
                 daily limit, timeout); both runs stay and are reported
    choice       eligible: complete records, at least 90% coverage, and a model
                 configuration that passes acceptance on at least 2 of the 3
                 runs; then events at k=10, wrong promotions, F1, fallbacks,
                 seconds, data terms; none eligible -> O1

Run (each command makes live model requests, except `report`):
    python -m src.model_compare pilot <configuration>
    python -m src.model_compare run <configuration>
    python -m src.model_compare report
    python -m src.model_compare batch <3|4|5>      the batch-size experiment, one size per run
    python -m src.model_compare batch-report       its results beside the comparison run (size 10)
"""

import hashlib
import json
import sys
import time

from src import ranking as rk

CONFIGS = {
    "gemini-3.8-flash": {"provider": "gemini", "model": "gemini-3.8-flash",
                         "generation": {"temperature": 1.0, "thinking_level": "medium"}},
    "gpt-oss-120b": {"provider": "groq", "model": "openai/gpt-oss-120b",
                     "generation": {"temperature": 1.0, "reasoning_effort": "medium"}},
    "gpt-oss-20b": {"provider": "groq", "model": "openai/gpt-oss-20b",
                    "generation": {"temperature": 1.0, "reasoning_effort": "medium"}},
    "qwen3.8-27b": {"provider": "groq", "model": "qwen/qwen3.8-27b",
                    "generation": {"temperature": 1.0, "top_p": 0.95, "reasoning_effort": "medium",
                                   "reasoning_format": "hidden"}},
}
PILOT_ARTICLES = 5
QUERY_GAP = 61          # seconds between two dev queries: each starts with a clear tokens-a-minute window
DATA_TERMS = {"groq": 0, "gemini": 1}      # Groq with zero data retention, then Gemini's free tier
CHOICE_NOTE = ("The configuration chosen on the development runs goes to the fresh test only. It becomes the "
               "default only if it passes the acceptance rule on the fresh test.")
RELEVANT = {"direct", "indirect"}
OLD_ERRORS = {"timeout": "timeout", "no valid answer": "invalid_output", "too large": "too_large"}  # records without kind
MIN_COVERAGE = 0.9                  # a run is valid for quality only when the model answers this share of articles
PROVIDER_FAILURES = {"server", "rate_limit", "daily_limit", "timeout"}


def valid_for_quality(answered: int, articles: int) -> bool:
    return articles > 0 and answered / articles >= MIN_COVERAGE


def second_run_allowed(first: dict) -> bool:
    """One more full run, only when the first falls below the coverage line mainly through provider failures."""
    kinds = first.get("failure_kinds") or {}
    provider = sum(n for kind, n in kinds.items() if kind in PROVIDER_FAILURES)
    return first["coverage"] < MIN_COVERAGE and provider > sum(kinds.values()) / 2


def with_fallback_kinds(runs: dict[str, dict]) -> dict[str, dict]:
    """A copy in which old fallbacks without a kind take their query's failure kind, when that query
    failed in one kind only; otherwise "unknown". A kind already recorded is kept. The saved record never changes."""
    from src.relevance_check import FALLBACK_KINDS
    fixed = json.loads(json.dumps(runs))
    for run in fixed.values():
        kinds = [FALLBACK_KINDS.get(k, "unknown") for k, n in (run.get("llm_stats") or {}).get("failures", {}).items()
                 if n]
        for answer in run["answers"].values():
            if answer.get("source") != "model" and not answer.get("kind"):
                answer["kind"] = kinds[0] if len(kinds) == 1 else "unknown"
    return fixed


# ---------- the batch-size experiment (separate from the comparison) ----------

BATCH_SIZES = (3, 4, 5)
BATCH_CONFIG = "gpt-oss-120b"
BATCH_QUERY = "2026-10-14"                 # a 25-article query


def batch_file_name(size: int) -> str:
    return f"batch_{BATCH_CONFIG}_size{size}.json"


def batch_rows(records: list[dict], labels: dict[str, str]) -> list[dict]:
    rows = []
    for record in records:
        result, stats = record["result"], record["result"]["llm_stats"]
        measures = model_measures(result["answers"], labels)
        rows.append({"batch_size": record["batch_size"], "wall_seconds": result["wall_seconds"],
                     "meets_180_s": result["wall_seconds"] <= rk.MAX_MODEL_SECONDS,
                     "calls": stats.get("calls"), "splits": stats.get("splits"),
                     "rate_limited": stats.get("rate_limited"), "waited_seconds": stats.get("waited_seconds"),
                     "tokens_counted": sum(c["input"] - c.get("cached", 0) + c["output"]
                                           for c in stats.get("per_call", [])),
                     "coverage": measures["coverage"], "precision": measures["precision"],
                     "model_recall": measures["model_recall"], "f1": measures["f1"],
                     "fallbacks": measures["fallbacks"]})
    return rows


def run_file_name(name: str, attempt: int = 1) -> str:
    return f"run_{name}.json" if attempt == 1 else f"run_{name}_attempt{attempt}.json"


# ---------- configurations ----------

def configs_after_pilot(pilot_seconds: dict[str, float]) -> dict[str, dict]:
    """The fixed configurations, plus a low reasoning version of each Groq one whose pilot estimate is too slow."""
    configs = {name: json.loads(json.dumps(c)) for name, c in CONFIGS.items()}
    for name, config in CONFIGS.items():
        if config["provider"] == "groq" and pilot_seconds.get(name, 0) > rk.MAX_MODEL_SECONDS:
            low = json.loads(json.dumps(config))
            low["generation"]["reasoning_effort"] = "low"
            configs[f"{name}-low"] = low
    return configs


def make_config_llm(config: dict, **options):
    from src.llm import make_llm
    return make_llm(config["provider"], model=config["model"], generation=config["generation"], **options)


# ---------- pilot ----------

def pilot_estimate(per_call: list[dict], pilot_articles: int, query_articles: int, tpm: int) -> dict:
    tokens = sum(c["input"] + c["output"] for c in per_call)        # output already holds the reasoning tokens
    per_article = tokens / pilot_articles
    query_tokens = round(per_article * query_articles)
    return {"kind": "estimate", "tokens_per_article": per_article, "query_tokens": query_tokens,
            "seconds": query_tokens / tpm * 60}


def time_row(estimate: dict, measured_seconds: float) -> dict:
    return {"estimated_seconds": estimate["seconds"], "measured_seconds": measured_seconds,
            "difference_seconds": measured_seconds - estimate["seconds"]}


# ---------- one query ----------

def _fingerprint(item: dict, scope: str) -> str:
    from src.relevance_check import PROMPT_VERSION
    text = json.dumps([PROMPT_VERSION, scope, item["title"], item["passage"]], ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_query(candidates: list[dict], llm, company: str, aliases: dict | None, clock=time.perf_counter,
              batch_size: int | None = None) -> dict:
    """One dev run as one real query: no saved answers, wall-clock time, a record for every article.
    batch_size: the project's BATCH_SIZE unless the batch-size experiment names another."""
    from src.ranking_eval import query_items
    from src.relevance_check import BATCH_SIZE, WORKERS, classify, company_scope
    batch_size = batch_size or BATCH_SIZE
    items = query_items(candidates, company, aliases)
    scope = company_scope(company, aliases)
    started = clock()
    answers = classify(items, llm, company, batch_size=batch_size, cache=None, scope=scope, workers=WORKERS)
    seconds = clock() - started
    if set(answers) != {c["key"] for c in candidates}:
        raise ValueError("Every candidate needs a record, an answer or a fallback.")
    return {"cached": False, "batch_size": batch_size, "wall_seconds": round(seconds, 1), "answers": answers,
            "inputs": {item["id"]: _fingerprint(item, scope) for item in items},
            "llm_stats": json.loads(json.dumps(getattr(llm, "stats", {})))}


# ---------- measures ----------

def model_measures(answers: dict[str, dict], labels: dict[str, str]) -> dict:
    answered = {k: a for k, a in answers.items() if a.get("source") == "model"}
    claimed = [k for k, a in answered.items() if a.get("label") in RELEVANT]
    gold = [k for k in answers if labels[k] in RELEVANT]
    right = [k for k in claimed if labels[k] in RELEVANT]
    precision = len(right) / len(claimed) if claimed else None
    recall = len(right) / len(gold) if gold else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    fallbacks = {}
    for a in answers.values():
        if a.get("source") != "model":
            kind = a.get("kind") or OLD_ERRORS.get(a.get("error"), "unknown")
            fallbacks[kind] = fallbacks.get(kind, 0) + 1
    return {"articles": len(answers), "answered": len(answered), "coverage": len(answered) / len(answers),
            "precision": precision, "model_recall": recall, "f1": f1,
            "unsure_share": (sum(1 for a in answered.values() if a.get("certainty") == "unsure") / len(answered)
                             if answered else None),
            "fallbacks": fallbacks}


def wrong_promotions(report: dict, labels: dict[str, str], k: int = 10) -> dict[str, list[str]]:
    """Not-relevant articles a model configuration puts in the top k that O1+S1 leaves out."""
    configs = report["configurations"]
    base = set(configs[rk.BASELINE][k]["selected_keys"])
    return {name: [key for key in by_k[k]["selected_keys"] if labels[key] not in RELEVANT and key not in base]
            for name, by_k in configs.items() if not name.startswith("O1")}


def consistency(runs: dict[str, dict]) -> dict:
    seen: dict[str, list[str]] = {}
    for name, run in runs.items():
        for key in run["answers"]:
            seen.setdefault(key, []).append(name)
    repeated = sorted(k for k, names in seen.items() if len(names) > 1)
    same, different, label_changes, relevance_changes = 0, [], [], []
    for key in repeated:
        names = seen[key]
        if len({runs[n]["inputs"][key] for n in names}) > 1:
            different.append(key)
            continue
        same += 1
        found = [runs[n]["answers"][key].get("label") for n in names]
        if len(set(found)) > 1:
            label_changes.append(key)
        if len({label in RELEVANT for label in found}) > 1:
            relevance_changes.append(key)
    return {"repeated": len(repeated), "same_input": same, "different_input": different,
            "label_changes": label_changes, "relevance_changes": relevance_changes}


def pooled_answers(runs: dict[str, dict], order: list[str]) -> dict[str, dict]:
    pooled = {}
    for name in order:
        for key, answer in runs[name]["answers"].items():
            pooled.setdefault(key, answer)
    return pooled


# ---------- the choice ----------

def choose(summaries: dict[str, dict]) -> str:
    eligible = {n: s for n, s in summaries.items()
                if s["complete"] and s["runs_passed"] >= 2 and s.get("coverage", 1.0) >= MIN_COVERAGE}
    if not eligible:
        return rk.BASELINE.split("+")[0]
    return min(eligible, key=lambda n: (-eligible[n]["events_k10"], eligible[n]["wrong_promotions"],
                                        -eligible[n]["f1"], eligible[n]["fallbacks"], eligible[n]["seconds"],
                                        DATA_TERMS[eligible[n]["provider"]]))


def _shown(report: dict, name: str) -> dict:
    """The configuration's results; in a query with no model answer, the app shows the plain ranking
    with the same selection (O2/O3 fall back to O1), so those results count."""
    configs = report["configurations"]
    return configs[name] if name in configs else configs["O1+" + name.split("+")[1]]


def summarize(provider: str, reports: dict[str, dict], wrong: dict[str, dict], measures: dict,
              seconds: dict[str, float], complete: bool, k: int = 10) -> dict:
    """One configuration over the three runs, on its ranking configuration that passes most runs."""
    names = [n for n in rk.SIMPLICITY if not n.startswith("O1")]
    passed = {n: sum(1 for r in reports.values() if r["acceptance"].get(n, {}).get("passed")) for n in names}
    best = max(names, key=lambda n: (passed[n], -names.index(n)))
    return {"provider": provider, "ranking": best, "runs_passed": passed[best],
            "events_k10": sum(len(_shown(r, best)[k]["events_covered"]) for r in reports.values()),
            "wrong_promotions": sum(len(w.get(best, [])) for w in wrong.values()),
            "f1": measures["f1"], "fallbacks": sum(measures["fallbacks"].values()),
            "coverage": measures["coverage"], "failure_kinds": dict(measures["fallbacks"]),
            "seconds": max(seconds.values()), "complete": complete}


# ---------- running it ----------

def _folder():
    from src.config import EVAL_DIR
    folder = EVAL_DIR / "model_compare"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _dev_runs(company: str):
    from src.config import DATA_DIR
    from src.fulltext_eval import _inputs
    from src.query import load_aliases
    from src.ranking_eval import DEV_RUNS, _load_run, stage1_candidates
    _, _, full_texts = _inputs()
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    return {name: stage1_candidates(_load_run(files), full_texts, company, aliases)
            for name, files in DEV_RUNS.items()}, aliases


def _pilot_command(name: str, company: str = "Tata Motors") -> None:
    from src.config import settings
    runs, aliases = _dev_runs(company)
    first = next(iter(runs.values()))[:PILOT_ARTICLES]
    llm = make_config_llm(CONFIGS[name])
    result = run_query(first, llm, company, aliases)
    tpm = settings.groq_tpm if CONFIGS[name]["provider"] == "groq" else None
    sizes = {run: len(c) for run, c in runs.items()}
    estimates = ({run: pilot_estimate(result["llm_stats"]["per_call"], len(first), size, tpm)
                  for run, size in sizes.items()} if tpm and result["llm_stats"]["per_call"] else None)
    record = {"configuration": name, "settings": CONFIGS[name], "articles": len(first),
              "wall_seconds": result["wall_seconds"], "llm_stats": result["llm_stats"], "estimates": estimates}
    (_folder() / f"pilot_{name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("configuration", "wall_seconds", "estimates")}, indent=1))
    print(f"tokens per call: {result['llm_stats'].get('per_call')}")
    print(f"rate-limit headers received: {result['llm_stats'].get('header_updates')}")


def _run_command(name: str, company: str = "Tata Motors") -> None:
    from src.experiment import current_commit
    configs = configs_after_pilot(_pilot_seconds())
    if name not in configs:
        raise SystemExit(f"Unknown configuration {name}; choose from {sorted(configs)}.")
    path, attempt = _folder() / run_file_name(name), 1
    if path.exists():
        first = json.loads(path.read_text(encoding="utf-8"))
        pooled = pooled_answers(with_fallback_kinds(first["runs"]), order=list(first["runs"]))
        measures = model_measures(pooled, {k: "not" for k in pooled})      # coverage and kinds only, no labels read
        second = _folder() / run_file_name(name, attempt=2)
        if second.exists() or not second_run_allowed({"coverage": measures["coverage"],
                                                       "failure_kinds": measures["fallbacks"]}):
            raise SystemExit(f"{path.name} already exists. A second run is allowed once, only when the first "
                             f"covers under {MIN_COVERAGE:.0%} of the articles mainly through provider failures.")
        path, attempt = second, 2
        print(f"Second run (first run coverage {measures['coverage']:.0%}, failures {measures['fallbacks']}); "
              "both runs are reported.", flush=True)
    runs, aliases = _dev_runs(company)
    results = {}
    for number, (run, candidates) in enumerate(runs.items()):
        if number and configs[name]["provider"] == "groq":
            print(f"   waiting {QUERY_GAP} s, so the next query starts with a clear minute (not counted)", flush=True)
            time.sleep(QUERY_GAP)
        llm = make_config_llm(configs[name])                   # a new model function: its own counts per query
        print(f"{name}: query {run}, {len(candidates)} candidates", flush=True)
        results[run] = run_query(candidates, llm, company, aliases)
        stats = results[run]["llm_stats"]
        print(f"   {results[run]['wall_seconds']} s, waits {stats.get('waited_seconds')} s, "
              f"rate-limit errors {stats.get('rate_limited')}, headers received {stats.get('header_updates')}",
              flush=True)
    record = {"configuration": name, "attempt": attempt, "settings": configs[name], "commit": current_commit(),
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": results}
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {path}")


def _pilot_seconds() -> dict[str, float]:
    seconds = {}
    for path in _folder().glob("pilot_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("estimates"):
            seconds[record["configuration"]] = max(e["seconds"] for e in record["estimates"].values())
    return seconds


def build_report(company: str = "Tata Motors", label_file: str = "v3_dev_tata_motors.csv") -> dict:
    """The comparison of every saved run, scored with `label_file`; it reads saved answers only, never a model."""
    from src.config import EVAL_DIR
    from src.fulltext import article_id
    from src.fulltext_eval import _inputs
    from src.ranking_eval import choose_threshold, evaluate_run
    from src.summary_run import EVENTS_FILE, load_events
    _, labels_by_url, _ = _inputs(label_file)
    labels = {article_id(url): label for url, label in labels_by_url.items()}
    events = {article_id(url): e for url, e in load_events(EVAL_DIR / EVENTS_FILE).items()}
    runs, _ = _dev_runs(company)
    threshold = choose_threshold([c for run in runs.values() for c in run], events)["threshold"]
    summaries, details = {}, {}
    for path in sorted(_folder().glob("run_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        results = with_fallback_kinds(record["runs"])
        name = record["configuration"] + ("" if record.get("attempt", 1) == 1 else f" (attempt {record['attempt']})")
        reports, wrong = {}, {}
        for run, candidates in runs.items():
            checked = [dict(c, check=results[run]["answers"][c["key"]]) for c in candidates]
            reports[run] = evaluate_run(checked, labels, events, threshold=threshold,
                                        seconds=results[run]["wall_seconds"])
            wrong[run] = wrong_promotions(reports[run], labels)
        pooled = pooled_answers(results, order=list(runs))
        measures = model_measures(pooled, labels)
        complete = all(set(results[run]["answers"]) == {c["key"] for c in runs[run]} for run in runs)
        summaries[name] = summarize(record["settings"]["provider"], reports, wrong, measures,
                                    {run: results[run]["wall_seconds"] for run in runs}, complete)
        summaries[name]["attempt"] = record.get("attempt", 1)
        details[name] = {"measures_pooled": measures,
                         "measures_by_run": {run: model_measures(results[run]["answers"], labels) for run in runs},
                         "consistency": consistency(results), "wrong_promotions": wrong,
                         "seconds": {run: results[run]["wall_seconds"] for run in runs},
                         "waits": {run: results[run]["llm_stats"].get("waited_seconds") for run in runs},
                         "acceptance": {run: r["acceptance"] for run, r in reports.items()}}
    return {"label_file": label_file, "threshold": threshold, "summaries": summaries, "details": details,
            "choice": choose(summaries), "note": CHOICE_NOTE}


def _report_command(company: str = "Tata Motors") -> None:
    record = build_report(company, "v3_dev_tata_motors.csv")
    for name, s in record["summaries"].items():
        measures = record["details"].get(name, {}).get("measures_pooled", {})
        if measures and not valid_for_quality(measures["answered"], measures["articles"]):
            print(f"{name}: coverage {measures['coverage']:.0%} (failures {measures['fallbacks']}): a reliability "
                  f"result, not valid for quality")
            continue
        print(f"{name}: passes {s['runs_passed']}/3 ({s['ranking']}), events@10 {s['events_k10']}, "
              f"wrong promotions {s['wrong_promotions']}, F1 {s['f1']:.3f}, coverage "
              f"{s['coverage']:.2f}, fallbacks {s['fallbacks']}, slowest query {s['seconds']} s")
    (_folder() / "report.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print(f"\nDevelopment choice: {record['choice']}\n{CHOICE_NOTE}")


def _batch_command(size: int, company: str = "Tata Motors") -> None:
    path = _folder() / batch_file_name(size)
    if path.exists():
        raise SystemExit(f"{path.name} already exists; each batch size runs once.")
    runs, aliases = _dev_runs(company)
    candidates = runs[BATCH_QUERY]
    llm = make_config_llm(CONFIGS[BATCH_CONFIG])
    print(f"{BATCH_CONFIG}, batch size {size}: query {BATCH_QUERY}, {len(candidates)} candidates", flush=True)
    result = run_query(candidates, llm, company, aliases, batch_size=size)
    stats = result["llm_stats"]
    print(f"   {result['wall_seconds']} s, calls {stats.get('calls')}, splits {stats.get('splits')}, "
          f"rate-limit errors {stats.get('rate_limited')}, headers received {stats.get('header_updates')}", flush=True)
    record = {"configuration": BATCH_CONFIG, "settings": CONFIGS[BATCH_CONFIG], "batch_size": size,
              "query": BATCH_QUERY, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "result": result}
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {path}")


def _batch_report_command() -> None:
    from src.fulltext import article_id
    from src.fulltext_eval import _inputs
    _, labels_by_url, _ = _inputs()
    labels = {article_id(url): label for url, label in labels_by_url.items()}
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(_folder().glob("batch_*.json"))]
    comparison = _folder() / run_file_name(BATCH_CONFIG)
    if comparison.exists():                    # the comparison run of the same query, batch size 10, for reference
        run = json.loads(comparison.read_text(encoding="utf-8"))["runs"][BATCH_QUERY]
        records.append({"batch_size": run.get("batch_size", 10), "result": with_fallback_kinds(
            {BATCH_QUERY: run})[BATCH_QUERY]})
    rows = sorted(batch_rows(records, labels), key=lambda r: r["batch_size"])
    for r in rows:
        print(f"size {r['batch_size']:>2}: {r['wall_seconds']} s ({'meets' if r['meets_180_s'] else 'over'} 180 s), "
              f"calls {r['calls']}, tokens {r['tokens_counted']}, 429s {r['rate_limited']}, "
              f"coverage {r['coverage']:.0%}, F1 {r['f1']:.3f}, fallbacks {r['fallbacks']}")
    (_folder() / "batch_report.json").write_text(json.dumps({"query": BATCH_QUERY, "rows": rows}, indent=1),
                                                 encoding="utf-8")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["pilot"] and len(args) == 2 and args[1] in CONFIGS:
        _pilot_command(args[1])
    elif args[:1] == ["run"] and len(args) == 2:
        _run_command(args[1])
    elif args == ["report"]:
        _report_command()
    elif args[:1] == ["batch"] and len(args) == 2 and args[1].isdigit() and int(args[1]) in BATCH_SIZES:
        _batch_command(int(args[1]))
    elif args == ["batch-report"]:
        _batch_report_command()
    else:
        print(__doc__)
