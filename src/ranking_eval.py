"""
The development ranking report: the six ranking configurations on the
development runs, measured against the hand-made events.

Each run is rebuilt as the app would see it:
    - the run's searches merged, each article once (src/multi_search.py)
    - Stage 1 on full text, when the fetched text passes the sanity check
      (src/fulltext.py, check_full_text); copies of one headline merged by
      the cleaning rules
    - where the company is named (title, description or only the body) is
      read from the article's own fields, so a name found only in the full
      text counts as "body"
Then O1-O3 x S1-S2 are measured at k = 5, 10 and 15 (src/ranking.py), and
each configuration is judged against the baseline O1 + S1 at k = 10 by the
rule fixed before the fresh run.

The S2 similarity threshold is chosen here, on development data only, by a
rule fixed in advance: the fewest grouping errors (false merges + missed
merges) against the hand-made events; on a tie, the higher threshold.

Development results are design evidence only. The decision is made once,
on the fresh run.

Run:
    python -m src.ranking_eval            (Stage 1 orderings only, no API keys)
    python -m src.ranking_eval --check    (with model answers, prompt version 3; saved and reused)
    python -m src.ranking_eval --check gpt-oss-120b
                                          (a named setting from the model comparison, src/model_compare.py)
                                          the provider is LLM_PROVIDER in .env; the record saves
                                          the experiment record, token use and rate-limit errors
"""

import json
import sys
import time

from src.config import DATA_DIR, EVAL_DIR, RAW_DIR
from src.evaluate import is_relevant
from src.preprocess import _mention_level, clean_articles, normalize_url
from src.query import load_aliases, parse_query
from src import ranking as rk
from src.ranking import RankingError

DEV_RUNS = {
    "2026-10-14": ["20261014_182548_tata-motors.json"],
    "2026-10-23": ["20261023_161430_tata-motors.json"],
    "2026-10-24": ["20261024_111739_tata-motors.json", "20261024_111739_tata-motors_2.json"],
}
KS = (5, 10, 15)
PRIMARY_K = 10
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
RECORD_FILE = "ranking_dev_record.json"
CHECK_CACHE = EVAL_DIR / "relevance_checks"

__all__ = ["RankingError"]          # used by the tests through this module


# ---------- the runs ----------

def merge_run(responses: list[dict]) -> list[dict]:
    """The articles of a run's searches, in order, each once."""
    seen, merged = set(), []
    for response in responses:
        for article in response.get("articles", []):
            key = normalize_url(article.get("url") or "")
            if key not in seen:
                seen.add(key)
                merged.append(article)
    return merged


def stage1_candidates(raw_articles: list[dict], full_texts: dict[str, str], company: str,
                      aliases: dict | None) -> list[dict]:
    """What Stage 1 keeps from one run, in NewsAPI's order, ready for ranking."""
    from src.fulltext import article_id
    from src.fulltext_eval import usable_full_texts
    if not raw_articles:
        return []
    texts, _ = usable_full_texts(raw_articles, full_texts, company, aliases)
    texts = {normalize_url(url): text for url, text in texts.items()}
    by_key, widened = {}, []
    for article in raw_articles:
        key = normalize_url(article.get("url") or "")
        by_key[key] = article
        text = texts.get(key)
        copy = dict(article)
        if text:            # added to the description: the cleaning step drops a content field that repeats it
            copy["description"] = f"{article.get('description') or ''} {text}".strip()
        widened.append(copy)
    kept, _ = clean_articles(widened, max_articles=len(widened), query=company, aliases=aliases,
                             remove_junk=True, drop_listings="both")
    rule = parse_query(company, aliases=aliases)
    candidates = []
    for article in kept:
        raw = by_key[normalize_url(article["url"])]
        fields = {"title": raw.get("title") or "", "description": raw.get("description") or "",
                  "content": raw.get("content") or ""}
        candidates.append({"key": article_id(raw["url"]), "url": raw["url"], "title": fields["title"],
                           "description": fields["description"],
                           "source": (raw.get("source") or {}).get("name") or "",
                           "mention_level": _mention_level(fields, rule), "check": None,
                           "text": texts.get(normalize_url(raw["url"]), "")})
    return candidates


# ---------- one run ----------

def _configurations(with_model: bool) -> list[str]:
    orderings = rk.ORDERINGS if with_model else ("O1",)
    return [name for name in rk.SIMPLICITY if name.split("+")[0] in orderings]


def evaluate_run(candidates: list[dict], labels: dict[str, str], events: dict[str, str], ks=KS,
                 threshold: float = 0.5, primary_k: int = PRIMARY_K, seconds: float | None = None) -> dict:
    """
    labels - {article id: direct / indirect / not}, guide version 3
    events - {article id: event name}, for every relevant candidate
    """
    relevant = {c["key"]: is_relevant(labels[c["key"]]) for c in candidates}
    keys = [c["key"] for c in candidates]
    with_model = any((c.get("check") or {}).get("source") == "model" for c in candidates)
    names = _configurations(with_model)
    plain = [dict(c, check=None) for c in candidates]
    o1_plain = [c["key"] for c in rk.order(plain, "O1")]
    fallback_ok = all([c["key"] for c in rk.order(plain, method)] == o1_plain for method in ("O2", "O3"))

    configurations, retention_same = {}, True
    for name in names:
        ordering, selection = name.split("+")
        ordered = rk.order(candidates, ordering)
        configurations[name] = {}
        for k in ks:
            chosen = rk.select(ordered, k, selection, threshold)
            selected = [c["key"] for c in chosen["selected"]]
            shown = selected + [c["key"] for c in chosen["additional"]]
            retention_same = retention_same and sorted(shown) == sorted(keys)
            configurations[name][k] = {**rk.measures(selected, keys, relevant, events, labels, k),
                                       "selected_keys": selected, "also_reported_by": chosen["also_reported_by"]}

    acceptance = {}
    if primary_k in ks:
        base = configurations[rk.BASELINE][primary_k]
        for name in names:
            if name == rk.BASELINE:
                continue
            uses_model = not name.startswith("O1")
            acceptance[name] = rk.acceptance(base, configurations[name][primary_k], uses_model=uses_model,
                                             retention_same=retention_same,
                                             seconds=seconds if uses_model else None,
                                             fallback_ok=fallback_ok if uses_model else None)
    winner_input = {name: {**verdict, **configurations[name][primary_k]} for name, verdict in acceptance.items()}
    return {"candidates": len(candidates), "relevant_candidates": sum(relevant.values()),
            "configurations": configurations, "acceptance": acceptance,
            "winner": rk.choose_winner(winner_input) if acceptance else rk.BASELINE,
            "not_run": sorted(name for name in rk.SIMPLICITY if name not in names),
            "title_protection": rk.title_protection_effect(candidates, relevant) if with_model
            else {"rescued": [], "promoted": []},
            "retention_same": retention_same, "fallback_ok": fallback_ok}


def choose_threshold(candidates: list[dict], events: dict[str, str], thresholds=THRESHOLDS) -> dict:
    """Fewest false + missed merges against the hand-made events; on a tie, the higher threshold."""
    quality = {t: rk.grouping_quality(candidates, events, t) for t in thresholds}
    errors = {t: q["false_merges"] + q["missed_merges"] for t, q in quality.items()}
    best = sorted(thresholds, key=lambda t: (errors[t], -t))[0]
    return {"threshold": best, "errors": errors, "quality": quality}


# ---------- running it ----------

def _load_run(files: list[str]) -> list[dict]:
    return merge_run([json.loads((RAW_DIR / name).read_text(encoding="utf-8")) for name in files])


def ranking_settings(threshold) -> dict:
    """The ranking part of the experiment record; the dev and fresh runs build it the same way."""
    return {"threshold": threshold, "ks": list(KS), "primary_k": PRIMARY_K, "order": list(rk.SIMPLICITY)}


def query_items(candidates: list[dict], company: str, aliases: dict | None) -> list[dict]:
    """What the model reads for each candidate: id, title and the relevance passage."""
    from src.relevance_check import build_passage
    items = []
    for c in candidates:
        passage = build_passage(c["title"], c["text"], company, aliases, short_text=c["description"])
        items.append({"id": c["key"], "title": c["title"], "passage": passage["text"]})
    return items


def cache_hits(items: list[dict], llm, company: str, scope: str, cache) -> list[str]:
    """The ids whose model answer is already saved for this exact question and model setting."""
    from src.relevance_check import _key
    model, tag = getattr(llm, "model_name", "unknown"), getattr(llm, "cache_tag", "") or ""
    return [item["id"] for item in items if cache.get(_key(item, company, model, scope, tag)) is not None]


def live_seconds(runtime: dict | None) -> float | None:
    """A runtime counts as live only when no answer came from the cache."""
    if not runtime or runtime.get("from_cache", 0):
        return None
    return runtime.get("seconds_total")


def _ask_model(candidates: list[dict], company: str, aliases: dict | None, llm=None,
               cache="default") -> tuple[dict, dict]:
    from src.llm import make_llm
    from src.relevance_check import WORKERS, CheckCache, classify, company_scope
    llm = llm or make_llm()
    cache = CheckCache(CHECK_CACHE) if cache == "default" else cache
    items = query_items(candidates, company, aliases)
    times = []

    def progress(done, total, seconds):
        times.append(seconds)
        print(f"   batch {done}/{total} answered in {seconds:.1f} s", flush=True)

    scope = company_scope(company, aliases)
    from_cache = cache_hits(items, llm, company, scope, cache) if cache else []
    print(f"Model checks: {len(items)} candidates ({len(from_cache)} already saved), {WORKERS} batches at a time",
          flush=True)
    started = time.perf_counter()
    answers = classify(items, llm, company, cache=cache, progress=progress,
                       scope=scope, workers=WORKERS)
    runtime = {"provider": llm.provider, "model": llm.model_name, "llm_stats": getattr(llm, "stats", None),
               "candidates": len(items), "batches_asked": len(times), "workers": WORKERS,
               "batch_seconds": [round(t, 1) for t in times],
               "seconds_total": round(time.perf_counter() - started, 1),
               "fallbacks": sum(1 for a in answers.values() if a["source"] != "model"),
               "from_cache": len(from_cache),
               "kind": "live" if not from_cache else ("cached" if len(from_cache) == len(items) else "mixed")}
    return answers, runtime


def check_argument(argv: list[str]):
    """False without --check; the setting name after --check (a name from src/model_compare.py, CONFIGS); None
    when --check names no setting (the project defaults from .env)."""
    if "--check" not in argv:
        return False
    position = argv.index("--check") + 1
    name = argv[position] if position < len(argv) else None
    return name if name and not name.startswith("--") else None


def check_llm(name: str | None):
    """The model for the check: a named setting from the model comparison, or the project defaults."""
    from src import llm as llm_module
    from src.model_compare import CONFIGS
    if name is None:
        return llm_module.make_llm()
    if name not in CONFIGS:
        raise ValueError(f"Unknown model setting {name!r}; use one of {sorted(CONFIGS)}.")
    config = CONFIGS[name]
    return llm_module.make_llm(config["provider"], model=config["model"], generation=config["generation"])


def main(company: str = "Tata Motors", check=False) -> None:
    from src.fulltext import article_id
    from src.fulltext_eval import _inputs
    from src.summary_run import EVENTS_FILE, load_events
    _, labels_by_url, full_texts = _inputs()
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    labels = {article_id(url): label for url, label in labels_by_url.items()}
    events = {article_id(url): event for url, event in load_events(EVAL_DIR / EVENTS_FILE).items()}
    runs = {name: stage1_candidates(_load_run(files), full_texts, company, aliases)
            for name, files in DEV_RUNS.items()}
    unique = {c["key"]: c for run in runs.values() for c in run}

    runtime = llm = None
    if check is not False:
        llm = check_llm(check)
        answers, runtime = _ask_model(list(unique.values()), company, aliases, llm=llm)
        for run in runs.values():
            for c in run:
                c["check"] = answers[c["key"]]
    choice = choose_threshold(list(unique.values()), events)
    print(f"\nS2 threshold chosen on dev data: {choice['threshold']} "
          f"(errors by threshold: {choice['errors']})")
    q = choice["quality"][choice["threshold"]]
    print(f"   at {choice['threshold']}: {q['true_merges']} true merges, {q['false_merges']} false merges, "
          f"{q['missed_merges']} missed merges, of {q['pairs']} pairs")

    reports = {}
    for name, candidates in runs.items():
        report = evaluate_run(candidates, labels, events, threshold=choice["threshold"],
                              seconds=live_seconds(runtime))          # a cached runtime never passes
        reports[name] = report
        print(f"\n=== Run {name}: {report['candidates']} candidates, {report['relevant_candidates']} relevant ===")
        print(f"{'config':<7}{'k':>3}{'sel':>5}{'rel':>5}{'prec':>7}{'art.rec':>9}"
              f"{'ev.rec':>8}{'slot':>7}{'dir.ev':>8}")
        for config, by_k in report["configurations"].items():
            for k, m in by_k.items():
                print(f"{config:<7}{k:>3}{m['selected']:>5}{m['relevant_selected']:>5}{m['precision']:>7.2f}"
                      f"{m['article_recall']:>9.2f}{m['event_recall']:>8.2f}{m['slot_use']:>7.2f}"
                      f"{m['direct_event_recall']:>8.2f}   events {len(m['events_covered'])}/{m['relevant_events']}")
        for config, verdict in report["acceptance"].items():
            failed = [c for c, ok in verdict["conditions"].items() if not ok]
            e = verdict["events"]
            print(f"   {config}: {'PASS' if verdict['passed'] else 'fail'}  gained {e['gained']} lost {e['lost']}"
                  f"  direct gained {e['direct_gained']} lost {e['direct_lost']}"
                  + (f"  failed: {failed}" if failed else ""))
        print(f"   title protection: {report['title_protection']}; retention same: {report['retention_same']}; "
              f"fallback gives O1: {report['fallback_ok']}; winner on this run: {report['winner']}")
    if runtime:
        print(f"\nRuntime (prompt version 3, {runtime['workers']} batches at a time): {runtime}")
        if runtime["kind"] != "live":
            print(f"   {runtime['from_cache']} of {runtime['candidates']} answers come from the cache: this runtime "
                  f"is {runtime['kind']}, not a live measurement. The live runtimes are in data/eval/model_compare/.")
    from src.experiment import build_record, current_commit
    record = {"about": "Ranking configurations on the development runs (design evidence only).",
              "experiment": (build_record(llm, ranking_settings(choice["threshold"]), current_commit())
                             if llm else None),
              "runs": DEV_RUNS, "threshold": {"chosen": choice["threshold"],
                                              "errors": {str(t): e for t, e in choice["errors"].items()}},
              "runtime": runtime,
              "reports": {name: {**r, "configurations": {c: {str(k): m for k, m in by_k.items()}
                                                          for c, by_k in r["configurations"].items()}}
                          for name, r in reports.items()}}
    (EVAL_DIR / RECORD_FILE).write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str),
                                        encoding="utf-8")
    print(f"Saved: data/eval/{RECORD_FILE}")


if __name__ == "__main__":
    main(check=check_argument(sys.argv[1:]))
