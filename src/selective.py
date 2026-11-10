"""
Selective ranking: the model checks only the articles that do not name the
company in the title (Route 1 of the free-plan work), so a query sends fewer
tokens. Simulated first on saved model answers, with no new model calls.

    SA  title articles first, in O1 order, with no model call; the rest
        below them, by the model's answer (direct, indirect, not; no answer
        goes with indirect), ties by O1, then NewsAPI order
    SB  title articles and the rest answered `direct` form one top group in
        NewsAPI order, so strong body news can compete for the top places;
        then indirect (and no answer), then not, ties by O1, then NewsAPI order

With no model answers at all, SA and SB give the O1 order (the fallback).
A title article's saved answer is never used by SA or SB.

The simulation compares O1, O2, O3, SA and SB with both selections (S1, S2)
under the unchanged acceptance rule, and names the single-report events each
configuration loses against O1, and the wrong promotions (not-relevant
articles it puts in the top k that O1 leaves out). Runtime from tokens is labelled an estimate.

Run (no model calls; reads the saved comparison runs):
    python -m src.selective
"""

import json
import sys

from src import ranking as rk

METHODS = ("SA", "SB")
CONFIGURATIONS = ["O1+S1", "O1+S2", "O2+S1", "O2+S2", "O3+S1", "O3+S2", "SA+S1", "SA+S2", "SB+S1", "SB+S2"]
CONFIGS_TO_SIMULATE = ("gpt-oss-120b", "gpt-oss-20b")


def _title(candidate: dict) -> bool:
    return candidate.get("mention_level") == "title"


def _group(candidate: dict) -> int:
    label = rk._model_label(candidate)
    return rk.LABEL_GROUP["indirect"] if label is None else rk.LABEL_GROUP[label]


def order(candidates: list[dict], method: str) -> list[dict]:
    if method not in METHODS:
        raise ValueError(f"Unknown selective method {method}; use one of {METHODS}.")
    position = {c["key"]: i for i, c in enumerate(candidates)}
    titles = [c for c in candidates if _title(c)]
    rest = [c for c in candidates if not _title(c)]
    if method == "SA":
        return (sorted(titles, key=rk._mention)
                + sorted(rest, key=lambda c: (_group(c), rk._mention(c), position[c["key"]])))
    top = sorted(titles + [c for c in rest if rk._model_label(c) == "direct"], key=lambda c: position[c["key"]])
    lower = [c for c in rest if rk._model_label(c) != "direct"]
    return top + sorted(lower, key=lambda c: (_group(c), rk._mention(c), position[c["key"]]))


def sent_keys(candidates: list[dict], method: str) -> list[str]:
    """The articles the model checks: all of them for O2 and O3, only those without a title mention for SA and SB."""
    if method in METHODS:
        return [c["key"] for c in candidates if not _title(c)]
    return [c["key"] for c in candidates]


def token_estimate(query_tokens: int, articles: int, sent: int, tpm: int) -> dict:
    tokens = round(query_tokens * sent / articles) if articles else 0
    return {"kind": "estimate", "tokens": tokens, "seconds": tokens / tpm * 60}


def single_report_losses(base: list[str], other: list[str], events: dict[str, str],
                         candidate_keys: list[str]) -> list[str]:
    """Events covered by the baseline but not by the other selection, that only one candidate reports."""
    reports: dict[str, int] = {}
    for key in candidate_keys:
        if key in events:
            reports[events[key]] = reports.get(events[key], 0) + 1
    covered = lambda keys: {events[k] for k in keys if k in events}
    return sorted(e for e in covered(base) - covered(other) if reports.get(e) == 1)


def _ordered(candidates: list[dict], method: str) -> list[dict]:
    return order(candidates, method) if method in METHODS else rk.order(candidates, method)


def simulate(candidates: list[dict], labels: dict[str, str], events: dict[str, str], ks=(5, 10, 15),
             primary_k: int = 10, threshold: float = 0.5, seconds: dict[str, float] | None = None) -> dict:
    seconds = seconds or {}
    relevant = {c["key"]: labels[c["key"]] in ("direct", "indirect") for c in candidates}
    keys = [c["key"] for c in candidates]
    plain = [dict(c, check=None) for c in candidates]
    o1_plain = [c["key"] for c in rk.order(plain, "O1")]
    configurations, retention_same = {}, True
    for name in CONFIGURATIONS:
        method, selection = name.split("+")
        ordered = _ordered(candidates, method)
        configurations[name] = {}
        for k in ks:
            chosen = rk.select(ordered, k, selection, threshold)
            selected = [c["key"] for c in chosen["selected"]]
            shown = selected + [c["key"] for c in chosen["additional"]]
            retention_same = retention_same and sorted(shown) == sorted(keys)
            configurations[name][k] = {**rk.measures(selected, keys, relevant, events, labels, k),
                                       "selected_keys": selected}
    for name in CONFIGURATIONS:
        for k in ks:
            configurations[name][k]["single_report_losses"] = single_report_losses(
                configurations[rk.BASELINE][k]["selected_keys"], configurations[name][k]["selected_keys"],
                events, keys)
            base_keys = set(configurations[rk.BASELINE][k]["selected_keys"])
            configurations[name][k]["wrong_promotions"] = [
                key for key in configurations[name][k]["selected_keys"] if not relevant[key] and key not in base_keys]
    acceptance = {}
    base = configurations[rk.BASELINE][primary_k]
    for name in CONFIGURATIONS:
        if name == rk.BASELINE:
            continue
        method = name.split("+")[0]
        uses_model = method != "O1"
        fallback_ok = [c["key"] for c in _ordered(plain, method)] == o1_plain if uses_model else None
        acceptance[name] = rk.acceptance(base, configurations[name][primary_k], uses_model=uses_model,
                                         retention_same=retention_same,
                                         seconds=seconds.get(method) if uses_model else None,
                                         fallback_ok=fallback_ok)
    return {"configurations": configurations, "acceptance": acceptance}


# ---------- running it ----------

def main() -> None:
    from src.config import settings
    from src.fulltext import article_id
    from src.fulltext_eval import _inputs
    from src.model_compare import _dev_runs, _folder, run_file_name
    from src.ranking_eval import KS, PRIMARY_K, choose_threshold
    from src.summary_run import EVENTS_FILE, load_events
    from src.config import EVAL_DIR
    _, labels_by_url, _ = _inputs()
    labels = {article_id(url): label for url, label in labels_by_url.items()}
    events = {article_id(url): e for url, e in load_events(EVAL_DIR / EVENTS_FILE).items()}
    runs, _ = _dev_runs("Tata Motors")
    threshold = choose_threshold([c for run in runs.values() for c in run], events)["threshold"]
    record = {"label_file": "v3_dev_tata_motors.csv", "threshold": threshold, "kind": "simulation on saved answers",
              "results": {}}
    for config in CONFIGS_TO_SIMULATE:
        saved = json.loads((_folder() / run_file_name(config)).read_text(encoding="utf-8"))["runs"]
        record["results"][config] = {}
        print(f"\n===== {config} (saved answers; SA/SB times are estimates)")
        for run, candidates in runs.items():
            result = saved[run]
            checked = [dict(c, check=result["answers"][c["key"]]) for c in candidates]
            stats = result["llm_stats"]
            query_tokens = sum(c["input"] - c.get("cached", 0) + c["output"] for c in stats.get("per_call", []))
            sent = len(sent_keys(checked, "SA"))
            estimate = token_estimate(query_tokens, len(candidates), sent, settings.groq_tpm)
            seconds = {"O2": result["wall_seconds"], "O3": result["wall_seconds"],
                       "SA": estimate["seconds"], "SB": estimate["seconds"]}
            sim = simulate(checked, labels, events, ks=KS, primary_k=PRIMARY_K, threshold=threshold, seconds=seconds)
            record["results"][config][run] = {"sent_selective": sent, "articles": len(candidates),
                                              "estimate": estimate, "simulation": sim}
            print(f"  {run}: {len(candidates)} articles, SA/SB send {sent} "
                  f"(estimated {estimate['seconds']:.0f} s from {estimate['tokens']} tokens)")
            for name in CONFIGURATIONS:
                m = sim["configurations"][name][PRIMARY_K]
                verdict = sim["acceptance"].get(name)
                status = "baseline" if verdict is None else ("PASS" if verdict["passed"] else "fail")
                failed = [] if verdict is None else [c for c, ok in verdict["conditions"].items() if not ok]
                change = "" if verdict is None else (f" +{len(verdict['events']['gained'])}/"
                                                     f"-{len(verdict['events']['lost'])} events, direct "
                                                     f"+{len(verdict['events']['direct_gained'])}/"
                                                     f"-{len(verdict['events']['direct_lost'])}")
                print(f"     {name:<6} {status:<8} events {len(m['events_covered'])}/{m['relevant_events']}, "
                      f"precision {m['precision']:.2f}{change}, wrong promotions {len(m['wrong_promotions'])}, "
                      f"single-report lost {m['single_report_losses']}"
                      + (f"  failed {failed}" if failed else ""))
    (_folder() / "selective_simulation.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print(f"\nSaved: {_folder() / 'selective_simulation.json'}")


if __name__ == "__main__":
    main()
