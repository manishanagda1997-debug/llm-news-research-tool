"""
Ranking variants for O1's headline weakness (second development cycle, round 1).

O1 puts every candidate that names the company in its title first, so a relevant article whose headline does
not name the company ranks below all of them (Stage 7: place 1 in the retrieval order, place 47 of 53 in O1).
Four variants use no model:

    O1R1  the first candidate of the retrieval order keeps place 1; the rest follow in O1 order
    O1R2  the same with the first two candidates
    O1M3  a candidate that does not name the company in its title but names it at least 3 times in its text
          joins the title group, keeping its retrieval position within the group
    O1M5  the same with at least 5 names

"The retrieval order" is the order of the candidates Stage 1 keeps (search A as returned, then search B as
returned, each article once). Every candidate stays in the list; only the order changes. Selection is S1: the
first k of the order (src/ranking.py). The frozen src/ranking.py is not changed.

The names counted are the ones of Stage 1's own matching rule (parse_query with the alias file), matched as
whole words without case, the longest name first and without overlap; a name that counts only next to a context
word counts only when one of its context words appears in the text; a text with an excluded term counts 0, as
in Stage 1's matching. The text counted is the full text Stage 1 uses, or the short description without it.

The rule, fixed before any result (plan): on each of the three Tata Motors development days at k = 10, a variant
loses no direct event against O1 + S1, and its event recall and precision are not lower (a tie is not lower). A
variant passes only on all three days. Among passing variants: the most direct events covered, summed over the
days; then the most events covered, summed; then pooled precision; then simplicity (O1R1, O1R2, O1M3, O1M5).
With no passing variant, O1 + S1 stays. k = 5 and 15 are reported only.

Commands, in order:
    python -m src.ranking_variants plan     the rule and the fingerprints of every input and of the code
                                            (ranking_variants_plan.json); then commit, then:
                                            git tag freeze-ranking-variants-plan
    python -m src.ranking_variants run      once: refused unless the plan equals its tag and every fingerprint
                                            matches; the result (ranking_variants_dev_record.json)
    python -m src.ranking_variants week1    after the run, for the chosen variant only: whether it keeps the
                                            article the Stage 7 transfer test loses (information only)
"""

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from src import ranking as rk
from src.config import DATA_DIR, EVAL_DIR, RAW_DIR, ROOT_DIR
from src.evaluate import is_relevant

COMPANY = "Tata Motors"
VARIANTS = ("O1R1", "O1R2", "O1M3", "O1M5")
SETTINGS = {"O1R1": ("retrieval", 1), "O1R2": ("retrieval", 2), "O1M3": ("mentions", 3), "O1M5": ("mentions", 5)}
PRIMARY_K = 10
KS = (5, 10, 15)
LABEL_FILE = "v3_dev_tata_motors_checked.csv"
PLAN_FILE = "ranking_variants_plan.json"
RECORD_FILE = "ranking_variants_dev_record.json"
WEEK1_FILE = "ranking_variants_week1_check.json"
PLAN_TAG = "freeze-ranking-variants-plan"
CODE_FILES = ("src/ranking_variants.py", "src/ranking.py", "src/ranking_eval.py", "src/preprocess.py",
              "src/query.py", "src/fulltext.py", "src/fulltext_eval.py", "src/evaluate.py", "src/summary_run.py",
              "src/config.py")
RULE = {
    "primary_k": PRIMARY_K,
    "baseline": rk.BASELINE,
    "no_harm": "on each development day at k = 10: no direct event lost against O1 + S1 (compared by event name), "
               "event recall not lower and precision not lower (a tie is not lower); a variant passes only on "
               "all three days",
    "ranking": "the most direct events covered, summed over the days; then the most events covered, summed; then "
               "pooled precision (relevant selected over all days / selected over all days); then simplicity",
    "simplicity": list(VARIANTS),
    "none_passes": "O1 + S1 stays",
    "reported_only": "k = 5 and k = 15",
    "week1": "after the choice, for the chosen variant only; information only, never a reason to change the choice",
}


class VariantError(Exception):
    pass


# ---------- counting the company's names ----------

def _term_pattern(term: str) -> str:
    return re.escape(term).replace(r"\ ", r"\s+")


def name_count(text: str, rule) -> int:
    """Whole-word, case-free count of the rule's names in the text, the longest name first, no overlap."""
    from src.query import _contains
    text = (text or "").lower()
    if any(_contains(text, term) for term in rule.exclude):
        return 0
    names = list(rule.include)
    names += [name for name, words in rule.include_with_context if any(_contains(text, word) for word in words)]
    names = sorted(set(names), key=len, reverse=True)
    if not names:
        return 0
    pattern = "|".join(_term_pattern(name) for name in names)
    return len(re.findall(rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])", text))


def counted_text(candidate: dict) -> str:
    return candidate.get("text") or candidate.get("description") or ""


# ---------- the variants ----------

def order_variant(candidates: list[dict], variant: str, rule=None) -> list[dict]:
    """candidates: in retrieval order. The baseline is "O1"."""
    if variant == "O1":
        return rk.order(candidates, "O1")
    if variant not in SETTINGS:
        raise ValueError(f"Unknown variant {variant}; use one of {VARIANTS}.")
    kind, value = SETTINGS[variant]
    if kind == "retrieval":
        head = list(candidates[:value])
        return head + rk.order(candidates[value:], "O1")
    if rule is None:
        raise ValueError("The mention variants need the matching rule.")

    def group(candidate):
        if candidate.get("mention_level") == "title" or name_count(counted_text(candidate), rule) >= value:
            return 0
        return rk._mention(candidate)
    return sorted(candidates, key=group)                       # sorted() keeps ties in the retrieval order


# ---------- one day and the choice ----------

def day_result(candidates: list[dict], labels: dict[str, str], events: dict[str, str], rule) -> dict:
    """labels and events: {article id: ...} for the day's candidates."""
    keys = [c["key"] for c in candidates]
    relevant = {key: is_relevant(labels[key]) for key in keys}
    result = {}
    for name in ("O1",) + VARIANTS:
        ordered = order_variant(candidates, name, rule)
        if sorted(c["key"] for c in ordered) != sorted(keys):
            raise VariantError(f"{name} does not keep every candidate.")
        result[name] = {str(k): {**rk.measures([c["key"] for c in rk.select(ordered, k, "S1")["selected"]],
                                                keys, relevant, events, labels, k),
                                 "selected_keys": [c["key"] for c in ordered[:k]]} for k in KS}
    base = result["O1"][str(PRIMARY_K)]
    checks = {}
    for name in VARIANTS:
        other = result[name][str(PRIMARY_K)]
        change = rk.compare_events(base, other)
        checks[name] = {"no_direct_event_lost": not change["direct_lost"],
                        "event_recall_not_lower": rk._not_lower(other["event_recall"], base["event_recall"]),
                        "precision_not_lower": rk._not_lower(other["precision"], base["precision"]),
                        "events": change}
        checks[name]["passed"] = all(checks[name][c] for c in ("no_direct_event_lost", "event_recall_not_lower",
                                                                "precision_not_lower"))
    return {"candidates": len(candidates), "measures": result, "checks": checks}


def choose(days: dict[str, dict]) -> dict:
    """The rule of the plan, applied to the day results."""
    k = str(PRIMARY_K)
    passing = [name for name in VARIANTS if all(day["checks"][name]["passed"] for day in days.values())]

    def totals(name):
        ms = [day["measures"][name][k] for day in days.values()]
        selected = sum(m["selected"] for m in ms)
        return {"direct_events_covered": sum(len(m["direct_events_covered"]) for m in ms),
                "events_covered": sum(len(m["events_covered"]) for m in ms),
                "pooled_precision": sum(m["relevant_selected"] for m in ms) / selected if selected else 0.0}
    pooled = {name: totals(name) for name in ("O1",) + VARIANTS}
    ranked = sorted(passing, key=lambda n: (-pooled[n]["direct_events_covered"], -pooled[n]["events_covered"],
                                            -pooled[n]["pooled_precision"], VARIANTS.index(n)))
    return {"passing": passing, "pooled": pooled, "ranked": ranked, "chosen": ranked[0] if ranked else rk.BASELINE}


# ---------- inputs and fingerprints ----------

def _sha_text(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def code_fingerprint(names=CODE_FILES) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8") + b"\0" + (ROOT_DIR / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def candidates_fingerprint(days: dict[str, list[dict]]) -> str:
    """Every candidate of every day, with the fields the variants and the measures read."""
    shaped = {day: [[c["key"], c.get("title", ""), c.get("description", ""), c.get("mention_level"),
                     c.get("text", "")] for c in cands] for day, cands in days.items()}
    return _sha_text(json.dumps(shaped, ensure_ascii=False, sort_keys=True))


def load_inputs() -> dict:
    """The Stage 3 development candidates, labels and events, keyed by article id."""
    from src.fresh_eval import file_fingerprint
    from src.fulltext import article_id
    from src.fulltext_eval import _inputs
    from src.query import load_aliases, parse_query
    from src.ranking_eval import DEV_RUNS, _load_run, stage1_candidates
    from src.summary_run import EVENTS_FILE, load_events
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    _, labels_by_url, full_texts = _inputs(LABEL_FILE)
    days = {day: stage1_candidates(_load_run(files), full_texts, COMPANY, aliases) for day, files in DEV_RUNS.items()}
    labels = {article_id(url): label for url, label in labels_by_url.items()}
    events = {article_id(url): event for url, event in load_events(EVAL_DIR / EVENTS_FILE).items()}
    fingerprints = {
        "labels": file_fingerprint(EVAL_DIR / LABEL_FILE), "events": file_fingerprint(EVAL_DIR / EVENTS_FILE),
        "aliases": file_fingerprint(DATA_DIR / "company_aliases.json"),
        "raw_files": {name: file_fingerprint(RAW_DIR / name) for files in DEV_RUNS.values() for name in files},
        "candidates": candidates_fingerprint(days), "code": code_fingerprint(),
    }
    return {"days": days, "labels": labels, "events": events, "rule": parse_query(COMPANY, aliases=aliases),
            "fingerprints": fingerprints, "dev_runs": DEV_RUNS}


def plan_record(fingerprints: dict, dev_runs: dict, source_commit: str) -> dict:
    return {"variants": {name: {"kind": SETTINGS[name][0], "value": SETTINGS[name][1]} for name in VARIANTS},
            "rule": RULE, "label_file": LABEL_FILE, "dev_runs": dev_runs, "code_files": list(CODE_FILES),
            "fingerprints": fingerprints, "source_commit": source_commit, "tag": PLAN_TAG,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def run_problems(plan_text: str | None, tag_text: str | None, plan: dict | None, fingerprints: dict,
                 record_exists: bool) -> list[str]:
    def same(a, b):
        return a is not None and b is not None and a.replace("\r\n", "\n") == b.replace("\r\n", "\n")
    found = [f"{RECORD_FILE} already exists: the run is made once"] if record_exists else []
    if not same(plan_text, tag_text):
        found.append(f"{PLAN_FILE} is missing or differs from the tag {PLAN_TAG}")
    if plan is not None:
        found += [f"changed since the plan: {name}" for name in sorted(set(plan["fingerprints"]) | set(fingerprints))
                  if plan["fingerprints"].get(name) != fingerprints.get(name)]
    return found


def evaluate(inputs: dict) -> dict:
    days = {}
    for day, candidates in inputs["days"].items():
        keys = [c["key"] for c in candidates]
        days[day] = day_result(candidates, {k: inputs["labels"][k] for k in keys},
                               {k: v for k, v in inputs["events"].items() if k in keys}, inputs["rule"])
    return {"days": days, "choice": choose(days)}


# ---------- commands ----------

def _git(*args) -> str:
    from src.fresh_eval import git
    return git(*args)


def _tag_text() -> str | None:
    try:
        return _git("show", f"{PLAN_TAG}:{(EVAL_DIR / PLAN_FILE).relative_to(ROOT_DIR).as_posix()}")
    except Exception:
        return None


def _plan_command() -> None:
    path = EVAL_DIR / PLAN_FILE
    if path.exists():
        raise VariantError(f"{PLAN_FILE} already exists; it is not overwritten.")
    if (EVAL_DIR / RECORD_FILE).exists():
        raise VariantError(f"{RECORD_FILE} exists: the plan must come before the run.")
    inputs = load_inputs()
    record = plan_record(inputs["fingerprints"], inputs["dev_runs"], _git("rev-parse", "HEAD"))
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=1))
    print(f"Variants: {', '.join(VARIANTS)}; days: {', '.join(inputs['days'])}")
    print(f"Saved: {path}\nNext: commit it, then: git tag {PLAN_TAG}")


def _run_command() -> None:
    plan_path, record_path = EVAL_DIR / PLAN_FILE, EVAL_DIR / RECORD_FILE
    plan_text = plan_path.read_text(encoding="utf-8") if plan_path.exists() else None
    plan = json.loads(plan_text) if plan_text else None
    inputs = load_inputs()
    found = run_problems(plan_text, _tag_text(), plan, inputs["fingerprints"], record_path.exists())
    if found:
        raise VariantError("The run is refused:\n" + "\n".join(found))
    result = evaluate(inputs)
    record = {"plan_fingerprint": _sha_text(plan_text), "fingerprints": inputs["fingerprints"], **result,
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(record_path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=1))
    choice = result["choice"]
    for day, r in result["days"].items():
        print(day, {name: ("pass" if c["passed"] else "FAIL") for name, c in r["checks"].items()})
    print(f"Passing on all three days: {choice['passing'] or 'none'}; chosen: {choice['chosen']}")
    print(f"Saved: {record_path}")


def week1_check(candidates: list[dict], chosen: str, rule, lost_keys: list[str]) -> dict:
    """Information only: the places of the lost article(s) in the chosen variant and in O1."""
    o1 = [c["key"] for c in order_variant(candidates, "O1", rule)]
    variant = [c["key"] for c in order_variant(candidates, chosen, rule)]
    return {"chosen": chosen, "candidates": len(candidates),
            "articles": {key: {"retrieval": [c["key"] for c in candidates].index(key) + 1,
                               "O1": o1.index(key) + 1, chosen: variant.index(key) + 1,
                               "in_top_10": variant.index(key) < PRIMARY_K} for key in lost_keys}}


def _week1_command() -> None:
    from src import transfer_eval as te
    from src.query import load_aliases, parse_query
    record_path, path = EVAL_DIR / RECORD_FILE, EVAL_DIR / WEEK1_FILE
    if not record_path.exists():
        raise VariantError(f"{RECORD_FILE} does not exist: the choice comes first.")
    if path.exists():
        raise VariantError(f"{WEEK1_FILE} already exists; it is not overwritten.")
    chosen = json.loads(record_path.read_text(encoding="utf-8"))["choice"]["chosen"]
    if chosen == rk.BASELINE:
        raise VariantError("No variant is chosen, so there is nothing to check.")
    score = json.loads((te.RUN_DIR / te.SCORE_FILE).read_text(encoding="utf-8"))
    aliases = load_aliases(te.ALIAS_FILE)
    candidates = te._run_candidates(aliases)
    from src.fulltext import article_id
    from src.summary_run import load_events
    events = {article_id(u): e for u, e in load_events(te.RUN_DIR / te.EVENTS_FILE).items()}
    keys = {c["key"] for c in candidates}
    lost = [k for k, e in events.items() if e in score["ranking"]["events"]["direct_lost"] and k in keys]
    result = {**week1_check(candidates, chosen, parse_query(te.COMPANY, aliases=aliases), lost),
              "note": "information only; the Stage 7 result stays as scored, and the choice does not change",
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result["articles"], indent=1))
    print(f"Saved: {path}")


COMMANDS = {"plan": _plan_command, "run": _run_command, "week1": _week1_command}

if __name__ == "__main__":
    try:
        if len(sys.argv) == 2 and sys.argv[1] in COMMANDS:
            COMMANDS[sys.argv[1]]()
        else:
            print(__doc__)
    except VariantError as error:
        print(f"[STOP] {error}")
        sys.exit(1)
