"""
Ranking and selection of the candidates Stage 1 keeps.

Stage 1 keeps every candidate. Ranking decides the order; selection decides
which k articles feed the summary. The others are shown as "Additional
articles — lower relevance ranking", with the model's reason when there is one.

A candidate is a dict with:
    key            - the article's id (or tidy link)
    title, description, source
    mention_level  - where the company is named: "title", "description" or "content"
    check          - the Stage 2 answer (src/relevance_check.py), or None

Orderings:
    O1 - Stage 1: company named in the title, then the description, then
         only the body; ties keep NewsAPI's order
    O2 - Gemini: direct, then indirect, then not; ties keep NewsAPI's order
    O3 - Gemini with title protection: an article that names the company in
         its title but is answered `not` goes with the indirect group; ties
         broken by O1, then NewsAPI's order
    An article without a model answer (fallback) goes with the indirect group.
    With no model answers at all, O2 and O3 give the O1 order.

Selection:
    S1 - the first k of the ordering
    S2 - the first k, skipping an article too similar to one already chosen;
         free slots at the end are filled from the skipped ones, in order,
         after the distinct ones. Similarity is the Jaccard share of content words of title and
         description: shared words / all words of the two.

Measures (each kept separate), comparison with the baseline O1 + S1 by event
name, the acceptance rule and the winner rule are below.
"""

from itertools import combinations

from src.fulltext import _content_words
from src.preprocess import MENTION_ORDER

ORDERINGS = ("O1", "O2", "O3")
SELECTIONS = ("S1", "S2")
BASELINE = "O1+S1"
SIMPLICITY = ["O1+S1", "O1+S2", "O3+S1", "O3+S2", "O2+S1", "O2+S2"]
MAX_MODEL_SECONDS = 180
LABEL_GROUP = {"direct": 0, "indirect": 1, "not": 2}


class RankingError(Exception):
    pass


# ---------- orderings ----------

def _mention(candidate: dict) -> int:
    return MENTION_ORDER.get(candidate.get("mention_level"), MENTION_ORDER[None])


def _model_label(candidate: dict) -> str | None:
    check = candidate.get("check") or {}
    return check.get("label") if check.get("source") == "model" else None


def _group(candidate: dict, protect_titles: bool) -> int:
    label = _model_label(candidate)
    if label is None:
        return LABEL_GROUP["indirect"]                           # no model answer: the middle group
    if protect_titles and label == "not" and candidate.get("mention_level") == "title":
        return LABEL_GROUP["indirect"]
    return LABEL_GROUP[label]


def order(candidates: list[dict], method: str) -> list[dict]:
    if method not in ORDERINGS:
        raise ValueError(f"Unknown ordering {method}; use one of {ORDERINGS}.")
    if method == "O1" or not any(_model_label(c) for c in candidates):
        return sorted(candidates, key=_mention)                  # sorted() keeps ties in NewsAPI's order
    if method == "O2":
        return sorted(candidates, key=lambda c: _group(c, protect_titles=False))
    return sorted(candidates, key=lambda c: (_group(c, protect_titles=True), _mention(c)))


def title_protection_effect(candidates: list[dict], relevant: dict[str, bool]) -> dict:
    """Articles O3 moves up: title mentions answered `not`. Rescued if relevant, promoted wrongly if not."""
    moved = [c["key"] for c in candidates if _model_label(c) == "not" and c.get("mention_level") == "title"]
    return {"rescued": [k for k in moved if relevant.get(k)], "promoted": [k for k in moved if not relevant.get(k)]}


# ---------- similarity and selection ----------

def _words(candidate: dict) -> set[str]:
    return set(_content_words(f"{candidate.get('title') or ''} {candidate.get('description') or ''}"))


def similarity(a: dict, b: dict) -> float:
    first, second = _words(a), _words(b)
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def select(ordered: list[dict], k: int, method: str, threshold: float = 0.5) -> dict:
    if method not in SELECTIONS:
        raise ValueError(f"Unknown selection {method}; use one of {SELECTIONS}.")
    if method == "S1":
        chosen = ordered[:k]
    else:
        chosen, skipped = [], []
        for candidate in ordered:
            if len(chosen) == k:
                break
            if any(similarity(candidate, other) >= threshold for other in chosen):
                skipped.append(candidate)
            else:
                chosen.append(candidate)
        chosen += skipped[:k - len(chosen)]                      # near-copies only fill free slots, last
    chosen_keys = {c["key"] for c in chosen}
    also = {c["key"]: [o["key"] for o in ordered if o["key"] != c["key"] and o["key"] not in chosen_keys
                       and similarity(c, o) >= threshold]
            for c in chosen}
    return {"selected": chosen, "additional": [c for c in ordered if c["key"] not in chosen_keys],
            "also_reported_by": also}


def grouping_quality(candidates: list[dict], events: dict[str, str], threshold: float) -> dict:
    """S2's idea of "the same event", checked pair by pair against the hand-made events."""
    labelled = [c for c in candidates if c["key"] in events]
    counts = {"true_merges": 0, "false_merges": 0, "missed_merges": 0, "pairs": 0}
    examples = {"false_merges": [], "missed_merges": []}
    for a, b in combinations(labelled, 2):
        counts["pairs"] += 1
        merged = similarity(a, b) >= threshold
        same = events[a["key"]] == events[b["key"]]
        if merged and same:
            counts["true_merges"] += 1
        elif merged:
            counts["false_merges"] += 1
            examples["false_merges"].append((a["key"], b["key"]))
        elif same:
            counts["missed_merges"] += 1
            examples["missed_merges"].append((a["key"], b["key"]))
    return {**counts, "examples": examples}


# ---------- measures ----------

def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def measures(selected_keys: list[str], candidate_keys: list[str], relevant: dict[str, bool],
             events: dict[str, str], labels: dict[str, str], k: int) -> dict:
    relevant_candidates = [key for key in candidate_keys if relevant.get(key)]
    missing = [key for key in relevant_candidates if key not in events]
    if missing:
        raise RankingError(f"Relevant candidate(s) without an event name: {', '.join(missing)}")
    all_events = {events[key] for key in relevant_candidates}
    direct_events = {events[key] for key in relevant_candidates if labels.get(key) == "direct"}
    chosen_relevant = [key for key in selected_keys if relevant.get(key)]
    covered = {events[key] for key in chosen_relevant}
    return {
        "selected": len(selected_keys), "relevant_selected": len(chosen_relevant),
        "precision": _share(len(chosen_relevant), len(selected_keys)),
        "article_recall": _share(len(chosen_relevant), len(relevant_candidates)),
        "relevant_events": len(all_events), "events_covered": sorted(covered),
        "event_recall": _share(len(covered), len(all_events)),
        "slot_use": _share(len(covered), min(k, len(all_events))),
        "direct_events": len(direct_events), "direct_events_covered": sorted(covered & direct_events),
        "direct_event_recall": _share(len(covered & direct_events), len(direct_events)),
    }


def compare_events(base: dict, other: dict) -> dict:
    base_all, other_all = set(base["events_covered"]), set(other["events_covered"])
    base_direct, other_direct = set(base["direct_events_covered"]), set(other["direct_events_covered"])
    return {"gained": sorted(other_all - base_all), "lost": sorted(base_all - other_all),
            "net": len(other_all) - len(base_all),
            "direct_gained": sorted(other_direct - base_direct), "direct_lost": sorted(base_direct - other_direct),
            "direct_net": len(other_direct) - len(base_direct)}


# ---------- acceptance and the winner ----------

def _not_lower(value, base) -> bool:
    if base is None:
        return True
    return value is not None and value >= base


def acceptance(base: dict, other: dict, uses_model: bool, retention_same: bool,
               seconds: float | None, fallback_ok: bool | None) -> dict:
    """The rule fixed before the fresh run; every condition must hold."""
    change = compare_events(base, other)
    conditions = {
        "gain": change["direct_net"] >= 1 or change["net"] >= 2,
        "no_direct_loss": not change["direct_lost"],
        "event_recall_not_lower": _not_lower(other["event_recall"], base["event_recall"]),
        "precision_not_lower": _not_lower(other["precision"], base["precision"]),
        "retention": bool(retention_same),
    }
    if uses_model:
        conditions["runtime"] = seconds is not None and seconds <= MAX_MODEL_SECONDS
        conditions["fallback"] = bool(fallback_ok)
    return {"passed": all(conditions.values()), "conditions": conditions, "events": change}


def choose_winner(results: dict[str, dict]) -> str:
    """Among passing configurations: direct-event recall, event recall, precision, then simplicity."""
    passing = [name for name, r in results.items() if r.get("passed") and name != BASELINE]
    if not passing:
        return BASELINE

    def key(name):
        r = results[name]
        return (-(r["direct_event_recall"] or 0), -(r["event_recall"] or 0), -(r["precision"] or 0),
                SIMPLICITY.index(name))
    return sorted(passing, key=key)[0]

