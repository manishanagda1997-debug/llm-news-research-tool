"""
The transfer test on a second company (Stage 7): Reliance Industries, one week of news, labelled blind,
with the frozen Tata Motors pipeline. Reliance is a test set only: nothing in the pipeline changes because
of its labels or its results.

The run has a fixed id (RUN_ID) and its own folder, data/eval/<RUN_ID>/. Each stage is recorded in
state.json, so a stopped run resumes and never mixes two fetches.

Stages, in order:
    protocol - before any fetch: the company, the window, both search strings (search B built from the alias
               file), the fingerprints of the alias file and the labelling guide, the sample rule, the
               definitions of every measure, the pass levels and how the result is read
               (transfer_protocol.json). Then: commit, and git tag FREEZE_TAG.
    fetch    - refused without the freeze tag, when the tagged protocol differs from the file, when the alias
               file or a search string differs from the protocol, or before FETCH_NOT_BEFORE (the free
               NewsAPI plan shows articles a day late; 6 hours of margin). Two requests, settings without keys.
    sample   - search A, then search B, each article once by tidy link; articles outside the window or with
               no date are excluded and logged; a count at each step. No top-up and no replacement.
    sheet    - full text (robots.txt respected, cached in data/fulltext/, never in Git), then a blind label
               sheet (labels.csv) and a reading page (in data/fulltext/): shuffled, with no search, filter or
               model output shown. The same code as the Tata Motors fresh test.
    events   - after the labels: an event sheet (events.csv) of the relevant articles.
    seal     - every label valid and every relevant article with an event; then the fingerprints of
               labels.csv and events.csv are saved (transfer_seal.json), before any model runs on Reliance.

    amend    - amendment 1 (transfer_amendment.json), written before any ranking, model or summary run and
               without reading the labels or events: the protocol's baseline "O1" equals O1+S1 by definition
               (S1 takes the first k of the ordering), so the ranking conditions could not fail. The baseline
               becomes the retrieval order: search A's articles as returned, then search B's, through the
               frozen Stage 1 rules, the surviving articles in their relative order. The conditions, k values
               and definitions stay the same; absolute values are reported without a level. Then: commit, and
               git tag AMEND_TAG.
    amend2   - amendment 2 (transfer_amendment2.json), written before the run and before its summary exists: the
               industry evidence of the five selected articles, judged first and from the articles alone, the
               split of the industry points, and two industry checks (no industry point not supported; the
               industry section matches the evidence), all reported without a pass level. The pass levels do not
               change. Then: commit, and git tag AMEND2_TAG.
    budget   - no model request: the prompts of the run are built and measured (a relevance batch too large for
               one request splits in two, as the frozen relevance check does); every request must stay within
               the minute limit with the largest observed extra Groq counts, and the conservative total of the
               run within the daily limit; otherwise the decision is "stop". The whole daily limit counts only
               after 24 hours with no Groq use, which the user confirms; the code cannot see the Groq usage.
               Made once, before the run and from committed code: the report (transfer_budget.json) holds the
               source commit, the fingerprints of the code that builds the prompts, the protocol, both
               amendments, the seal, the freeze and the token evidence, a fingerprint of every planned prompt,
               the estimates, the assumptions and the decision. Then: review it and commit it. The run refuses
               when the report is missing, says stop, or no longer matches the code, the inputs or the prompts.
    run      - once. Refused when the record, the attempt marker or the saved passages exist; without the three
               tags (protocol, amendment 1, amendment 2) holding exactly the files in the folder; without the
               seal; without a Stage 4 freeze that validates the protocol's summary k; when Git shows a change
               or an untracked file in a protected path (PROTECTED); or without --no-groq-24h, the user's
               confirmation of no Groq use in the 24 hours before the run. Never reads labels.csv or events.csv.
               Then:
               Stage 1 on the sample and the two selections at every k (no model); the summary's passages (no
               model); the attempt marker, created before any model request and never removed, with an attempt id,
               the source commit and the confirmation; the passages, saved to data/fulltext/ with their
               fingerprints in the record; one summary with the frozen design on the default's top k, with no
               answer cache (first: it carries the pass levels); a pause of RELEVANCE_PAUSE seconds; one first
               live pass of the relevance model with no answer cache; the record (transfer_run.json), with the
               same attempt id and source commit. A failure after the marker ends the test run for good: no retry. The command
               prints counts and the summary status only, no ranking result.
    judge    - the Part 4 sheets of the one summary, on the saved passages after checking their fingerprints
               (nothing is rebuilt): the industry evidence row, the items and the events, the key, and the reading
               page (industry evidence first, then the items, then the summary for the events)
    lock     - the run record, the attempt marker and the saved passages from the same attempt and commit, and the
               budget report unchanged since the run started (its fingerprint is in the run record); every mark
               present and valid; then the fingerprints of the three sheets, the key, the judging
               code (JUDGING_CODE), the guides (Parts 1 to 3, 4 and 6), the protocol, both amendments, the seal,
               the labels and events, the Stage 4 freeze, the run record, the attempt marker, the passages and the
               budget report
    score    - made once, after the lock (or at once when the run gives no summary), and refused when anything
               fixed by the seal or the lock changed: the ranking conditions against the amended baseline, the
               relevance check against the labels (no level), the unsure labels on their own, the summary levels
               from the protocol (compared unrounded) and the amendment 2 industry report (no level)
               (transfer_score.json)

Run, in order:
    python -m src.transfer_eval protocol    (then commit, then: git tag freeze-reliance-transfer)
    python -m src.transfer_eval fetch       (2 NewsAPI requests, no model)
    python -m src.transfer_eval sample
    python -m src.transfer_eval sheet
    python -m src.transfer_eval events      (after labels.csv is filled)
    python -m src.transfer_eval seal        (after events.csv is filled; then commit)
    python -m src.transfer_eval amend       (then commit, then: git tag freeze-reliance-transfer-amend1)
    python -m src.transfer_eval amend2      (then commit, then: git tag freeze-reliance-transfer-amend2)
    python -m src.transfer_eval budget      (no model request; then review and commit transfer_budget.json)
    python -m src.transfer_eval run --no-groq-24h   (only after 24 hours with no Groq use; model calls;
                                                     then commit the record and marker)
    python -m src.transfer_eval judge       (then fill the three sheets: industry evidence first)
    python -m src.transfer_eval lock
    python -m src.transfer_eval score
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import DATA_DIR, EVAL_DIR, RAW_DIR, ROOT_DIR
from src.evaluate import VALID_LABELS, is_relevant
from src.fresh_eval import _published, file_fingerprint, git, load_state as _load_state, progress_order, safe_request, save_state
from src.ranking_eval import merge_run

COMPANY = "Reliance Industries"
RUN_ID = "transfer_reliance_2026-10-26_11-01"
RUN_DIR = EVAL_DIR / RUN_ID
WINDOW_FROM = "2026-10-26T00:00:00Z"
WINDOW_TO = "2026-11-01T23:59:59Z"
FETCH_NOT_BEFORE = datetime(2026, 11, 3, 6, 0, tzinfo=timezone.utc)   # the one-day delay, plus 6 hours
FREEZE_TAG = "freeze-reliance-transfer"
PROTOCOL_FILE = "transfer_protocol.json"
SEAL_FILE = "transfer_seal.json"
LABEL_FILE = "labels.csv"
EVENTS_FILE = "events.csv"
ALIAS_FILE = DATA_DIR / "company_aliases.json"
AMENDMENT_FILE = "transfer_amendment.json"
AMEND_TAG = "freeze-reliance-transfer-amend1"
RUN_FILE = "transfer_run.json"
SCORE_FILE = "transfer_score.json"
JUDGING_ITEMS = "summary_judging.csv"
JUDGING_EVENTS = "summary_events.csv"
JUDGING_KEY = "summary_judging_key.json"
JUDGING_LOCK = "summary_judging_lock.json"
JUDGING_PAGE = f"summary_judging_{RUN_ID}.html"                       # in data/fulltext/, outside Git
BASELINE = "retrieval_order"
DEFAULT = "O1+S1"
AMENDMENT2_FILE = "transfer_amendment2.json"
AMEND2_TAG = "freeze-reliance-transfer-amend2"
ATTEMPT_FILE = "transfer_run_attempt.json"                             # in the run folder, in Git; never removed
BUDGET_FILE = "transfer_budget.json"                                   # in the run folder, in Git; made before the run
PASSAGES_FILE = f"transfer_passages_{RUN_ID}.json"                     # in data/fulltext/, outside Git
JUDGING_INDUSTRY = "summary_industry.csv"
JUDGING_SEED = 47
SCORING_VERSION = "1"
RELEVANCE_PAUSE = 61               # seconds between the summary and the relevance pass: an empty minute window
CACHE_NOTE = "No answer cache is read or written; the run result is saved in this record."
CONFIRMATION = ("The user confirms no Groq use in the 24 hours before the run. The program cannot see the Groq "
                "usage; this is the user's statement, not a check made by the code.")
PROTECTED = ("src/", "requirements.txt", "data/company_aliases.json", f"data/eval/{RUN_ID}/",
             "data/eval/LABELLING_GUIDE.md", "data/eval/SUMMARY_JUDGING_GUIDE.md",
             "data/eval/SUMMARY_JUDGING_GUIDE_PART4.md", "data/eval/SUMMARY_JUDGING_GUIDE_PART6.md",
             "data/eval/summary_freeze.json", "data/eval/summary_freeze_decision.md")
BUDGET_CODE = ("src/transfer_eval.py", "src/relevance_check.py", "src/ranking_eval.py", "src/ranking.py",
               "src/summarizer.py", "src/summary_v2.py", "src/summary_passage.py", "src/boilerplate.py",
               "src/preprocess.py", "src/fulltext.py", "src/query.py", "src/llm.py", "data/company_aliases.json")
JUDGING_GUIDES = ("SUMMARY_JUDGING_GUIDE.md", "SUMMARY_JUDGING_GUIDE_PART4.md", "SUMMARY_JUDGING_GUIDE_PART6.md")
JUDGING_CODE = ("src/transfer_eval.py", "src/summary_judging.py", "src/summary_judging_p4.py",
                "src/summary_judging_fresh.py", "src/summary_fresh.py", "src/summary_v2.py", "src/summary_run.py",
                "src/evaluate.py", "src/preprocess.py")

# the budget check (no model request)
MINUTE_LIMIT = 8000                # Groq free plan: tokens a minute
DAY_LIMIT = 200000                 # Groq free plan: tokens a day
DAY_SHARE = 1.0                    # the whole daily limit, only after 24 hours with no Groq use (confirmed by the user)
CONFIRM_FLAG = "--no-groq-24h"      # the user confirms no Groq use in the 24 hours before the run
GROQ_EXTRA = 551                   # the largest observed difference between Groq's requested tokens and its input count
BOUND_MIN_CHARS = 10000            # the token rate comes from prompts of at least this length
RELEVANCE_TRIES = 2                # one try and one retry per batch (src/relevance_check.py, _ask_batch)
RELEVANCE_OUTPUT_MARGIN = 1.25     # the relevance requests have no output limit: the largest recorded answer, plus a quarter
RELEVANCE_EVIDENCE = ("model_compare/batch_gpt-oss-120b_size3.json", "model_compare/batch_gpt-oss-120b_size4.json",
                      "model_compare/batch_gpt-oss-120b_size5.json", "model_compare/run_gpt-oss-120b.json",
                      "model_compare/setaside_gpt-oss-120b_old-pacer_unscored.json", "ranking_dev_record.json")
SUMMARY_EVIDENCE = ("summary_v2_dev_k5_c7_p3.json", "summary_v2_dev_k5_c7_p3_run2.json", "summary_v2_dev_k5_c7_p4.json",
                    "summary_v2_dev_k5_c7_p5.json", "summary_v2_dev_k10_c7_p4.json",
                    "summary_v2_dev_k10_c7_p4_run2.json", "summary_v2_fresh_k5_c7_p4.json",
                    "summary_v2_smoke_c7_p3.json", "summary_v2_diagnosis_c3_p3.json",
                    "summary_v2_diagnosis_c4_p3.json", "summary_v2_diagnosis_c5_p3.json",
                    "summary_v2_diagnosis_p3.json", "summary_v2_diagnosis_p3_run2.json",
                    "summary_v2_diagnosis_p3_run3.json")
GUIDE_FILE = EVAL_DIR / "LABELLING_GUIDE.md"
SEARCH_SETTINGS = {"page_size": 30, "sort_by": "relevancy", "exact_phrase": True, "title_only": False}

SAMPLE_RULE = ("Search A (up to 30 articles), then search B (up to 30), merged in that order; each article kept "
               "once by tidy link; articles published outside the window or without a date are excluded and "
               "logged; the sample is what remains, even when it is small or unbalanced: no top-up and no "
               "replacement. An article whose page cannot be fetched stays and is labelled on its short text.")

DEFINITIONS = {
    "relevant_event": "an event with at least one article labelled direct or indirect",
    "direct_event": "an event with at least one article labelled direct",
    "event_recall": "distinct relevant events among the relevant articles in the top k, divided by the relevant "
                    "events among the articles Stage 1 keeps; undefined when there is no relevant event",
    "precision": "relevant articles among the selected articles, divided by the number selected (k, or fewer "
                 "when fewer articles are available; the number available is reported)",
    "no_direct_event_lost": "every direct event that O1's top 10 covers is also covered by O1+S1's top 10, on "
                            "the same articles and the same event names",
    "model_coverage": "the share of the articles Stage 1 keeps that have a model answer after the first live "
                      "pass, started with no saved answers; it applies only to configurations that use the model",
    "model_time": "the seconds of that first pass; it applies only to configurations that use the model",
    "summary_event_coverage": "event rows marked covered = yes, divided by all event rows: one row per relevant "
                              "event among the summary's own top 5 articles (SUMMARY_JUDGING_GUIDE.md, Part 3)",
    "contradicted_rate": "items with contradicted = yes, divided by all judged items (company points, industry "
                         "points and visible article briefs)",
    "produced": "the one summary comes out (1 of 1)",
}

PASS_LEVELS = {
    "ranking": {"configuration": "O1+S1", "baseline": "O1", "primary_k": 10, "reported_k": [5, 15],
                "conditions": ["no_direct_event_lost", "event_recall_not_lower", "precision_not_lower"],
                "min_model_coverage": 0.9, "max_model_seconds": 180,
                "note": "the Tata 'gain' condition is not used: it decided whether O1+S1 replaces the baseline, "
                        "which is settled; this test checks that the default does no worse on a new company"},
    "summary": {"design": "the design frozen in Stage 4", "k": 5, "produced": True, "min_event_coverage": 0.8,
                "max_contradicted": 0.05, "no_central_contradiction": True},
    "reported_without_level": ["the relevance check against the labels", "unsure labels, on their own"],
}

READING = {
    "candidate_pool": "event recall is recall within the articles Stage 1 keeps, not within all news published "
                      "about the company",
    "one_summary": "the test judges one summary for factual support, event coverage and completion; it does not "
                   "establish production reliability across repeated searches",
    "no_tuning": "Reliance labels and results do not change the pipeline, the Stage 4 design or any setting; "
                 "an experimental configuration never becomes the default because of this test",
    "fixes": "a failure is reported with its diagnosis; a fix is built and tested on Tata Motors development "
             "data only, then checked on a new, unseen Reliance week",
    "balance": "a sample with few indirect or not articles is reported as it is; the searches are not changed",
}


class TransferError(Exception):
    pass


def load_state(run_dir) -> dict:
    """The run's state, always under this run's id (a new state is not given the Tata Motors run id)."""
    state = _load_state(run_dir)
    state["run_id"] = RUN_ID
    return state


# ---------- the searches and the protocol ----------

def search_strings(aliases: dict) -> dict[str, str]:
    from src.query import build_search_plan
    a = build_search_plan(COMPANY, exact_phrase=True, aliases=None, part="all")["search_query"]
    b = build_search_plan(COMPANY, exact_phrase=True, aliases=aliases, part="subsidiaries")["search_query"]
    if not b:
        raise TransferError("Search B is empty: the alias file has no subsidiaries for the company.")
    return {"A": a, "B": b}


def _text_fingerprint(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def protocol_record(aliases: dict, alias_text: str, guide_text: str, source_commit: str) -> dict:
    from src.relevance_check import company_scope
    return {"run_id": RUN_ID, "company": COMPANY, "window": [WINDOW_FROM, WINDOW_TO],
            "fetch_not_before": FETCH_NOT_BEFORE.isoformat(), "searches": search_strings(aliases),
            "search_settings": SEARCH_SETTINGS, "alias_fingerprint": _text_fingerprint(alias_text),
            "guide_fingerprint": _text_fingerprint(guide_text), "company_scope": company_scope(COMPANY, aliases),
            "sample_rule": SAMPLE_RULE, "definitions": DEFINITIONS, "pass_levels": PASS_LEVELS,
            "reading": READING, "source_commit": source_commit, "freeze_tag": FREEZE_TAG,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def fetch_problems(protocol: dict, aliases: dict, alias_text: str, now: datetime,
                   tag_protocol: str | None, protocol_text: str) -> None:
    """Every guard before the fetch; raises with the first problem found."""
    if tag_protocol is None:
        raise TransferError(f"The tag {FREEZE_TAG} does not exist or does not hold {PROTOCOL_FILE}. "
                            f"Commit the protocol, then: git tag {FREEZE_TAG}")
    if tag_protocol.replace("\r\n", "\n") != protocol_text.replace("\r\n", "\n"):
        raise TransferError(f"{PROTOCOL_FILE} differs from the one in the tag {FREEZE_TAG}.")
    if _text_fingerprint(alias_text) != protocol["alias_fingerprint"]:
        raise TransferError("The alias file differs from the one fixed in the protocol.")
    if search_strings(aliases) != protocol["searches"]:
        raise TransferError("A search string differs from the protocol.")
    if now < FETCH_NOT_BEFORE:
        raise TransferError(f"Too early: fetch on or after {FETCH_NOT_BEFORE:%d %B %Y %H:%M} (UTC).")


# ---------- the sample ----------

def _window() -> tuple[datetime, datetime]:
    return (datetime.fromisoformat(WINDOW_FROM.replace("Z", "+00:00")),
            datetime.fromisoformat(WINDOW_TO.replace("Z", "+00:00")))


def sample(responses: dict[str, dict]) -> dict:
    start, end = _window()
    merged = merge_run([responses["A"], responses["B"]])
    kept, excluded = [], []
    for article in merged:
        published = _published(article)
        if published is None:
            excluded.append({"url": article.get("url"), "title": article.get("title"), "reason": "no_date"})
        elif not start <= published <= end:
            excluded.append({"url": article.get("url"), "title": article.get("title"), "reason": "outside_window"})
        else:
            kept.append(article)
    reasons = [e["reason"] for e in excluded]
    counts = {"search_A": len(responses["A"].get("articles", [])), "search_B": len(responses["B"].get("articles", [])),
              "after_merge": len(merged), "outside_window": reasons.count("outside_window"),
              "no_date": reasons.count("no_date"), "sample": len(kept)}
    return {"kept": kept, "excluded": excluded, "counts": counts}


# ---------- the seal ----------

def seal_problems(labels: dict[str, str], events: dict[str, str]) -> list[str]:
    """labels: {article id: label}; events: {article id: event name}."""
    problems = [f"{key}: the label must be one of {', '.join(VALID_LABELS)}"
                for key, label in labels.items() if label not in VALID_LABELS]
    problems += [f"{key}: a relevant article needs an event name" for key, label in labels.items()
                 if label in VALID_LABELS and is_relevant(label) and not events.get(key)]
    return problems


# ---------- amendment 1: the retrieval-order baseline ----------

AMENDMENT = {
    "amendment": 1,
    "reason": ("A logical flaw in the protocol's definition, found before any Reliance ranking, model or summary "
               "run: the baseline \"O1\" equals O1+S1 by definition, because S1 takes the first k of the "
               "ordering, so the ranking conditions pass on any data. The amendment responds to that flaw, not "
               "to any Reliance result."),
    "baseline": {"name": BASELINE,
                 "definition": "search A's articles in the order returned, then search B's in the order "
                               "returned; each article once; the frozen Stage 1 rules (cleaning and duplicates) "
                               "applied with no other sorting; the surviving articles keep their relative "
                               "order; the first k",
                 "note": "the retrieval order of two searches, not NewsAPI's relevance ranking of one search"},
    "independence": ("not blind in the strictest sense: the labels and events exist when the amendment is made; "
                     "it is written without reading them, and before any ranking result exists"),
    "absolute_values": "precision, event recall and direct-event recall of both lists at every k, without a level",
}


def amendment_record(protocol_text: str, protocol: dict, source_commit: str, now: datetime) -> dict:
    ranking = protocol["pass_levels"]["ranking"]
    return {**AMENDMENT, "configuration": ranking["configuration"], "conditions": list(ranking["conditions"]),
            "primary_k": ranking["primary_k"], "reported_k": list(ranking["reported_k"]),
            "protocol_fingerprint": _text_fingerprint(protocol_text), "source_commit": source_commit,
            "tag": AMEND_TAG, "created": now.isoformat(timespec="seconds")}


def write_amendment(run_dir, source_commit: str, now: datetime) -> dict:
    """Reads the protocol only; labels.csv and events.csv are never opened."""
    path = Path(run_dir) / AMENDMENT_FILE
    if path.exists():
        raise TransferError(f"{AMENDMENT_FILE} already exists; it is not overwritten.")
    protocol_text = (Path(run_dir) / PROTOCOL_FILE).read_text(encoding="utf-8")
    record = amendment_record(protocol_text, json.loads(protocol_text), source_commit, now)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


# ---------- amendment 2: the industry-context checks ----------

AMENDMENT_2 = {
    "amendment": 2,
    "reason": ("The Stage 4 design adds an industry-context rule (prompt version 4): an industry point needs an "
               "article that itself reports on the wider industry, the market or other companies, and a fact "
               "about the company alone is never stated as a trend for the sector. The protocol judges industry "
               "points only as items among all the others, so it cannot show whether this rule holds for a "
               "second company. Reliance Industries runs several businesses in different sectors, where a fact "
               "about one business is easy to state as a sector trend."),
    "follows": ("The amendment follows the Stage 4 fresh-week result (summary_v2_fresh_judging_score.json), which "
                "reports these checks for Tata Motors."),
    "timing": ("Written before the Reliance run and before its summary exists: transfer_run.json and the attempt "
               "marker do not exist. The labels and events are sealed, and the amendment does not read them."),
    "judging": ("The rules of SUMMARY_JUDGING_GUIDE_PART6.md: first, from the five selected articles alone and "
                "before the summary is read, does any article report a fact about the sector, the market or "
                "another company in its own right; then every item of the summary; then the events."),
    "added": {
        "industry_evidence": "the answer to the industry-evidence question (yes or no), with a note naming the "
                             "article and the fact, or the reason for no",
        "industry_points": "the industry points judged supported, partly supported and not supported, and the "
                           "check that no industry point is not supported",
        "industry_section": "the check that the industry section matches the evidence answer: points only with "
                            "evidence, and an empty section only without evidence",
    },
    "level": "all three are reported without a pass level and never decide whether the test passes",
    "unchanged": ("The ranking conditions, the summary pass levels (produced, event coverage, contradicted items, "
                  "central contradiction) and every definition of the protocol and amendment 1 stay as they are."),
}


def amendment2_record(protocol_text: str, amendment_text: str, source_commit: str, now: datetime) -> dict:
    return {**AMENDMENT_2, "protocol_fingerprint": _text_fingerprint(protocol_text),
            "amendment1_fingerprint": _text_fingerprint(amendment_text), "source_commit": source_commit,
            "tag": AMEND2_TAG, "created": now.isoformat(timespec="seconds")}


def write_amendment2(run_dir, source_commit: str, now: datetime) -> dict:
    """Before the run only. Reads the protocol and amendment 1; labels.csv and events.csv are never opened."""
    run_dir = Path(run_dir)
    path = run_dir / AMENDMENT2_FILE
    if path.exists():
        raise TransferError(f"{AMENDMENT2_FILE} already exists; it is not overwritten.")
    for name in (RUN_FILE, ATTEMPT_FILE):
        if (run_dir / name).exists():
            raise TransferError(f"{name} exists: amendment 2 must come before the run.")
    record = amendment2_record((run_dir / PROTOCOL_FILE).read_text(encoding="utf-8"),
                               (run_dir / AMENDMENT_FILE).read_text(encoding="utf-8"), source_commit, now)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


# ---------- the run: selections without labels ----------

def selections(candidates: list[dict], ks) -> dict[str, dict[str, list[str]]]:
    """The keys each list selects at every k. Candidates come in retrieval order from Stage 1."""
    from src import ranking as rk
    plain = [dict(c, check=None) for c in candidates]
    ordered = rk.order(plain, "O1")
    return {BASELINE: {str(k): [c["key"] for c in plain[:k]] for k in ks},
            DEFAULT: {str(k): [c["key"] for c in rk.select(ordered, k, "S1")["selected"]] for k in ks}}


def protected_changes(status_text: str) -> list[str]:
    """The changed or untracked paths (git status --porcelain) that the run protects."""
    found = set()
    for line in status_text.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].split(" -> ")[-1].strip().strip('"').replace("\\", "/")
        if any(path == p or (p.endswith("/") and path.startswith(p)) for p in PROTECTED):
            found.add(path)
    return sorted(found)


def run_paths(run_dir=None, fulltext_dir=None) -> dict:
    """The run's three files. The record and the attempt marker are kept in the run folder (in Git); the
    passages are publisher text, so they stay in data/fulltext/, which Git ignores."""
    from src.fulltext import FULLTEXT_DIR
    run_dir = Path(run_dir or RUN_DIR)
    return {"record": run_dir / RUN_FILE, "attempt": run_dir / ATTEMPT_FILE,
            "passages": Path(fulltext_dir or FULLTEXT_DIR) / PASSAGES_FILE}


def run_problems(protocol_text: str | None, tag_protocol: str | None, amendment_text: str | None,
                 tag_amendment: str | None, amendment2_text: str | None, tag_amendment2: str | None,
                 seal_exists: bool, freeze: list[str], existing: list[str], changed: list[str],
                 confirmed: bool, budget: list[str]) -> list[str]:
    def same(a, b):
        return a is not None and b is not None and a.replace("\r\n", "\n") == b.replace("\r\n", "\n")
    found = [f"{name} already exists: the run is attempted once only" for name in existing]
    if not same(protocol_text, tag_protocol):
        found.append(f"{PROTOCOL_FILE} is missing or differs from the tag {FREEZE_TAG}")
    if not same(amendment_text, tag_amendment):
        found.append(f"{AMENDMENT_FILE} is missing or differs from the tag {AMEND_TAG}")
    if not same(amendment2_text, tag_amendment2):
        found.append(f"{AMENDMENT2_FILE} is missing or differs from the tag {AMEND2_TAG}")
    if not seal_exists:
        found.append(f"{SEAL_FILE} is missing: the labels and events are not sealed")
    if changed:
        found.append("not committed: " + ", ".join(changed))
    if not confirmed:
        found.append(f"no Groq use in the 24 hours before the run is not confirmed: add {CONFIRM_FLAG} only when it "
                     "is true; otherwise wait")
    found += [f"budget: {problem}" for problem in budget]
    return found + [f"Stage 4 freeze: {problem}" for problem in freeze]


def relevance_call(candidates: list[dict], aliases: dict, llm) -> tuple[dict, dict]:
    """The first live relevance pass: no saved answer is read and none is written (cache=None)."""
    from src.ranking_eval import _ask_model
    return _ask_model(candidates, COMPANY, aliases, llm=llm, cache=None)


def summary_call(design: str, articles: list[dict], llm, k: int, correction: str) -> dict:
    """The frozen summary on the saved passages: no saved answer is read and none is written (NoCache)."""
    from src import summary_v2 as sv
    return sv._one(design, articles, llm, sv.NoCache(), k, correction)


def execute_run(run_dir, candidates: list[dict], ks, summary_k: int, design: str, prepare, relevance, summarize,
                fingerprints: dict, source_commit: str = "", fulltext_dir=None, pause=None,
                confirmation: str = CONFIRMATION) -> dict:
    """prepare(chosen) -> the summary's articles as passages (no model); summarize(articles, k) -> result;
    relevance(candidates) -> (answers, runtime). Labels and events are not inputs.

    Order: none of the three files may exist; the selections and the passages (no model); the attempt marker,
    created exclusively before any model request and never removed, with its own attempt id, the source commit
    and the user's 24-hour confirmation; the passages, saved exclusively with their fingerprints; the summary
    first (it carries the pass levels, so a later daily-limit failure can only reach the reported-only stage);
    a pause of RELEVANCE_PAUSE seconds, so the relevance pass starts in an empty minute window; the relevance
    pass; the record, created exclusively, with the same attempt id and source commit. A failure after the
    marker stops the run for good: the marker refuses any second attempt."""
    import uuid
    from src.summary_fresh import passage_entry
    paths = run_paths(run_dir, fulltext_dir)
    for path in paths.values():
        if path.exists():
            raise TransferError(f"{path.name} already exists: the run is attempted once only.")
    chosen = selections(candidates, ks)
    by_key = {c["key"]: c for c in candidates}
    articles = prepare([by_key[key] for key in chosen[DEFAULT][str(summary_k)]])
    passages = [passage_entry(a) for a in articles]
    attempt_id = uuid.uuid4().hex
    with open(paths["attempt"], "x", encoding="utf-8") as handle:     # first, before any model request; never removed
        handle.write(json.dumps({"attempt_id": attempt_id, "source_commit": source_commit,
                                 "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "candidates": len(candidates), "confirmation": confirmation}, indent=1))
    with open(paths["passages"], "x", encoding="utf-8") as handle:    # the evidence for judging, outside Git
        handle.write(json.dumps({"attempt_id": attempt_id, "source_commit": source_commit, "passages": passages},
                                ensure_ascii=False, indent=1))
    result = summarize(articles, summary_k)
    (pause or (lambda seconds: None))(RELEVANCE_PAUSE)
    answers, runtime = relevance(candidates)
    answered = sum(1 for a in answers.values() if a.get("source") == "model")
    record = {"run_id": RUN_ID, "company": COMPANY, "attempt_id": attempt_id, "source_commit": source_commit,
              "confirmation": confirmation, "order": ["summary", f"pause {RELEVANCE_PAUSE} s", "relevance"],
              "candidates": [{"key": c["key"], "title": c["title"], "url": c.get("url", "")} for c in candidates],
              "selections": chosen,
              "relevance": {"answers": answers, "coverage": answered / len(candidates) if candidates else None,
                            "seconds": runtime.get("seconds_total"), "runtime": runtime},
              "summary_record": {"records": [{"run": RUN_ID, "design": design, "k": summary_k,
                                              "articles": [{"id": a["id"], "key": a["key"], "title": a["title"]}
                                                           for a in articles],
                                              "result": result}]},
              "passage_fingerprints": [{"id": p["id"], "key": p["key"], "sha256": p["sha256"]} for p in passages],
              "passages_file": f"data/fulltext/{PASSAGES_FILE}", "attempt_file": ATTEMPT_FILE,
              "cache": CACHE_NOTE, "fingerprints": fingerprints, "pacer_version": _pacer_version(),
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(paths["record"], "x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=1, default=str))
    return record


# ---------- the budget check: no model request ----------

def token_bound(calls: list[tuple[int, int]]) -> dict:
    """calls: (Groq's input count, prompt characters) of answered requests. The rate is the largest tokens per
    character among prompts of at least BOUND_MIN_CHARS characters; the base is the largest amount by which any
    request exceeds that rate. Every recorded request lies on or under base + rate * characters."""
    large = [(tokens, chars) for tokens, chars in calls if chars >= BOUND_MIN_CHARS]
    if not large:
        raise TransferError("No recorded request is large enough to set a token rate.")
    rate = max(tokens / chars for tokens, chars in large)
    base = max(0.0, max(tokens - rate * chars for tokens, chars in calls))
    return {"rate": rate, "base": base, "requests": len(calls)}


def _estimate(bound: dict, chars: int) -> int:
    import math
    return math.ceil(bound["base"] + bound["rate"] * chars)


def planned_requests(batch: list[dict], measure, fits) -> tuple[list[int], list[str]]:
    """The relevance requests one batch becomes under the frozen rule (src/relevance_check.py, _ask_batch): a
    batch too large for the minute limit splits in two, again and again; one article too large on its own is
    not sent and falls back. measure(items) -> prompt characters; fits(characters) -> bool. Returns the
    characters of every request sent and the ids that fall back."""
    chars = measure(batch)
    if fits(chars):
        return [chars], []
    if len(batch) == 1:
        return [], [batch[0]["id"]]
    half = len(batch) // 2
    first, second = planned_requests(batch[:half], measure, fits), planned_requests(batch[half:], measure, fits)
    return first[0] + second[0], first[1] + second[1]


def planned_prompts(batch: list[dict], build, fits) -> tuple[list[str], list[str]]:
    """The same split as planned_requests, keeping each prompt's text: build(items) -> prompt text."""
    text = build(batch)
    if fits(len(text)):
        return [text], []
    if len(batch) == 1:
        return [], [batch[0]["id"]]
    half = len(batch) // 2
    first, second = planned_prompts(batch[:half], build, fits), planned_prompts(batch[half:], build, fits)
    return first[0] + second[0], first[1] + second[1]


def fits_minute(bound: dict):
    return lambda chars: _estimate(bound, chars) + GROQ_EXTRA <= MINUTE_LIMIT


def budget_report(relevance_chars: list[int], summary_chars: int, bounds: dict, relevance_output: int,
                  repair_calls: int) -> dict:
    """The conservative token budget of the run, before it is made. relevance_chars: one prompt per relevance
    request after the frozen batch split (planned_requests); summary_chars: the summary prompt. Nothing here
    sends a request."""
    from src import summarizer as sm
    from src import summary_v2 as sv
    relevance = [_estimate(bounds["relevance"], c) for c in relevance_chars]
    summary = _estimate(bounds["summary"], summary_chars)
    first = [("relevance request", n, t) for n, t in enumerate(relevance, start=1)] + [("summary", 1, summary)]
    over = [f"{name} {n}: {t:,} + {GROQ_EXTRA} > {MINUTE_LIMIT:,}" for name, n, t in first
            if t + GROQ_EXTRA > MINUTE_LIMIT]
    relevance_total = RELEVANCE_TRIES * sum(t + relevance_output for t in relevance)
    summary_total = (sm.MAX_ATTEMPTS + repair_calls) * (summary + sv.OUTPUT_LIMIT)
    total = relevance_total + summary_total
    day_limit = int(DAY_LIMIT * DAY_SHARE)
    return {"requests": {"relevance_requests": len(relevance), "relevance_input_estimates": relevance,
                         "summary_input_estimate": summary},
            "per_request": {"limit": MINUTE_LIMIT - GROQ_EXTRA, "over": over},
            "day": {"relevance": relevance_total, "summary": summary_total, "total": total, "limit": day_limit},
            "assumptions": {
                "groq_extra": f"{GROQ_EXTRA}: the largest difference between Groq's requested tokens and its input "
                              "count in its refusal messages; an observed value, not a proven maximum",
                "token_bounds": {name: {"rate": round(b["rate"], 4), "base": round(b["base"], 1),
                                        "requests": b["requests"]} for name, b in bounds.items()},
                "relevance_output": f"{relevance_output} per request: the largest recorded relevance answer times "
                                    f"{RELEVANCE_OUTPUT_MARGIN} (the relevance requests have no output limit)",
                "relevance_tries": f"{RELEVANCE_TRIES} per request (one try and one retry)",
                "relevance_split": "a batch over the minute limit splits in two, as the frozen relevance check "
                                   "does; the split is planned with the conservative token bound",
                "summary": f"{sm.MAX_ATTEMPTS} full attempts and {repair_calls} repairs (the most repair requests "
                           f"recorded for one summary at k = 5), each at the summary's input estimate plus the "
                           f"full output limit of {sv.OUTPUT_LIMIT:,}",
                "no_groq_24h": "the whole daily limit is used only after 24 hours with no Groq use; the user "
                               "confirms this when starting the run; the code cannot see the Groq usage",
                "estimate": "a conservative estimate from recorded requests, not a guaranteed maximum"},
            "decision": ("stop" if over or total > day_limit else
                         "go only if no Groq use in the 24 hours before the run is confirmed; otherwise stop")}


def code_fingerprint(names) -> str:
    """One fingerprint of the listed files, line endings made plain."""
    import hashlib
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8") + b"\0" + (ROOT_DIR / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def budget_plan(candidates: list[dict], aliases: dict, bounds: dict) -> dict:
    """Every prompt of the run, built exactly as the run builds it, with no model request: the relevance requests
    after the frozen batch split, and the summary's first request. plan_fingerprint covers every prompt text, so
    any change in the articles, the passages or the prompt code changes it."""
    import hashlib
    from src import summarizer as sm
    from src.ranking_eval import query_items
    from src.relevance_check import BATCH_SIZE, build_prompt, company_scope
    items, scope = query_items(candidates, COMPANY, aliases), company_scope(COMPANY, aliases)
    batches = [items[start:start + BATCH_SIZE] for start in range(0, len(items), BATCH_SIZE)]
    prompts, fallbacks = [], []
    for batch in batches:
        sent, fallen = planned_prompts(batch, lambda part: build_prompt(part, COMPANY, "", scope),
                                       fits_minute(bounds["relevance"]))
        prompts += sent
        fallbacks += fallen
    summary_prompt = sm.build_prompt(_summary_articles(candidates, aliases, PASS_LEVELS["summary"]["k"]))
    every = prompts + [summary_prompt]
    return {"candidates": len(candidates), "relevance_batches_before_split": len(batches),
            "relevance_prompt_chars": [len(t) for t in prompts], "relevance_fallbacks_too_large": len(fallbacks),
            "summary_prompt_chars": len(summary_prompt),
            "plan_fingerprint": hashlib.sha256(json.dumps(every, ensure_ascii=False).encode("utf-8")).hexdigest()}


def budget_inputs(run_dir=None) -> dict:
    """The fingerprints the budget report and the run must share."""
    from src.summary_freeze import FREEZE_FILE
    run_dir = Path(run_dir or RUN_DIR)
    return {"code": code_fingerprint(BUDGET_CODE), "protocol": file_fingerprint(run_dir / PROTOCOL_FILE),
            "amendment1": file_fingerprint(run_dir / AMENDMENT_FILE),
            "amendment2": file_fingerprint(run_dir / AMENDMENT2_FILE), "seal": file_fingerprint(run_dir / SEAL_FILE),
            "freeze": file_fingerprint(EVAL_DIR / FREEZE_FILE),
            "evidence": code_fingerprint([f"data/eval/{name}" for name in RELEVANCE_EVIDENCE + SUMMARY_EVIDENCE])}


def budget_problems(saved: dict | None, inputs: dict, plan_fingerprint: str) -> list[str]:
    """The saved budget report must exist, allow the run, and match the current code, inputs and prompts."""
    if saved is None:
        return [f"{BUDGET_FILE} does not exist: run the budget, review it and commit it first"]
    found = []
    if saved.get("decision", "").startswith("stop"):
        found.append("the budget decision is stop")
    changed = [name for name in sorted(set(inputs) | set(saved.get("inputs", {})))
               if saved.get("inputs", {}).get(name) != inputs.get(name)]
    if changed:
        found.append("changed since the budget report: " + ", ".join(changed) + "; make and review a new report")
    if saved.get("plan", {}).get("plan_fingerprint") != plan_fingerprint:
        found.append("the prompts of the run differ from the ones in the budget report")
    return found


def _evidence_calls(names) -> list[tuple[int, int]]:
    """Every answered request with Groq's input count and its prompt length, in the committed records."""
    calls = []

    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get("input"), int) and value.get("prompt_chars"):
                calls.append((value["input"], value["prompt_chars"]))
            for inner in value.values():
                walk(inner)
        elif isinstance(value, list):
            for inner in value:
                walk(inner)
    for name in names:
        walk(json.loads((EVAL_DIR / name).read_text(encoding="utf-8")))
    return calls


def _largest_output(names) -> int:
    found = []

    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get("output"), int) and value.get("prompt_chars"):
                found.append(value["output"])
            for inner in value.values():
                walk(inner)
        elif isinstance(value, list):
            for inner in value:
                walk(inner)
    for name in names:
        walk(json.loads((EVAL_DIR / name).read_text(encoding="utf-8")))
    return max(found)


def _most_repairs(names) -> int:
    most = 0
    for name in names:
        for entry in json.loads((EVAL_DIR / name).read_text(encoding="utf-8")).get("records", []):
            if entry.get("design") == "direct" and entry.get("k", 5) == 5:
                most = max(most, sum(r.get("requests", 0) for r in entry["result"].get("repairs") or []))
    return most


# ---------- the score ----------

def _pacer_version() -> str:
    from src.llm import PACER_VERSION
    return PACER_VERSION


def _share(part, whole):
    return part / whole if whole else None


def ranking_score(chosen: dict, candidate_keys: list[str], labels: dict[str, str], events: dict[str, str],
                  protocol: dict, amendment: dict) -> dict:
    """The protocol's conditions at its primary k, against the amended baseline; absolute values at every k."""
    from src import ranking as rk
    relevant = {key: is_relevant(labels.get(key, "")) for key in candidate_keys}
    k_values = sorted({amendment["primary_k"], *amendment["reported_k"]})
    measured = {name: {str(k): rk.measures(chosen[name][str(k)], candidate_keys, relevant, events, labels, k)
                       for k in k_values} for name in (amendment["baseline"]["name"], amendment["configuration"])}
    primary = str(amendment["primary_k"])
    base, other = measured[amendment["baseline"]["name"]][primary], measured[amendment["configuration"]][primary]
    change = rk.compare_events(base, other)
    tests = {"no_direct_event_lost": not change["direct_lost"],
             "event_recall_not_lower": rk._not_lower(other["event_recall"], base["event_recall"]),
             "precision_not_lower": rk._not_lower(other["precision"], base["precision"])}
    conditions = {name: tests[name] for name in amendment["conditions"]}
    absolute = {name: {k: {"selected": m["selected"], "precision": m["precision"], "event_recall": m["event_recall"],
                           "direct_event_recall": m["direct_event_recall"]} for k, m in by_k.items()}
                for name, by_k in measured.items()}
    return {"configuration": amendment["configuration"], "baseline": amendment["baseline"]["name"],
            "primary_k": amendment["primary_k"], "conditions": conditions, "passed": all(conditions.values()),
            "events": change, "absolute": absolute,
            "events_covered": {name: {k: m["events_covered"] for k, m in by_k.items()}
                               for name, by_k in measured.items()}}


def seal_changes(seal: dict, labels_sha: str, events_sha: str) -> list[str]:
    return ([LABEL_FILE] if seal["labels_sha256"] != labels_sha else []) + \
           ([EVENTS_FILE] if seal["events_sha256"] != events_sha else [])


def relevance_report(answers: dict, labels: dict[str, str]) -> dict:
    """The model's answers against the labels: reported, with no pass level."""
    answered = {k: a for k, a in answers.items() if a.get("source") == "model" and k in labels}
    same = sum(1 for k, a in answered.items() if a.get("label") == labels[k])
    said = [k for k, a in answered.items() if is_relevant(a.get("label", ""))]
    truly = [k for k in answered if is_relevant(labels[k])]
    hits = len(set(said) & set(truly))
    return {"articles": len(answers), "answered": len(answered), "same_label": same,
            "agreement": _share(same, len(answered)), "relevant_precision": _share(hits, len(said)),
            "relevant_recall": _share(hits, len(truly))}


def summary_result(judged: dict | None, protocol: dict) -> dict:
    """judged: one design's entry of the judging score, or None while the judgements are not locked. The shares
    are compared unrounded when the counts are present."""
    if judged is None:
        return {"status": "pending", "reason": "the blind judgements are not locked and scored yet"}
    level = protocol["pass_levels"]["summary"]
    items, events = judged["items"], judged["events"]
    coverage = events["covered"] / events["total"] if events.get("total") else events.get("coverage")
    contradicted = (items["contradicted"] / items["total"] if items.get("total") else items.get("contradicted_share"))
    checks = {"produced": judged["produced"]["count"] == judged["produced"]["total"] == 1,
              "event_coverage": coverage is not None and coverage >= level["min_event_coverage"],
              "contradicted": contradicted is not None and contradicted <= level["max_contradicted"],
              "no_central_contradiction": items["central_contradicted"] == 0}
    return {"status": "scored", "checks": checks, "passed": all(checks.values()), "values": judged}


def no_summary_result() -> dict:
    """The run gives no summary: nothing is judged and the summary fails on 'produced'."""
    return {"status": "scored", "checks": {"produced": False}, "passed": False, "values": None}


def industry_report(items: list[dict], industry: list[dict], key: dict, record: dict) -> dict:
    """Amendment 2: the industry evidence answer, the split of the industry points and the two checks, all
    reported without a pass level."""
    from src import summary_judging as sj
    from src import summary_judging_p4 as p4
    rows = sj._rows_by_number(items)
    points = [rows[n] for n, entry in key["items"].items() if entry["section"] == "industry_context"]
    counts = p4._counts(points)
    evidence = sj._value(industry[0]["industry_evidence"]) if industry else None
    produced = len(record["records"][0]["result"]["summary"].get("industry_context", []))
    return {"industry_evidence": evidence, "industry_points_produced": produced, "industry_points": counts,
            "checks": {"no_unsupported_industry_point": counts["not_supported"] == 0,
                       "industry_section_matches_evidence": (produced > 0) == (evidence == "yes")},
            "level": "reported without a pass level (amendment 2)"}


# ---------- the judging lock ----------

def _code_fingerprint() -> str:
    import hashlib
    digest = hashlib.sha256()
    for name in JUDGING_CODE:
        digest.update((ROOT_DIR / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def judging_inputs(run_dir=None, fulltext_dir=None) -> dict:
    """The fingerprints of everything the judgements and the score depend on, apart from the sheets and key."""
    from src.summary_freeze import FREEZE_FILE
    run_dir = Path(run_dir or RUN_DIR)
    paths = run_paths(run_dir, fulltext_dir)
    named = {"protocol": run_dir / PROTOCOL_FILE, "amendment1": run_dir / AMENDMENT_FILE,
             "amendment2": run_dir / AMENDMENT2_FILE, "seal": run_dir / SEAL_FILE, "labels": run_dir / LABEL_FILE,
             "events": run_dir / EVENTS_FILE, "freeze": EVAL_DIR / FREEZE_FILE, "run_record": paths["record"],
             "attempt": paths["attempt"], "passages": paths["passages"], "budget": run_dir / BUDGET_FILE}
    missing = [name for name, path in named.items() if not path.exists()]
    if missing:
        raise TransferError("Missing: " + ", ".join(missing))
    return {"guides": [file_fingerprint(EVAL_DIR / name) for name in JUDGING_GUIDES],
            **{name: file_fingerprint(path) for name, path in named.items()}}


def judging_lock(items, events, industry, key, inputs: dict) -> dict:
    from src import summary_judging as sj
    from src import summary_judging_p4 as p4
    found = p4.problems(items, events, industry, key)
    if found:
        raise TransferError("The sheets are not complete:\n" + "\n".join(found))
    return {"items_fingerprint": sj.sheet_fingerprint(items), "events_fingerprint": sj.sheet_fingerprint(events),
            "industry_fingerprint": sj.sheet_fingerprint(industry), "key_fingerprint": sj._hash(key),
            "code_fingerprint": _code_fingerprint(), "inputs": inputs, "scoring_version": SCORING_VERSION,
            "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def lock_changes(lock: dict, items, events, industry, key, inputs: dict) -> list[str]:
    """Everything that changed after the lock; empty when the score may go ahead."""
    from src import summary_judging as sj
    found = [f"the {name} sheet changed after the lock" for name, rows in
             (("items", items), ("events", events), ("industry", industry))
             if lock[f"{name}_fingerprint"] != sj.sheet_fingerprint(rows)]
    if lock["key_fingerprint"] != sj._hash(key):
        found.append("the key changed after the lock")
    if lock["code_fingerprint"] != _code_fingerprint():
        found.append("the judging code changed after the lock")
    found += [f"{name} changed after the lock" for name in sorted(set(lock["inputs"]) | set(inputs))
              if lock["inputs"].get(name) != inputs.get(name)]
    return found


def saved_passages(run: dict, saved: dict) -> dict:
    """{article key: passage} from the passages file the run saved: the same attempt and source commit as the run
    record, and every passage checked against its fingerprint in the record (IDs, order and text). Nothing is
    rebuilt."""
    from src import summary_judging_fresh as sjf
    if not isinstance(saved, dict) or saved.get("attempt_id") != run.get("attempt_id") or not saved.get("attempt_id"):
        raise TransferError("The saved passages come from another attempt than the run record.")
    if saved.get("source_commit") != run.get("source_commit"):
        raise TransferError("The saved passages name another source commit than the run record.")
    shaped = {"records": run["summary_record"]["records"], "passage_fingerprints": run.get("passage_fingerprints")}
    try:
        return sjf.passages_from_record(shaped, saved.get("passages") or [])
    except sjf.JudgingError as error:
        raise TransferError(str(error)) from None


def budget_file_problems(record: dict, actual: str | None) -> list[str]:
    """The budget report must still be the one the run started from: the record holds its fingerprint, taken when
    the run starts. actual: the report's fingerprint now, or None when the file is missing."""
    expected = (record.get("fingerprints") or {}).get("budget")
    if not expected:
        return ["the run record holds no budget report fingerprint"]
    if actual is None:
        return [f"{BUDGET_FILE} is missing"]
    if actual != expected:
        return [f"{BUDGET_FILE} differs from the report the run started from (modified or replaced)"]
    return []


def run_fingerprints(protocol_text: str, amendment_text: str, amendment2_text: str, run_dir=None) -> dict:
    """The fingerprints stored in the run record when the run starts."""
    from src.summary_freeze import FREEZE_FILE
    run_dir = Path(run_dir or RUN_DIR)
    return {"protocol": _text_fingerprint(protocol_text), "amendment": _text_fingerprint(amendment_text),
            "amendment2": _text_fingerprint(amendment2_text), "seal": file_fingerprint(run_dir / SEAL_FILE),
            "freeze": file_fingerprint(EVAL_DIR / FREEZE_FILE), "budget": file_fingerprint(run_dir / BUDGET_FILE)}


def attempt_problems(marker: dict, record: dict, passages: dict | None = None) -> list[str]:
    """The run record, the attempt marker and (when given) the saved passages file must come from the same attempt
    and the same source commit."""
    found = []
    if passages is not None and (not isinstance(passages, dict) or not passages.get("attempt_id")
                                 or passages.get("attempt_id") != record.get("attempt_id")
                                 or passages.get("source_commit") != record.get("source_commit")):
        found.append("the saved passages come from another attempt or source commit than the run record")
    if not marker.get("attempt_id") or marker.get("attempt_id") != record.get("attempt_id"):
        found.append("the run record and the attempt marker come from different attempts")
    if not marker.get("source_commit") or marker.get("source_commit") != record.get("source_commit"):
        found.append("the run record and the attempt marker name different source commits")
    if record.get("attempt_file") != ATTEMPT_FILE:
        found.append("the run record names another attempt marker")
    return found


# ---------- commands ----------

def _protocol_text() -> str:
    path = RUN_DIR / PROTOCOL_FILE
    if not path.exists():
        raise TransferError(f"{PROTOCOL_FILE} does not exist yet: python -m src.transfer_eval protocol")
    return path.read_text(encoding="utf-8")


def _tag_protocol() -> str | None:
    try:
        return git("show", f"{FREEZE_TAG}:{(RUN_DIR / PROTOCOL_FILE).relative_to(ROOT_DIR).as_posix()}")
    except Exception:
        return None


def _protocol_command() -> None:
    from src.query import load_aliases
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if (RUN_DIR / PROTOCOL_FILE).exists():
        raise TransferError(f"{PROTOCOL_FILE} already exists; it is not overwritten.")
    record = protocol_record(load_aliases(ALIAS_FILE), ALIAS_FILE.read_text(encoding="utf-8"),
                             GUIDE_FILE.read_text(encoding="utf-8"), git("rev-parse", "HEAD"))
    (RUN_DIR / PROTOCOL_FILE).write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Search A: {record['searches']['A']}\nSearch B: {record['searches']['B']}")
    print(f"Window: {WINDOW_FROM} to {WINDOW_TO}\nSaved: {RUN_DIR / PROTOCOL_FILE}")
    print(f"Next: commit it, then: git tag {FREEZE_TAG}")


def _fetch_command() -> None:
    from src.news_fetcher import fetch_news, save_raw
    from src.query import load_aliases
    protocol_text = _protocol_text()
    protocol, aliases = json.loads(protocol_text), load_aliases(ALIAS_FILE)
    fetch_problems(protocol, aliases, ALIAS_FILE.read_text(encoding="utf-8"), datetime.now(timezone.utc),
                   _tag_protocol(), protocol_text)
    state = load_state(RUN_DIR)
    if state["stages"].get("fetch", {}).get("done"):
        raise TransferError("The fetch is already done; a second fetch would mix two sets of results.")
    stage = {"done": False, "raw_files": {}, "requests": {}}
    state["stages"]["fetch"] = stage
    save_state(RUN_DIR, state)
    window = {"from_time": WINDOW_FROM, "to_time": WINDOW_TO, "save": False}
    for name in ("A", "B"):
        if name == "A":
            response = fetch_news(COMPANY, aliases=None, **SEARCH_SETTINGS, **window)
        else:
            response = fetch_news(COMPANY, aliases=aliases, part="subsidiaries", **SEARCH_SETTINGS, **window)
        response["request"] = safe_request(response.get("request", {}))
        stage["raw_files"][name] = save_raw(response).name
        stage["requests"][name] = response["request"]
        save_state(RUN_DIR, state)
        print(f"Search {name}: {len(response.get('articles', []))} articles -> data/raw/{stage['raw_files'][name]}")
    stage["done"] = True
    save_state(RUN_DIR, state)


def _responses(state: dict) -> dict[str, dict]:
    stage = state["stages"].get("fetch")
    if not stage or not stage.get("done"):
        raise TransferError("Run the fetch first.")
    return {name: json.loads((RAW_DIR / file).read_text(encoding="utf-8")) for name, file in stage["raw_files"].items()}


def _sample_command() -> None:
    state = load_state(RUN_DIR)
    result = sample(_responses(state))
    state["stages"]["sample"] = {"done": True, "kept": [a["url"] for a in result["kept"]],
                                 "excluded": result["excluded"], "counts": result["counts"]}
    save_state(RUN_DIR, state)
    print("Counts: " + ", ".join(f"{k} {v}" for k, v in result["counts"].items()))


def _kept(state: dict) -> list[dict]:
    stage = state["stages"].get("sample")
    if not stage:
        raise TransferError("Run the sample step first.")
    by_url = {a["url"]: a for a in merge_run(list(_responses(state).values()))}
    return [by_url[url] for url in stage["kept"]]


def _sheet_command() -> None:
    from src.fresh_eval import build_sheet, label_sheet_is_empty
    from src.fulltext import FULLTEXT_DIR, LABEL_COLUMNS, FullTextFetcher, labelling_page
    from src.query import load_aliases
    if (RUN_DIR / LABEL_FILE).exists() and not label_sheet_is_empty(RUN_DIR / LABEL_FILE):
        raise TransferError(f"{LABEL_FILE} already holds labels; it is not overwritten.")
    state = load_state(RUN_DIR)
    articles, aliases, fetcher, results = _kept(state), load_aliases(ALIAS_FILE), FullTextFetcher(), {}
    for number, article in enumerate(progress_order(articles), start=1):
        results[article["url"]] = fetcher.fetch(article["url"])
        print(f"{number:>3}/{len(articles)}  {results[article['url']]['status']:<18} {article['title'][:70]}")
    sheet = build_sheet(articles, results, COMPANY, aliases)
    with open(RUN_DIR / LABEL_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(sheet["rows"])
    page = Path(FULLTEXT_DIR) / f"labelling_view_{RUN_ID}.html"           # publisher text: never in Git
    page.write_text(labelling_page(sheet["items"]), encoding="utf-8")
    state["stages"]["sheet"] = {"done": True, "article_ids": [r["article_id"] for r in sheet["rows"]],
                                "evidence": sheet["evidence"]}
    save_state(RUN_DIR, state)
    counts = {}
    for kind in sheet["evidence"].values():
        counts[kind] = counts.get(kind, 0) + 1
    print(f"\n{len(sheet['rows'])} articles to label -> {RUN_DIR / LABEL_FILE}\nReading page: {page}")
    print("Evidence: " + ", ".join(f"{kind} {n}" for kind, n in sorted(counts.items())))


def _read_labels() -> dict[str, str]:
    with open(RUN_DIR / LABEL_FILE, encoding="utf-8-sig", newline="") as handle:
        return {row["article_id"]: (row.get("label") or "").strip().lower() for row in csv.DictReader(handle)}


def _events_command() -> None:
    from src.fulltext import article_id
    from src.summary_run import EVENT_COLUMNS, relevant_event_rows
    state = load_state(RUN_DIR)
    labels = _read_labels()
    bad = [key for key, label in labels.items() if label not in VALID_LABELS]
    if bad:
        raise TransferError(f"Missing or invalid labels in {LABEL_FILE}: {', '.join(bad)}")
    if (RUN_DIR / EVENTS_FILE).exists():
        raise TransferError(f"{EVENTS_FILE} already exists; it is not overwritten.")
    articles = _kept(state)
    rows = relevant_event_rows(articles, {a["url"]: labels[article_id(a["url"])] for a in articles})
    with open(RUN_DIR / EVENTS_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    state["stages"]["events"] = {"done": True}
    save_state(RUN_DIR, state)
    print(f"{len(rows)} relevant articles to group into events -> {RUN_DIR / EVENTS_FILE}")


def _seal_command() -> None:
    from src.fulltext import article_id
    from src.summary_run import load_events
    if (RUN_DIR / SEAL_FILE).exists():
        raise TransferError(f"{SEAL_FILE} already exists: the labels and events are sealed.")
    by_url = load_events(RUN_DIR / EVENTS_FILE)
    events = {article_id(url): event for url, event in by_url.items()}
    problems = seal_problems(_read_labels(), events)
    if problems:
        raise TransferError("Not complete:\n" + "\n".join(problems))
    seal = {"labels_sha256": file_fingerprint(RUN_DIR / LABEL_FILE),
            "events_sha256": file_fingerprint(RUN_DIR / EVENTS_FILE),
            "sealed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (RUN_DIR / SEAL_FILE).write_text(json.dumps(seal, indent=1), encoding="utf-8")
    print(f"Sealed: {RUN_DIR / SEAL_FILE}  (commit it before any model runs on Reliance)")


def _amend_command() -> None:
    record = write_amendment(RUN_DIR, git("rev-parse", "HEAD"), datetime.now(timezone.utc))
    print(f"Baseline: {record['baseline']['name']}; conditions unchanged: {', '.join(record['conditions'])}")
    print(f"Saved: {RUN_DIR / AMENDMENT_FILE}\nNext: commit it, then: git tag {AMEND_TAG}")


def _amend2_command() -> None:
    record = write_amendment2(RUN_DIR, git("rev-parse", "HEAD"), datetime.now(timezone.utc))
    print(f"Added, reported without a pass level: {', '.join(record['added'])}")
    print(f"Saved: {RUN_DIR / AMENDMENT2_FILE}\nNext: commit it, then: git tag {AMEND2_TAG}")


def _git_text(tag: str, name: str) -> str | None:
    try:
        return git("show", f"{tag}:{(RUN_DIR / name).relative_to(ROOT_DIR).as_posix()}")
    except Exception:
        return None


def _optional_text(name: str) -> str | None:
    path = RUN_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else None


def _run_candidates(aliases: dict) -> list[dict]:
    """Stage 1 on the sample, in retrieval order, with the full texts saved by the sheet step."""
    from src.fulltext import FullTextFetcher
    from src.ranking_eval import stage1_candidates
    articles, fetcher, full_texts = _kept(load_state(RUN_DIR)), FullTextFetcher(), {}
    for article in articles:
        cached = fetcher._cache_path(article["url"])
        if cached.exists():
            result = json.loads(cached.read_text(encoding="utf-8"))
            if result.get("status") == "ok":
                full_texts[article["url"]] = result["text"]
    return stage1_candidates(articles, full_texts, COMPANY, aliases)


def _summary_articles(candidates: list[dict], aliases: dict, k: int) -> list[dict]:
    from src import summary_v2 as sv
    by_key = {c["key"]: c for c in candidates}
    chosen = selections(candidates, [k])[DEFAULT][str(k)]
    return sv.passage_articles([by_key[key] for key in chosen], aliases, COMPANY)


def _bounds() -> tuple[dict, int, int]:
    bounds = {"relevance": token_bound(_evidence_calls(RELEVANCE_EVIDENCE)),
              "summary": token_bound(_evidence_calls(SUMMARY_EVIDENCE))}
    return bounds, round(_largest_output(RELEVANCE_EVIDENCE) * RELEVANCE_OUTPUT_MARGIN), _most_repairs(SUMMARY_EVIDENCE)


def budget_record(plan: dict, report: dict, inputs: dict, source_commit: str) -> dict:
    return {"source_commit": source_commit, "code_files": list(BUDGET_CODE), "inputs": inputs, "plan": plan,
            "report": report, "decision": report["decision"],
            "confirmation_note": ("The 24-hour condition is the user's confirmation, given with --no-groq-24h when the "
                                  "run starts; this report does not and cannot verify the Groq usage."),
            "model_requests": 0, "answer_cache": "not read and not written",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _budget_command() -> None:
    """No model request and no answer cache: the prompts are built and measured only. The report is saved once
    (transfer_budget.json), before the run, from committed code."""
    from src.query import load_aliases
    path = RUN_DIR / BUDGET_FILE
    for done in (path, *run_paths().values()):
        if done.exists():
            raise TransferError(f"{done.name} exists: the budget report is made once, before the run.")
    changed = protected_changes(git("status", "--porcelain", "--untracked-files=all"))
    if changed:
        raise TransferError("Commit these first, so the report names the exact code: " + ", ".join(changed))
    aliases = load_aliases(ALIAS_FILE)
    bounds, output, repairs = _bounds()
    plan = budget_plan(_run_candidates(aliases), aliases, bounds)
    report = budget_report(plan["relevance_prompt_chars"], plan["summary_prompt_chars"], bounds, output, repairs)
    record = budget_record(plan, report, budget_inputs(), git("rev-parse", "HEAD"))
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=1))
    print(json.dumps(record, indent=1))
    print(f"Decision: {report['decision']}")
    print(f"Saved: {path}  (review it, then commit it before the run)")


def _run_command() -> None:
    import time
    from src import summary_freeze as sf
    from src import summary_v2 as sv
    from src.llm import make_llm
    from src.model_compare import CONFIGS
    from src.query import load_aliases
    protocol_text = _optional_text(PROTOCOL_FILE)
    amendment_text, amendment2_text = _optional_text(AMENDMENT_FILE), _optional_text(AMENDMENT2_FILE)
    protocol = json.loads(protocol_text) if protocol_text else {"pass_levels": PASS_LEVELS}
    summary_k = protocol["pass_levels"]["summary"]["k"]
    freeze_path = EVAL_DIR / sf.FREEZE_FILE
    if freeze_path.exists():
        freeze_text = freeze_path.read_text(encoding="utf-8")
        freeze = json.loads(freeze_text)
        freeze_found = sf.freeze_problems(freeze, summary_k, sf.current_settings(), sf.code_fingerprints(),
                                          sf._tag_text(git), freeze_text)
    else:
        freeze, freeze_found = None, [f"{sf.FREEZE_FILE} does not exist: the Stage 4 summary is not frozen yet"]
    paths = run_paths()
    aliases = load_aliases(ALIAS_FILE)
    candidates = _run_candidates(aliases)                       # Stage 1 only: no model
    budget_text = _optional_text(BUDGET_FILE)
    try:
        plan = budget_plan(candidates, aliases, _bounds()[0])
        budget_found = budget_problems(json.loads(budget_text) if budget_text else None, budget_inputs(),
                                       plan["plan_fingerprint"])
    except (OSError, ValueError) as error:
        budget_found = [f"the budget check cannot be made: {error}"]
    problems = run_problems(protocol_text, _git_text(FREEZE_TAG, PROTOCOL_FILE), amendment_text,
                            _git_text(AMEND_TAG, AMENDMENT_FILE), amendment2_text,
                            _git_text(AMEND2_TAG, AMENDMENT2_FILE), (RUN_DIR / SEAL_FILE).exists(), freeze_found,
                            [path.name for path in paths.values() if path.exists()],
                            protected_changes(git("status", "--porcelain", "--untracked-files=all")),
                            CONFIRM_FLAG in sys.argv[2:], budget_found)
    if problems:
        raise TransferError("The run is refused:\n" + "\n".join(problems))
    commit = git("rev-parse", "HEAD")
    amendment = json.loads(amendment_text)
    stage3 = CONFIGS[sv.MODEL_SETTING]
    relevance_llm = make_llm(stage3["provider"], model=stage3["model"], generation=stage3["generation"])
    summary_llm = sv.summary_llm()

    def summarize(articles, k):
        result = summary_call(freeze["design"], articles, summary_llm, k, freeze["settings"]["correction"])
        for message in sv.take_messages(result):
            print(f"    {message}")
        return result

    ks = sorted({amendment["primary_k"], *amendment["reported_k"]})
    record = execute_run(RUN_DIR, candidates, ks, summary_k, freeze["design"],
                         lambda chosen: sv.passage_articles(chosen, aliases, COMPANY),
                         lambda items: relevance_call(items, aliases, relevance_llm), summarize,
                         run_fingerprints(protocol_text, amendment_text, amendment2_text),
                         source_commit=commit, pause=time.sleep)
    print(run_message(record))
    print(f"Saved: {paths['record']} and {paths['attempt']}  (commit both before judging; do not open the record)")


def run_message(record: dict) -> str:
    """What the run prints: counts and the summary status only, never a selection or a ranking result."""
    result = record["summary_record"]["records"][0]["result"]
    return (f"Candidates: {len(record['candidates'])}; relevance coverage {record['relevance']['coverage']}, "
            f"{record['relevance']['seconds']} s; summary {result.get('status')}, "
            f"failures {result.get('failures', [])}")


def _load(name: str) -> dict:
    path = RUN_DIR / name
    if not path.exists():
        raise TransferError(f"{name} does not exist yet.")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_seal() -> None:
    changed = seal_changes(_load(SEAL_FILE), file_fingerprint(RUN_DIR / LABEL_FILE), file_fingerprint(RUN_DIR / EVENTS_FILE))
    if changed:
        raise TransferError(f"Changed after the seal: {', '.join(changed)}. The sealed versions are the references.")


def _labels_by_url() -> dict[str, str]:
    from src.preprocess import normalize_url
    with open(RUN_DIR / LABEL_FILE, encoding="utf-8-sig", newline="") as handle:
        return {normalize_url(row["url"]): (row.get("label") or "").strip().lower() for row in csv.DictReader(handle)}


def _check_attempt(run: dict) -> None:
    path = run_paths()["passages"]
    if not path.exists():
        raise TransferError(f"{path} does not exist: the evidence the model received is missing.")
    found = attempt_problems(_load(ATTEMPT_FILE), run, json.loads(path.read_text(encoding="utf-8")))
    budget = RUN_DIR / BUDGET_FILE
    found += budget_file_problems(run, file_fingerprint(budget) if budget.exists() else None)
    if found:
        raise TransferError("; ".join(found))


def _produced(run: dict) -> bool:
    from src import summary_judging as sj
    return bool(sj._produced(run["summary_record"]))


def _judge_command() -> None:
    """The Part 4 sheets on the saved passages: industry evidence first, then the items, then the events."""
    from src import summary_judging as sj
    from src import summary_judging_p4 as p4
    from src.fulltext import FULLTEXT_DIR
    from src.summary_run import load_events
    run = _load(RUN_FILE)
    _check_seal()
    _check_attempt(run)
    if not _produced(run):
        raise TransferError("The run gives no summary: 'produced' fails and there is nothing to judge. "
                            "Run the score.")
    path = run_paths()["passages"]
    if not path.exists():
        raise TransferError(f"{path} does not exist: the evidence the model received is missing.")
    passages = saved_passages(run, json.loads(path.read_text(encoding="utf-8")))
    sheets = p4.build_sheets(run["summary_record"], passages, load_events(RUN_DIR / EVENTS_FILE), _labels_by_url(),
                             {}, seed=JUDGING_SEED)
    sheets["key"]["record_file"] = RUN_FILE
    files = {RUN_DIR / JUDGING_ITEMS: sj._csv_text(sheets["items"], sj.ITEM_COLUMNS),
             RUN_DIR / JUDGING_EVENTS: sj._csv_text(sheets["events"], sj.EVENT_COLUMNS),
             RUN_DIR / JUDGING_INDUSTRY: sj._csv_text(sheets["industry"], p4.INDUSTRY_COLUMNS),
             RUN_DIR / JUDGING_KEY: json.dumps(sheets["key"], indent=1),
             Path(FULLTEXT_DIR) / JUDGING_PAGE: sheets["page"]}
    for target in files:
        if target.exists():
            raise TransferError(f"{target.name} already exists; it is not overwritten.")
    for target, text in files.items():
        sj._write_new(target, text)
    print(f"Industry evidence: 1 row in {RUN_DIR / JUDGING_INDUSTRY}  (judge it first, from the articles alone)")
    print(f"Items: {len(sheets['items'])} rows in {RUN_DIR / JUDGING_ITEMS}")
    print(f"Events: {len(sheets['events'])} rows in {RUN_DIR / JUDGING_EVENTS}")
    print(f"Reading page: {Path(FULLTEXT_DIR) / JUDGING_PAGE}")
    print(f"Key: {RUN_DIR / JUDGING_KEY}  (do not open it until the sheets are judged and locked)")


def _sheets() -> tuple:
    from src import summary_judging as sj
    return tuple(sj._read_csv(RUN_DIR / name) for name in (JUDGING_ITEMS, JUDGING_EVENTS, JUDGING_INDUSTRY))


def _lock_command() -> None:
    from src import summary_judging as sj
    if (RUN_DIR / JUDGING_LOCK).exists():
        raise TransferError(f"{JUDGING_LOCK} already exists: the sheets are locked.")
    _check_seal()
    _check_attempt(_load(RUN_FILE))
    lock = judging_lock(*_sheets(), _load(JUDGING_KEY), judging_inputs())
    sj._write_new(RUN_DIR / JUDGING_LOCK, json.dumps(lock, indent=1))
    print(f"Locked: {RUN_DIR / JUDGING_LOCK}")


def unsure_report(rows: list[dict], chosen: dict, primary_k: int) -> list[dict]:
    """The articles labelled unsure, on their own: their label and whether each list selects them."""
    return [{"article_id": row["article_id"], "label": (row.get("label") or "").strip().lower(),
             **{name: row["article_id"] in by_k[str(primary_k)] for name, by_k in chosen.items()}}
            for row in rows if (row.get("unsure") or "").strip().lower() == "yes"]


def _score_command() -> None:
    """After the lock (or when the run gives no summary): the ranking, the relevance report, the unsure labels,
    the protocol's summary levels and the amendment 2 industry report, all in one file, made once."""
    from src import summary_judging as sj
    from src.fulltext import article_id
    from src.summary_run import load_events
    if (RUN_DIR / SCORE_FILE).exists():
        raise TransferError(f"{SCORE_FILE} already exists; it is not overwritten.")
    run, amendment, protocol = _load(RUN_FILE), _load(AMENDMENT_FILE), _load(PROTOCOL_FILE)
    _check_seal()
    _check_attempt(run)
    if _produced(run):
        if not (RUN_DIR / JUDGING_LOCK).exists():
            raise TransferError("The judgements are not locked: the score waits for the lock, so no ranking result "
                                "is seen while judging.")
        items, event_rows, industry = _sheets()
        key = _load(JUDGING_KEY)
        changed = lock_changes(_load(JUDGING_LOCK), items, event_rows, industry, key, judging_inputs())
        if changed:
            raise TransferError("The score cannot be made:\n" + "\n".join(changed))
        design = run["summary_record"]["records"][0]["design"]
        judged = sj.score(items, event_rows, key, run["summary_record"])["designs"][design]
        summary, industry_result = summary_result(judged, protocol), industry_report(items, industry, key,
                                                                                     run["summary_record"])
    else:
        summary, industry_result = no_summary_result(), None
    labels = _read_labels()
    events = {article_id(url): event for url, event in load_events(RUN_DIR / EVENTS_FILE).items()}
    keys = [c["key"] for c in run["candidates"]]
    ranking = ranking_score(run["selections"], keys, labels, events, protocol, amendment)
    with open(RUN_DIR / LABEL_FILE, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {"ranking": ranking,
              "relevance_check": {**relevance_report(run["relevance"]["answers"], labels),
                                  "coverage": run["relevance"]["coverage"], "seconds": run["relevance"]["seconds"]},
              "unsure": unsure_report(rows, run["selections"], amendment["primary_k"]),
              "summary": summary, "industry": industry_result,
              "scored": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(RUN_DIR / SCORE_FILE, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    print(f"Ranking at k = {ranking['primary_k']} against the {ranking['baseline']} baseline: passed {ranking['passed']} "
          f"{ranking['conditions']}")
    print(f"Summary: passed {summary['passed']} {summary['checks']}")
    if industry_result:
        print(f"Industry (reported only): evidence {industry_result['industry_evidence']}, points "
              f"{industry_result['industry_points']}, checks {industry_result['checks']}")
    print(f"Saved: {RUN_DIR / SCORE_FILE}")


COMMANDS = {"protocol": _protocol_command, "fetch": _fetch_command, "sample": _sample_command,
            "sheet": _sheet_command, "events": _events_command, "seal": _seal_command, "amend": _amend_command,
            "amend2": _amend2_command, "budget": _budget_command, "run": _run_command, "judge": _judge_command,
            "lock": _lock_command, "score": _score_command}

if __name__ == "__main__":
    try:
        if (len(sys.argv) == 2 and sys.argv[1] in COMMANDS) or sys.argv[1:] == ["run", CONFIRM_FLAG]:
            COMMANDS[sys.argv[1]]()
        else:
            print(__doc__)
    except TransferError as error:
        print(f"[STOP] {error}")
        sys.exit(1)
