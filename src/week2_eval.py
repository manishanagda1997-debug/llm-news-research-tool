"""
Stage 7, week 2: a fresh test of the final default on one new week of Tata Motors news and one new week of
Reliance Industries news. This part covers the protocol, the data steps and the ranking test, none of which uses a
model. The summary runs and their judging are a later part of this module; their rules are fixed here, in the
protocol, before any fetch.

Each week has its own run folder (data/eval/<run id>/) and its own protocol and tag. src/transfer_eval.py, the
Reliance week 1 folder and its lock are only read, never changed.

The ranking test (no model):
    primary     O1M5 + S1 (the default, src/research.py) against the retrieval order (Reliance amendment 1): the
                three conditions at k = 10 decide "passed"
    second      O1M5 + S1 against O1 + S1, the same three conditions, reported only: the round 1 no-harm rule on
                fresh data; it never changes the primary result
    reported    O1 + S1 against the retrieval order too, and the relevant articles whose title does not name the
                company, with their places under each order and whether they reach the top 10

Window convention: NewsAPI's publishedAt has whole seconds, so the window is start <= published < the next
midnight after the last day, which is the same as published <= 23:59:59 of the last day. The request asks for the
same times as week 1 (from 00:00:00 to 23:59:59).

Commands, for each week (tata or reliance), in order:
    python -m src.week2_eval tata protocol   then commit, then: git tag freeze-week2-tata
    python -m src.week2_eval tata fetch      2 NewsAPI requests; refused without the tag or before the fetch time
    python -m src.week2_eval tata overlap    then decide overlap_review.csv when it has rows
    python -m src.week2_eval tata sheet      the blind label sheet and the reading page
    python -m src.week2_eval tata events     after every label
    python -m src.week2_eval tata seal       labels, events and the optional expectation pairs; then commit
    python -m src.week2_eval tata rank       once, after the seal; no model
"""

import csv
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import DATA_DIR, EVAL_DIR, RAW_DIR, ROOT_DIR
from src.evaluate import VALID_LABELS, is_relevant

IST = timezone(timedelta(hours=5, minutes=30))
ALIAS_FILE = DATA_DIR / "company_aliases.json"
GUIDE_FILE = EVAL_DIR / "LABELLING_GUIDE.md"
PROTOCOL_FILE = "week2_protocol.json"
STATE_FILE = "state.json"
REVIEW_FILE = "overlap_review.csv"
LABEL_FILE = "labels.csv"
EVENTS_FILE = "events.csv"
PAIRS_FILE = "pairs.csv"
SEAL_FILE = "week2_seal.json"
RANK_FILE = "week2_ranking.json"
PAIR_COLUMNS = ["expectation_event", "outcome_event", "note"]
SEARCH_SETTINGS = {"page_size": 30, "sort_by": "relevancy", "exact_phrase": True, "title_only": False}
FETCH_DELAY = timedelta(hours=30)        # NewsAPI's one-day delay plus 6 hours, after the window's last second
RUN_CUTOFF_IST = "2026-11-10T22:00:00+05:30"
JUDGING_CUTOFF_IST = "2026-11-11T05:30:00+05:30"
RETRIEVAL, OLD_DEFAULT, DEFAULT = "retrieval_order", "O1+S1", "O1M5+S1"
PRIMARY_K, REPORTED_K = 10, (5, 15)
CONDITIONS = ("no_direct_event_lost", "event_recall_not_lower", "precision_not_lower")

WEEKS = {
    "tata": {"company": "Tata Motors", "run_id": "week2_tata_2026-10-31_11-06",
             "window": ("2026-10-31T00:00:00Z", "2026-11-06T23:59:59Z"), "tag": "freeze-week2-tata",
             "earlier": "the development raw files and Tata Motors fresh week 1 (fresh_2026-10-24_30)"},
    "reliance": {"company": "Reliance Industries", "run_id": "week2_reliance_2026-11-02_11-08",
                 "window": ("2026-11-02T00:00:00Z", "2026-11-08T23:59:59Z"), "tag": "freeze-week2-reliance",
                 "earlier": "Reliance Industries week 1 (transfer_reliance_2026-10-26_11-01)"},
}

SAMPLE_RULE = ("Search A (up to 30 articles, by relevance), then search B (up to 30), merged in that order; each "
               "article kept once by tidy link; excluded and logged: an article with the same tidy link as an "
               "article of the company's earlier data, an article published outside the window, an article with "
               "no date, and an article judged a repost of an earlier article in the overlap review (near-identical "
               "titles: `same` only for the same article or a repost). The sample is what remains, even when small "
               "or unbalanced: no top-up and no replacement. An article whose page cannot be fetched stays and is "
               "labelled on its short text.")

DEFINITIONS = {
    "relevant_event": "an event with at least one article labelled direct or indirect",
    "direct_event": "an event with at least one article labelled direct",
    "event_recall": "distinct relevant events among the relevant articles in the top k, divided by the relevant "
                    "events among the articles Stage 1 keeps; undefined when there is no relevant event",
    "precision": "relevant articles among the selected articles, divided by the number selected (k, or fewer "
                 "when fewer articles are available; the number available is reported)",
    "no_direct_event_lost": "every direct event that the baseline's top 10 covers is also covered by the tested "
                            "list's top 10, on the same article ids and the same event names from the sealed sheet",
    "retrieval_order": "search A's articles in the order returned, then search B's; each once; the frozen Stage 1 "
                       "rules (cleaning and duplicates) with no other sorting; the surviving articles keep their "
                       "relative order; the first k (Reliance amendment 1)",
    "duplicates": "only the frozen Stage 1 rules; S1 takes the first k with no grouping",
    "summary_event_coverage": "event rows marked covered = yes, divided by all event rows: one row per relevant "
                              "event among the summary's own top 5 articles (SUMMARY_JUDGING_GUIDE.md, Part 3)",
    "contradicted_rate": "items with contradicted = yes, divided by all judged items (company points, industry "
                         "points and visible article briefs)",
    "produced": "the one summary comes out (1 of 1)",
}

RANKING_RULE = {
    "tested": "O1M5+S1, the default (src/research.py, default_order then S1)",
    "primary": {"baseline": RETRIEVAL, "conditions": list(CONDITIONS), "primary_k": PRIMARY_K,
                "reported_k": list(REPORTED_K), "decides": "passed"},
    "second": {"baseline": OLD_DEFAULT, "conditions": list(CONDITIONS), "primary_k": PRIMARY_K,
               "reported_k": list(REPORTED_K), "decides": "nothing: the round 1 no-harm rule on fresh data, "
               "reported only; it never changes the primary result"},
    "also_reported": ["O1+S1 against the retrieval order, as in week 1",
                      "each relevant article whose title does not name the company: its place in the retrieval "
                      "order, O1+S1 and O1M5+S1, and whether it reaches the top 10"],
    "no_model": "the ranking test uses no model call",
}

SUMMARY_RULE = {
    "design": "the summary design frozen after round 3 of the second development cycle (its freeze tag), k = 5, on "
              "the default's top 5; one run per week; no answer cache; no retry",
    "pass_levels": {"produced": True, "min_event_coverage": 0.8, "max_contradicted": 0.05,
                    "no_central_contradiction": True},
    "industry": "amendment 2 of week 1, built in: the industry evidence judged first from the five articles alone; "
                "the industry points split into supported, partly supported and not supported; the checks 'no "
                "industry point not supported' and 'the section matches the evidence'; reported without a level",
    "pairs": "the expectation-and-outcome pairs listed in pairs.csv at the seal (event names from the sealed event "
             "sheet), before any run: for each pair whose two events are among the summary's articles, the Part 7 "
             "rows outlook_stated and kept_apart; reported without a level",
    "judging": "blind, with SUMMARY_JUDGING_GUIDE.md, Parts 4, 6 and 7; the sheets locked before the score",
    "relevance_check": "not run: it is experimental and not part of the default; reported as not run",
    "tokens": "before each run: the logged tokens of the last 24 hours (round 2, round 3 and week 2 records) + the "
              "stated other use + 45,000 <= 200,000; each call guarded by the round 3 allowance guard (the "
              "estimated worst case before, the real count after); no retry",
    "order": "a run only after round 3 is scored and the final design frozen, and only on a sealed week",
    "run_cutoff": f"no week 2 summary run starts after {RUN_CUTOFF_IST}; a run not started is reported as not run",
    "judging_cutoff": f"summaries not judged and locked by {JUDGING_CUTOFF_IST} are reported as run but not judged, "
                      "with no result claimed",
}

READING = {
    "candidate_pool": "the measures cover the articles the two searches return (up to 30 each, by relevance) and "
                      "Stage 1 keeps, not all news of the week",
    "no_tuning": "nothing in the pipeline, the aliases or the searches changes because of these weeks",
    "one_week": "each result covers one week of one company",
}


class Week2Error(Exception):
    pass


# ---------- the week ----------

def week(name: str) -> dict:
    if name not in WEEKS:
        raise Week2Error(f"Unknown week {name}; use one of {', '.join(WEEKS)}.")
    found = dict(WEEKS[name], name=name)
    found["run_dir"] = EVAL_DIR / found["run_id"]
    return found


def window(w: dict) -> tuple[datetime, datetime]:
    """(start, end) with the end exclusive: the next midnight after the last day."""
    start = datetime.fromisoformat(w["window"][0].replace("Z", "+00:00"))
    last = datetime.fromisoformat(w["window"][1].replace("Z", "+00:00"))
    return start, last.replace(hour=0, minute=0, second=0) + timedelta(days=1)


def fetch_not_before(w: dict) -> datetime:
    """30 hours after the window's end (the next midnight after the last day)."""
    return window(w)[1] + FETCH_DELAY


def search_strings(company: str, aliases: dict) -> dict[str, str]:
    from src.query import build_search_plan
    a = build_search_plan(company, exact_phrase=True, aliases=None, part="all")["search_query"]
    b = build_search_plan(company, exact_phrase=True, aliases=aliases, part="subsidiaries")["search_query"]
    if not b:
        raise Week2Error("Search B is empty: the alias file has no subsidiaries for the company.")
    return {"A": a, "B": b}


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def protocol_record(w: dict, aliases: dict, alias_text: str, guide_text: str, source_commit: str) -> dict:
    from src.relevance_check import company_scope
    return {"run_id": w["run_id"], "company": w["company"], "week": w["name"], "window": list(w["window"]),
            "window_convention": "start <= publishedAt < the next midnight after the last day (whole seconds, so "
                                 "the same as publishedAt <= 23:59:59 of the last day); the request asks for the "
                                 "same from and to times as week 1",
            "fetch_not_before": fetch_not_before(w).isoformat(), "searches": search_strings(w["company"], aliases),
            "search_settings": SEARCH_SETTINGS, "alias_fingerprint": text_fingerprint(alias_text),
            "guide_fingerprint": text_fingerprint(guide_text), "company_scope": company_scope(w["company"], aliases),
            "earlier_data": w["earlier"], "sample_rule": SAMPLE_RULE, "definitions": DEFINITIONS,
            "ranking": RANKING_RULE, "summary": SUMMARY_RULE, "reading": READING,
            "replaces": "the earlier decision to skip the week 2 tests; round 3 is not shortened",
            "source_commit": source_commit, "freeze_tag": w["tag"],
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def fetch_problems(w: dict, protocol: dict, aliases: dict, alias_text: str, now: datetime,
                   tag_text: str | None, protocol_text: str) -> list[str]:
    found = []
    if tag_text is None or tag_text.replace("\r\n", "\n") != protocol_text.replace("\r\n", "\n"):
        found.append(f"{PROTOCOL_FILE} is missing from the tag {w['tag']} or differs from it")
    if text_fingerprint(alias_text) != protocol["alias_fingerprint"]:
        found.append("the alias file differs from the protocol")
    if text_fingerprint(GUIDE_FILE.read_text(encoding="utf-8")) != protocol["guide_fingerprint"]:
        found.append("the labelling guide differs from the protocol")
    if search_strings(w["company"], aliases) != protocol["searches"]:
        found.append("a search string differs from the protocol")
    if now < fetch_not_before(w):
        found.append(f"too early: fetch on or after {fetch_not_before(w).astimezone(IST):%d %B %Y %H:%M} IST")
    return found


# ---------- the state of a week ----------

def load_state(run_dir: Path) -> dict:
    path = Path(run_dir) / STATE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stages": {}}


def save_state(run_dir: Path, state: dict) -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def _published(article: dict) -> datetime | None:
    from src.fresh_eval import _published as published
    return published(article)


def day_counts(w: dict, articles: list[dict]) -> dict[str, int]:
    start, end = window(w)
    counts, day = {}, start
    while day < end:
        counts[day.date().isoformat()] = 0
        day += timedelta(days=1)
    for article in articles:
        published = _published(article)
        if published is not None and published.date().isoformat() in counts:
            counts[published.date().isoformat()] += 1
    return counts


# ---------- the sample and the overlap ----------

def earlier_articles(w: dict) -> list[dict]:
    """The company's earlier articles: their raw files, as each earlier run records them."""
    files = []
    if w["name"] == "tata":
        state = json.loads((EVAL_DIR / "fresh_2026-10-24_30" / STATE_FILE).read_text(encoding="utf-8"))
        files = list(state["stages"]["fetch"]["seen_files"]) + list(state["stages"]["fetch"]["raw_files"].values())
    else:
        state = json.loads((EVAL_DIR / "transfer_reliance_2026-10-26_11-01" / STATE_FILE).read_text(encoding="utf-8"))
        files = list(state["stages"]["fetch"]["raw_files"].values())
    articles = []
    for name in sorted(set(files)):
        articles += json.loads((RAW_DIR / name).read_text(encoding="utf-8")).get("articles", [])
    return articles


def sample(w: dict, responses: dict[str, dict], earlier: list[dict]) -> dict:
    from src.fresh_eval import _entry, _near_title
    from src.preprocess import normalize_url
    from src.ranking_eval import merge_run
    start, end = window(w)
    merged = merge_run([responses["A"], responses["B"]])
    earlier_links = {normalize_url(a.get("url") or "") for a in earlier}
    kept, excluded, review = [], [], []
    for article in merged:
        published = _published(article)
        if normalize_url(article.get("url") or "") in earlier_links:
            excluded.append(_entry(article, "earlier_link"))
        elif published is None:
            excluded.append(_entry(article, "no_date"))
        elif not start <= published < end:
            excluded.append(_entry(article, "outside_window"))
        else:
            kept.append(article)
            match = next((e for e in earlier if _near_title(article.get("title"), e.get("title"))), None)
            if match:
                review.append({"fresh_url": article["url"], "dev_url": match["url"],
                               "fresh_title": article.get("title"), "dev_title": match.get("title"),
                               "fresh_source": (article.get("source") or {}).get("name", ""),
                               "dev_source": (match.get("source") or {}).get("name", ""),
                               "fresh_published": article.get("publishedAt"),
                               "dev_published": match.get("publishedAt")})
    reasons = [e["reason"] for e in excluded]
    counts = {"search_A": len(responses["A"].get("articles", [])), "search_B": len(responses["B"].get("articles", [])),
              "after_merge": len(merged), **{r: reasons.count(r) for r in ("earlier_link", "no_date", "outside_window")},
              "to_review": len(review), "kept_before_review": len(kept)}
    return {"kept": kept, "excluded": excluded, "review": review, "counts": counts}


# ---------- the seal ----------

def pairs_problems(pairs: list[dict], event_names: set[str]) -> list[str]:
    """No row means no pairs. Every row needs two different event names of the sealed sheet, compared as the event
    sheet itself is read (outer spaces removed, lower case); a blank or half-blank row is an error."""
    found = []
    for number, row in enumerate(pairs, start=1):
        first = (row.get("expectation_event") or "").strip().lower()
        second = (row.get("outcome_event") or "").strip().lower()
        if not first or not second:
            found.append(f"pairs row {number}: both event names are needed (delete the row if there is no pair)")
            continue
        for name in (first, second):
            if name not in event_names:
                found.append(f"pairs row {number}: '{name}' is not an event name of the sealed event sheet")
        if first == second:
            found.append(f"pairs row {number}: the two events are the same")
    return found


def seal_record(run_dir: Path) -> dict:
    from src.fresh_eval import file_fingerprint
    record = {"labels_sha256": file_fingerprint(run_dir / LABEL_FILE),
              "events_sha256": file_fingerprint(run_dir / EVENTS_FILE),
              "pairs_sha256": file_fingerprint(run_dir / PAIRS_FILE) if (run_dir / PAIRS_FILE).exists() else None,
              "sealed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return record


def seal_changes(seal: dict, run_dir: Path) -> list[str]:
    current = seal_record(run_dir)
    return [name for name, field in ((LABEL_FILE, "labels_sha256"), (EVENTS_FILE, "events_sha256"),
                                     (PAIRS_FILE, "pairs_sha256")) if seal.get(field) != current[field]]


# ---------- the ranking test (no model) ----------

def orders(candidates: list[dict], company: str, aliases: dict) -> dict[str, list[dict]]:
    """The three orders on the candidates Stage 1 keeps (in retrieval order)."""
    from src import ranking as rk
    from src.research import default_order
    plain = [dict(c, check=None) for c in candidates]
    return {RETRIEVAL: plain, OLD_DEFAULT: rk.order(plain, "O1"), DEFAULT: default_order(plain, company, aliases)}


def chosen(ordered: dict[str, list[dict]], ks) -> dict[str, dict[str, list[str]]]:
    from src import ranking as rk
    return {RETRIEVAL: {str(k): [c["key"] for c in ordered[RETRIEVAL][:k]] for k in ks},
            **{name: {str(k): [c["key"] for c in rk.select(ordered[name], k, "S1")["selected"]] for k in ks}
               for name in (OLD_DEFAULT, DEFAULT)}}


def comparison(picked: dict, keys: list[str], labels: dict, events: dict, baseline: str, tested: str) -> dict:
    from src.transfer_eval import ranking_score
    rule = {"baseline": {"name": baseline}, "configuration": tested, "primary_k": PRIMARY_K,
            "reported_k": list(REPORTED_K), "conditions": list(CONDITIONS)}
    return ranking_score(picked, keys, labels, events, {}, rule)


def untitled_relevant(ordered: dict[str, list[dict]], labels: dict) -> list[dict]:
    """The relevant candidates whose title does not name the company, with their places (1 = first)."""
    found = []
    for c in ordered[RETRIEVAL]:
        if c.get("mention_level") == "title" or not is_relevant(labels.get(c["key"], "")):
            continue
        places = {name: [x["key"] for x in order].index(c["key"]) + 1 for name, order in ordered.items()}
        found.append({"key": c["key"], "title": c.get("title"), "label": labels[c["key"]], "places": places,
                      "in_top_10": {name: place <= PRIMARY_K for name, place in places.items()}})
    return found


def ranking_result(candidates: list[dict], company: str, aliases: dict, labels: dict, events: dict) -> dict:
    ordered = orders(candidates, company, aliases)
    keys = [c["key"] for c in candidates]
    picked = chosen(ordered, sorted({PRIMARY_K, *REPORTED_K}))
    primary = comparison(picked, keys, labels, events, RETRIEVAL, DEFAULT)
    second = comparison(picked, keys, labels, events, OLD_DEFAULT, DEFAULT)
    week1_form = comparison(picked, keys, labels, events, RETRIEVAL, OLD_DEFAULT)
    return {"candidates": len(candidates), "primary": primary, "passed": primary["passed"],
            "second": {**second, "decides": "nothing (reported only)"},
            "old_default_against_retrieval": {**week1_form, "decides": "nothing (reported only)"},
            "untitled_relevant": untitled_relevant(ordered, labels),
            "orders": {name: [c["key"] for c in order] for name, order in ordered.items()}}


# ---------- files and commands ----------

def _git(*args) -> str:
    from src.fresh_eval import git
    return git(*args)


def _tag_text(w: dict) -> str | None:
    try:
        return _git("show", f"{w['tag']}:{(w['run_dir'] / PROTOCOL_FILE).relative_to(ROOT_DIR).as_posix()}")
    except Exception:
        return None


def _protocol_text(w: dict) -> str:
    path = w["run_dir"] / PROTOCOL_FILE
    if not path.exists():
        raise Week2Error(f"{PROTOCOL_FILE} does not exist yet: python -m src.week2_eval {w['name']} protocol")
    return path.read_text(encoding="utf-8")


def _protocol_command(w: dict) -> None:
    from src.query import load_aliases
    path = w["run_dir"] / PROTOCOL_FILE
    if path.exists():
        raise Week2Error(f"{PROTOCOL_FILE} already exists; it is not overwritten.")
    record = protocol_record(w, load_aliases(ALIAS_FILE), ALIAS_FILE.read_text(encoding="utf-8"),
                             GUIDE_FILE.read_text(encoding="utf-8"), _git("rev-parse", "HEAD"))
    w["run_dir"].mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=1))
    print(f"Search A: {record['searches']['A']}\nSearch B: {record['searches']['B']}")
    print(f"Window: {w['window'][0]} to {w['window'][1]}; fetch from {fetch_not_before(w).astimezone(IST):%d %b %H:%M} IST")
    print(f"Saved: {path}\nNext: commit it, then: git tag {w['tag']}")


def _fetch_command(w: dict, fetch=None, save=None, now=None) -> None:
    from src.fresh_eval import safe_request
    from src.query import load_aliases
    if fetch is None or save is None:
        from src.news_fetcher import fetch_news, save_raw
        fetch, save = fetch or fetch_news, save or save_raw
    protocol_text = _protocol_text(w)
    aliases = load_aliases(ALIAS_FILE)
    found = fetch_problems(w, json.loads(protocol_text), aliases, ALIAS_FILE.read_text(encoding="utf-8"),
                           now or datetime.now(timezone.utc), _tag_text(w), protocol_text)
    state = load_state(w["run_dir"])
    earlier = state["stages"].get("fetch")
    if earlier and (earlier.get("done") or earlier.get("raw_files")):
        found.append("the fetch is done or has saved a response; a second fetch would mix two sets of results")
    if found:
        raise Week2Error("The fetch is refused:\n" + "\n".join(found))
    # a fetch that saved nothing (its first request failed) can start again: there is nothing to mix
    stage = {"done": False, "raw_files": {}, "requests": {}, "day_counts": {}, "empty_days": {},
             "attempt": (earlier or {}).get("attempt", 0) + 1,
             "fetched_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")}
    state["stages"]["fetch"] = stage
    save_state(w["run_dir"], state)
    times = {"from_time": w["window"][0], "to_time": w["window"][1], "save": False}
    for name in ("A", "B"):
        if name == "A":
            response = fetch(w["company"], aliases=None, **SEARCH_SETTINGS, **times)
        else:
            response = fetch(w["company"], aliases=aliases, part="subsidiaries", **SEARCH_SETTINGS, **times)
        response["request"] = safe_request(response.get("request", {}))
        stage["raw_files"][name] = save(response).name
        stage["requests"][name] = response["request"]
        stage["day_counts"][name] = day_counts(w, response.get("articles", []))
        counts = stage["day_counts"][name]
        stage["empty_days"][name] = [day for day, n in counts.items() if n == 0]
        save_state(w["run_dir"], state)
        print(f"Search {name}: {len(response.get('articles', []))} articles -> data/raw/{stage['raw_files'][name]}")
        print("   per day (UTC): " + ", ".join(f"{day[5:]} {n}" for day, n in counts.items()))
        if not counts[max(counts)]:
            print(f"   NOTE: no article on the last day ({max(counts)}); it is reported as it is.")
    stage["done"] = True
    save_state(w["run_dir"], state)


def _responses(w: dict, state: dict) -> dict[str, dict]:
    stage = state["stages"].get("fetch")
    if not stage or not stage.get("done"):
        raise Week2Error("Run the fetch first.")
    return {name: json.loads((RAW_DIR / file).read_text(encoding="utf-8")) for name, file in stage["raw_files"].items()}


def _overlap_command(w: dict) -> None:
    state = load_state(w["run_dir"])
    if state["stages"].get("overlap"):
        raise Week2Error("The overlap step is already done.")
    result = sample(w, _responses(w, state), earlier_articles(w))
    state["stages"]["overlap"] = {"done": True, "kept": [a["url"] for a in result["kept"]],
                                  "excluded": result["excluded"], "review": result["review"],
                                  "counts": result["counts"]}
    save_state(w["run_dir"], state)
    if result["review"]:
        from src.fresh_eval import REVIEW_COLUMNS
        with open(w["run_dir"] / REVIEW_FILE, "x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerows({"row": n, "decision": "", **pair} for n, pair in enumerate(result["review"], start=1))
    print("Counts: " + ", ".join(f"{k} {v}" for k, v in result["counts"].items()))
    if result["review"]:
        print(f"{len(result['review'])} near-identical title(s) to review -> {w['run_dir'] / REVIEW_FILE}\n"
              "Write `same` only for the same article or a repost, otherwise `different`.")


def kept_articles(w: dict, state: dict) -> list[dict]:
    from src.fresh_eval import apply_review
    from src.ranking_eval import merge_run
    stage = state["stages"].get("overlap")
    if not stage:
        raise Week2Error("Run the overlap step first.")
    by_url = {a["url"]: a for a in merge_run(list(_responses(w, state).values()))}
    result = {"kept": [by_url[url] for url in stage["kept"]], "excluded": stage["excluded"], "review": stage["review"]}
    decisions = []
    if stage["review"]:
        with open(w["run_dir"] / REVIEW_FILE, encoding="utf-8-sig", newline="") as handle:
            decisions = list(csv.DictReader(handle))
    return apply_review(result, decisions)["kept"]


def _sheet_command(w: dict) -> None:
    from src.fresh_eval import build_sheet, label_sheet_is_empty, progress_order
    from src.fulltext import FULLTEXT_DIR, LABEL_COLUMNS, FullTextFetcher, labelling_page
    from src.query import load_aliases
    if (w["run_dir"] / LABEL_FILE).exists() and not label_sheet_is_empty(w["run_dir"] / LABEL_FILE):
        raise Week2Error(f"{LABEL_FILE} already holds labels; it is not overwritten.")
    state = load_state(w["run_dir"])
    articles, aliases, fetcher, results = kept_articles(w, state), load_aliases(ALIAS_FILE), FullTextFetcher(), {}
    for number, article in enumerate(progress_order(articles), start=1):
        results[article["url"]] = fetcher.fetch(article["url"])
        print(f"{number:>3}/{len(articles)}  {results[article['url']]['status']:<18} {article['title'][:70]}")
    sheet = build_sheet(articles, results, w["company"], aliases)
    with open(w["run_dir"] / LABEL_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(sheet["rows"])
    page = Path(FULLTEXT_DIR) / f"labelling_view_{w['run_id']}.html"          # publisher text: never in Git
    page.write_text(labelling_page(sheet["items"]), encoding="utf-8")
    state["stages"]["sheet"] = {"done": True, "article_ids": [r["article_id"] for r in sheet["rows"]],
                                "evidence": sheet["evidence"]}
    save_state(w["run_dir"], state)
    print(f"\n{len(sheet['rows'])} articles to label -> {w['run_dir'] / LABEL_FILE}\nReading page: {page}")


def read_labels(run_dir: Path) -> dict[str, str]:
    with open(Path(run_dir) / LABEL_FILE, encoding="utf-8-sig", newline="") as handle:
        return {row["article_id"]: (row.get("label") or "").strip().lower() for row in csv.DictReader(handle)}


def _events_command(w: dict) -> None:
    from src.fulltext import article_id
    from src.summary_run import EVENT_COLUMNS, relevant_event_rows
    labels = read_labels(w["run_dir"])
    bad = [key for key, label in labels.items() if label not in VALID_LABELS]
    if bad:
        raise Week2Error(f"Missing or invalid labels in {LABEL_FILE}: {', '.join(bad)}")
    if (w["run_dir"] / EVENTS_FILE).exists():
        raise Week2Error(f"{EVENTS_FILE} already exists; it is not overwritten.")
    state = load_state(w["run_dir"])
    articles = kept_articles(w, state)
    rows = relevant_event_rows(articles, {a["url"]: labels[article_id(a["url"])] for a in articles})
    with open(w["run_dir"] / EVENTS_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    if not (w["run_dir"] / PAIRS_FILE).exists():
        with open(w["run_dir"] / PAIRS_FILE, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=PAIR_COLUMNS).writeheader()
    state["stages"]["events"] = {"done": True}
    save_state(w["run_dir"], state)
    print(f"{len(rows)} relevant articles to group into events -> {w['run_dir'] / EVENTS_FILE}")
    print(f"Optional: expectation-and-outcome pairs of event names -> {w['run_dir'] / PAIRS_FILE}")


def sealed_events(run_dir: Path) -> dict[str, str]:
    from src.fulltext import article_id
    from src.summary_run import SummaryRunError, load_events
    try:
        by_url = load_events(Path(run_dir) / EVENTS_FILE)
    except SummaryRunError as error:
        raise Week2Error(f"{EVENTS_FILE}: {error} Every relevant article needs an event name.") from error
    return {article_id(url): event for url, event in by_url.items()}


def _read_pairs(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / PAIRS_FILE
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _seal_command(w: dict) -> None:
    from src.transfer_eval import seal_problems
    if (w["run_dir"] / SEAL_FILE).exists():
        raise Week2Error(f"{SEAL_FILE} already exists: the week is sealed.")
    events = sealed_events(w["run_dir"])
    found = seal_problems(read_labels(w["run_dir"]), events)
    found += pairs_problems(_read_pairs(w["run_dir"]), set(events.values()))
    if found:
        raise Week2Error("Not complete:\n" + "\n".join(found))
    with open(w["run_dir"] / SEAL_FILE, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(seal_record(w["run_dir"]), indent=1))
    print(f"Sealed: {w['run_dir'] / SEAL_FILE}  (commit it before any model runs on this week)")


def _candidates(w: dict, state: dict, aliases: dict) -> list[dict]:
    from src.fulltext import FullTextFetcher
    from src.ranking_eval import stage1_candidates
    articles = kept_articles(w, state)
    fetcher, texts = FullTextFetcher(), {}
    for article in articles:                                  # the cached pages of the sheet step
        result = fetcher.fetch(article["url"])
        if result.get("status") == "ok" and result.get("text"):
            texts[article["url"]] = result["text"]
    return stage1_candidates(articles, texts, w["company"], aliases)


def _rank_command(w: dict) -> None:
    from src.query import load_aliases
    path = w["run_dir"] / RANK_FILE
    if path.exists():
        raise Week2Error(f"{RANK_FILE} already exists: the ranking test is made once.")
    seal_path = w["run_dir"] / SEAL_FILE
    if not seal_path.exists():
        raise Week2Error("The week is not sealed: python -m src.week2_eval <week> seal")
    changed = seal_changes(json.loads(seal_path.read_text(encoding="utf-8")), w["run_dir"])
    if changed:
        raise Week2Error("Changed after the seal: " + ", ".join(changed))
    protocol_text = _protocol_text(w)
    if _tag_text(w) is None or _tag_text(w).replace("\r\n", "\n") != protocol_text.replace("\r\n", "\n"):
        raise Week2Error(f"{PROTOCOL_FILE} differs from the tag {w['tag']}.")
    aliases = load_aliases(ALIAS_FILE)
    state = load_state(w["run_dir"])
    result = ranking_result(_candidates(w, state, aliases), w["company"], aliases, read_labels(w["run_dir"]),
                            sealed_events(w["run_dir"]))
    out = {"week": w["name"], "company": w["company"], "tested": DEFAULT, "rule": RANKING_RULE, **result,
           "seal": seal_record(w["run_dir"]) | {"sealed_at": None},
           "code": _git("rev-parse", "HEAD"), "scored": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(out, ensure_ascii=False, indent=1))
    p = result["primary"]
    print(f"{w['company']}: {result['candidates']} candidates")
    print(f"Primary, O1M5+S1 against the retrieval order at k = 10: {p['conditions']} -> passed {p['passed']}")
    print(f"Second (reported only), O1M5+S1 against O1+S1: {result['second']['conditions']}")
    print(f"Relevant articles without the company in the title: {len(result['untitled_relevant'])}; in O1M5+S1's top "
          f"10: {sum(a['in_top_10'][DEFAULT] for a in result['untitled_relevant'])}, in O1+S1's: "
          f"{sum(a['in_top_10'][OLD_DEFAULT] for a in result['untitled_relevant'])}")
    print(f"Saved: {path}")


COMMANDS = {"protocol": _protocol_command, "fetch": _fetch_command, "overlap": _overlap_command,
            "sheet": _sheet_command, "events": _events_command, "seal": _seal_command, "rank": _rank_command}

if __name__ == "__main__":
    try:
        if len(sys.argv) == 3 and sys.argv[2] in COMMANDS:
            COMMANDS[sys.argv[2]](week(sys.argv[1]))
        else:
            print(__doc__)
    except Week2Error as error:
        print(f"[STOP] {error}")
        sys.exit(1)
