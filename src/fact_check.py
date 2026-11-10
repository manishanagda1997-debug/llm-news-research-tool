"""
The hand check of important facts (Stage 4.1b): which input should the
summary model read for a long article?

The check covers the relevant dev articles judged on full text that are
longer than 3,500 characters (shorter ones are given whole by every method),
except an article whose saved text fails the wrong-page check (the saved page
holds another story): it has no real full text to select from.
Three inputs are compared:

    short     the NewsAPI short text (title, description, first 200 characters)
    passage   the relevance passage (src/relevance_check.py)
    evidence  the summary evidence (src/summary_evidence.py)

Steps (each refuses to overwrite a file or to run out of order):
    python -m src.fact_check facts    reading page with the full articles and an
                                      empty fact sheet (8 slots per article)
    python -m src.fact_check inputs   after the fact sheet is complete: the fact
                                      list is fingerprinted, then a page with the
                                      three inputs as X, Y and Z (shuffled per
                                      article by a fixed seed, stored nowhere) and
                                      a marks sheet
    python -m src.fact_check score    un-blinds, scores and applies the choice rule

Facts are the important facts for an analyst who follows the company (Tata
Motors and its group): main development, who is involved, quantities, status
and timing, cause, impact and risks, qualifications. In a round-up or a live
blog, only the parts about the company count.

Each fact has a scope: company (Tata Motors, its businesses and its deals)
or context (industry, market or competitor facts that help explain the
development). Scope and central are separate: a context fact can be central,
and a company fact can be supporting.

Marks: kept yes / partly / no; distorted yes / no (the meaning changes, for
example "not approved" becomes "approved"); a fact that is not kept cannot be
distorted. Only yes counts as kept; partly and no are counted on their own.

Scores: for each input, central, company and context facts, each with the
counts of yes, partly, no and distorted, and the share kept in full.

Choice: an input that distorts a central fact is out; then the highest share
of central facts kept, then of company facts, then fewer other distortions,
then the highest share of context facts; on a tie, the relevance passage,
because it already exists. If every input distorts a central fact, none is
chosen and the selection is fixed first.

Publisher text stays in data/fulltext/ (outside Git); Git keeps the fact list,
the marks, the record and the score.
"""

import csv
import hashlib
import html as html_module
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

METHODS = ("short", "passage", "evidence")
LETTERS = ("X", "Y", "Z")
CATEGORIES = ("main development", "who is involved", "quantities", "status and timing", "cause",
              "impact and risks", "qualifications")
SLOTS = 8
SEED = 42
LIMIT = 3500
KEPT_VALUES = ("yes", "partly", "no")
SCOPES = ("company", "context")
GROUPS = ("central", "company", "context", "all")
FACT_COLUMNS = ["article_id", "title", "fact_no", "category", "fact", "central", "scope", "note"]
MARK_COLUMNS = ["article_id", "fact_no", "fact", "input", "kept", "distorted", "note"]
FACTS_FILE = "fact_check_facts.csv"
MARKS_FILE = "fact_check_marks.csv"
RECORD_FILE = "fact_check_record.json"
SCORE_FILE = "fact_check_score.json"
READING_PAGE = "fact_check_reading.html"
INPUTS_PAGE = "fact_check_inputs.html"


class FactCheckError(Exception):
    pass


# ---------- the sample ----------

def sample(rows: list[dict], lengths: dict[str, int], limit: int = LIMIT, wrong_page: set | None = None) -> list[dict]:
    """Relevant full-text articles over the limit, except those whose saved text fails the wrong-page check."""
    wrong_page = wrong_page or set()
    return [r for r in rows if r["basis"] == "full text" and r["label"] in ("direct", "indirect")
            and lengths.get(r["article_id"], 0) > limit and r["article_id"] not in wrong_page]


def facts_for(facts: list[dict], article_ids: list[str]) -> list[dict]:
    """The facts of the sample articles; other rows stay in the sheet, untouched and unused."""
    ids = set(article_ids)
    return [f for f in facts if f["article_id"] in ids]


# ---------- step A: facts ----------

def fact_rows(articles: list[dict]) -> list[dict]:
    return [{"article_id": a["article_id"], "title": a["title"], "fact_no": str(n), "category": "", "fact": "",
             "central": "", "scope": "", "note": ""} for a in articles for n in range(1, SLOTS + 1)]


def _value(text) -> str:
    return (text or "").strip().lower()


def filled(facts: list[dict]) -> list[dict]:
    return [f for f in facts if (f.get("fact") or "").strip()]


def fact_problems(facts: list[dict], article_ids: list[str]) -> list[str]:
    problems = []
    rows = filled(facts)
    for f in rows:
        where = f"{f['article_id']} fact {f['fact_no']}"
        if _value(f.get("category")) not in CATEGORIES:
            problems.append(f"{where}: the category must be one of the seven categories")
        if _value(f.get("central")) not in ("yes", "no"):
            problems.append(f"{where}: central must be yes or no")
        if _value(f.get("scope")) not in SCOPES:
            problems.append(f"{where}: scope must be company or context")
    for article_id in article_ids:
        mine = [f for f in rows if f["article_id"] == article_id]
        if not mine:
            problems.append(f"{article_id}: no fact listed")
        elif not any(_value(f.get("central")) == "yes" for f in mine):
            problems.append(f"{article_id}: no central fact")
    return problems


def facts_fingerprint(facts: list[dict]) -> str:
    items = sorted([f["article_id"], f["fact_no"], _value(f.get("category")), (f.get("fact") or "").strip(),
                    _value(f.get("central")), _value(f.get("scope"))] for f in filled(facts))
    return hashlib.sha256(json.dumps(items, ensure_ascii=False).encode("utf-8")).hexdigest()


def reading_page(articles: list[dict]) -> str:
    e = html_module.escape
    parts = [_page_head("Fact check: full articles"),
             "<h1>Fact check: full articles</h1><p class='meta'>List the important facts of each article in "
             f"{FACTS_FILE}: category, fact in a few words, central yes or no. Only the parts about Tata Motors "
             "and its group count.</p>"]
    for a in articles:
        parts.append(f"<section><h2>{e(a['article_id'])} — {e(a['title'])}</h2>"
                     + "".join(f"<p>{e(p)}</p>" for p in a["paragraphs"]) + "</section>")
    return "".join(parts) + "</body></html>"


# ---------- step B: blind inputs ----------

def xyz_order(article_id: str, seed: int = SEED) -> dict[str, str]:
    """Letter -> method, fixed for each article by the seed; never written to a file."""
    methods = random.Random(f"{seed}-{article_id}").sample(list(METHODS), len(METHODS))
    return dict(zip(LETTERS, methods))


def mark_rows(facts: list[dict]) -> list[dict]:
    return [{"article_id": f["article_id"], "fact_no": f["fact_no"], "fact": f["fact"].strip(), "input": letter,
             "kept": "", "distorted": "", "note": ""} for f in filled(facts) for letter in LETTERS]


def _page_head(title: str) -> str:
    return ("<!doctype html><html><head><meta charset='utf-8'><title>" + html_module.escape(title) + "</title>"
            "<style>body{font-family:Georgia,serif;max-width:52rem;margin:2rem auto;padding:0 1rem;line-height:1.55}"
            "section{border-top:2px solid #444;margin-top:2.5rem;padding-top:1rem}"
            ".meta{font-family:Arial,sans-serif;font-size:.85rem;color:#555}"
            ".box{border:1px solid #bbb;padding:.6rem 1rem;margin:.8rem 0;white-space:pre-wrap}"
            "h3{font-family:Arial,sans-serif}</style></head><body>")


def inputs_page(items: list[dict]) -> str:
    e = html_module.escape
    parts = [_page_head("Fact check: inputs"),
             f"<h1>Fact check: inputs</h1><p class='meta'>For each fact and each text X, Y and Z, write in "
             f"{MARKS_FILE}: kept yes / partly / no, and distorted yes / no.</p>"]
    for item in items:
        facts = "".join(f"<li>{e(f['fact_no'])}. {e(f['fact'].strip())}"
                        f"{' (central)' if _value(f.get('central')) == 'yes' else ''}</li>"
                        for f in filled(item["facts"]))
        inputs = item["inputs"]
        names = [l for l in LETTERS if l in inputs] if set(inputs) <= set(LETTERS) else list(inputs)
        texts = "".join(f"<h3>{e(name)}</h3><div class='box'>{e(inputs[name])}</div>" for name in names)
        parts.append(f"<section><h2>{e(item['article_id'])} — {e(item['title'])}</h2><ol>{facts}</ol>{texts}"
                     "</section>")
    return "".join(parts) + "</body></html>"


# ---------- step C: marks, score and choice ----------

def mark_problems(marks: list[dict]) -> list[str]:
    problems = []
    for m in marks:
        where = f"{m['article_id']} fact {m['fact_no']} input {m['input']}"
        kept, distorted = _value(m.get("kept")), _value(m.get("distorted"))
        if kept not in KEPT_VALUES:
            problems.append(f"{where}: kept must be yes, partly or no")
        if distorted not in ("yes", "no"):
            problems.append(f"{where}: distorted must be yes or no")
        if kept == "no" and distorted == "yes":
            problems.append(f"{where}: a fact that is not kept cannot be distorted")
    return problems


def _empty_group() -> dict:
    return {"facts": 0, "yes": 0, "partly": 0, "no": 0, "distorted": 0}


def score(facts: list[dict], marks: list[dict]) -> dict[str, dict]:
    """
    For each input: central, company, context and all facts, each with its counts of yes, partly, no and
    distorted, and `kept`, the share of facts kept in full (yes only; None when the group has no facts).
    Distortions are counted for central facts and for the other facts.
    """
    rows = filled(facts)
    expected = {(f["article_id"], f["fact_no"], letter) for f in rows for letter in LETTERS}
    given = {(m["article_id"], m["fact_no"], m["input"]) for m in marks}
    if given != expected or len(marks) != len(expected):
        raise FactCheckError("The marks must hold every fact against X, Y and Z, each once.")
    problems = mark_problems(marks)
    if problems:
        raise FactCheckError("Marks to fix:\n  " + "\n  ".join(problems))
    by_fact = {(f["article_id"], f["fact_no"]): f for f in rows}
    results = {method: {**{group: _empty_group() for group in GROUPS}, "central_distortions": 0,
                        "other_distortions": 0} for method in METHODS}
    for m in marks:
        method = xyz_order(m["article_id"])[m["input"]]
        fact = by_fact[(m["article_id"], m["fact_no"])]
        is_central = _value(fact["central"]) == "yes"
        groups = ["all", _value(fact["scope"])] + (["central"] if is_central else [])
        kept, distorted = _value(m["kept"]), _value(m["distorted"]) == "yes"
        r = results[method]
        for group in groups:
            r[group]["facts"] += 1
            r[group][kept] += 1
            r[group]["distorted"] += int(distorted)
        if distorted:
            r["central_distortions" if is_central else "other_distortions"] += 1
    for r in results.values():
        for group in GROUPS:
            g = r[group]
            g["kept"] = g["yes"] / g["facts"] if g["facts"] else None
    return results


def choose_input(scores: dict[str, dict]) -> str | None:
    """
    An input that distorts a central fact is out. Then the highest share of central facts kept, then of company
    facts, then fewer other distortions, then the highest share of context facts; on a tie, the relevance passage.
    """
    allowed = [m for m in METHODS if scores[m]["central_distortions"] == 0]
    if not allowed:
        return None
    share = lambda m, group: scores[m][group]["kept"] or 0.0
    return min(allowed, key=lambda m: (-share(m, "central"), -share(m, "company"), scores[m]["other_distortions"],
                                       -share(m, "context"), m != "passage"))


# ---------- Step A: the summary passage, version 2 ----------
#
# Only articles whose summary passage differs from the relevance passage are marked again, against the new
# passage alone (input P). Unchanged articles keep their passage marks. The new passage is accepted only when
# it keeps every fact the old one kept in full, adds no distortion, and stays within the length limit.

PASSAGE2_LETTER = "P"
PASSAGE2_MARKS_FILE = "fact_check_marks_passage2.csv"
PASSAGE2_RECORD_FILE = "fact_check_passage2_record.json"
PASSAGE2_SCORE_FILE = "fact_check_passage2_score.json"
PASSAGE2_PAGE = "fact_check_passage2.html"
ADDED_FILE = "fact_check_passage2_added.csv"
ADDED_COLUMNS = ["article_id", "added_text", "attribution_correct", "meaning_preserved", "note"]


def passage2_mark_rows(facts: list[dict], changed: set[str]) -> list[dict]:
    return [{"article_id": f["article_id"], "fact_no": f["fact_no"], "fact": f["fact"], "input": PASSAGE2_LETTER,
             "kept": "", "distorted": "", "note": ""} for f in filled(facts) if f["article_id"] in changed]


def added_rows(changed: list[dict]) -> list[dict]:
    """One row per added paragraph: does it keep the speaker, the company and the meaning?"""
    return [{"article_id": c["article_id"], "added_text": text, "attribution_correct": "", "meaning_preserved": "",
             "note": ""} for c in changed for text in c["added"]]


def added_problems(added: list[dict]) -> list[str]:
    return [f"{a['article_id']}: attribution_correct and meaning_preserved must be yes or no"
            for a in added if _value(a.get("attribution_correct")) not in ("yes", "no")
            or _value(a.get("meaning_preserved")) not in ("yes", "no")]


def _passage_kept(old_marks: list[dict]) -> dict[tuple, dict]:
    """The first check's marks for the relevance passage, by (article, fact)."""
    return {(m["article_id"], m["fact_no"]): m for m in old_marks
            if xyz_order(m["article_id"])[m["input"]] == "passage"}


def _group_counts(rows: list[dict], marks: dict[tuple, dict]) -> dict:
    groups = {group: _empty_group() for group in GROUPS}
    for f in rows:
        m = marks[(f["article_id"], f["fact_no"])]
        kept, distorted = _value(m["kept"]), _value(m["distorted"]) == "yes"
        for group in ["all", _value(f["scope"])] + (["central"] if _value(f["central"]) == "yes" else []):
            groups[group]["facts"] += 1
            groups[group][kept] += 1
            groups[group]["distorted"] += int(distorted)
    for g in groups.values():
        g["kept"] = g["yes"] / g["facts"] if g["facts"] else None
    return groups


def passage2_compare(facts: list[dict], old_marks: list[dict], new_marks: list[dict], changed: set[str],
                     lengths: dict[str, int], added: list[dict], limit: int | None = None) -> dict:
    from src.relevance_check import SUMMARY_MAX_CHARS
    limit = limit or SUMMARY_MAX_CHARS
    rows = filled(facts)
    old = _passage_kept(old_marks)
    problems = mark_problems(new_marks) + added_problems(added)
    expected = {(f["article_id"], f["fact_no"]) for f in rows if f["article_id"] in changed}
    if {(m["article_id"], m["fact_no"]) for m in new_marks} != expected or len(new_marks) != len(expected):
        problems.append("the new marks must hold every fact of every changed article, each once")
    if problems:
        raise FactCheckError("Marks to fix:\n  " + "\n  ".join(problems))
    new = {**old, **{(m["article_id"], m["fact_no"]): m for m in new_marks}}
    gained = sorted([list(k) for k in expected if _value(old[k]["kept"]) != "yes" and _value(new[k]["kept"]) == "yes"])
    lost = sorted([list(k) for k in expected if _value(old[k]["kept"]) == "yes" and _value(new[k]["kept"]) != "yes"])
    distorted = sorted([list(k) for k in expected if _value(new[k]["distorted"]) == "yes"])
    too_long = sorted(a for a in changed if lengths.get(a, 0) > limit)
    bad_added = [a["article_id"] for a in added if _value(a["attribution_correct"]) != "yes"
                 or _value(a["meaning_preserved"]) != "yes"]
    before, after = _group_counts(rows, old), _group_counts(rows, new)
    refused = ([("lost", lost)] if lost else []) + ([("distortions", distorted)] if distorted else []) \
        + ([("length", too_long)] if too_long else []) \
        + ([("added text", bad_added)] if bad_added else []) \
        + ([("central", [])] if after["central"]["yes"] < before["central"]["yes"] else [])
    return {"old": before, "new": after, "changed_articles": sorted(changed), "gained": gained, "lost": lost,
            "distorted": distorted, "too_long": too_long, "added_text_problems": bad_added, "accepted": not refused,
            "refused_because": [name for name, _ in refused]}


# ---------- files ----------

def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    if path.exists():
        raise FactCheckError(f"{path.name} already exists; it is not overwritten.")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FactCheckError(f"{path.name} already exists; it is not overwritten.")
    path.write_text(text, encoding="utf-8")


def _articles() -> list[dict]:
    """The sample, each with its full text, title and short text."""
    from src.config import EVAL_DIR
    from src.fulltext import FULLTEXT_DIR
    from src.fulltext import _paragraphs, article_id
    from src.recheck import FULLTEXT_SHEET, LABEL_FILE
    eval_dir, fulltext_dir = Path(EVAL_DIR), Path(FULLTEXT_DIR)
    fetched = {r["article_id"]: r["url"] for r in _read_csv(eval_dir / FULLTEXT_SHEET)}
    texts = {}
    for i, url in fetched.items():
        path = fulltext_dir / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.json"
        if path.exists():
            texts[i] = json.loads(path.read_text(encoding="utf-8")).get("text", "")
    short = {}
    for path in sorted(eval_dir.glob("dev_*.csv")):
        for row in _read_csv(path):
            short.setdefault(article_id(row["url"]), row)
    from src.config import DATA_DIR
    from src.query import load_aliases
    from src.recheck import saved_text_is_another_story
    aliases = load_aliases(Path(DATA_DIR) / "company_aliases.json")
    rows = _read_csv(eval_dir / LABEL_FILE)
    wrong_page = {r["article_id"] for r in rows if r["article_id"] in texts and saved_text_is_another_story(
        r["title"], short.get(r["article_id"], {}).get("description", ""), texts[r["article_id"]], aliases)}
    chosen = sample(rows, {i: len(t) for i, t in texts.items()}, wrong_page=wrong_page)
    return [{"article_id": r["article_id"], "title": r["title"], "text": texts[r["article_id"]],
             "paragraphs": _paragraphs(texts[r["article_id"]]), "short": short.get(r["article_id"], {})}
            for r in chosen]


def _inputs_for(article: dict, aliases) -> dict[str, str]:
    from src.relevance_check import build_passage
    from src.summary_evidence import build_summary_evidence
    s = article["short"]
    short = "\n".join(p for p in (article["title"], s.get("description", ""), s.get("content_first_200_chars", ""))
                      if p and p.strip())
    return {"short": short,
            "passage": build_passage(article["title"], article["text"], "Tata Motors", aliases)["text"],
            "evidence": build_summary_evidence(article["title"], article["text"], "Tata Motors", aliases)["text"]}


def _paths():
    from src.config import EVAL_DIR
    from src.fulltext import FULLTEXT_DIR
    return Path(EVAL_DIR), Path(FULLTEXT_DIR)


def _facts_command() -> None:
    eval_dir, fulltext_dir = _paths()
    articles = _articles()
    _write_csv(eval_dir / FACTS_FILE, fact_rows(articles), FACT_COLUMNS)
    _write_new(fulltext_dir / READING_PAGE, reading_page(articles))
    print(f"{len(articles)} articles")
    print(f"Fact sheet:   {eval_dir / FACTS_FILE}")
    print(f"Reading page: {fulltext_dir / READING_PAGE}  (open it in your browser)")


def _inputs_command() -> None:
    from src.config import DATA_DIR
    from src.query import load_aliases
    eval_dir, fulltext_dir = _paths()
    articles = _articles()
    all_facts = _read_csv(eval_dir / FACTS_FILE)
    facts = facts_for(all_facts, [a["article_id"] for a in articles])
    ignored = sorted({f["article_id"] for f in filled(all_facts)} - {a["article_id"] for a in articles})
    problems = fact_problems(facts, [a["article_id"] for a in articles])
    if problems:
        raise FactCheckError("The fact sheet is not complete:\n  " + "\n  ".join(problems))
    aliases = load_aliases(Path(DATA_DIR) / "company_aliases.json")
    items, fingerprints = [], {}
    for a in articles:
        by_method = _inputs_for(a, aliases)
        order = xyz_order(a["article_id"])
        fingerprints[a["article_id"]] = {m: hashlib.sha256(t.encode("utf-8")).hexdigest()
                                         for m, t in by_method.items()}
        items.append({"article_id": a["article_id"], "title": a["title"],
                      "facts": [f for f in facts if f["article_id"] == a["article_id"]],
                      "inputs": {letter: by_method[order[letter]] for letter in LETTERS}})
    _write_csv(eval_dir / MARKS_FILE, mark_rows(facts), MARK_COLUMNS)
    _write_new(fulltext_dir / INPUTS_PAGE, inputs_page(items))
    record = {"articles": [a["article_id"] for a in articles], "ignored_articles": {
                  i: "the saved text holds another story (wrong-page check), so it is not a long full-text article"
                  for i in ignored}, "seed": SEED,
              "facts_fingerprint": facts_fingerprint(facts), "input_fingerprints": fingerprints,
              "inputs_made": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    _write_new(eval_dir / RECORD_FILE, json.dumps(record, indent=1))
    print(f"Marks sheet: {eval_dir / MARKS_FILE}")
    print(f"Inputs page: {fulltext_dir / INPUTS_PAGE}  (open it in your browser)")


def _score_command() -> None:
    eval_dir, _ = _paths()
    record = json.loads((eval_dir / RECORD_FILE).read_text(encoding="utf-8"))
    facts = facts_for(_read_csv(eval_dir / FACTS_FILE), record["articles"])
    if facts_fingerprint(facts) != record["facts_fingerprint"]:
        raise FactCheckError("The fact list differs from the one fingerprinted before the inputs were shown. "
                             "Restore it, and record any correction in a separate file with its reason.")
    scores = score(facts, _read_csv(eval_dir / MARKS_FILE))
    choice = choose_input(scores)
    for method, s in scores.items():
        parts = []
        for group in ("central", "company", "context"):
            g = s[group]
            share = "-" if g["kept"] is None else f"{g['kept']:.0%}"
            parts.append(f"{group} {g['yes']}/{g['facts']} kept ({share}), partly {g['partly']}, no {g['no']}")
        print(f"{method:<9} " + "; ".join(parts)
              + f"; distortions: central {s['central_distortions']}, other {s['other_distortions']}")
    print(f"Chosen input: {choice or 'none: every input distorts a central fact'}")
    _write_new(eval_dir / SCORE_FILE, json.dumps({"scores": scores, "choice": choice,
                                                  "facts_fingerprint": record["facts_fingerprint"]}, indent=1))


def _changed_passages(aliases) -> list[dict]:
    from src.relevance_check import build_passage, build_summary_passage
    changed = []
    for a in _articles():
        old = build_passage(a["title"], a["text"], "Tata Motors", aliases)["text"]
        new = build_summary_passage(a["title"], a["text"], "Tata Motors", aliases)
        if new["text"] != old or new["not_fitted"]:
            changed.append({"article_id": a["article_id"], "title": a["title"], "old": old, "new": new["text"],
                            "added": new["added_text"], "not_fitted": new["not_fitted"]})
    return changed


def passage2_files(changed: list[dict], facts: list[dict], facts_fp: str, found: list[dict] | None = None) -> dict:
    """Every file of Step A, built in memory: {file name: rows or text}. Nothing is written here."""
    found = found if found is not None else changed
    ids = {c["article_id"] for c in changed}
    items = [{"article_id": c["article_id"], "title": c["title"],
              "facts": [f for f in filled(facts) if f["article_id"] == c["article_id"]],
              "inputs": {PASSAGE2_LETTER: c["new"], "Added text (also inside P)": "\n\n".join(c["added"])}}
             for c in changed]
    record = {"changed_articles": sorted(ids), "lengths": {c["article_id"]: len(c["new"]) for c in changed},
              "new_passage_sha256": {c["article_id"]: hashlib.sha256(c["new"].encode("utf-8")).hexdigest()
                                     for c in changed},
              "not_fitted": {c["article_id"]: c["not_fitted"] for c in found if c["not_fitted"]},
              "facts_fingerprint": facts_fp, "made": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    return {PASSAGE2_MARKS_FILE: passage2_mark_rows(facts, ids), ADDED_FILE: added_rows(changed),
            PASSAGE2_PAGE: inputs_page(items), PASSAGE2_RECORD_FILE: json.dumps(record, indent=1)}


def _passage2_command() -> None:
    """The page and marks sheets for the changed summary passages (input P); files are written only at the end."""
    from src.query import load_aliases
    from src.config import DATA_DIR
    eval_dir, fulltext_dir = _paths()
    if not (eval_dir / SCORE_FILE).exists():
        raise FactCheckError("The first fact check is scored first: python -m src.fact_check score")
    record = json.loads((eval_dir / RECORD_FILE).read_text(encoding="utf-8"))
    facts = facts_for(_read_csv(eval_dir / FACTS_FILE), record["articles"])
    if facts_fingerprint(facts) != record["facts_fingerprint"]:
        raise FactCheckError("The fact list differs from the fingerprinted one.")
    found = [c for c in _changed_passages(load_aliases(Path(DATA_DIR) / "company_aliases.json"))
             if c["article_id"] in record["articles"]]
    changed = [c for c in found if c["added"]]
    files = passage2_files(changed, facts, record["facts_fingerprint"], found)
    targets = {PASSAGE2_MARKS_FILE: eval_dir, ADDED_FILE: eval_dir, PASSAGE2_PAGE: fulltext_dir,
               PASSAGE2_RECORD_FILE: eval_dir}
    existing = [name for name, folder in targets.items() if (folder / name).exists()]
    if existing:
        raise FactCheckError(f"Already made: {', '.join(existing)}; nothing is overwritten.")
    _write_csv(eval_dir / PASSAGE2_MARKS_FILE, files[PASSAGE2_MARKS_FILE], MARK_COLUMNS)
    _write_csv(eval_dir / ADDED_FILE, files[ADDED_FILE], ADDED_COLUMNS)
    _write_new(fulltext_dir / PASSAGE2_PAGE, files[PASSAGE2_PAGE])
    _write_new(eval_dir / PASSAGE2_RECORD_FILE, files[PASSAGE2_RECORD_FILE])
    print(f"{len(changed)} of {len(record['articles'])} passages change: {sorted(c['article_id'] for c in changed)}")
    for c in changed:
        print(f"   {c['article_id']} ({len(c['old'])} -> {len(c['new'])} characters, {len(c['added'])} added) "
              f"{c['title'][:55]}")
    for c in found:
        if c["not_fitted"]:
            print(f"   {c['article_id']}: {c['not_fitted']} continued quote(s) did not fit the hard cap (recorded)")
    print(f"Marks sheet: {eval_dir / PASSAGE2_MARKS_FILE}  ({len(files[PASSAGE2_MARKS_FILE])} rows)")
    print(f"Added-text sheet: {eval_dir / ADDED_FILE}  ({len(files[ADDED_FILE])} rows)")
    print(f"Page: {fulltext_dir / PASSAGE2_PAGE}  (open it in your browser)")


def _passage2_score_command() -> None:
    eval_dir, _ = _paths()
    record = json.loads((eval_dir / RECORD_FILE).read_text(encoding="utf-8"))
    step = json.loads((eval_dir / PASSAGE2_RECORD_FILE).read_text(encoding="utf-8"))
    facts = facts_for(_read_csv(eval_dir / FACTS_FILE), record["articles"])
    result = passage2_compare(facts, _read_csv(eval_dir / MARKS_FILE), _read_csv(eval_dir / PASSAGE2_MARKS_FILE),
                              set(step["changed_articles"]), step["lengths"], _read_csv(eval_dir / ADDED_FILE))
    for name in ("old", "new"):
        g = result[name]
        print(f"{name:<4} central {g['central']['yes']}/{g['central']['facts']}, company {g['company']['yes']}/"
              f"{g['company']['facts']}, context {g['context']['yes']}/{g['context']['facts']} kept in full; "
              f"distorted {g['all']['distorted']}")
    print(f"gained {result['gained']}  lost {result['lost']}  distorted {result['distorted']}  "
          f"too long {result['too_long']}  added-text problems {result['added_text_problems']}")
    print("Accepted: the summaries use passage version 2" if result["accepted"]
          else f"Refused ({', '.join(result['refused_because'])}): the summaries use the relevance passage")
    _write_new(eval_dir / PASSAGE2_SCORE_FILE, json.dumps(result, indent=1))


COMMANDS = {"facts": _facts_command, "inputs": _inputs_command, "score": _score_command,
            "passage2": _passage2_command, "passage2-score": _passage2_score_command}

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in COMMANDS:
        COMMANDS[sys.argv[1]]()
    else:
        print(__doc__)
