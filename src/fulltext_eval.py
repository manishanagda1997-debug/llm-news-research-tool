"""
The relevance filter on short text and on full text, scored on the same labels.

The same filter (query rules, verified aliases, junk removal, sale-listing
check: the cleaning step itself) decides each labelled article twice:
    short - on the text NewsAPI gives (title, description, first 200 characters)
    full  - on the title, description and cleaned full text when the page is
            fetched and passes the sanity check (src/fulltext.py,
            check_full_text); otherwise on the short text
Both are scored against the labels of guide version 3, so the difference
between the two comes from the text alone.

With model checks (stage 2, src/relevance_check.py), a third row is scored:
    full+check - articles kept on full text, then checked by the model;
                 borderline (not + unsure) and fallback articles are kept
                 and counted separately

Two kinds of numbers:
    article level - each labelled article is one record (precision, recall, F1)
    story level   - copies of the same story (same link, same or nearly the
                    same title) form one story; a relevant story is covered
                    when at least one copy is kept

Acceptance checks for stage 2 (broad labels), fixed before the version 2 run:
    recall_kept - relevant articles kept after stage 2 / relevant articles stage 1 keeps   >= 43/45
    stories     - relevant stories covered after stage 2 / all relevant stories             >= 41/43
    fp_removed  - stage 1 false positives dropped by stage 2 / all stage 1 false positives  >= 2/3
The required count is the total times the threshold, rounded up. With fewer
than 20 articles, 20 stories or 8 false positives, a check is "too_few".

Run:
    python -m src.fulltext_eval            (no network, no API keys)
    python -m src.fulltext_eval --check    (stage 2: Gemini calls, answers saved and reused)
"""

import json
import math
import sys
import time
from difflib import SequenceMatcher
from fractions import Fraction

from src.config import DATA_DIR, EVAL_DIR, RAW_DIR
from src.evaluate import confusion, is_relevant, load_label_set, scores
from src.preprocess import NEAR_DUPLICATE_MIN_LENGTH, NEAR_DUPLICATE_RATIO, _title_key, clean_articles, normalize_url
from src.query import load_aliases

LABEL_FILE = "v3_dev_tata_motors.csv"
CHECKED_LABEL_FILE = "v3_dev_tata_motors_checked.csv"     # after the blind re-check (src/recheck.py)
STAGE1_FILE = "stage1_label_versions.json"
TRACKED_FILE = "stage2_tracked_v1.json"
ANSWERS_V1_FILE = "stage2_answers_v1.json"
RECORD_FILE = "stage2_dev_v2_record.json"
THRESHOLDS = {"recall_kept": (Fraction(43, 45), 20),
              "stories": (Fraction(41, 43), 20),
              "fp_removed": (Fraction(2, 3), 8)}
DECISIONS = ("kept", "dropped", "borderline", "fallback")


class FulltextEvalError(Exception):
    pass


def keeps(raw: dict, full_text: str | None, company: str, aliases: dict | None) -> bool:
    """Does the filter keep this article? With full text, the text it checks also holds the full article."""
    article = dict(raw)
    if full_text:
        # added to the description: the cleaning step drops a content field that repeats the description
        article["description"] = f"{raw.get('description') or ''} {full_text}".strip()
    kept, _ = clean_articles([article], max_articles=1, query=company, aliases=aliases,
                             remove_junk=True, drop_listings="both")
    return len(kept) == 1


def stories(unique: dict[str, dict]) -> list[set[str]]:
    """Groups of copies of the same story, by the duplicate rules of cleaning."""
    keys = list(unique)
    parent = {key: key for key in keys}

    def root(key):
        while parent[key] != key:
            key = parent[key]
        return key

    titles = {key: _title_key(unique[key].get("title") or "") for key in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            same = titles[a] == titles[b] or (
                min(len(titles[a]), len(titles[b])) >= NEAR_DUPLICATE_MIN_LENGTH
                and SequenceMatcher(None, titles[a], titles[b]).ratio() >= NEAR_DUPLICATE_RATIO)
            if same:
                parent[root(b)] = root(a)
    groups = {}
    for key in keys:
        groups.setdefault(root(key), set()).add(key)
    return list(groups.values())


def usable_full_texts(articles: list[dict], full_texts: dict[str, str], company: str,
                      aliases: dict | None) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Full texts that pass the sanity check, and the reasons for each one that does not."""
    from src.fulltext import check_full_text
    by_key = {normalize_url(url): text for url, text in full_texts.items() if text}
    texts, fallbacks = {}, {}
    for article in articles:
        url = article.get("url") or ""
        text = by_key.get(normalize_url(url))
        if not text or url in texts or url in fallbacks:
            continue
        check = check_full_text(article.get("title") or "", article.get("description") or "", text, company, aliases,
                                content=article.get("content") or "")
        if check["use_full_text"]:
            texts[url] = text
        else:
            fallbacks[url] = check["reasons"]
    return texts, fallbacks


def required_count(total: int, threshold: Fraction) -> int:
    """The smallest whole number of articles that reaches the threshold."""
    return math.ceil(total * threshold)


def judge(count: int, total: int, threshold: Fraction, minimum: int) -> dict:
    required = required_count(total, threshold)
    status = "too_few" if total < minimum else ("pass" if count >= required else "fail")
    return {"status": status, "count": count, "total": total, "required": required}


def load_tracked(path=None) -> dict[str, list[str]]:
    data = json.loads((path or EVAL_DIR / TRACKED_FILE).read_text(encoding="utf-8"))
    return {"false_negatives": data["false_negatives"], "false_positives": data["false_positives"]}


def tracked_status(tracked: dict[str, list[str]], kept_ids: set[str], known_ids: set[str]) -> dict:
    missing = [i for ids in tracked.values() for i in ids if i not in known_ids]
    if missing:
        raise FulltextEvalError(f"Tracked article(s) missing from this run: {', '.join(missing)}")
    return {group: {i: ("kept" if i in kept_ids else "dropped") for i in ids} for group, ids in tracked.items()}


def compare(articles: list[dict], labels: dict[str, str], full_texts: dict[str, str], company: str,
            aliases: dict | None, strict: bool = False, checks: dict[str, dict] | None = None) -> dict:
    from src.fulltext import article_id
    from src.relevance_check import decide
    labels = {normalize_url(url): label for url, label in labels.items()}
    unique = {}
    for article in articles:
        unique.setdefault(normalize_url(article.get("url") or ""), article)
    missing = [key for key in unique if key not in labels]
    if missing:
        raise FulltextEvalError(f"{len(missing)} article(s) have no label, for example {missing[0]}.")

    usable, fallbacks = usable_full_texts(list(unique.values()), full_texts, company, aliases)
    full_texts = {normalize_url(url): text for url, text in usable.items()}
    fallbacks = {normalize_url(url): reasons for url, reasons in fallbacks.items()}
    checks = {normalize_url(url): value for url, value in (checks or {}).items()}
    ids = {key: article_id(key) for key in unique}
    relevant = {key: is_relevant(labels[key], strict=strict) for key in unique}
    groups = stories(unique)
    result = {"articles": len(unique), "full_text_used": sum(1 for key in unique if key in full_texts),
              "full_text_fallbacks": {unique[k]["title"]: reasons for k, reasons in fallbacks.items()}}
    kept_by_mode = {
        "short": {k for k, a in unique.items() if keeps(a, None, company, aliases)},
        "full": {k for k, a in unique.items() if keeps(a, full_texts.get(k), company, aliases)},
    }
    full_kept = kept_by_mode["full"]
    decisions = {k: decide(checks[k]) for k in full_kept if k in checks}
    if checks:
        kept_by_mode["full+check"] = {k for k in full_kept if decisions.get(k, "kept") != "dropped"}
        result["check_counts"] = {
            "model": sum(1 for k in full_kept if checks.get(k, {}).get("source") == "model"),
            "fallback": sum(1 for k in full_kept if checks.get(k, {}).get("source") == "fallback"),
            "not_checked": sum(1 for k in full_kept if k not in checks),
        }
        result["decision_counts"] = {d: sum(1 for v in decisions.values() if v == d) for d in DECISIONS}
    relevant_groups = [g for g in groups if any(relevant[k] for k in g)]
    for mode, kept in kept_by_mode.items():
        counts = confusion(relevant, kept)
        result[mode] = {
            **counts, **scores(counts),
            "missed": [unique[k]["title"] for k in unique if relevant[k] and k not in kept],
            "wrongly_kept": [unique[k]["title"] for k in unique if not relevant[k] and k in kept],
            "stories_relevant": len(relevant_groups),
            "stories_covered": sum(1 for g in relevant_groups if g & kept),
            "kept_ids": {ids[k] for k in kept},
        }
    if checks:
        checked = kept_by_mode["full+check"]
        result["full+check"]["wrongly_dropped_by_check"] = [
            unique[k]["title"] for k in unique if relevant[k] and k in full_kept and k not in checked]
        result["full+check"]["borderline"] = [unique[k]["title"] for k in unique if decisions.get(k) == "borderline"]
        result["copy_disagreements"] = sorted(
            sorted(unique[k]["title"] for k in g if checks.get(k, {}).get("source") == "model")
            for g in groups
            if len({checks[k]["label"] for k in g if checks.get(k, {}).get("source") == "model"}) > 1)
        result["records"] = [_record(k, unique[k], labels[k], ids[k], k in full_texts, fallbacks.get(k),
                                     checks.get(k), decisions.get(k, "not_checked"))
                             for k in unique if k in full_kept]
        if not strict:
            stage1_relevant = [k for k in full_kept if relevant[k]]
            stage1_wrong = [k for k in full_kept if not relevant[k]]
            counts = {"recall_kept": (sum(1 for k in stage1_relevant if k in checked), len(stage1_relevant)),
                      "stories": (result["full+check"]["stories_covered"], len(relevant_groups)),
                      "fp_removed": (sum(1 for k in stage1_wrong if k not in checked), len(stage1_wrong))}
            result["acceptance"] = {name: judge(*counts[name], *THRESHOLDS[name]) for name in THRESHOLDS}
    return result


def _record(key, article, label, identifier, used_full, fallback_reasons, check, decision) -> dict:
    check = check or {}
    source = "full" if used_full else ("short_after_fallback" if fallback_reasons else "short")
    return {"article_id": identifier, "title": article.get("title"), "label": label, "text_source": source,
            "fallback_reasons": fallback_reasons or [], "model_label": check.get("label"),
            "certainty": check.get("certainty"), "reason": check.get("reason"), "decision": decision,
            "omitted_mentions": check.get("omitted_mentions"), "omitted_business": check.get("omitted_business")}


# ---------- running it ----------

def label_versions() -> tuple[str, str]:
    return (LABEL_FILE, CHECKED_LABEL_FILE)


def stage1_record(result: dict, label_file: str) -> dict:
    """Stage 1 on full text (broad: direct and indirect are relevant), with its denominators and definitions."""
    full = result["full"]
    return {"label_file": label_file, "guide_version": "3",
            "relevant_articles": full["tp"] + full["fn"], "relevant_kept": full["tp"], "relevant_missed": full["fn"],
            "kept": full["tp"] + full["fp"], "kept_not_relevant": full["fp"],
            "precision": full["precision"], "recall": full["recall"], "f1": full["f1"],
            "stories_relevant": full.get("stories_relevant"), "stories_covered": full.get("stories_covered"),
            "definitions": {
                "recall": "relevant articles Stage 1 keeps / all relevant articles (direct or indirect)",
                "precision": "relevant articles Stage 1 keeps / all articles Stage 1 keeps",
                "f1": "the harmonic mean of precision and recall",
                "stories": "relevant stories (copies of one headline counted once) with at least one copy kept"}}


def _inputs(label_file: str = LABEL_FILE):
    from src.fulltext import FullTextFetcher
    labels = load_label_set([EVAL_DIR / label_file])
    articles = {}
    for path in sorted(RAW_DIR.glob("*.json")):
        for article in json.loads(path.read_text(encoding="utf-8")).get("articles", []):
            key = normalize_url(article.get("url") or "")
            if key in labels:
                articles.setdefault(key, article)
    fetcher, full_texts = FullTextFetcher(), {}
    for key, article in articles.items():
        cached = fetcher._cache_path(article["url"])
        if cached.exists():
            result = json.loads(cached.read_text(encoding="utf-8"))
            if result.get("status") == "ok":
                full_texts[article["url"]] = result["text"]
    return list(articles.values()), labels, full_texts


CHECK_CACHE = EVAL_DIR / "relevance_checks"


def _run_checks(articles: list[dict], full_texts: dict[str, str], company: str, aliases: dict | None) -> tuple:
    """Stage 2 on every article stage 1 keeps, using the full texts that passed the sanity check."""
    from src.fulltext import article_id
    from src.llm import make_llm
    from src.relevance_check import (BATCH_SIZE, PROMPT_VERSION, CheckCache, _key, build_passage, classify,
                                      company_scope)
    llm = make_llm()
    texts, _ = usable_full_texts(articles, full_texts, company, aliases)
    items, url_of, passages = [], {}, {}
    for article in articles:
        if not keeps(article, texts.get(article["url"]), company, aliases):
            continue
        short = f"{article.get('description') or ''} {article.get('content') or ''}"
        identifier = article_id(article["url"])
        passage = build_passage(article.get("title") or "", texts.get(article["url"], ""), company, aliases,
                                short_text=short)
        items.append({"id": identifier, "title": article.get("title") or "", "passage": passage["text"]})
        url_of[identifier], passages[identifier] = article["url"], passage
    times = []

    def progress(done, total, seconds):
        times.append(seconds)
        print(f"   batch {done}/{total} answered in {seconds:.1f} s", flush=True)

    cache, scope = CheckCache(CHECK_CACHE), company_scope(company, aliases)
    saved = sum(1 for item in items if cache.get(_key(item, company, llm.model_name, scope)) is not None)
    print(f"Model checks: {len(items)} articles, model {llm.model_name}, prompt version {PROMPT_VERSION}, "
          f"{saved} answered from saved answers, {len(items) - saved} to ask in batches of {BATCH_SIZE}", flush=True)
    started = time.perf_counter()
    answers = classify(items, llm, company, cache=cache, progress=progress, scope=scope)
    runtime = {"articles": len(items), "answered_from_saved": saved, "batches_asked": len(times),
               "batch_seconds": [round(t, 1) for t in times],
               "seconds_total": round(time.perf_counter() - started, 1)}
    checks = {}
    for identifier, answer in answers.items():
        passage = passages[identifier]
        checks[url_of[identifier]] = {**answer, "omitted_mentions": passage["omitted_mentions"],
                                      "omitted_business": passage["omitted_business"]}
    return checks, runtime


def _print_scores(result: dict, title: str, strict: bool) -> None:
    print(f"\n=== {title}: {result['articles']} labelled articles (guide v3), "
          f"full text used for {result['full_text_used']} ===\n")
    print(f"{'Text':<11}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}{'Precision':>11}{'Recall':>8}{'F1':>7}")
    modes = [m for m in ("short", "full", "full+check") if m in result]
    for mode in modes:
        r = result[mode]
        print(f"{mode:<11}{r['tp']:>4}{r['fp']:>4}{r['fn']:>4}{r['tn']:>4}"
              f"{r['precision']:>11.3f}{r['recall']:>8.3f}{r['f1']:>7.3f}"
              f"   stories covered {r['stories_covered']}/{r['stories_relevant']}")
    if strict:
        return
    for title_text, reasons in result["full_text_fallbacks"].items():
        print(f"   FULL TEXT NOT USED ({', '.join(reasons)}): {title_text[:80]}")
    for mode in modes:
        for title_text in result[mode]["missed"]:
            print(f"   {mode} missed:       {title_text[:90]}")
        for title_text in result[mode]["wrongly_kept"]:
            print(f"   {mode} wrongly kept: {title_text[:90]}")
    if "full+check" in result:
        for title_text in result["full+check"]["wrongly_dropped_by_check"]:
            print(f"   STAGE 2 WRONGLY DROPPED: {title_text[:90]}")
        for title_text in result["full+check"]["borderline"]:
            print(f"   BORDERLINE (not + unsure, kept): {title_text[:85]}")
        print(f"\nStage 2 answers: {result['check_counts']}; decisions: {result['decision_counts']}")
        for titles in result["copy_disagreements"]:
            print(f"   COPIES ANSWERED DIFFERENTLY: {titles[0][:80]} ({len(titles)} copies)")
        print("\nAcceptance checks (fixed before this run):")
        for name, verdict in result["acceptance"].items():
            print(f"   {name:<12} {verdict['status'].upper():<8} {verdict['count']}/{verdict['total']} "
                  f"(at least {verdict['required']} needed)")


def main(company: str = "Tata Motors", with_checks: bool = False) -> None:
    articles, labels, full_texts = _inputs()
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    checks, runtime = _run_checks(articles, full_texts, company, aliases) if with_checks else (None, None)
    results = {}
    for strict, title in ((False, "BROAD (direct + indirect)"), (True, "STRICT (direct only)")):
        results[strict] = compare(articles, labels, full_texts, company, aliases, strict=strict, checks=checks)
        _print_scores(results[strict], title, strict)
    if not checks:
        return
    broad = results[False]
    known = {r["article_id"] for r in broad["records"]}
    status = tracked_status(load_tracked(), broad["full+check"]["kept_ids"], known)
    records = {r["article_id"]: r for r in broad["records"]}
    v1 = json.loads((EVAL_DIR / ANSWERS_V1_FILE).read_text(encoding="utf-8"))["answers"]
    print("\nTracked articles (version 1 answer -> version 2 answer, decision):")
    for group, wanted in (("false_negatives", "kept"), ("false_positives", "dropped")):
        good = sum(1 for v in status[group].values() if v == wanted)
        print(f"  {group}: {good}/{len(status[group])} now {wanted}")
        for identifier, decision in status[group].items():
            r = records[identifier]
            print(f"   {identifier} {v1.get(identifier, {}).get('label', '?'):<8} -> "
                  f"{str(r['model_label']):<8} {str(r['certainty']):<6} {r['decision']:<10} {r['title'][:50]}")
    changed = [r for r in broad["records"] if r["article_id"] in v1 and v1[r["article_id"]]["label"] != r["model_label"]]
    print(f"\nAnswers changed from version 1: {len(changed)} of {len(records)}")
    for r in changed:
        print(f"   {r['article_id']} {v1[r['article_id']]['label']} -> {r['model_label']} ({r['certainty']}), "
              f"label {r['label']}: {r['title'][:60]}")
    cut = [r for r in broad["records"] if r["omitted_mentions"]]
    print(f"Passages that left out a paragraph naming the company: {len(cut)}")
    print(f"Runtime: {runtime}")
    record = {"about": "Stage 2 version 2, development run on the guide version 3 labels.",
              "scores": {("strict" if strict else "broad"): {mode: {k: v for k, v in r[mode].items() if k != "kept_ids"}
                                                            for mode in ("short", "full", "full+check")}
                         for strict, r in results.items()},
              "acceptance": broad["acceptance"], "decision_counts": broad["decision_counts"],
              "check_counts": broad["check_counts"], "full_text_fallbacks": broad["full_text_fallbacks"],
              "copy_disagreements": broad["copy_disagreements"], "tracked": status, "runtime": runtime,
              "records": broad["records"]}
    (EVAL_DIR / RECORD_FILE).write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved: data/eval/{RECORD_FILE}")


def _stage1_command(company: str = "Tata Motors") -> None:
    """Stage 1 on full text with each label version: the effect of the re-check corrections."""
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    records = []
    for label_file in label_versions():
        articles, labels, full_texts = _inputs(label_file)
        records.append(stage1_record(compare(articles, labels, full_texts, company, aliases), label_file))
    for r in records:
        print(f"{r['label_file']}: recall {r['relevant_kept']}/{r['relevant_articles']} = {r['recall']:.3f}, "
              f"precision {r['relevant_kept']}/{r['kept']} = {r['precision']:.3f}, f1 {r['f1']:.3f}, "
              f"stories {r['stories_covered']}/{r['stories_relevant']}")
    (EVAL_DIR / STAGE1_FILE).write_text(json.dumps({"stage": "Stage 1 on full text, broad", "records": records},
                                                   indent=1), encoding="utf-8")
    print(f"Saved: data/eval/{STAGE1_FILE}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["stage1"]:
        _stage1_command()
    else:
        main(with_checks="--check" in sys.argv[1:])
