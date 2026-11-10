"""
Blind relevance re-check of the guide version 3 labels.

The re-check labels a set of already judged articles a second time, without
the first label, and measures how well the two labellings agree. Real label
mistakes found this way are corrected in a separate checked label file, with
a reason for each change. v3_dev_tata_motors.csv itself never changes.

    pool      the 74 articles judged on their own text: 71 from full text and
              3 from short text because the publisher blocks the page
    Part A    40 articles from the pool, stratified by first label in the
              pool's own mix, fixed seed; agreement and kappa use Part A only
    Part B    the 8 articles where the model and the label disagree; reported
              article by article
    sheet     Part A and Part B together, each article once, shuffled; no
              column shows the part or the first label, and no file stores
              Part A: scoring rebuilds it from the seed and checks the pool
              fingerprint first

Steps (each refuses to run out of order or to overwrite a file):
    python -m src.recheck sheet        the sheet, the reading page and the record
    python -m src.recheck check        the page against the old evidence and the saved source text
    python -m src.recheck score        agreement on Part A and the Part B list, before any adjudication
    python -m src.recheck adjudicate   a sheet of the articles whose two labels differ
    python -m src.recheck apply        the checked label file and the corrections file
"""

import csv
import hashlib
import html as html_module
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SEED = 42
PART_A_SIZE = 40
LABELS = ("direct", "indirect", "not")
RELEVANT = {"direct", "indirect"}
FULL_TEXT = "full text"
BLOCKED = "short text (page blocked by robots.txt)"
PART_B = (                              # the 7 tracked misses (stage2_tracked_v1.json), then the Waymo Madrid article
    "A24e6d8662d", "A79f20394bc", "Ab5f09d2bd1", "A58b7f6f1f1", "Afda0bbac4b", "Adc9449a897", "A02b25c8c80",
    "Aac357bfe35",
)
LABEL_FILE = "v3_dev_tata_motors.csv"
GUIDE_VERSION = "3"
OLD_PAGE = "labelling_view_20261024.html"
FULLTEXT_SHEET = "fulltext_labels_v3_20261024.csv"
FIRST_LABELLING = "2026-10-24"          # the full-text labelling of version 3
LAST_SEEN = "2026-11-02"                # first labels named in a labelling-training message (see EXPOSED)
LABEL_SHOWN = "first label shown"       # the label itself was named in a labelling-training message on 2 November
FACT_SHEET = "in the relevant-only fact sheet"  # read for the fact check, whose sample holds relevant articles only
_LABEL_SHOWN_IDS = ("A0ae8a2cfab", "Aaa3d8d3035", "A639826ac02", "A690a6ec804", "A02b25c8c80", "A58b7f6f1f1",
                    "A43ea61cb65", "Aac357bfe35", "A7595fd1c77", "Ab5f09d2bd1", "A7830fe9dbf", "A20c83edcbc")
_FACT_SHEET_IDS = ("A02b25c8c80", "A24e6d8662d", "A29c1a29e1d", "A391b491c75", "A58b1377275", "A58b7f6f1f1",
                   "A79f20394bc", "A81fc8ba892", "A8619a7e886", "A8d5b4a5afc", "A986c846cdc", "Ab5f09d2bd1",
                   "Ad74eeaf8f2", "Adb9ff93da2", "Ae689fc3060", "Aee4703af04", "Af3ec3babc9")
EXPOSED = {**{i: FACT_SHEET for i in _FACT_SHEET_IDS}, **{i: LABEL_SHOWN for i in _LABEL_SHOWN_IDS}}
AMENDMENT = {
    "date": "2026-11-02", "before_labelling": True,
    "original_plan": "Part A: 40 articles from all 74 pool articles, stratified by first label, fixed seed.",
    "amended_plan": "Part A: 40 articles from the 48 pool articles still blind, stratified by first label in "
                    "their own mix, fixed seed. Part B is unchanged and each article carries its exposure.",
    "reason": "26 pool articles are no longer blind: 12 had their first label named in a labelling-training "
              "message on 2 November, and 14 more were read for the fact check, whose sample holds relevant "
              "articles only.",
}
MEASURES = "agreement on the unexposed development subset"
LIMITS = ("Part A covers the unexposed development subset only, not all 74 articles. The fact-check sample holds "
          "the long relevant articles, so long relevant articles are underrepresented in Part A.")
NOT_BEFORE = datetime(2026, 11, 1, tzinfo=timezone.utc)

SHEET_COLUMNS = ["row", "article_id", "title", "url", "label", "unsure", "note"]
ADJUDICATION_COLUMNS = ["article_id", "title", "first_label", "recheck_label", "final_label", "reason"]
CORRECTION_COLUMNS = ["article_id", "title", "old_label", "new_label", "reason", "decided"]
SHEET_FILE = "recheck_v3_sheet.csv"
RECORD_FILE = "recheck_v3_record.json"
SCORE_FILE = "recheck_v3_score.json"
ADJUDICATION_FILE = "recheck_v3_adjudication.csv"
CHECKED_FILE = "v3_dev_tata_motors_checked.csv"
CORRECTIONS_FILE = "v3_dev_tata_motors_corrections.csv"
RESULT_FILE = "recheck_v3_result.json"
PAGE_FILE = "recheck_view_v3.html"      # publisher text: kept in data/fulltext/, outside Git


class RecheckError(Exception):
    pass


# ---------- the pool, Part A and Part B ----------

def recheck_pool(rows: list[dict]) -> list[dict]:
    """The articles judged on their own text: full text, or short text when the publisher blocks the page."""
    return [r for r in rows if r["basis"] in (FULL_TEXT, BLOCKED)]


def stratum_sizes(pool: list[dict], n: int) -> dict[str, int]:
    """Articles per first label, in the pool's own mix (largest remainder, so the sizes always add up to n)."""
    counts = {label: sum(1 for r in pool if r["label"] == label) for label in LABELS}
    shares = {label: n * count / len(pool) for label, count in counts.items()}
    sizes = {label: int(share) for label, share in shares.items()}
    by_remainder = sorted(LABELS, key=lambda label: (-(shares[label] - sizes[label]), label))
    for label in by_remainder[:n - sum(sizes.values())]:
        sizes[label] += 1
    return sizes


def part_a(pool: list[dict], n: int = PART_A_SIZE, seed: int = SEED, exclude=()) -> set[str]:
    """The original rule; `exclude` leaves articles out before the draw."""
    pool = [r for r in pool if r["article_id"] not in set(exclude)]
    rng = random.Random(seed)
    chosen = set()
    for label, size in stratum_sizes(pool, n).items():
        ids = sorted(r["article_id"] for r in pool if r["label"] == label)
        chosen |= set(rng.sample(ids, size))
    return chosen


def blind_pool(pool: list[dict]) -> list[dict]:
    """The pool articles whose first label is still unknown to the labeller."""
    return [r for r in pool if r["article_id"] not in EXPOSED]


def amended_part_a(pool: list[dict], n: int = PART_A_SIZE, seed: int = SEED) -> set[str]:
    """Part A after the amendment of 2 November: drawn from the articles still blind only."""
    return part_a(pool, n=n, seed=seed, exclude=EXPOSED)


def pool_fingerprint(pool: list[dict]) -> str:
    """Changes when any article, first label or basis in the pool changes."""
    items = sorted([r["article_id"], r["label"], r["basis"]] for r in pool)
    return hashlib.sha256(json.dumps(items).encode("utf-8")).hexdigest()


# ---------- the sheet ----------

def check_start(now: datetime) -> None:
    if now < NOT_BEFORE:
        raise RecheckError(f"The re-check sheet is made from {NOT_BEFORE:%d %B %Y} (UTC).")


def sheet_rows(pool: list[dict], seed: int = SEED) -> list[dict]:
    by_id = {r["article_id"]: r for r in pool}
    ids = sorted(amended_part_a(pool, seed=seed) | set(PART_B))
    random.Random(seed).shuffle(ids)
    return [{"row": number, "article_id": i, "title": by_id[i]["title"], "url": by_id[i]["url"],
             "label": "", "unsure": "", "note": ""} for number, i in enumerate(ids, start=1)]


ANOTHER_STORY = "the saved page holds another story"


def saved_text_is_another_story(title: str, description: str, text: str, aliases: dict | None) -> bool:
    """The project's wrong-page check (src/fulltext.py): the saved text does not match the title and description."""
    from src.fulltext import check_full_text
    return not check_full_text(title, description, text, "Tata Motors", aliases)["use_full_text"]


def sheet_record(pool: list[dict], evidence: dict[str, str], created: str, overrides: dict | None = None) -> dict:
    """Settings, dates and text fingerprints. It never lists Part A."""
    return {"label_file": LABEL_FILE, "guide_version": GUIDE_VERSION, "pool_size": len(pool),
            "pool_fingerprint": pool_fingerprint(pool), "seed": SEED, "part_a_size": PART_A_SIZE,
            "part_a_sample": "stratified by first label, proportional, fixed seed, from the articles still blind",
            "part_b": list(PART_B), "amendment": AMENDMENT, "exposed": dict(EXPOSED),
            "first_labelling": FIRST_LABELLING, "last_seen": LAST_SEEN, "sheet_created": created,
            "evidence": evidence, "evidence_overrides": dict(overrides or {})}


def check_pool(record: dict, pool: list[dict]) -> None:
    if record["pool_fingerprint"] != pool_fingerprint(pool):
        raise RecheckError("The pool or its first labels differ from the sheet record; Part A is not the same.")


# ---------- the same evidence ----------

SECTION = re.compile(r'<section id="([^"]+)">(.*?)</section>', re.S)
PARAGRAPH = re.compile(r"<p>(.*?)</p>", re.S)          # article paragraphs only: the meta lines carry a class


def parse_page(page: str) -> dict[str, list[str]]:
    return {i: [html_module.unescape(p) for p in PARAGRAPH.findall(body)] for i, body in SECTION.findall(page)}


def fingerprint(paragraphs: list[str]) -> str:
    return hashlib.sha256("\n".join(paragraphs).encode("utf-8")).hexdigest()


def short_text_paragraphs(row: dict) -> list[str]:
    parts = (row.get("title"), row.get("description"), row.get("content_first_200_chars"))
    return [p.strip() for p in parts if p and p.strip()]


def one_short_text(article_id: str, rows: list[dict]) -> list[str]:
    """The short text of a blocked article; every sheet that holds it must show the same text."""
    texts = {tuple(short_text_paragraphs(r)) for r in rows}
    if len(texts) != 1:
        raise RecheckError(f"{article_id}: {'no' if not texts else 'more than one'} short text in the label sheets.")
    return list(texts.pop())


def recheck_page(items: list[dict]) -> str:
    e = html_module.escape
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Relevance re-check</title>",
             "<style>body{font-family:Georgia,serif;max-width:46rem;margin:2rem auto;padding:0 1rem;line-height:1.55}"
             "section{border-top:2px solid #444;padding-top:1rem;margin-top:2.5rem}"
             ".meta{font-family:Arial,sans-serif;font-size:.85rem;color:#555}"
             ".status{font-family:Arial,sans-serif;font-size:.85rem;font-weight:bold}</style></head><body>",
             "<h1>Relevance re-check</h1><p class='meta'>Read each article here. Write the label in the CSV row "
             "with the same row number and article id. Follow LABELLING_GUIDE.md (version 3).</p>"]
    for item in items:
        parts.append(
            '<section id="' + e(item["article_id"]) + '">'
            f"<h2>Row {item['row']} — {e(item['title'])}</h2>"
            f"<p class='meta'>Article id: {e(item['article_id'])} | {e(item['source'])} | {e(item['published'])}</p>"
            f"<p class='status'>Text: {e(item['status'])}</p>"
            + "".join(f"<p>{e(paragraph)}</p>" for paragraph in item["paragraphs"])
            + "</section>")
    parts.append("</body></html>")
    return "".join(parts)


def evidence_problems(old: dict[str, str], page: str) -> list[str]:
    """Each article on the page must show exactly the text of the first labelling."""
    shown = {i: fingerprint(p) for i, p in parse_page(page).items()}
    problems = [f"{i}: missing from the page" for i in sorted(old) if i not in shown]
    problems += [f"{i}: the text differs from the first labelling" for i in sorted(old)
                 if i in shown and shown[i] != old[i]]
    problems += [f"{i}: on the page but not in the first labelling" for i in sorted(shown) if i not in old]
    return problems


def source_problems(page: dict[str, list[str]], saved: dict[str, str]) -> list[str]:
    """The old page against the saved full text it is built from."""
    from src.fulltext import _paragraphs
    problems = []
    for i in sorted(page):
        if i not in saved:
            problems.append(f"{i}: no saved source text")
        elif _paragraphs(saved[i]) != page[i]:
            problems.append(f"{i}: the page differs from the saved source text")
    return problems


# ---------- scoring ----------

def _label(value) -> str:
    return (value or "").strip().lower()


def cohen_kappa(first: list[str], second: list[str]) -> float | None:
    """None when chance agreement is 1 (both labellings use one and the same label only)."""
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / n
    chance = sum((first.count(x) / n) * (second.count(x) / n) for x in set(first) | set(second))
    return None if chance == 1 else (observed - chance) / (1 - chance)


def _agreement(first: list[str], second: list[str]) -> tuple[float, float | None]:
    return sum(a == b for a, b in zip(first, second)) / len(first), cohen_kappa(first, second)


def recheck_labels(rows: list[dict], expected: set[str]) -> dict[str, str]:
    labels = {}
    for row in rows:
        value = _label(row.get("label"))
        if value not in LABELS:
            raise RecheckError(f"Row {row.get('row')}: the label must be direct, indirect or not.")
        labels[row["article_id"]] = value
    if set(labels) != expected or len(labels) != len(rows):
        raise RecheckError("The sheet does not hold exactly the re-check articles, each once.")
    return labels


def score(original: dict[str, str], recheck_rows: list[dict], part_a_ids: set[str], part_b_ids) -> dict:
    labels = recheck_labels(recheck_rows, set(part_a_ids) | set(part_b_ids))
    ids = sorted(part_a_ids)
    first, second = [original[i] for i in ids], [labels[i] for i in ids]
    agreement, kappa = _agreement(first, second)
    relevant_agreement, relevant_kappa = _agreement([x in RELEVANT for x in first], [x in RELEVANT for x in second])
    per_label = {}
    for label in LABELS:
        pairs = [(x, y) for x, y in zip(first, second) if x == label]
        if pairs:
            same = sum(1 for x, y in pairs if x == y)
            per_label[label] = {"articles": len(pairs), "same": same, "agreement": same / len(pairs)}
    return {
        "label_file": LABEL_FILE, "guide_version": GUIDE_VERSION, "seed": SEED,
        "part_a": {"articles": len(ids), "measures": MEASURES, "limits": LIMITS,
                   "sample": "stratified by first label, proportional, fixed seed, from the articles still blind",
                   "agreement": agreement, "kappa": kappa, "per_label": per_label,
                   "relevant_agreement": relevant_agreement, "relevant_kappa": relevant_kappa,
                   "pairs": {f"{a} -> {b}": sum(1 for x, y in zip(first, second) if (x, y) == (a, b))
                             for a in LABELS for b in LABELS}},
        "part_b": [{"article_id": i, "first_label": original[i], "recheck_label": labels[i],
                    "exposure": EXPOSED.get(i, "blind")} for i in part_b_ids],
        "part_b_note": "Part B is the chosen model-label disagreements: an error review, not combined with Part A.",
    }


# ---------- adjudication and the checked label file ----------

def require_score(path: Path) -> None:
    if not Path(path).exists():
        raise RecheckError("The score comes first: run `python -m src.recheck score`.")


def adjudication_rows(original: dict[str, str], recheck_rows: list[dict], titles: dict[str, str]) -> list[dict]:
    rows = []
    for row in recheck_rows:
        i, label = row["article_id"], _label(row.get("label"))
        if label != original[i]:
            rows.append({"article_id": i, "title": titles.get(i, ""), "first_label": original[i],
                         "recheck_label": label, "final_label": "", "reason": ""})
    return rows


def _check_decision(row: dict, first: dict[str, str]) -> str:
    i = row["article_id"]
    if i not in first:
        raise RecheckError(f"{i}: not in {LABEL_FILE}.")
    if _label(row["first_label"]) != first[i]:
        raise RecheckError(f"{i}: the first label in the adjudication sheet differs from {LABEL_FILE}.")
    final = _label(row.get("final_label"))
    if final not in LABELS:
        raise RecheckError(f"{i}: final_label must be direct, indirect or not.")
    if not (row.get("reason") or "").strip():
        raise RecheckError(f"{i}: every decision needs a reason.")
    return final


def apply_adjudication(v3_rows: list[dict], decided: list[dict], when: str) -> tuple[list[dict], list[dict]]:
    """The checked rows (only decided labels change) and the corrections (final label differs from the first)."""
    first = {r["article_id"]: r["label"] for r in v3_rows}
    finals, corrections = {}, []
    for row in decided:
        final = _check_decision(row, first)
        finals[row["article_id"]] = final
        if final != first[row["article_id"]]:
            corrections.append({"article_id": row["article_id"], "title": row["title"],
                                "old_label": first[row["article_id"]], "new_label": final,
                                "reason": row["reason"].strip(), "decided": when})
    checked = [{**r, "label": finals.get(r["article_id"], r["label"])} for r in v3_rows]
    return checked, corrections


def decision_counts(decided: list[dict]) -> dict[str, int]:
    finals = [(_label(r["first_label"]), _label(r["recheck_label"]), _label(r["final_label"])) for r in decided]
    return {"decisions": len(finals),
            "first_label_kept": sum(1 for a, _, c in finals if c == a),
            "recheck_label_taken": sum(1 for a, b, c in finals if c == b != a),
            "third_label": sum(1 for a, b, c in finals if c not in (a, b)),
            "corrections": sum(1 for a, _, c in finals if c != a)}


# ---------- files ----------

def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_new(path: Path, write) -> None:
    if path.exists():
        raise RecheckError(f"{path.name} already exists; it is not overwritten.")
    write(path)


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    def write(p):
        with open(p, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    _write_new(path, write)


def _write_json(path: Path, data: dict) -> None:
    _write_new(path, lambda p: p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _paths():
    from src.config import EVAL_DIR
    from src.fulltext import FULLTEXT_DIR
    return Path(EVAL_DIR), Path(FULLTEXT_DIR)


def _pool():
    eval_dir, _ = _paths()
    return recheck_pool(_read_csv(eval_dir / LABEL_FILE))


def _old_evidence(pool: list[dict]) -> tuple[dict[str, list[str]], dict[str, dict]]:
    """Paragraphs of the first labelling for every pool article, and source and date for the page.
    A saved full text that fails the wrong-page check is replaced by the short text, and its meta says why."""
    from src.config import DATA_DIR
    from src.fulltext import article_id
    from src.query import load_aliases
    aliases = load_aliases(Path(DATA_DIR) / "company_aliases.json")
    saved = _saved_source(pool)
    eval_dir, fulltext_dir = _paths()
    old_page = parse_page((fulltext_dir / OLD_PAGE).read_text(encoding="utf-8"))
    sheets = [row for path in sorted(eval_dir.glob("dev_*.csv")) for row in _read_csv(path)]
    by_id: dict[str, list[dict]] = {}
    for row in sheets:
        by_id.setdefault(article_id(row["url"]), []).append(row)
    paragraphs, meta = {}, {}
    for r in pool:
        i = r["article_id"]
        found = by_id.get(i, [])
        first = found[0] if found else {}
        another = r["basis"] == FULL_TEXT and saved_text_is_another_story(
            r["title"], first.get("description", ""), saved.get(i, ""), aliases)
        if r["basis"] == FULL_TEXT and not another:
            if i not in old_page:
                raise RecheckError(f"{i}: missing from {OLD_PAGE}.")
            paragraphs[i] = old_page[i]
            status = "complete"
        else:
            paragraphs[i] = one_short_text(i, found)
            status = (f"short text only ({ANOTHER_STORY})" if another
                      else "short text only (the publisher blocks the page)")
        meta[i] = {"source": first.get("source", ""), "published": first.get("published", ""), "status": status,
                   "override": ANOTHER_STORY if another else None}
    return paragraphs, meta


def _saved_source(pool: list[dict]) -> dict[str, str]:
    """The saved full text behind the old page, found through the link it was fetched with."""
    eval_dir, fulltext_dir = _paths()
    fetched = {r["article_id"]: r["url"] for r in _read_csv(eval_dir / FULLTEXT_SHEET)}
    saved = {}
    for r in pool:
        if r["basis"] != FULL_TEXT or r["article_id"] not in fetched:
            continue
        path = fulltext_dir / f"{hashlib.sha1(fetched[r['article_id']].encode('utf-8')).hexdigest()}.json"
        if path.exists():
            saved[r["article_id"]] = json.loads(path.read_text(encoding="utf-8")).get("text", "")
    return saved


def _evidence_report(pool: list[dict], page_text: str) -> tuple[dict[str, str], list[str]]:
    paragraphs, meta = _old_evidence(pool)
    evidence = {i: fingerprint(p) for i, p in paragraphs.items()}
    full = {r["article_id"] for r in pool if r["basis"] == FULL_TEXT and not meta[r["article_id"]]["override"]}
    problems = source_problems({i: p for i, p in paragraphs.items() if i in full}, _saved_source(pool))
    sheet_ids = {r["article_id"] for r in sheet_rows(pool)}
    problems += evidence_problems({i: f for i, f in evidence.items() if i in sheet_ids}, page_text)
    return evidence, problems


def _sheet_command() -> None:
    check_start(datetime.now(timezone.utc))
    eval_dir, fulltext_dir = _paths()
    pool = _pool()
    rows = sheet_rows(pool)
    paragraphs, meta = _old_evidence(pool)
    items = [{**row, **meta[row["article_id"]], "paragraphs": paragraphs[row["article_id"]]} for row in rows]
    page = recheck_page(items)
    evidence, problems = _evidence_report(pool, page)
    if problems:
        raise RecheckError("The page does not show the first labelling's text:\n  " + "\n  ".join(problems))
    _write_csv(eval_dir / SHEET_FILE, rows, SHEET_COLUMNS)
    (fulltext_dir / PAGE_FILE).write_text(page, encoding="utf-8")
    overrides = {i: m["override"] for i, m in meta.items() if m["override"]}
    _write_json(eval_dir / RECORD_FILE, sheet_record(pool, evidence, _now(), overrides))
    for i, reason in overrides.items():
        print(f"{i}: shown with its short text ({reason})")
    print(f"{len(rows)} articles to label")
    print(f"Label sheet:  {eval_dir / SHEET_FILE}")
    print(f"Reading page: {fulltext_dir / PAGE_FILE}  (open it in your browser)")


def _checked_inputs():
    eval_dir, fulltext_dir = _paths()
    pool = _pool()
    record = json.loads((eval_dir / RECORD_FILE).read_text(encoding="utf-8"))
    check_pool(record, pool)
    evidence, problems = _evidence_report(pool, (fulltext_dir / PAGE_FILE).read_text(encoding="utf-8"))
    if {i: evidence[i] for i in record["evidence"]} != record["evidence"]:
        problems.append("The evidence differs from the sheet record.")
    return pool, record, problems


def _check_command() -> None:
    _, _, problems = _checked_inputs()
    print("The page shows the first labelling's text for every article." if not problems
          else "Problems:\n  " + "\n  ".join(problems))


def _score_command() -> None:
    eval_dir, _ = _paths()
    pool, record, problems = _checked_inputs()
    if problems:
        raise RecheckError("Problems:\n  " + "\n  ".join(problems))
    original = {r["article_id"]: r["label"] for r in pool}
    result = score(original, _read_csv(eval_dir / SHEET_FILE), amended_part_a(pool), PART_B)
    result.update({"pool_fingerprint": record["pool_fingerprint"], "first_labelling": record["first_labelling"],
                   "last_seen": record["last_seen"], "sheet_created": record["sheet_created"], "scored": _now()})
    _write_json(eval_dir / SCORE_FILE, result)
    a = result["part_a"]
    kappa = "undefined" if a["kappa"] is None else f"{a['kappa']:.3f}"
    print(f"Part A ({a['articles']} articles, {a['measures']}): agreement {a['agreement']:.3f}, kappa {kappa}")
    for label, n in a["per_label"].items():
        print(f"   first label {label}: {n['same']} of {n['articles']} the same")
    print("   pairs (first -> re-check): " + ", ".join(f"{k} {v}" for k, v in a["pairs"].items() if v))
    print(f"Part B (error review, not in the agreement): "
          + ", ".join(f"{b['article_id']} {b['first_label']}->{b['recheck_label']} ({b['exposure']})"
                      for b in result["part_b"]))
    print(f"Saved: {eval_dir / SCORE_FILE}")


def _adjudicate_command() -> None:
    eval_dir, _ = _paths()
    require_score(eval_dir / SCORE_FILE)
    pool = _pool()
    original = {r["article_id"]: r["label"] for r in pool}
    rows = adjudication_rows(original, _read_csv(eval_dir / SHEET_FILE),
                             titles={r["article_id"]: r["title"] for r in pool})
    _write_csv(eval_dir / ADJUDICATION_FILE, rows, ADJUDICATION_COLUMNS)
    print(f"{len(rows)} articles to decide  ->  {eval_dir / ADJUDICATION_FILE}")


def _apply_command() -> None:
    eval_dir, _ = _paths()
    v3 = _read_csv(eval_dir / LABEL_FILE)
    decided = _read_csv(eval_dir / ADJUDICATION_FILE)
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    checked, corrections = apply_adjudication(v3, decided, when)
    _write_csv(eval_dir / CHECKED_FILE, checked, list(v3[0]))
    _write_csv(eval_dir / CORRECTIONS_FILE, corrections, CORRECTION_COLUMNS)
    counts = decision_counts(decided)
    _write_json(eval_dir / RESULT_FILE, {"label_file": LABEL_FILE, "checked_file": CHECKED_FILE,
                                         "corrections_file": CORRECTIONS_FILE, "decided": when, **counts})
    print(f"{counts['decisions']} decisions, {counts['corrections']} corrections")


COMMANDS = {"sheet": _sheet_command, "check": _check_command, "score": _score_command,
            "adjudicate": _adjudicate_command, "apply": _apply_command}

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in COMMANDS:
        COMMANDS[sys.argv[1]]()
    else:
        print(__doc__)
