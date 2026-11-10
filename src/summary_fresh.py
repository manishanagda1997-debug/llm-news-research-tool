"""
The fresh-week summary run (SUMMARY_JUDGING_GUIDE_PART6.md): the frozen summary design, once, on the week of 24 to
30 October. It lives apart from src/summary_v2.py, so the frozen code stays exactly as frozen.

Each check runs before anything after it:
    1. none of the run's three files (fresh_paths: the record, the attempt marker, the passages) may exist: the
       fresh-week summary is attempted once only. The marker is the first file written, before the evidence and
       before any model call, and it is never removed, so a run that stops at any later point (writing the
       evidence, during the call, before the record) can never be sent again
    2. frozen_summary(5) must accept the current code and settings, and the freeze must be the agreed design
       (direct, k = 5, prompt version 4, summary passage 3, pacer version 3)
    3. Git must show no change or untracked file under src/, in the alias file, in the fresh week's labels and
       events, or in the judging guides (other paths do not stop the run)
Then: the fresh week's Stage 1 candidates in their order and the default ranking's top five; the attempt marker;
the exact passages the model receives, saved to data/fulltext/ (publisher text stays outside Git), with their
fingerprints in the record; one call path with no answer cache read or written (the approved 429 and 5xx
resends of the same request apply); and the record: commit, freeze fingerprint, every candidate in order, the
five selected with their ranks, the passage fingerprints and the full result.

Run:
    python -m src.summary_fresh
"""

import json
import sys
from pathlib import Path

from src import summary_v2 as sv
import hashlib
from datetime import datetime, timezone

from src.config import DATA_DIR, EVAL_DIR
from src.fulltext import FULLTEXT_DIR

FRESH_RUN = "fresh_2026-10-24_30"
FRESH_FILE = "summary_v2_fresh_k5_c7_p4.json"
ATTEMPT_FILE = "summary_v2_fresh_k5_c7_p4_attempt.json"
PASSAGES_FILE = "summary_v2_fresh_k5_c7_p4_passages.json"  # in data/fulltext/, outside Git
CACHE_NOTE = "No answer cache is read or written; the run result is saved in this record."
FRESH_K = 5
FRESH_SETTINGS = {"design": "direct", "prompt_version": "4", "passage_version": "3", "pacer_version": "3"}
FRESH_PROTECTED = ("src/", "data/company_aliases.json", f"data/eval/{FRESH_RUN}/labels.csv",
                   f"data/eval/{FRESH_RUN}/events.csv", "data/eval/SUMMARY_JUDGING_GUIDE.md",
                   "data/eval/SUMMARY_JUDGING_GUIDE_PART4.md", "data/eval/SUMMARY_JUDGING_GUIDE_PART6.md")


class FreshRunError(Exception):
    pass


def fresh_path_problems(status_text: str) -> list[str]:
    """The changed or untracked paths (git status --porcelain) that the fresh run protects."""
    found = set()
    for line in status_text.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].split(" -> ")[-1].strip().strip('"').replace("\\", "/")
        if any(path == p or (p.endswith("/") and path.startswith(p)) for p in FRESH_PROTECTED):
            found.add(path)
    return sorted(found)


def fresh_paths() -> dict:
    """The run's three files, in one place. The record and the attempt marker are kept in data/eval/ (in Git);
    the passages are publisher text, so they stay in data/fulltext/, which Git ignores."""
    return {"record": EVAL_DIR / FRESH_FILE, "attempt": EVAL_DIR / ATTEMPT_FILE,
            "passages": Path(FULLTEXT_DIR) / PASSAGES_FILE}


def passage_entry(article: dict) -> dict:
    """The exact text the model receives for one article, and its fingerprint."""
    text = {"title": article.get("title", ""), "description": article.get("description", ""),
            "content": article.get("content", "")}
    digest = hashlib.sha256(json.dumps(text, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return {"id": article["id"], "key": article["key"], **text, "sha256": digest}


def _fresh_aliases() -> dict:
    from src.query import load_aliases
    return load_aliases(DATA_DIR / "company_aliases.json")


def fresh_command() -> None:
    from src import fresh_eval as fe
    from src import summary_freeze as sf
    paths = fresh_paths()
    path, attempt, passages_path = paths["record"], paths["attempt"], paths["passages"]
    for done in (path, attempt, passages_path):
        if done.exists():
            raise FreshRunError(f"{done} already exists: the fresh-week summary is attempted once only.")
    try:
        freeze = sf.frozen_summary(FRESH_K)
    except sf.FreezeError as error:
        raise FreshRunError(str(error)) from None
    found = ["design"] if freeze.get("design") != FRESH_SETTINGS["design"] else []
    found += [name for name in ("prompt_version", "passage_version", "pacer_version")
              if freeze.get("settings", {}).get(name) != FRESH_SETTINGS[name]]
    if FRESH_K not in freeze.get("ks", []):
        found.append("k")
    if found:
        raise FreshRunError("The freeze is not the agreed design: " + ", ".join(found))
    changed = fresh_path_problems(fe.git("status", "--porcelain", "--untracked-files=all"))
    if changed:
        raise FreshRunError("Commit or remove these first: " + ", ".join(changed))
    commit = fe.git("rev-parse", "HEAD")
    aliases = _fresh_aliases()
    candidates = fe._candidates(fe.load_state(fe.RUN_DIR), "Tata Motors", aliases)
    articles = sv.summary_articles(candidates, FRESH_K, aliases)
    passages = [passage_entry(a) for a in articles]
    with open(attempt, "x", encoding="utf-8") as handle:     # first: before the evidence and any model call; never removed
        handle.write(json.dumps({"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "source_commit": commit, "selected": [a["key"] for a in articles]}, indent=1))
    with open(passages_path, "x", encoding="utf-8") as handle:     # the evidence for judging, outside Git
        handle.write(json.dumps(passages, ensure_ascii=False, indent=1))
    llm = sv.summary_llm()
    result = sv._one(freeze["design"], articles, llm, sv.NoCache(), FRESH_K)   # no answer cache read or written
    for message in sv.take_messages(result):
        print(f"    {message}")
    record = {"run": FRESH_RUN, "design": freeze["design"], "k": FRESH_K, "source_commit": commit,
              "freeze_fingerprint": fe.file_fingerprint(sf.EVAL_DIR / sf.FREEZE_FILE),
              "prompt_version": sv.PROMPT_VERSION, "number_check": sv.NUMBER_CHECK_VERSION,
              "pacer_version": sv.PACER_VERSION, "passage_version": sv.PASSAGE_VERSION,
              "model_setting": sv.MODEL_SETTING, "model": llm.model_name,
              "generation": getattr(llm, "generation", None), "output_limit_sent": getattr(llm, "output_limit", None),
              "cache": CACHE_NOTE, "passages_file": f"data/fulltext/{PASSAGES_FILE}",
              "attempt_file": f"data/eval/{ATTEMPT_FILE}",
              "passage_fingerprints": [{"id": p["id"], "key": p["key"], "sha256": p["sha256"]} for p in passages],
              "candidates": [{"position": n, "key": c["key"], "title": c.get("title", "")}
                             for n, c in enumerate(candidates, start=1)],
              "selected": [{"rank": rank, "id": a["id"], "key": a["key"], "title": a["title"]}
                           for rank, a in enumerate(articles, start=1)],
              "records": [sv._record(FRESH_RUN, freeze["design"], articles, result)]}
    with open(path, "x", encoding="utf-8") as handle:          # "x": never overwrites
        handle.write(json.dumps(record, ensure_ascii=False, indent=1))
    print(f"{FRESH_RUN} {freeze['design']} {result['status']} {result.get('seconds', '-')} s  calls "
          f"{result.get('usage', {}).get('calls', '-')}  failures {result.get('failures', [])}  "
          f"removed {len(result.get('removed', []))}")
    print(f"Saved: data/eval/{FRESH_FILE}")


if __name__ == "__main__":
    try:
        fresh_command()
    except FreshRunError as error:
        print(f"[STOP] {error}")
        sys.exit(1)
