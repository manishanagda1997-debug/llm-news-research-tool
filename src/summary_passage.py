"""
The summary input of one article (summary passage version 3).

Version 3 is version 2 (src/relevance_check.py, build_summary_passage: the relevance passage plus whole
continued quotes, within the hard cap) built on the text after press-release boilerplate is removed
(src/boilerplate.py). When boilerplate is removed, the room it took goes back to the article: the paragraphs the
selection leaves out are added in article order, whole when they fit, otherwise their opening part, within the
passage limit. An article without boilerplate gets exactly its version 2 passage.

The relevance passage of Stages 2 and 3 (build_passage) is not changed.

Check (no model, no network; publisher text stays in data/fulltext/, outside Git):
    python -m src.summary_passage check
        version 2 and version 3 for the 16 fact-check articles and the top 10 of each development run (the
        passages the fact list and the development comparison read), and for every saved page that has
        boilerplate: for these, every news line of version 2 must still be in version 3. The counts go to
        data/eval/summary_passage_v3_check.json.
"""

from src.boilerplate import clean
from src.relevance_check import build_summary_passage

PASSAGE_VERSION = "3"


def build(title: str, text: str, company: str, aliases: dict | None, short_text: str = "") -> dict:
    full, short = clean(text), clean(short_text)
    passage = build_summary_passage(title, full["text"], company, aliases, short_text=short["text"],
                                    fill=bool(full["removed"]))
    removed = full["removed"] + ([{**r, "part": "short_text"} for r in short["removed"]] if not full["text"] else [])
    return {**passage, "removed": removed, "removed_chars": sum(r["chars"] for r in removed)}


def news_lines_lost(v2: str, v3: str, boilerplate: set[str]) -> list[str]:
    """Lines of the version 2 passage that come from the article's news (not from removed boilerplate) and are
    missing in version 3. A trimmed line counts as kept when its text is inside version 3."""
    lost = []
    for line in v2.split("\n")[1:]:
        core = line.strip().strip("…").strip()
        if not core or core == "[…]" or any(core in part for part in boilerplate):
            continue
        if core not in v3:
            lost.append(core)
    return lost


def _check_command() -> None:
    import glob
    import hashlib
    import json
    from src import fact_check as fc
    from src import ranking as rk
    from src.config import DATA_DIR, EVAL_DIR
    from src.query import load_aliases
    from src.summary_v2 import _dev_candidates
    aliases = load_aliases(DATA_DIR / "company_aliases.json")

    def same(title, text, short=""):
        old = build_summary_passage(title, text, "Tata Motors", aliases, short_text=short)["text"]
        return old == build(title, text, "Tata Motors", aliases, short_text=short)["text"]

    facts = json.loads((EVAL_DIR / "fact_check_record.json").read_text(encoding="utf-8"))["articles"]
    articles = [a for a in fc._articles() if a["article_id"] in facts]
    fact_changed = sorted(a["article_id"] for a in articles if not same(a["title"], a["text"]))
    runs, dev_aliases = _dev_candidates()
    dev = [c for cands in runs.values() for c in rk.order([dict(x, check=None) for x in cands], "O1")[:10]]
    dev_changed = sorted(c["key"] for c in dev if not same(c["title"], c.get("text") or "", c.get("description") or ""))

    pages, total = [], 0
    for path in sorted(glob.glob(str(DATA_DIR / "fulltext" / "*.json"))):
        try:
            saved = json.loads(open(path, encoding="utf-8").read())
        except (OSError, ValueError):
            continue
        if not isinstance(saved, dict) or saved.get("status") != "ok" or not saved.get("text"):
            continue
        total += 1
        cleaned = clean(saved["text"])
        if not cleaned["removed"]:
            continue
        paragraphs = [p.strip() for p in saved["text"].split("\n") if p.strip()]
        boilerplate = {p for r in cleaned["removed"] for p in paragraphs[r["first"]:r["last"] + 1]}
        title = saved["url"].rstrip("/").split("/")[-1].replace("-", " ")[:150]
        v2 = build_summary_passage(title, saved["text"], "Tata Motors", aliases)["text"]
        v3 = build(title, saved["text"], "Tata Motors", aliases)["text"]
        pages.append({"page": hashlib.sha1(saved["url"].encode("utf-8")).hexdigest(),
                      "site": saved["url"].split("/")[2], "removed_chars": cleaned["removed_chars"],
                      "removed": [{"kind": r["kind"], "heading": r["heading"]} for r in cleaned["removed"]],
                      "v2_chars": len(v2), "v3_chars": len(v3), "news_lines_lost": len(news_lines_lost(v2, v3, boilerplate))})
    result = {"passage_version": PASSAGE_VERSION,
              "fact_check": {"articles": len(articles), "changed": fact_changed},
              "development_top10": {"passages": len(dev), "changed": dev_changed},
              "saved_pages": {"with_text": total, "with_boilerplate": len(pages), "pages": pages},
              "accepted": not fact_changed and not dev_changed and all(p["news_lines_lost"] == 0 for p in pages)}
    (EVAL_DIR / "summary_passage_v3_check.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"Fact-check passages changed: {len(fact_changed)} of {len(articles)}; development top-10 passages "
          f"changed: {len(dev_changed)} of {len(dev)}")
    print(f"Saved pages with boilerplate: {len(pages)} of {total}")
    for p in pages:
        print(f"   {p['site']:<30} removed {p['removed_chars']:>6} characters; passage {p['v2_chars']} -> "
              f"{p['v3_chars']}; news lines lost {p['news_lines_lost']}")
    print(f"Accepted: {result['accepted']}  (saved: data/eval/summary_passage_v3_check.json)")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["check"]:
        _check_command()
    else:
        print(__doc__)
