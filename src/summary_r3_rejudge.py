"""
Round 3 consistency check: a second judgement of 20% of the Round 3 items by the same judge.

After the Round 3 lock, 20% of the items (decoys left out: they come from earlier locks) are drawn with a fixed seed,
shuffled and given new row numbers. The sheet shows each item's text and evidence only, never the first judgement.
The score is the share of matching judgements and of matching contradiction decisions, with the three-by-three
table, as in the Stage 4 consistency check (src/summary_judging.py). It is reported only: it changes no judgement,
no score and no choice.

Run:
    python -m src.summary_r3_rejudge sheet    after the Round 3 lock; writes the sheet, its key and the reading page
    python -m src.summary_r3_rejudge score    after every row is judged
"""

import json
import sys
from pathlib import Path

from src import summary_judging as sj
from src import summary_round3 as r3
from src.config import EVAL_DIR

SEED = 83
SHARE = 0.2
SHEET_FILE = "summary_v2_r3_rejudge.csv"
KEY_FILE = "summary_v2_r3_rejudge_key.json"
SCORE_FILE = "summary_v2_r3_rejudge_score.json"
PAGE_FILE = "summary_v2_r3_rejudge.html"


class RejudgeError(Exception):
    pass


def candidates(items: list[dict], key: dict) -> list[dict]:
    """The Round 3 items that are not decoys."""
    return [row for row in items if not key["items"][str(row["row"]).strip()].get("decoy")]


def sheet_rows(items: list[dict], key: dict) -> tuple[list[dict], dict]:
    return sj.rejudge_rows(candidates(items, key), seed=SEED, share=SHARE)


def score(first: list[dict], second: list[dict], rejudge_key: dict, key: dict) -> dict:
    fake_key = {n: {"fingerprint": key["items"][str(original)]["fingerprint"]} for n, original in rejudge_key.items()}
    found = sj.item_problems(second, fake_key)
    if found:
        raise RejudgeError("The consistency sheet is not complete:\n" + "\n".join(found))
    result = sj.agreement(first, second, rejudge_key)
    result.update(seed=SEED, share=SHARE, scope="Round 3 items without the decoys; the same judge; reported only")
    return result


def _sheet_command(eval_dir=EVAL_DIR, fulltext_dir=None) -> None:
    from src.fulltext import FULLTEXT_DIR
    eval_dir = Path(eval_dir)
    if not (eval_dir / r3.LOCK_FILE).exists():
        raise RejudgeError(f"{r3.LOCK_FILE} does not exist: the first judgements are locked first.")
    key = json.loads((eval_dir / r3.KEY_FILE).read_text(encoding="utf-8"))
    rows, rejudge_key = sheet_rows(sj._read_csv(eval_dir / r3.ITEMS_FILE), key)
    records, _, _ = r3._load_round()
    first_record = records[r3.run_id(key["kept"][0], "4")]
    titles = {a["key"]: a["title"] for record in records.values() for r in record["records"] for a in r["articles"]}
    page = sj.reading_page(rows, [], sj._passages(first_record), titles)
    sj._write_new(eval_dir / SHEET_FILE, sj._csv_text(rows, sj.ITEM_COLUMNS))
    sj._write_new(eval_dir / KEY_FILE, json.dumps(rejudge_key, indent=1))
    sj._write_new(Path(fulltext_dir or FULLTEXT_DIR) / PAGE_FILE, page)
    print(f"Consistency sheet: {len(rows)} rows in data/eval/{SHEET_FILE}; reading page {PAGE_FILE} in data/fulltext")


def _score_command(eval_dir=EVAL_DIR) -> None:
    eval_dir = Path(eval_dir)
    result = score(sj._read_csv(eval_dir / r3.ITEMS_FILE), sj._read_csv(eval_dir / SHEET_FILE),
                   json.loads((eval_dir / KEY_FILE).read_text(encoding="utf-8")),
                   json.loads((eval_dir / r3.KEY_FILE).read_text(encoding="utf-8")))
    sj._write_new(eval_dir / SCORE_FILE, json.dumps(result, indent=1))
    print(f"Same judgement: {result['judgement_agreement']} of {result['items']} items; same contradiction "
          f"decision: {result['contradicted_agreement']}. Saved: data/eval/{SCORE_FILE}")


if __name__ == "__main__":
    try:
        {"sheet": _sheet_command, "score": _score_command}[sys.argv[1]]()
    except (KeyError, IndexError):
        print(__doc__)
    except (RejudgeError, sj.JudgingError) as error:
        print(f"[STOP] {error}")
        sys.exit(1)
