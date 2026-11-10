"""Tests for src/summary_r3_rejudge.py: the Round 3 consistency check (no model, no network)."""

import json

import pytest

from src import summary_judging as sj
from src import summary_r3_rejudge as rj


def _items(n=12):
    return [{"row": str(i), "kind": "company point", "text": f"point {i}", "evidence": "A1", "judgement": "supported",
             "contradicted": "no", "central": "", "note": ""} for i in range(1, n + 1)]


def _key(n=12, decoys=(2, 5)):
    return {"items": {str(i): {"fingerprint": sj._hash("company point", f"point {i}", "A1"), **({"decoy": True} if i in decoys else {})}
                      for i in range(1, n + 1)}}


def test_decoys_are_left_out_and_the_draw_is_fixed():
    items, key = _items(), _key()
    rows, rk = rj.sheet_rows(items, key)
    assert len(rows) == 2 and set(rk.values()).isdisjoint({2, 5})            # 20% of the 10 non-decoy items
    assert rj.sheet_rows(items, key) == (rows, rk)
    assert all(r["judgement"] == "" and "point" in r["text"] for r in rows)   # the first answer is never shown


def test_the_score_needs_every_row_and_reports_both_agreements():
    items, key = _items(), _key()
    rows, rk = rj.sheet_rows(items, key)
    with pytest.raises(rj.RejudgeError, match="not complete"):
        rj.score(items, rows, rk, key)
    for r in rows:
        r.update(judgement="supported", contradicted="no")
    rows[0].update(judgement="partly_supported", note="one detail is missing")
    result = rj.score(items, rows, rk, key)
    assert result["items"] == 2 and result["judgement_agreement"] == 0.5 and result["contradicted_agreement"] == 1.0
    assert result["seed"] == rj.SEED and "reported only" in result["scope"]


def test_the_sheet_waits_for_the_round_3_lock(tmp_path):
    with pytest.raises(rj.RejudgeError, match="locked first"):
        rj._sheet_command(eval_dir=tmp_path, fulltext_dir=tmp_path)
