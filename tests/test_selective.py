"""
Tests for selective ranking, simulated on saved model answers (no network, no model calls).

Route 1 of the free-plan work: the model checks only the articles that do not
name the company in the title, so a query sends fewer tokens.

    SA  title articles first, in O1 order, with no model call; the rest
        below them, by the model's answer (direct, indirect, not; no
        answer goes with indirect), ties by O1, then NewsAPI order
    SB  title articles and the rest answered `direct` form one top group in
        NewsAPI order, so strong body news can compete for the top places;
        then indirect (and no answer), then not

The simulation uses only answers that exist for the same article in the
same saved run; a title article's saved answer is never used by SA or SB.
Runtime from tokens is labelled an estimate; the real time comes from a
later free-plan run of only the articles SA or SB sends.
"""

import pytest

from src import selective as sv


def _c(key, mention, label=None, source="model"):
    check = None if label is None and source == "model" else {
        "label": label, "certainty": "sure", "reason": "", "source": source}
    return {"key": key, "title": key, "description": "", "mention_level": mention, "check": check}


CANDIDATES = [
    _c("T1", "title", "not"), _c("B1", "content", "direct"), _c("T2", "title", "direct"),
    _c("D1", "description", "not"), _c("B2", "content", "indirect"),
    _c("B3", "content", None, source="fallback"),
]


def _keys(ordered):
    return [c["key"] for c in ordered]


def test_sa_keeps_title_articles_on_top_in_o1_order_and_orders_the_rest_by_the_model():
    assert _keys(sv.order(CANDIDATES, "SA")) == ["T1", "T2", "B1", "B2", "B3", "D1"]


def test_sb_lets_a_body_article_answered_direct_compete_with_the_title_articles():
    assert _keys(sv.order(CANDIDATES, "SB")) == ["T1", "B1", "T2", "B2", "B3", "D1"]


def test_a_title_articles_saved_answer_is_never_used():
    flipped = [dict(c, check={"label": "direct" if c["check"] and c["check"]["label"] == "not" else "not",
                              "certainty": "sure", "reason": "", "source": "model"})
               if c["mention_level"] == "title" else c for c in CANDIDATES]
    for method in ("SA", "SB"):
        assert _keys(sv.order(flipped, method)) == _keys(sv.order(CANDIDATES, method))


def test_only_the_articles_without_a_title_mention_are_sent():
    assert sv.sent_keys(CANDIDATES, "SA") == sv.sent_keys(CANDIDATES, "SB") == ["B1", "D1", "B2", "B3"]
    assert sv.sent_keys(CANDIDATES, "O2") == [c["key"] for c in CANDIDATES]


def test_unknown_methods_are_refused():
    with pytest.raises(ValueError):
        sv.order(CANDIDATES, "SC")


def test_the_token_estimate_scales_with_the_articles_sent_and_says_so():
    estimate = sv.token_estimate(query_tokens=21_000, articles=25, sent=10, tpm=8000)
    assert estimate == {"kind": "estimate", "tokens": 8400, "seconds": pytest.approx(8400 / 8000 * 60)}


def test_lost_single_report_events_are_named():
    events = {"T1": "e1", "B1": "e2", "B2": "e2", "T2": "e3"}
    lost = sv.single_report_losses(base=["T1", "B1", "T2"], other=["B1", "B2", "X"], events=events,
                                   candidate_keys=["T1", "B1", "B2", "T2", "X"])
    assert lost == ["e1", "e3"]                          # e2 has two reports and stays covered


def test_the_simulation_measures_every_method_and_keeps_the_acceptance_rule():
    labels = {"T1": "not", "B1": "direct", "T2": "direct", "D1": "not", "B2": "indirect", "B3": "indirect"}
    events = {"B1": "e1", "T2": "e2", "B2": "e3", "B3": "e4"}
    result = sv.simulate(CANDIDATES, labels, events, ks=(2, 4), primary_k=2, threshold=0.5,
                         seconds={"SA": 100.0, "SB": 100.0, "O2": 300.0, "O3": 300.0})
    assert set(result["configurations"]) >= {"O1+S1", "O2+S1", "O3+S1", "SA+S1", "SB+S1", "SA+S2", "SB+S2"}
    assert set(result["acceptance"]) == set(result["configurations"]) - {"O1+S1"}
    assert result["acceptance"]["O2+S1"]["conditions"]["runtime"] is False       # 300 s
    for name in ("SA+S1", "SB+S1"):
        assert "single_report_losses" in result["configurations"][name][2]
        assert "wrong_promotions" in result["configurations"][name][2]
    assert result["configurations"]["O1+S1"][2]["wrong_promotions"] == []
