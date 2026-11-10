"""
Tests for ranking and selection (no network, no model calls).

Stage 1 keeps every candidate. Ranking decides the order, and selection
decides which k articles feed the summary; the rest are shown as
"Additional articles — lower relevance ranking".

Orderings:
    O1 - Stage 1: company named in the title, then the description, then
         only the body; ties keep NewsAPI's order
    O2 - Gemini: direct, then indirect, then not; ties keep NewsAPI's order
    O3 - Gemini with title protection: an article that names the company in
         its title but is answered `not` is placed with the indirect group;
         ties broken by O1, then NewsAPI's order
    An article without a model answer (fallback) is placed with the indirect
    group. With no model answers at all, O2 and O3 give the O1 order.

Selection:
    S1 - the first k of the ordering
    S2 - the first k, skipping an article too similar to one already chosen
         (shared content words of title + description, Jaccard >= threshold);
         free slots at the end are filled from the skipped ones, in order.
         Each chosen article lists the similar candidates as "also reported by".

Measures, at k (each kept separate):
    precision         - relevant selected / selected (fewer than k: divide by the number selected)
    article recall    - relevant selected / relevant candidates
    event recall      - relevant events covered / relevant events among the candidates
    slot use          - relevant events covered / min(k, relevant events)
    direct events     - the same two, for events with at least one `direct` article
Comparison with the baseline (O1 + S1) uses event names: gained, lost, net.
"""

import pytest

from src import ranking as rk


def _c(key, mention="title", label=None, certainty="sure", title=None, description="", source="News"):
    check = None if label is None else (
        {"label": None, "certainty": None, "reason": "", "source": "fallback"} if label == "fallback"
        else {"label": label, "certainty": certainty, "reason": f"reason {key}", "source": "model"})
    return {"key": key, "title": title or f"Story {key}", "description": description, "source": source,
            "mention_level": mention, "check": check}


def _keys(articles):
    return [a["key"] for a in articles]


# ---------- orderings ----------

CANDIDATES = [_c("a", "content", "direct"), _c("b", "title", "not"), _c("c", "description", "indirect"),
              _c("d", "title", "direct"), _c("e", "content", "not"), _c("f", "description", "fallback")]


def test_o1_orders_by_where_the_company_is_named_and_keeps_newsapi_order_in_ties():
    assert _keys(rk.order(CANDIDATES, "O1")) == ["b", "d", "c", "f", "a", "e"]


def test_o2_orders_by_the_model_label_and_puts_a_fallback_with_indirect():
    assert _keys(rk.order(CANDIDATES, "O2")) == ["a", "d", "c", "f", "b", "e"]


def test_o3_protects_a_title_mention_answered_not_and_breaks_ties_by_o1():
    # group direct: d (title), a (content); group indirect: b (title, protected), c (description), f (description)
    assert _keys(rk.order(CANDIDATES, "O3")) == ["d", "a", "b", "c", "f", "e"]


def test_without_any_model_answer_every_ordering_is_o1():
    plain = [dict(c, check=None) for c in CANDIDATES]
    assert _keys(rk.order(plain, "O2")) == _keys(rk.order(plain, "O3")) == _keys(rk.order(plain, "O1"))


def test_ordering_does_not_drop_or_change_candidates():
    for method in ("O1", "O2", "O3"):
        ordered = rk.order(CANDIDATES, method)
        assert sorted(_keys(ordered)) == sorted(_keys(CANDIDATES))
    assert CANDIDATES[0] == _c("a", "content", "direct")


def test_an_unknown_ordering_is_an_error():
    with pytest.raises(ValueError):
        rk.order(CANDIDATES, "O9")


def test_title_protection_effect_lists_rescued_and_promoted_articles():
    relevant = {"a": True, "b": True, "c": True, "d": True, "e": False, "f": True,
                "g": False}
    candidates = CANDIDATES + [_c("g", "title", "not")]
    effect = rk.title_protection_effect(candidates, relevant)
    assert effect == {"rescued": ["b"], "promoted": ["g"]}


# ---------- similarity and selection ----------

IVECO_1 = _c("i1", title="Tata Motors raises Iveco takeover offer to 14.40 euro per share",
             description="Tata Motors raised its offer for Iveco.", source="Reuters")
IVECO_2 = _c("i2", title="Tata Motors raises Iveco takeover offer price to 14.40 euro",
             description="The offer for Iveco was raised.", source="ET")
EV_SALES = _c("e1", title="Tata Motors electric vehicle sales grow 40% in October",
              description="EV sales rose.")
EV_PLANT = _c("e2", title="Tata Motors electric vehicle plant expansion in Sanand",
              description="A new EV plant line.")
OTHER = _c("o1", title="Jaguar unveils Type 01 electric car", description="Jaguar launched the Type 01.")


def test_similarity_is_the_share_of_shared_content_words():
    assert rk.similarity(IVECO_1, IVECO_2) > rk.similarity(IVECO_1, OTHER)
    assert rk.similarity(IVECO_1, IVECO_1) == 1.0
    assert 0.0 <= rk.similarity(EV_SALES, EV_PLANT) <= 1.0


def test_s1_takes_the_first_k():
    chosen = rk.select([IVECO_1, IVECO_2, OTHER], k=2, method="S1")
    assert _keys(chosen["selected"]) == ["i1", "i2"] and _keys(chosen["additional"]) == ["o1"]


def test_s2_skips_a_near_copy_and_fills_with_the_next_article():
    chosen = rk.select([IVECO_1, IVECO_2, OTHER], k=2, method="S2", threshold=0.4)
    assert _keys(chosen["selected"]) == ["i1", "o1"] and _keys(chosen["additional"]) == ["i2"]


def test_s2_records_the_other_sources_of_a_chosen_article():
    chosen = rk.select([IVECO_1, IVECO_2, OTHER], k=2, method="S2", threshold=0.4)
    assert chosen["also_reported_by"]["i1"] == ["i2"] and chosen["also_reported_by"]["o1"] == []


def test_s2_fills_free_slots_from_skipped_articles_in_order():
    chosen = rk.select([IVECO_1, IVECO_2, OTHER], k=3, method="S2", threshold=0.4)
    assert _keys(chosen["selected"]) == ["i1", "o1", "i2"]


def test_fewer_candidates_than_k_selects_them_all():
    for method in ("S1", "S2"):
        chosen = rk.select([IVECO_1, OTHER], k=10, method=method, threshold=0.4)
        assert sorted(_keys(chosen["selected"])) == ["i1", "o1"] and chosen["additional"] == []


def test_grouping_quality_counts_false_and_missed_merges_against_hand_made_events():
    candidates = [IVECO_1, IVECO_2, EV_SALES, EV_PLANT, OTHER]
    events = {"i1": "iveco_offer_raised", "i2": "iveco_offer_raised", "e1": "ev_sales",
              "e2": "ev_plant", "o1": "type01_launch"}
    quality = rk.grouping_quality(candidates, events, threshold=0.0)        # threshold 0: everything merges
    assert quality["true_merges"] == 1 and quality["false_merges"] == 9 and quality["missed_merges"] == 0
    strict = rk.grouping_quality(candidates, events, threshold=1.01)        # nothing merges
    assert strict["true_merges"] == 0 and strict["false_merges"] == 0 and strict["missed_merges"] == 1


# ---------- measures ----------

RELEVANT = {"i1": True, "i2": True, "e1": True, "e2": True, "o1": True, "x": False}
EVENTS = {"i1": "iveco", "i2": "iveco", "e1": "ev_sales", "e2": "ev_plant", "o1": "type01"}
LABELS = {"i1": "direct", "i2": "direct", "e1": "indirect", "e2": "direct", "o1": "direct", "x": "not"}


def test_measures_keep_articles_events_and_direct_events_apart():
    m = rk.measures(["i1", "i2", "x"], candidate_keys=list(RELEVANT), relevant=RELEVANT, events=EVENTS,
                    labels=LABELS, k=3)
    assert m["selected"] == 3 and m["relevant_selected"] == 2
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["article_recall"] == pytest.approx(2 / 5)
    assert m["events_covered"] == ["iveco"] and m["relevant_events"] == 4
    assert m["event_recall"] == pytest.approx(1 / 4)
    assert m["slot_use"] == pytest.approx(1 / 3)
    assert m["direct_events_covered"] == ["iveco"] and m["direct_events"] == 3      # iveco, ev_plant, type01
    assert m["direct_event_recall"] == pytest.approx(1 / 3)


def test_precision_with_fewer_selected_than_k_divides_by_the_number_selected():
    m = rk.measures(["i1", "x"], candidate_keys=["i1", "x"], relevant=RELEVANT, events=EVENTS, labels=LABELS, k=10)
    assert m["selected"] == 2 and m["precision"] == pytest.approx(1 / 2)


def test_events_of_relevant_articles_outside_the_candidates_do_not_count():
    m = rk.measures(["i1"], candidate_keys=["i1", "i2"], relevant=RELEVANT, events=EVENTS, labels=LABELS, k=1)
    assert m["relevant_events"] == 1 and m["event_recall"] == 1.0


def test_a_relevant_candidate_without_an_event_name_is_an_error():
    with pytest.raises(rk.RankingError):
        rk.measures(["i1"], candidate_keys=["i1", "e1"], relevant=RELEVANT, events={"i1": "iveco"},
                    labels=LABELS, k=1)


def test_comparison_with_the_baseline_uses_event_names():
    base = {"events_covered": ["a", "b", "c"], "direct_events_covered": ["a", "c"]}
    other = {"events_covered": ["a", "b", "d"], "direct_events_covered": ["a", "d"]}
    assert rk.compare_events(base, other) == {"gained": ["d"], "lost": ["c"], "net": 0,
                                              "direct_gained": ["d"], "direct_lost": ["c"], "direct_net": 0}


# ---------- acceptance and the winner ----------

BASE = {"events_covered": ["a", "b", "c"], "direct_events_covered": ["a"], "event_recall": 0.5,
        "direct_event_recall": 0.5, "precision": 0.7}


def _x(events, direct, event_recall, direct_recall, precision):
    return {"events_covered": events, "direct_events_covered": direct, "event_recall": event_recall,
            "direct_event_recall": direct_recall, "precision": precision}


def test_a_configuration_passes_with_a_direct_gain_and_no_losses():
    x = _x(["a", "b", "c", "d"], ["a", "d"], 0.6, 1.0, 0.7)
    verdict = rk.acceptance(BASE, x, uses_model=True, retention_same=True, seconds=150, fallback_ok=True)
    assert verdict["passed"] is True and all(verdict["conditions"].values())


@pytest.mark.parametrize("change, failed", [
    ({"events_covered": ["a", "b", "c"], "direct_events_covered": ["a"]}, "gain"),          # nothing gained
    ({"direct_events_covered": ["d"]}, "no_direct_loss"),                                     # lost direct a
    ({"event_recall": 0.4}, "event_recall_not_lower"),
    ({"precision": 0.6}, "precision_not_lower"),
])
def test_each_condition_can_fail_on_its_own(change, failed):
    x = {**_x(["a", "b", "c", "d"], ["a", "d"], 0.6, 1.0, 0.7), **change}
    verdict = rk.acceptance(BASE, x, uses_model=True, retention_same=True, seconds=150, fallback_ok=True)
    assert verdict["passed"] is False and verdict["conditions"][failed] is False


def test_two_events_net_is_enough_without_a_direct_gain():
    x = _x(["a", "b", "c", "d", "e"], ["a"], 0.7, 0.5, 0.7)
    assert rk.acceptance(BASE, x, uses_model=True, retention_same=True, seconds=150, fallback_ok=True)["passed"]


def test_a_model_configuration_fails_when_too_slow_or_without_a_working_fallback():
    x = _x(["a", "b", "c", "d"], ["a", "d"], 0.6, 1.0, 0.7)
    slow = rk.acceptance(BASE, x, uses_model=True, retention_same=True, seconds=181, fallback_ok=True)
    broken = rk.acceptance(BASE, x, uses_model=True, retention_same=True, seconds=100, fallback_ok=False)
    assert slow["conditions"]["runtime"] is False and broken["conditions"]["fallback"] is False


def test_runtime_and_fallback_do_not_apply_without_the_model():
    x = _x(["a", "b", "c", "d"], ["a", "d"], 0.6, 1.0, 0.7)
    verdict = rk.acceptance(BASE, x, uses_model=False, retention_same=True, seconds=None, fallback_ok=None)
    assert verdict["passed"] is True and "runtime" not in verdict["conditions"]


def test_retention_must_stay_the_same():
    x = _x(["a", "b", "c", "d"], ["a", "d"], 0.6, 1.0, 0.7)
    assert rk.acceptance(BASE, x, uses_model=False, retention_same=False, seconds=None,
                         fallback_ok=None)["passed"] is False


def test_the_winner_is_chosen_only_from_passing_configurations_in_the_fixed_order():
    results = {
        "O2+S2": {"passed": False, "direct_event_recall": 1.0, "event_recall": 1.0, "precision": 1.0},
        "O3+S2": {"passed": True, "direct_event_recall": 0.8, "event_recall": 0.7, "precision": 0.8},
        "O3+S1": {"passed": True, "direct_event_recall": 0.8, "event_recall": 0.7, "precision": 0.8},
        "O2+S1": {"passed": True, "direct_event_recall": 0.8, "event_recall": 0.6, "precision": 0.9},
    }
    assert rk.choose_winner(results) == "O3+S1"                     # tie on all measures: the simpler one


def test_with_no_passing_configuration_the_baseline_stays():
    assert rk.choose_winner({"O2+S1": {"passed": False, "direct_event_recall": 1.0, "event_recall": 1.0,
                                       "precision": 1.0}}) == "O1+S1"


def test_the_simplicity_order_is_fixed():
    assert rk.SIMPLICITY == ["O1+S1", "O1+S2", "O3+S1", "O3+S2", "O2+S1", "O2+S2"]
    assert rk.MAX_MODEL_SECONDS == 180
