"""
Tests for the development ranking report (no network, no model calls).

Each development run is rebuilt as the app would see it: the run's searches
merged, Stage 1 on full text (after the sanity check), duplicates merged by
the cleaning rules. Where the company is named (title, description or only
the body) is read from the article's own fields, so a name found only in the
full text counts as "body". Then all six configurations (O1-O3 x S1-S2) are
measured at k = 5, 10 and 15 against the hand-made events.

The similarity threshold for S2 is chosen on development data by a rule
fixed in advance: the fewest grouping errors (false merges + missed merges);
on a tie, the higher threshold (fewer merges, so fewer events pushed out).
"""

import pytest

from src import ranking_eval as re_
from src.fulltext import article_id


def _raw(n, title, description="", url=None, content=""):
    return {"source": {"id": None, "name": f"Source {n}"}, "author": "A", "title": title,
            "description": description, "content": content, "url": url or f"https://x.com/{n}",
            "urlToImage": None, "publishedAt": "2026-10-20T10:00:00Z"}


IVECO = _raw(1, "Tata Motors raises Iveco takeover offer to 14.40 euro", "The carmaker raised its offer.")
IVECO_COPY = _raw(2, "Tata Motors raises Iveco takeover offer to 14.40 euro", "Same story, another site.",
                  url="https://y.com/2")
NAVRATRI = _raw(3, "Car registrations plunge 22% this Navratri", "Registrations fell during the festival.")
NAVRATRI_FULL = ("Car registrations fell 22% during the Navratri festival, data from dealers showed.\n"
                 "Tata Motors fell 16.4% to 24,843 units, while Hyundai dropped 2.7%.")
CRUDE = _raw(4, "Crude oil rises above $114", "Brent rose on supply fears.")
JAGUAR = _raw(5, "Jaguar Land Rover opens new showroom in Delhi", "JLR expands its retail network.")


# ---------- the runs ----------

def test_dev_runs_mirror_the_app_searches():
    assert re_.DEV_RUNS == {
        "2026-10-14": ["20261014_182548_tata-motors.json"],
        "2026-10-23": ["20261023_161430_tata-motors.json"],
        "2026-10-24": ["20261024_111739_tata-motors.json", "20261024_111739_tata-motors_2.json"],
    }


def test_runs_merge_their_searches_each_article_once():
    merged = re_.merge_run([{"articles": [IVECO, CRUDE]}, {"articles": [dict(IVECO), JAGUAR]}])
    assert [a["url"] for a in merged] == [IVECO["url"], CRUDE["url"], JAGUAR["url"]]


# ---------- stage 1 candidates ----------

def test_candidates_are_the_articles_stage_1_keeps_in_newsapi_order():
    candidates = re_.stage1_candidates([IVECO, CRUDE, NAVRATRI], {NAVRATRI["url"]: NAVRATRI_FULL},
                                       "Tata Motors", aliases=None)
    assert [c["key"] for c in candidates] == [article_id(IVECO["url"]), article_id(NAVRATRI["url"])]


def test_a_name_found_only_in_the_full_text_counts_as_body():
    candidates = re_.stage1_candidates([NAVRATRI], {NAVRATRI["url"]: NAVRATRI_FULL}, "Tata Motors", aliases=None)
    assert candidates[0]["mention_level"] == "content"
    assert candidates[0]["description"] == "Registrations fell during the festival."   # not the full text


def test_copies_of_one_headline_become_one_candidate():
    candidates = re_.stage1_candidates([IVECO, IVECO_COPY], {}, "Tata Motors", aliases=None)
    assert len(candidates) == 1


def test_candidates_carry_the_fields_ranking_needs():
    candidate = re_.stage1_candidates([IVECO], {}, "Tata Motors", aliases=None)[0]
    assert {"key", "title", "description", "source", "mention_level", "url", "check"} <= set(candidate)
    assert candidate["mention_level"] == "title" and candidate["check"] is None


# ---------- one run, six configurations ----------

def _cand(key, mention, label, title, relevant_label):
    check = None if label is None else {"label": label, "certainty": "sure", "reason": "r", "source": "model"}
    return {"key": key, "title": title, "description": "", "source": "S", "mention_level": mention,
            "url": f"https://x.com/{key}", "check": check}, relevant_label


ROWS = [_cand("a", "content", "direct", "Tata Motors Q2 profit rebounds to 1257 crore", "direct"),
        _cand("b", "title", "not", "TMPV among stocks at 52-week low", "not"),
        _cand("c", "title", "direct", "Tata Motors raises Iveco offer", "direct"),
        _cand("d", "title", "direct", "Tata Motors raises Iveco offer to 14.40 euro", "direct"),
        _cand("e", "description", "indirect", "Navratri car registrations fall 22 percent", "indirect")]
CANDIDATES = [row[0] for row in ROWS]
LABELS = {row[0]["key"]: row[1] for row in ROWS}
EVENTS = {"a": "q2_results", "c": "iveco_offer", "d": "iveco_offer", "e": "navratri_sales"}


def test_a_run_is_measured_for_all_six_configurations_at_each_k():
    report = re_.evaluate_run(CANDIDATES, LABELS, EVENTS, ks=(2, 3), threshold=0.5)
    assert set(report["configurations"]) == {"O1+S1", "O1+S2", "O2+S1", "O2+S2", "O3+S1", "O3+S2"}
    assert set(report["configurations"]["O1+S1"]) == {2, 3}
    assert report["configurations"]["O1+S1"][2]["selected_keys"] == ["b", "c"]          # title mentions first
    assert report["configurations"]["O2+S1"][2]["selected_keys"] == ["a", "c"]          # model labels first


def test_acceptance_is_judged_against_the_baseline_at_the_primary_k():
    report = re_.evaluate_run(CANDIDATES, LABELS, EVENTS, ks=(2, 3), threshold=0.5, primary_k=2,
                              seconds=100)
    verdict = report["acceptance"]["O2+S1"]
    assert verdict["events"]["gained"] == ["q2_results"] and verdict["events"]["lost"] == []
    assert verdict["conditions"]["runtime"] is True and verdict["conditions"]["fallback"] is True
    assert "O1+S1" not in report["acceptance"]                                            # the baseline itself


def test_without_model_answers_only_the_o1_configurations_are_measured():
    plain = [dict(c, check=None) for c in CANDIDATES]
    report = re_.evaluate_run(plain, LABELS, EVENTS, ks=(2,), threshold=0.5)
    assert set(report["configurations"]) == {"O1+S1", "O1+S2"}
    assert report["not_run"] == ["O2+S1", "O2+S2", "O3+S1", "O3+S2"]


def test_the_report_shows_title_protection_and_retention():
    report = re_.evaluate_run(CANDIDATES, LABELS, EVENTS, ks=(2,), threshold=0.5)
    assert report["title_protection"] == {"rescued": [], "promoted": ["b"]}
    assert report["retention_same"] is True


def test_fallback_is_checked_on_the_real_candidates():
    report = re_.evaluate_run(CANDIDATES, LABELS, EVENTS, ks=(2,), threshold=0.5)
    assert report["fallback_ok"] is True            # without answers, O2 and O3 give the O1 order


def test_missing_event_names_stop_the_report():
    with pytest.raises(re_.RankingError):
        re_.evaluate_run(CANDIDATES, LABELS, {"a": "q2_results"}, ks=(2,), threshold=0.5)


# ---------- the S2 threshold ----------

def test_the_threshold_with_the_fewest_grouping_errors_is_chosen():
    choice = re_.choose_threshold(CANDIDATES, EVENTS, thresholds=(0.0, 0.5, 1.01))
    # 0.0 merges everything (many false merges); 1.01 merges nothing (misses c-d); 0.5 merges only c-d
    assert choice["threshold"] == 0.5
    assert set(choice["errors"]) == {0.0, 0.5, 1.01}


def test_a_tie_in_errors_goes_to_the_higher_threshold():
    candidates = [c for c in CANDIDATES if c["key"] in ("a", "e")]              # no true pairs at all
    choice = re_.choose_threshold(candidates, EVENTS, thresholds=(0.6, 0.8))
    assert choice["threshold"] == 0.8


def test_the_threshold_candidates_are_fixed():
    assert re_.THRESHOLDS == (0.3, 0.4, 0.5, 0.6, 0.7)
    assert re_.KS == (5, 10, 15) and re_.PRIMARY_K == 10


# ---------- cached answers are not a live runtime ----------

class _Llm:
    model_name, cache_tag = "openai/gpt-oss-120b", "groq|t"


def test_answers_already_in_the_cache_are_counted(tmp_path):
    from src import relevance_check as rc
    cache = rc.CheckCache(tmp_path)
    items = [{"id": "A1", "title": "T1", "passage": "P1"}, {"id": "A2", "title": "T2", "passage": "P2"}]
    cache.put(rc._key(items[0], "Tata Motors", _Llm.model_name, "scope", _Llm.cache_tag), {"source": "model"})
    assert re_.cache_hits(items, _Llm(), "Tata Motors", "scope", cache) == ["A1"]


def test_a_runtime_with_cached_answers_is_not_a_live_runtime():
    assert re_.live_seconds({"seconds_total": 40.0, "from_cache": 0}) == 40.0
    assert re_.live_seconds({"seconds_total": 1.2, "from_cache": 30}) is None
    assert re_.live_seconds(None) is None


# ---------- the model setting comes from the comparison's fixed list ----------

def test_the_check_uses_the_named_comparison_setting(monkeypatch):
    seen = {}
    monkeypatch.setattr("src.llm.make_llm", lambda provider=None, **kw: seen.update(provider=provider, **kw) or "llm")
    assert re_.check_llm("gpt-oss-120b") == "llm"
    assert seen == {"provider": "groq", "model": "openai/gpt-oss-120b",
                    "generation": {"temperature": 1.0, "reasoning_effort": "medium"}}


def test_an_unknown_setting_name_is_refused():
    with pytest.raises(ValueError):
        re_.check_llm("gpt-5")


@pytest.mark.parametrize("argv, name", [(["--check"], None), (["--check", "gpt-oss-120b"], "gpt-oss-120b"), ([], False)])
def test_the_command_line_names_the_setting(argv, name):
    assert re_.check_argument(argv) == name
