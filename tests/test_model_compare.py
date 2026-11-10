"""
Tests for the model comparison on the development runs (fake models, fake clock: no keys, no network).

Configurations are fixed before any run: provider, exact model, generation
settings. A `low` reasoning version of a Groq configuration is added only
when its pilot estimate is over the 180-second limit, never because of a
quality score.

    pilot        one batch gives a token estimate per article; the estimate for
                 a whole query is labelled as an estimate, never as a measurement
    query run    each dev run is one real query: its own candidates, no saved
                 answers, wall-clock seconds from start to the last answer or
                 fallback (waits, retries, splits included)
    model        coverage, precision on the answered articles, model recall
                 (a relevant article the model fails on is not found), F1,
                 unsure share, fallbacks by kind
    end to end   a failed article stays in the ranking, in the middle group
                 (with the articles the model calls indirect), ordered by its
                 Stage 1 mention level; it is never dropped and never treated
                 as not; the acceptance rule is unchanged; wrong promotions
                 are counted
    consistency  an article in several runs: same input text and prompt ->
                 a different label is an inconsistency; different input is
                 reported on its own
    choice       eligible: a model configuration passes acceptance on at least
                 2 of the 3 runs; then events at k=10, wrong promotions, F1,
                 fallbacks, seconds, data terms; none eligible -> O1
"""

import json

import pytest

from src import model_compare as mc
from src import ranking as rk


# ---------- configurations ----------

def test_the_configurations_are_fixed_with_exact_settings():
    assert mc.CONFIGS == {
        "gemini-3.8-flash": {"provider": "gemini", "model": "gemini-3.8-flash",
                             "generation": {"temperature": 1.0, "thinking_level": "medium"}},
        "gpt-oss-120b": {"provider": "groq", "model": "openai/gpt-oss-120b",
                         "generation": {"temperature": 1.0, "reasoning_effort": "medium"}},
        "gpt-oss-20b": {"provider": "groq", "model": "openai/gpt-oss-20b",
                        "generation": {"temperature": 1.0, "reasoning_effort": "medium"}},
        "qwen3.8-27b": {"provider": "groq", "model": "qwen/qwen3.8-27b",
                        "generation": {"temperature": 1.0, "top_p": 0.95, "reasoning_effort": "medium",
                                       "reasoning_format": "hidden"}},
    }


def test_low_reasoning_is_added_only_for_a_groq_configuration_over_the_time_limit():
    estimates = {"gemini-3.8-flash": 400.0, "gpt-oss-120b": 250.0, "gpt-oss-20b": 120.0, "qwen3.8-27b": 181.0}
    configs = mc.configs_after_pilot(estimates)
    assert set(configs) == set(mc.CONFIGS) | {"gpt-oss-120b-low", "qwen3.8-27b-low"}
    assert configs["gpt-oss-120b-low"]["generation"]["reasoning_effort"] == "low"
    assert configs["gpt-oss-120b"] == mc.CONFIGS["gpt-oss-120b"]            # the medium version stays as it is


def test_the_choice_of_configurations_never_reads_quality():
    import inspect
    assert list(inspect.signature(mc.configs_after_pilot).parameters) == ["pilot_seconds"]


# ---------- pilot ----------

def test_the_pilot_estimate_scales_tokens_per_article_and_says_it_is_an_estimate():
    per_call = [{"input": 4000, "output": 900, "reasoning": 700}]
    estimate = mc.pilot_estimate(per_call, pilot_articles=5, query_articles=25, tpm=8000)
    assert estimate == {"kind": "estimate", "tokens_per_article": 980.0, "query_tokens": 24500,
                        "seconds": pytest.approx(24500 / 8000 * 60)}


def test_the_report_puts_the_estimate_beside_the_measured_time():
    row = mc.time_row(estimate={"kind": "estimate", "seconds": 183.75}, measured_seconds=201.3)
    assert row == {"estimated_seconds": 183.75, "measured_seconds": 201.3, "difference_seconds": pytest.approx(17.55)}


# ---------- one query, no saved answers ----------

class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeLLM:
    def __init__(self, clock, seconds_per_call=10.0, fail_ids=(), label="direct", cache_tag="t"):
        self.clock, self.seconds, self.fail_ids, self.label = clock, seconds_per_call, set(fail_ids), label
        self.provider, self.model_name, self.generation, self.cache_tag = "groq", "m", {"temperature": 1.0}, cache_tag
        self.stats = {"calls": 0, "per_call": [], "waits": [], "waited_seconds": 0.0}
        self.prompts = []

    def __call__(self, prompt):
        from src.summarizer import LLMUnavailable
        self.clock.now += self.seconds
        self.stats["calls"] += 1
        self.prompts.append(prompt)
        ids = [p.split('"')[0] for p in prompt.split('<article id="')[1:]]
        if self.fail_ids & set(ids):
            raise LLMUnavailable("server")
        return json.dumps({"results": [{"id": i, "label": self.label, "certainty": "sure", "reason": "r"}
                                       for i in ids]})


def _candidates(n, prefix="A"):
    return [{"key": f"{prefix}{i}", "title": f"Tata Motors item {i}", "text": f"Tata Motors text {i}.",
             "description": f"d {i}", "mention": "title"} for i in range(n)]


def test_a_query_run_never_uses_saved_answers_and_measures_wall_clock(tmp_path):
    clock = Clock()
    llm = FakeLLM(clock)
    result = mc.run_query(_candidates(25), llm, "Tata Motors", None, clock=clock)
    assert result["cached"] is False and llm.stats["calls"] == 3              # 25 articles in batches of 10
    assert result["wall_seconds"] == pytest.approx(30.0)
    assert set(result["answers"]) == {f"A{i}" for i in range(25)}            # every article has a record


def test_a_query_run_records_the_input_of_every_article():
    clock = Clock()
    result = mc.run_query(_candidates(3), FakeLLM(clock), "Tata Motors", None, clock=clock)
    assert set(result["inputs"]) == {"A0", "A1", "A2"} and all(len(f) == 64 for f in result["inputs"].values())


def test_a_query_run_keeps_failed_articles_as_fallbacks():
    clock = Clock()
    result = mc.run_query(_candidates(12), FakeLLM(clock, fail_ids={"A11"}), "Tata Motors", None, clock=clock)
    assert result["answers"]["A11"]["source"] == "fallback" and len(result["answers"]) == 12


# ---------- model measures ----------

def _answer(label=None, certainty="sure", source="model", error=""):
    return {"label": label, "certainty": certainty, "reason": "", "source": source, "error": error, "kind": error}


def test_model_measures_keep_coverage_apart_and_count_a_failed_relevant_article_as_not_found():
    answers = {"A": _answer("direct"), "B": _answer("not"), "C": _answer(source="fallback", error="server"),
               "D": _answer("indirect", "unsure"), "E": _answer("direct")}
    labels = {"A": "direct", "B": "not", "C": "direct", "D": "not", "E": "indirect"}
    m = mc.model_measures(answers, labels)
    assert m["articles"] == 5 and m["answered"] == 4 and m["coverage"] == pytest.approx(0.8)
    assert m["precision"] == pytest.approx(2 / 3)                         # A, E right; D wrong; C claims nothing
    assert m["model_recall"] == pytest.approx(2 / 3)                      # A, E found; C (failed) not found
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["unsure_share"] == pytest.approx(1 / 4) and m["fallbacks"] == {"server": 1}


def test_a_model_that_answers_nothing_gets_zero_not_an_error():
    m = mc.model_measures({"A": _answer(source="fallback", error="daily_limit")}, {"A": "direct"})
    assert m["coverage"] == 0 and m["precision"] is None and m["model_recall"] == 0 and m["f1"] == 0


# ---------- end to end ----------

def test_a_wrong_promotion_is_a_not_relevant_article_a_model_moves_into_the_top_k():
    report = {"configurations": {
        "O1+S1": {10: {"selected_keys": ["A", "B", "C"]}},
        "O2+S1": {10: {"selected_keys": ["A", "X", "C"]}},
        "O3+S1": {10: {"selected_keys": ["A", "B", "Y"]}}}}
    labels = {"A": "direct", "B": "not", "C": "indirect", "X": "not", "Y": "direct"}
    assert mc.wrong_promotions(report, labels, k=10) == {"O2+S1": ["X"], "O3+S1": []}


def test_a_failed_article_stays_in_the_ranking_in_the_middle_group_never_as_not():
    candidates = [{"key": k, "title": k, "mention_level": "title", "check": c} for k, c in [
        ("A", _answer("not")), ("B", _answer(source="fallback", error="server")), ("C", _answer("direct"))]]
    assert [c["key"] for c in rk.order(candidates, "O2")] == ["C", "B", "A"]


# ---------- consistency ----------

def test_a_label_change_on_the_same_input_is_an_inconsistency():
    runs = {"r1": {"answers": {"A": _answer("direct"), "B": _answer("not")}, "inputs": {"A": "f1", "B": "f2"}},
            "r2": {"answers": {"A": _answer("indirect"), "B": _answer("not")}, "inputs": {"A": "f1", "B": "f2"}}}
    c = mc.consistency(runs)
    assert c == {"repeated": 2, "same_input": 2, "different_input": [], "label_changes": ["A"],
                 "relevance_changes": []}


def test_a_different_input_is_reported_apart_and_never_called_inconsistent():
    runs = {"r1": {"answers": {"A": _answer("direct")}, "inputs": {"A": "f1"}},
            "r2": {"answers": {"A": _answer("not")}, "inputs": {"A": "f9"}}}
    c = mc.consistency(runs)
    assert c["different_input"] == ["A"] and c["label_changes"] == [] and c["same_input"] == 0


def test_a_change_between_relevant_and_not_is_listed_too():
    runs = {"r1": {"answers": {"A": _answer("direct")}, "inputs": {"A": "f"}},
            "r2": {"answers": {"A": _answer("not")}, "inputs": {"A": "f"}}}
    assert mc.consistency(runs)["relevance_changes"] == ["A"]


def test_pooled_answers_use_each_article_once_from_its_first_run():
    runs = {"r1": {"answers": {"A": _answer("direct")}}, "r2": {"answers": {"A": _answer("not"), "B": _answer("not")}}}
    pooled = mc.pooled_answers(runs, order=["r1", "r2"])
    assert pooled["A"]["label"] == "direct" and set(pooled) == {"A", "B"}


# ---------- the choice ----------

def _summary(passes, events=10, wrong=0, f1=0.9, fallbacks=0, seconds=100.0, provider="groq", complete=True):
    return {"runs_passed": passes, "events_k10": events, "wrong_promotions": wrong, "f1": f1,
            "fallbacks": fallbacks, "seconds": seconds, "provider": provider, "complete": complete}


def test_only_configurations_passing_two_of_three_runs_are_eligible():
    summaries = {"a": _summary(1, events=20), "b": _summary(2, events=12), "c": _summary(3, events=11)}
    assert mc.choose(summaries) == "b"


@pytest.mark.parametrize("first, second, winner", [
    (_summary(2, events=12), _summary(2, events=11), "first"),
    (_summary(2, wrong=1), _summary(2, wrong=0), "second"),
    (_summary(2, f1=0.8), _summary(2, f1=0.85), "second"),
    (_summary(2, fallbacks=2), _summary(2, fallbacks=0), "second"),
    (_summary(2, seconds=150), _summary(2, seconds=90), "second"),
    (_summary(2, provider="gemini"), _summary(2, provider="groq"), "second"),
])
def test_the_order_of_choice(first, second, winner):
    assert mc.choose({"first": first, "second": second}) == winner


def test_with_no_eligible_configuration_the_dev_choice_is_o1():
    assert mc.choose({"a": _summary(1), "b": _summary(0)}) == "O1"


def test_an_incomplete_run_is_never_eligible():
    assert mc.choose({"a": _summary(3, complete=False), "b": _summary(2)}) == "b"


def test_the_choice_is_for_the_fresh_test_only():
    assert "fresh test" in mc.CHOICE_NOTE and "default" in mc.CHOICE_NOTE


def test_summarize_reads_a_real_evaluation_report():
    from src.ranking_eval import evaluate_run
    keys = [f"K{i}" for i in range(12)]
    labels = {k: ("direct" if i % 3 == 0 else "indirect" if i % 3 == 1 else "not") for i, k in enumerate(keys)}
    events = {k: f"e{i}" for i, k in enumerate(keys) if labels[k] != "not"}
    candidates = [{"key": k, "title": f"Tata Motors news {i}", "mention_level": "body" if i < 6 else "title",
                   "check": _answer(labels[k])} for i, k in enumerate(keys)]
    report = evaluate_run(candidates, labels, events, threshold=0.5, seconds=30.0)
    reports = {"r1": report, "r2": report, "r3": report}
    wrong = {r: mc.wrong_promotions(report, labels) for r in reports}
    measures = mc.model_measures({c["key"]: c["check"] for c in candidates}, labels)
    s = mc.summarize("groq", reports, wrong, measures, {"r1": 30.0, "r2": 20.0, "r3": 25.0}, complete=True)
    assert s["provider"] == "groq" and s["ranking"] in rk.SIMPLICITY and not s["ranking"].startswith("O1")
    assert s["runs_passed"] in (0, 3) and s["seconds"] == 30.0 and s["f1"] == 1.0 and s["fallbacks"] == 0
    assert s["events_k10"] == 3 * len(report["configurations"][s["ranking"]][10]["events_covered"])


def test_a_query_with_no_model_answer_counts_as_the_plain_ranking_and_never_crashes():
    from src.ranking_eval import evaluate_run
    keys = [f"K{i}" for i in range(6)]
    labels = {k: ("direct" if i % 2 == 0 else "not") for i, k in enumerate(keys)}
    events = {k: f"e{i}" for i, k in enumerate(keys) if labels[k] != "not"}
    fallback = {"label": None, "certainty": None, "reason": "", "source": "fallback", "error": "LLMUnavailable",
                "kind": "server"}
    answered = [{"key": k, "title": k, "mention_level": "title", "check": _answer(labels[k])} for k in keys]
    failed = [{"key": k, "title": k, "mention_level": "title", "check": fallback} for k in keys]
    with_model = evaluate_run(answered, labels, events, seconds=10.0)
    without = evaluate_run(failed, labels, events, seconds=10.0)
    assert "O3+S1" not in without["configurations"]                 # nothing to order with
    reports = {"r1": with_model, "r2": without}
    wrong = {r: mc.wrong_promotions(rep, labels) for r, rep in reports.items()}
    measures = mc.model_measures({k: fallback for k in keys}, labels)
    s = mc.summarize("gemini", reports, wrong, measures, {"r1": 10.0, "r2": 10.0}, complete=True)
    plain = s["ranking"].replace("O2", "O1").replace("O3", "O1")
    expected = (len(with_model["configurations"][s["ranking"]][10]["events_covered"])
                + len(without["configurations"][plain][10]["events_covered"]))
    assert s["events_k10"] == expected and s["runs_passed"] <= 1


# ---------- the report as a function (for the Stage 3 notebook) ----------

def test_the_report_command_saves_exactly_what_build_report_returns(monkeypatch, tmp_path):
    from src import model_compare as mc
    seen = {}
    monkeypatch.setattr(mc, "build_report", lambda company, label_file: seen.update(label_file=label_file) or
                        {"summaries": {}, "choice": "O1", "note": mc.CHOICE_NOTE, "label_file": label_file})
    monkeypatch.setattr(mc, "_folder", lambda: tmp_path)
    mc._report_command()
    assert seen["label_file"] == "v3_dev_tata_motors.csv"           # the historical report keeps its labels
    import json
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["choice"] == "O1"
