import json
from datetime import datetime, timedelta, timezone

import pytest

from src import summary_judging as sj
from src import summary_prompt6 as p6
from src import summary_round3 as r3

DAYS = ["2026-10-14", "2026-10-23", "2026-10-24"]
NOW = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)


# ---------- fake runs ----------

def day_entry(day, outcome="usable", points=("p",), industry=(), failures=()):
    status = {"usable": "ok", "partial": "partial"}.get(outcome, "unavailable")
    summary = None if status == "unavailable" else {
        "article_briefs": [{"id": 1, "brief": f"brief {day}", "enough_text": True}],
        "key_points": [{"text": f"{t} {day}", "citations": [1]} for t in points],
        "industry_context": [{"text": f"{t} {day}", "citations": [2]} for t in industry], "not_used": []}
    articles = [{"id": 1, "key": f"{day}-a", "title": "A"}, {"id": 2, "key": f"{day}-b", "title": "B"}]
    headlines = [{"id": 1, "title": "A", "url": f"https://a.com/{day}/1"},
                 {"id": 2, "title": "B", "url": f"https://a.com/{day}/2"}]
    return {"run": day, "design": "direct", "k": 2, "articles": articles,
            "result": {"status": status, "summary": summary, "headlines": headlines, "outcome": outcome,
                       "failures": list(failures), "repairs": [],
                       "usage": {"per_call": [{"sent_at": 10.0, "input": 3000, "output": 2000}]}}}


def run(pair, prompt, outcomes=None, industry=None):
    outcomes = outcomes or {}
    industry = industry or {}
    return {"run": r3.run_id(pair, prompt), "pair": pair, "prompt": prompt,
            "records": [day_entry(d, outcomes.get(d, "usable"), (f"pt{prompt}r{pair}",), industry.get(d, ()))
                        for d in DAYS]}


def complete(pairs=(1, 2, 3), **changes):
    records = {}
    for pair in pairs:
        for prompt in r3.PROMPTS:
            records[r3.run_id(pair, prompt)] = changes.get(r3.run_id(pair, prompt)) or run(pair, prompt)
    return records


# ---------- the plan ----------

def test_the_orders_are_fixed_by_the_seed_and_hold_both_prompts_and_all_runs():
    a, b = r3.seeded_orders(), r3.seeded_orders()
    assert a == b
    assert all(sorted(order) == ["4", "6"] for order in a["pair_order"].values())
    assert set(a["pair_order"]) == {"1", "2", "3"}
    assert all(sorted(order) == [1, 2, 3] for order in a["full_item_order"].values())


def test_a_daily_limit_that_cannot_hold_one_pair_is_refused_and_one_that_exactly_holds_it_is_accepted():
    with pytest.raises(r3.Round3Error):
        r3.plan_record(89_999, "base", "abc")
    assert r3.plan_record(90_000, "base", "abc")["daily_limit"] == 90_000


def test_the_plan_records_both_prompts_the_limit_and_the_rule():
    plan = r3.plan_record(200_000, "base", "abc")
    assert plan["daily_limit"] == 200_000 and plan["allowance_per_run"] == 45_000
    assert plan["prompts"]["6"]["fingerprint"] == p6.prompt_fingerprint(p6.prompt_v6(r3.prompt_text("4")))
    assert plan["prompts"]["4"]["fingerprint"] != plan["prompts"]["6"]["fingerprint"]
    assert plan["orders"] == r3.seeded_orders() and plan["rule"] == r3.RULE
    assert r3.plan_problems("x", "x", plan, "base") == []


@pytest.mark.parametrize("change, expected", [
    ({"tag": None}, "differs from the tag"), ({"base": "other"}, "differs from the plan"),
    ({"code": "0"}, "code differs"), ({"guide": "0"}, "SUMMARY_JUDGING_GUIDE_PART7.md"), ({"prompt": "0"}, "prompt 6")])
def test_plan_problems_finds_each_mismatch(change, expected):
    plan = r3.plan_record(200_000, "base", "abc")
    if "code" in change:
        plan["code_fingerprint"] = change["code"]
    if "guide" in change:
        plan["guides"][p6.GUIDE_FILE] = change["guide"]
    if "prompt" in change:
        plan["prompts"]["6"]["fingerprint"] = change["prompt"]
    found = r3.plan_problems("x", change.get("tag", "x"), plan, change.get("base", "base"))
    assert any(expected in f for f in found)


# ---------- outcomes and the state of the round ----------

@pytest.mark.parametrize("result, outcome", [
    ({"status": "ok"}, "usable"), ({"status": "partial"}, "partial"),
    ({"status": "unavailable", "failures": []}, "failed"),
    ({"status": "unavailable", "failures": ["rate_limit"]}, "provider_stop"),
    ({"status": "unavailable", "failures": ["daily_limit"]}, "provider_stop")])
def test_each_outcome_is_classified(result, outcome):
    assert r3.classify(result) == outcome


def test_an_empty_round_starts_with_pair_1():
    state = r3.round_state({}, set(), None)
    assert state == {"kept": [], "stopped_at": None, "next": 1, "finished": False}


def test_complete_pairs_are_kept_and_the_next_pair_follows():
    records = complete((1,))
    assert r3.round_state(records, set(records), None) == {"kept": [1], "stopped_at": None, "next": 2,
                                                            "finished": False}
    records = complete()
    assert r3.round_state(records, set(records), None)["finished"]


def test_a_provider_stop_drops_its_pair_and_every_later_pair():
    bad = run(2, "6", {"2026-10-23": "provider_stop"})
    bad["records"][1]["result"]["failures"] = ["rate_limit"]
    records = complete((1, 2), **{"pair2_p6": bad})
    state = r3.round_state(records, set(records), None)
    assert state["kept"] == [1] and state["stopped_at"] == 2 and state["finished"] and state["next"] is None


def test_a_started_run_without_a_record_stops_the_round():
    records = complete((1,))
    records["pair2_p4"] = run(2, "4")
    state = r3.round_state(records, set(records) | {"pair2_p6"}, None)
    assert state["kept"] == [1] and state["stopped_at"] == 2


def test_a_half_pair_without_a_marker_for_the_other_run_also_stops_the_round():
    records = complete((1,))
    records["pair2_p4"] = run(2, "4")
    state = r3.round_state(records, set(records), None)
    assert state["kept"] == [1] and state["stopped_at"] == 2 and state["finished"]


def test_a_refused_second_run_stops_the_round_at_its_pair():
    records = complete((1, 2))
    records["pair3_p4"] = run(3, "4")
    state = r3.round_state(records, set(records), {"pair": 3})
    assert state["kept"] == [1, 2] and state["stopped_at"] == 3 and state["finished"]


def test_a_failed_day_is_not_a_stop():
    records = complete((1,), **{"pair1_p6": run(1, "6", {"2026-10-24": "failed"})})
    assert r3.round_state(records, set(records), None)["kept"] == [1]


# ---------- tokens ----------

def test_calls_are_timed_from_the_run_clock():
    record = {"records": [day_entry("2026-10-14")]}
    entries = r3.call_entries(record, "2026-11-08T10:00:00+00:00")
    assert entries == [(datetime(2026, 11, 8, 10, 0, 10, tzinfo=timezone.utc), 5000)]


def test_a_started_run_without_a_record_counts_as_the_full_allowance():
    log = r3.token_log([(None, "2026-11-08T10:00:00+00:00")])
    assert log == [(datetime(2026, 11, 8, 10, 0, tzinfo=timezone.utc), r3.ALLOWANCE)]


def test_only_calls_inside_the_last_24_hours_count():
    log = [(NOW - timedelta(hours=25), 150_000), (NOW - timedelta(hours=2), 20_000)]
    check = r3.budget_check(log, 0, 90_000, 200_000, NOW)
    assert check["logged_tokens"] == 20_000 and check["fits"]
    assert check["now_ist"].endswith("+05:30") and check["window_start_utc"].endswith("+00:00")


def test_other_use_and_the_pair_allowance_decide_and_the_earliest_time_is_given():
    log = [(NOW - timedelta(hours=20), 60_000), (NOW - timedelta(hours=10), 50_000)]
    check = r3.budget_check(log, 10_000, 90_000, 200_000, NOW)
    assert not check["fits"]
    assert check["earliest_utc"] == (NOW + timedelta(hours=4)).isoformat(timespec="seconds")
    assert r3.budget_check(log, 0, 90_000, 200_000, NOW)["fits"]


def test_the_limit_itself_fits():
    assert r3.budget_check([(NOW - timedelta(hours=1), 110_000)], 0, 90_000, 200_000, NOW)["fits"]
    assert not r3.budget_check([(NOW - timedelta(hours=1), 110_001)], 0, 90_000, 200_000, NOW)["fits"]


# ---------- diagnostics ----------

class FakeRecorder:
    def __init__(self, answers, failures=(), messages=()):
        self.answers, self.failures, self.messages = list(answers), list(failures), list(messages)


def test_every_call_keeps_its_answer_and_problems_and_a_refused_call_its_error():
    log = [{"attempt": 1, "structural": ["unaccounted_article"], "numbers": [], "sources": []},
           {"attempt": 2, "structural": [], "numbers": [], "sources": []}]
    per_call = [{"finish_reason": "stop"}, {"finish_reason": "stop"}, {"finish_reason": "stop"}]
    rec = FakeRecorder(["a1", "a2", "repair"], ["rate_limit"], ["provider said: limit org_abc123"])
    found = r3.run_diagnostics(rec, log, per_call)
    assert [f["answer"] for f in found] == ["a1", "a2", "repair", ""]
    assert found[0]["problems"] == ["unaccounted_article"] and found[0]["classification"] == "answer with check problems"
    assert found[1]["classification"] == "answer" and found[2]["role"] == "repair"
    assert found[3]["classification"] == "provider: rate_limit" and "org_[hidden]" in found[3]["error"]
    assert "org_abc123" not in str(found)


# ---------- full-item evidence ----------

ORDER = {"4": [2, 1, 3], "6": [3, 1, 2]}


def test_the_evidence_follows_the_seeded_order():
    chosen = r3.full_item_evidence(complete(), [1, 2, 3], ORDER, DAYS)
    assert chosen["4"] == {d: "pair2_p4" for d in DAYS} and chosen["6"] == {d: "pair3_p6" for d in DAYS}


def test_a_failed_day_falls_back_down_the_order_and_records_the_run():
    records = complete(**{"pair3_p6": run(3, "6", {"2026-10-23": "failed"})})
    chosen = r3.full_item_evidence(records, [1, 2, 3], ORDER, DAYS)
    assert chosen["6"] == {"2026-10-14": "pair3_p6", "2026-10-23": "pair1_p6", "2026-10-24": "pair3_p6"}


def test_a_dropped_pair_is_never_eligible_and_no_usable_run_gives_unavailable():
    records = complete((1, 2, 3), **{"pair1_p6": run(1, "6", {"2026-10-24": "failed"}),
                                    "pair2_p6": run(2, "6", {"2026-10-24": "failed"})})
    chosen = r3.full_item_evidence(records, [1, 2], ORDER, DAYS)
    assert chosen["6"]["2026-10-14"] == "pair1_p6" and chosen["6"]["2026-10-24"] is None


def test_a_partial_day_is_eligible():
    records = complete(**{"pair3_p6": run(3, "6", {"2026-10-14": "partial"})})
    assert r3.full_item_evidence(records, [1, 2, 3], ORDER, DAYS)["6"]["2026-10-14"] == "pair3_p6"


# ---------- the sheets ----------

EVENTS = {f"https://a.com/{d}/1": f"event {d}" for d in DAYS}
LABELS = {url: "direct" for url in EVENTS}


def p4_items(n=12):
    return [{"row": i, "kind": "company point", "text": f"locked {i}", "evidence": "e", "judgement": "supported",
             "contradicted": "no", "central": "", "note": ""} for i in range(1, n + 1)]


def built(records=None, kept=(1, 2, 3)):
    return r3.build_sheets(records or complete(), list(kept), ORDER, DAYS, {}, EVENTS, LABELS, {}, p4_items())


def test_every_usable_summary_has_a_letter_and_the_paired_days_their_pair_rows():
    b = built()
    assert len(b["key"]["letters"]) == 18
    assert len(b["pairs"]) == 12 and len(b["events"]) == 18
    assert {v["prompt"] for v in b["key"]["letters"].values()} == {"4", "6"}


def test_failed_and_dropped_run_days_get_no_rows():
    records = complete(**{"pair1_p6": run(1, "6", {"2026-10-23": "failed"})})
    b = built(records, kept=(1, 2))
    assert len(b["key"]["letters"]) == 11 and len(b["pairs"]) == 7


def test_the_sheets_show_neither_prompt_nor_run():
    b = built()
    shown = str(b["pairs"]) + str(b["overreach"]) + str(b["items"]) + str(b["events"])
    for word in ("pair1", "pair2", "pair3", "_p4", "_p6", "prompt"):
        assert word not in shown


def test_items_come_only_from_the_full_item_evidence_plus_ten_decoys():
    b = built()
    real = [e for e in b["key"]["items"].values() if not e.get("decoy")]
    assert {e["source_run"] for e in real} == {"pair2_p4", "pair3_p6"}
    assert sum(1 for e in b["key"]["items"].values() if e.get("decoy")) == 10
    assert len(b["items"]) == len(real) + 10


def test_the_sheets_are_the_same_for_the_same_inputs():
    assert built()["key"] == built()["key"]


def fill(b, outlook="yes", apart="yes", over="no", covered="yes"):
    pairs = [dict(r, outlook_stated=outlook, kept_apart=apart, note="n" if "no" in (outlook, apart) else "")
             for r in b["pairs"]]
    overreach = [dict(r, overreach=over, note="n" if over == "yes" else "") for r in b["overreach"]]
    items = [dict(r, judgement="supported", contradicted="no") for r in b["items"]]
    events = [dict(r, covered=covered) for r in b["events"]]
    return pairs, overreach, items, events


def test_complete_sheets_have_no_problems_and_a_no_needs_a_note():
    b = built(complete(**{"pair1_p4": run(1, "4", industry={"2026-10-23": ("sector",)})}))
    sheets = fill(b)
    assert r3.sheet_problems(sheets, b["key"]) == []
    pairs = [dict(sheets[0][0], kept_apart="no", note="")] + sheets[0][1:]
    assert any("note" in f for f in r3.sheet_problems((pairs,) + sheets[1:], b["key"]))
    assert r3.sheet_problems((sheets[0][1:],) + sheets[1:], b["key"])
    key = dict(b["key"], pairs={k: v for k, v in b["key"]["pairs"].items() if k != "1"})
    assert any("one row for every" in f for f in r3.sheet_problems(sheets, key))


# ---------- the measures over all runs ----------

def measures(records, kept, **fill_args):
    b = built(records, kept)
    pairs, overreach, items, events = fill(b, **fill_args)
    n_events = {d: 1 for d in DAYS}
    return r3.all_run_measures(pairs, overreach, events, b["key"], r3.outcome_table(records, kept), n_events,
                               kept, records)


def test_a_failed_day_gets_explicit_zeros_with_the_same_denominators():
    records = complete(**{"pair1_p6": run(1, "6", {"2026-10-23": "failed"})})
    m = measures(records, [1, 2, 3])
    assert m["6"]["failed_run_days"] == 1 and m["4"]["failed_run_days"] == 0
    assert m["6"]["pair_score"] == {"2026-10-23": 4, "2026-10-24": 6}
    assert m["4"]["pair_score"] == {"2026-10-23": 6, "2026-10-24": 6}
    assert m["6"]["events"]["2026-10-23"] == {"covered": 2, "total": 3}
    assert m["4"]["events"]["2026-10-23"] == {"covered": 3, "total": 3}
    assert m["6"]["run_days"] == m["4"]["run_days"] == 9 and m["6"]["pair_scores_count"] == 6


def test_the_denominators_follow_the_kept_pairs():
    m = measures(complete((1, 2)), [1, 2])
    assert m["4"]["run_days"] == 6 and m["4"]["pair_scores_count"] == 4
    assert m["4"]["events"]["2026-10-14"]["total"] == 2


def test_overreach_counts_and_industry_points_are_reported_per_prompt():
    records = complete(**{"pair1_p6": run(1, "6", industry={"2026-10-23": ("s1", "s2")})})
    m = measures(records, [1, 2, 3], over="yes")
    assert m["6"]["overreach"] == 2 and m["6"]["industry_points"] == 2 and m["4"]["industry_points"] == 0


# ---------- the choice ----------

def measure(failed=0, pairs=(3, 3), covered=(3, 3, 3), over=0, kept=3):
    return {"failed_run_days": failed, "pair_score": dict(zip(sorted(p6.PAIRS), pairs)), "pair_total": sum(pairs),
            "overreach": over, "events": {d: {"covered": c, "total": 3} for d, c in zip(DAYS, covered)}}


def full(unavailable=(), core_not=0, ind_not=0, contra=0, supported=30, total=40, empty=(), without=(),
         evidence=None):
    evidence = evidence or {d: ("x" if d not in unavailable else None) for d in DAYS}
    return {"evidence": evidence, "unavailable_days": sorted(unavailable),
            "core": {"total": total, "supported": supported, "not_supported": core_not, "contradicted": contra},
            "industry": {"total": 1, "not_supported": ind_not, "contradicted": 0},
            "empty_with_evidence_days": list(empty), "points_without_evidence_days": list(without)}


def test_prompt6_wins_when_every_condition_holds():
    c = r3.choose({"4": measure(pairs=(2, 2)), "6": measure(pairs=(3, 3))},
                  {"4": full(empty=["2026-10-24"]), "6": full(empty=["2026-10-24"])}, [1, 2, 3])
    assert c["prompt"] == "6" and all(c["conditions"].values())


@pytest.mark.parametrize("m6, f6, kept, condition", [
    (measure(pairs=(3, 3), failed=1), full(), [1, 2, 3], "failures_not_more"),
    (measure(pairs=(2, 2)), full(), [1, 2, 3], "pair_total_higher"),
    (measure(pairs=(1, 5)), full(), [1, 2, 3], "pair_not_lower_on_each_day"),
    (measure(pairs=(3, 3), over=1), full(), [1, 2, 3], "overreach_not_higher"),
    (measure(pairs=(3, 3), covered=(3, 2, 3)), full(), [1, 2, 3], "events_not_lower_each_day"),
    (measure(pairs=(3, 3)), full(unavailable=["2026-10-24"]), [1, 2, 3], "prompt6_evidence_every_day"),
    (measure(pairs=(3, 3)), full(ind_not=1), [1, 2, 3], "no_unsupported_industry_point"),
    (measure(pairs=(3, 3)), full(core_not=1), [1, 2, 3], "no_unsupported_key_point_or_brief"),
    (measure(pairs=(3, 3)), full(contra=1), [1, 2, 3], "no_contradicted_item"),
    (measure(pairs=(3, 3)), full(supported=29, total=39), [1, 2, 3], "supported_share_at_least_0_75"),
    (measure(pairs=(3, 3)), full(without=["2026-10-14"]), [1, 2, 3], "no_industry_points_without_evidence"),
    (measure(pairs=(3, 3)), full(empty=["2026-10-23", "2026-10-24"]), [1, 2, 3], "empty_with_evidence_not_more"),
    (measure(pairs=(3, 3)), full(), [1], "at_least_2_kept_pairs"),
])
def test_each_condition_alone_keeps_prompt4(m6, f6, kept, condition):
    c = r3.choose({"4": measure(pairs=(2, 2)), "6": m6}, {"4": full(empty=["2026-10-24"]), "6": f6}, kept)
    assert c["prompt"] == "4" and not c["conditions"][condition]
    assert sum(not v for v in c["conditions"].values()) == 1


def test_a_tie_on_pairs_keeps_prompt4():
    c = r3.choose({"4": measure(), "6": measure()}, {"4": full(), "6": full()}, [1, 2, 3])
    assert c["prompt"] == "4"


def test_empty_days_are_compared_only_where_both_prompts_have_evidence():
    f4 = full(evidence={"2026-10-14": "x", "2026-10-23": "x", "2026-10-24": None})
    f6 = full(empty=["2026-10-24"])
    c = r3.choose({"4": measure(pairs=(2, 2)), "6": measure()}, {"4": f4, "6": f6}, [1, 2, 3])
    assert c["conditions"]["empty_with_evidence_not_more"] and c["days_with_both_evidence"] == DAYS[:2]
    assert c["days_left_out_of_the_empty_section_comparison"] == ["2026-10-24"]
    assert c["full_item_evidence"]["prompt4"]["2026-10-24"] is None


def test_the_share_is_unrounded():
    c = r3.choose({"4": measure(pairs=(2, 2)), "6": measure()}, {"4": full(), "6": full(supported=2999, total=4000)},
                  [1, 2, 3])
    assert not c["conditions"]["supported_share_at_least_0_75"]


# ---------- the pair command, end to end with a fake model ----------

class FakeLLM:
    model_name = "openai/gpt-oss-120b"
    generation = {"temperature": 1.0, "reasoning_effort": "medium", "max_completion_tokens": 4096}
    output_limit = 4096

    def __init__(self, answer=None, error=None, error_after=None, tokens=150):
        self.answer, self.error, self.error_after, self.tokens = answer, error, error_after, tokens
        self.stats = {"per_call": [], "requests": []}
        self.prompts = []

    def __call__(self, prompt):
        from src.summarizer import LLMUnavailable
        self.prompts.append(prompt)
        if self.error and (self.error_after is None or len(self.prompts) > self.error_after):
            err = LLMUnavailable("Rate limit reached for org_abc. Please try again in 1m0s.")
            err.kind = self.error
            raise err
        self.stats["per_call"].append({"sent_at": 1.0, "input": self.tokens - 50, "output": 50,
                                       "finish_reason": "stop"})
        return self.answer


@pytest.fixture
def round_dir(tmp_path, monkeypatch):
    import shutil
    from tests.raw_data import require_raw_data
    require_raw_data()                               # the round's plan fingerprints the development raw files
    from src.config import EVAL_DIR
    for name in (p6.P4_RECORD,) + r3.GUIDES + r3.DATA_FILES:
        shutil.copy(EVAL_DIR / name, tmp_path / name)
    monkeypatch.setattr(r3, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(p6, "EVAL_DIR", tmp_path)
    plan = r3.plan_record(200_000, (tmp_path / p6.P4_RECORD).read_text(encoding="utf-8"), "abc")
    text = __import__("json").dumps(plan)
    (tmp_path / r3.PLAN_FILE).write_text(text, encoding="utf-8")
    monkeypatch.setattr(r3, "_tag_text", lambda: text)
    return tmp_path


def use_llm(monkeypatch, llm):
    from src import summary_v2 as sv
    monkeypatch.setattr(sv, "summary_llm", lambda: llm)


def test_a_pair_with_bad_answers_records_failed_days_and_both_prompts(round_dir, monkeypatch):
    from src import summarizer as sm
    llm = FakeLLM(answer="not json")
    use_llm(monkeypatch, llm)
    r3._pair_command(0)
    records, markers, stop = r3._load_round()
    assert set(records) == {"pair1_p4", "pair1_p6"} and set(markers) == set(records) and stop is None
    for record in records.values():
        assert [e["result"]["outcome"] for e in record["records"]] == ["failed"] * 3
        diag = record["records"][0]["result"]["diagnostics"]
        assert [d["answer"] for d in diag] == ["not json", "not json"]
    v6 = [p for p in llm.prompts if r3.prompt_text("6").split("\n")[6] in p]
    assert len(v6) == 6 and len(llm.prompts) == 12
    assert sm.RULES == sm.PROMPT_TEXTS["4"]
    assert r3.round_state(records, set(markers), stop)["next"] == 2
    with pytest.raises(FileExistsError):
        r3._one_run(1, "4", {}, llm, "2026-11-08T00:00:00+00:00", 0, {})


def test_a_provider_stop_ends_the_pair_before_the_second_run(round_dir, monkeypatch):
    use_llm(monkeypatch, FakeLLM(error="rate_limit"))
    r3._pair_command(0)
    records, markers, stop = r3._load_round()
    assert len(records) == 1
    only = next(iter(records.values()))
    assert [e["result"]["outcome"] for e in only["records"]] == ["provider_stop"]
    assert only["records"][0]["result"]["diagnostics"][-1]["classification"] == "provider: rate_limit"
    assert "org_abc" not in str(only)
    state = r3.round_state(records, set(markers), stop)
    assert state["finished"] and state["kept"] == [] and state["stopped_at"] == 1
    with pytest.raises(r3.Round3Error):
        r3._pair_command(0)


def test_a_pair_that_does_not_fit_sends_nothing(round_dir, monkeypatch):
    llm = FakeLLM(answer="not json")
    use_llm(monkeypatch, llm)
    with pytest.raises(r3.Round3Error, match="does not fit"):
        r3._pair_command(150_000)
    assert llm.prompts == [] and r3._load_round() == ({}, {}, None)


def test_a_changed_plan_refuses_the_pair(round_dir, monkeypatch):
    llm = FakeLLM(answer="not json")
    use_llm(monkeypatch, llm)
    monkeypatch.setattr(r3, "_tag_text", lambda: None)
    with pytest.raises(r3.Round3Error, match="differs from the tag"):
        r3._pair_command(0)
    assert llm.prompts == []


# ---------- the run allowance ----------

def test_a_call_whose_worst_case_would_pass_the_allowance_is_not_sent():
    from src.summarizer import LLMUnavailable
    llm = FakeLLM(answer="x", tokens=41_000)
    guard = r3.AllowanceGuard(llm, allowance=45_000, output_limit=4096)
    assert guard("a" * 300) == "x"                  # 0 + 101 + 4096 fits; 41,000 + 101 + 4096 does not
    with pytest.raises(LLMUnavailable) as error:
        guard("a" * 300)
    assert error.value.kind == "run_allowance" and guard.refused and len(llm.prompts) == 1
    with pytest.raises(LLMUnavailable):
        guard("")                                   # nothing more is sent once a call is refused


def test_the_worst_case_counts_the_careful_input_and_the_output_limit():
    guard = r3.AllowanceGuard(FakeLLM(), allowance=45_000, output_limit=4096)
    assert guard.worst("a" * 30) == 11 + 4096


def test_a_run_over_its_allowance_ends_as_a_provider_stop_and_the_round_stops(round_dir, monkeypatch):
    use_llm(monkeypatch, FakeLLM(answer="not json", tokens=40_000))
    r3._pair_command(0)
    records, markers, stop = r3._load_round()
    assert len(records) == 1
    only = next(iter(records.values()))
    assert [e["result"]["outcome"] for e in only["records"]] == ["provider_stop"] and only["allowance_refused"]
    assert only["records"][0]["result"]["diagnostics"][-1]["classification"] == "provider: run_allowance"
    assert only["tokens"] <= r3.ALLOWANCE
    assert r3.round_state(records, set(markers), stop)["stopped_at"] == 1


# ---------- a provider stop after completed days ----------

def test_a_provider_stop_on_the_second_day_keeps_no_later_day(round_dir, monkeypatch):
    use_llm(monkeypatch, FakeLLM(answer="not json", error="rate_limit", error_after=2))
    r3._pair_command(0)
    records, _, _ = r3._load_round()
    only = next(iter(records.values()))
    assert [(e["run"], e["result"]["outcome"]) for e in only["records"]] == \
        [("2026-10-14", "failed"), ("2026-10-23", "provider_stop")]
    plan = __import__("json").loads((round_dir / r3.PLAN_FILE).read_text(encoding="utf-8"))
    baseline = __import__("json").loads((round_dir / p6.P4_RECORD).read_text(encoding="utf-8"))
    assert r3.record_problems(only["run"], only, r3._load_round()[1][only["run"]], plan, baseline) == []


# ---------- the second run refused ----------

def test_a_refused_second_run_is_not_sent_drops_the_pair_and_stops_the_round(round_dir, monkeypatch):
    llm = FakeLLM(answer="not json")
    use_llm(monkeypatch, llm)
    real = r3.budget_check
    calls = []

    def second_refused(log, other, need, limit, now):
        calls.append(need)
        check = real(log, other, need, limit, now)
        return dict(check, fits=len(calls) == 1)
    monkeypatch.setattr(r3, "budget_check", second_refused)
    with pytest.raises(r3.Round3Error, match="would not fit"):
        r3._pair_command(0)
    assert calls == [2 * r3.ALLOWANCE, r3.ALLOWANCE] and len(llm.prompts) == 6
    records, markers, stop = r3._load_round()
    assert len(records) == 1 and stop["pair"] == 1
    state = r3.round_state(records, set(markers), stop)
    assert state["kept"] == [] and state["stopped_at"] == 1 and state["finished"]


# ---------- a marker without a record ----------

def test_a_marker_without_a_record_counts_the_full_allowance_and_stops_the_round(round_dir, monkeypatch):
    import json
    marker = {"run": "pair1_p4", "started_at": "2026-11-08T10:00:00+00:00", "clock_start": "2026-11-08T10:00:00+00:00",
              "prompt_fingerprint": "x"}
    (round_dir / r3.marker_file(1, "4")).write_text(json.dumps(marker), encoding="utf-8")
    records, markers, stop = r3._load_round()
    log = r3.token_log(r3._log_runs(records, markers))
    assert (datetime(2026, 11, 8, 10, 0, tzinfo=timezone.utc), r3.ALLOWANCE) in log
    assert r3.round_state(records, set(markers), stop)["stopped_at"] == 1
    llm = FakeLLM(answer="not json")
    use_llm(monkeypatch, llm)
    with pytest.raises(r3.Round3Error, match="finished"):
        r3._pair_command(0)
    assert llm.prompts == []


# ---------- the frozen inputs before the sheets, the lock and the score ----------

def stopped_round(monkeypatch):
    use_llm(monkeypatch, FakeLLM(error="rate_limit"))
    r3._pair_command(0)


def test_the_sheets_are_refused_when_the_plan_no_longer_matches_its_tag(round_dir, monkeypatch):
    stopped_round(monkeypatch)
    monkeypatch.setattr(r3, "_tag_text", lambda: None)
    with pytest.raises(r3.Round3Error, match="refused"):
        r3._sheet_command()
    assert not (round_dir / r3.SCORE_FILE).exists()


def test_the_sheets_are_refused_when_a_saved_run_was_changed(round_dir, monkeypatch):
    import json
    stopped_round(monkeypatch)
    path = next(round_dir.glob("summary_v2_r3_pair1_p*[0-9].json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    record["prompt_fingerprint"] = "changed"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(r3.Round3Error, match="prompt differs"):
        r3._sheet_command()


def test_an_unchanged_stopped_round_with_no_kept_pair_keeps_prompt4(round_dir, monkeypatch):
    import json
    stopped_round(monkeypatch)
    r3._sheet_command()
    result = json.loads((round_dir / r3.SCORE_FILE).read_text(encoding="utf-8"))
    assert result["choice"]["prompt"] == "4" and result["kept"] == []
    assert result["comparison_made"] is False and "insufficient data" in result["conclusion"]
    assert "not as a winner" in result["conclusion"]
    assert list(result["runs"].values())[0]["outcomes"] == {"2026-10-14": "provider_stop"}


@pytest.mark.parametrize("command", ["_lock_command", "_score_command"])
def test_the_lock_and_the_score_check_the_frozen_inputs_first(round_dir, monkeypatch, command):
    stopped_round(monkeypatch)
    monkeypatch.setattr(r3, "_tag_text", lambda: None)
    with pytest.raises(r3.Round3Error, match="refused"):
        getattr(r3, command)()


# ---------- one code list for the plan, the lock and the score ----------

def test_the_lock_uses_the_plan_code_list_and_a_change_to_any_listed_file_shows(tmp_path, monkeypatch):
    import shutil
    from src.config import ROOT_DIR
    assert {"src/llm.py", "src/summarizer.py", "src/summary_v2.py", "src/summary_round3.py",
            "src/summary_judging_p5.py", "src/ranking_eval.py"} <= set(r3.CODE_FILES)
    for name in r3.CODE_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT_DIR / name, tmp_path / name)
    monkeypatch.setattr(p6, "ROOT_DIR", tmp_path)
    before = r3._code_fp()
    assert before == p6.code_fingerprint(r3.CODE_FILES)
    (tmp_path / "src/llm.py").write_text("# changed\n", encoding="utf-8")
    assert r3._code_fp() != before


# ---------- every saved run checked against the plan ----------

@pytest.fixture
def valid():
    import json
    from src.config import EVAL_DIR
    baseline = json.loads((EVAL_DIR / p6.P4_RECORD).read_text(encoding="utf-8"))
    plan = r3.plan_record(200_000, "base", "abc")
    records = []
    for e in baseline["records"]:
        e = json.loads(json.dumps(e))
        e["result"]["outcome"] = r3.classify(e["result"])
        records.append(e)
    st = plan["settings"]
    record = {"run": "pair1_p6", "pair": 1, "prompt": "6", "prompt_fingerprint": plan["prompts"]["6"]["fingerprint"],
              "clock_start": "2026-11-08T10:00:00+00:00", "version": st["configuration"],
              "pacer_version": st["pacer_version"], "model_setting": st["model_setting"], "model": st["model"],
              "generation": st["generation"], "correction": st["correction"], "number_check": st["number_check"],
              "k": 5, "cache": "none", "output_limit_sent": 4096, "allowance": 45_000, "records": records}
    record["tokens"] = sum(n for _, n in r3.call_entries(record, record["clock_start"]))
    record["allowance_exceeded"] = record["tokens"] > 45_000
    budget = {"fits": True, "need": 90_000, "limit": 200_000, "other_tokens": 0}
    marker = {"run": "pair1_p6", "prompt_fingerprint": record["prompt_fingerprint"],
              "clock_start": record["clock_start"], "other_tokens": 0, "budget": budget}
    return record, marker, plan, baseline


def problems_of(valid, change):
    import json
    record, marker, plan, baseline = valid
    record = json.loads(json.dumps(record))
    change(record)
    return r3.record_problems("pair1_p6", record, marker, plan, baseline)


def test_a_valid_record_has_no_problems(valid):
    assert problems_of(valid, lambda r: None) == []


@pytest.mark.parametrize("change, expected", [
    (lambda r: r["records"].pop(1), "the days are"),
    (lambda r: r["records"].append(r["records"][0]), "the days are"),
    (lambda r: r["records"][2].update(run="2026-10-30"), "the days are"),
    (lambda r: r.update(prompt_fingerprint="other"), "prompt differs"),
    (lambda r: r.update(model_setting="other"), "model_setting"),
    (lambda r: r.update(output_limit_sent=8192), "output_limit_sent"),
    (lambda r: r.update(pacer_version="2"), "pacer_version"),
    (lambda r: r.update(prompt="4"), "does not match its file"),
    (lambda r: r["records"][0]["result"].update(outcome="provider_stop", failures=["rate_limit"], status="unavailable",
                                                 summary=None), "a day follows a provider stop"),
    (lambda r: r["records"][1]["result"].update(outcome="failed"), "outcome does not match"),
    (lambda r: r["records"][0]["result"].update(outcome="lost"), "no valid outcome"),
    (lambda r: r["records"][2]["result"].update(outcome="provider_stop"), "without a recorded refusal"),
    (lambda r: r["records"][2]["result"].update(outcome="provider_stop", failures=["rate_limit"]),
     "with a usable summary"),
    (lambda r: r["records"][2]["result"].update(failures=["run_allowance"]), "not marked as a provider stop"),
    (lambda r: r.update(correction="full"), "correction"),
    (lambda r: r.update(number_check="1"), "number_check"),
    (lambda r: r.update(generation={"temperature": 0.0}), "generation"),
    (lambda r: r.update(tokens=1), "not the sum of its calls"),
    (lambda r: r.update(allowance_exceeded=True), "allowance_exceeded does not match"),
    (lambda r: r["records"][0]["articles"].reverse(), "articles differ"),
])
def test_each_change_to_a_record_is_found(valid, change, expected):
    assert any(expected in f for f in problems_of(valid, change))


def test_a_missing_or_different_marker_is_found(valid):
    record, marker, plan, baseline = valid
    assert any("marker" in f for f in r3.record_problems("pair1_p6", record, None, plan, baseline))
    other = dict(marker, clock_start="2026-11-09T00:00:00+00:00")
    assert any("marker" in f for f in r3.record_problems("pair1_p6", record, other, plan, baseline))


@pytest.mark.parametrize("change, expected", [
    ({"run": "pair2_p6"}, "names another run"),
    ({"other_tokens": -5}, "no valid stated other use"),
    ({"other_tokens": "0"}, "no valid stated other use"),
    ({"budget": {"fits": False, "need": 90_000, "limit": 200_000, "other_tokens": 0}}, "pair budget check"),
    ({"budget": {"fits": True, "need": 45_000, "limit": 200_000, "other_tokens": 0}}, "pair budget check"),
    ({"budget": {"fits": True, "need": 90_000, "limit": 300_000, "other_tokens": 0}}, "pair budget check"),
    ({"budget": {"fits": True, "need": 90_000, "limit": 200_000, "other_tokens": 7}}, "pair budget check"),
])
def test_each_change_to_the_marker_is_found(valid, change, expected):
    record, marker, plan, baseline = valid
    found = r3.record_problems("pair1_p6", record, dict(marker, **change), plan, baseline)
    assert any(expected in f for f in found)


def test_a_provider_stop_with_a_usable_summary_is_valid_only_for_a_run_allowance_refusal(valid):
    record, marker, plan, baseline = valid
    import json
    r = json.loads(json.dumps(record))
    r["records"][2]["result"].update(outcome="provider_stop", failures=["run_allowance"])
    assert r3.record_problems("pair1_p6", r, marker, plan, baseline) == []


def test_a_saved_provider_stop_label_stops_the_round_only_after_it_passed_the_check(valid):
    record, marker, plan, baseline = valid
    import json
    r = json.loads(json.dumps(record))
    r["records"][2]["result"].update(outcome="provider_stop")       # a label with no evidence
    assert any("without a recorded refusal" in f for f in r3.record_problems("pair1_p6", r, marker, plan, baseline))


# ---------- the round 2 run in the token log ----------

def test_a_round2_marker_without_its_record_counts_the_full_allowance(round_dir):
    import json
    (round_dir / p6.ATTEMPT_FILE).write_text(json.dumps({"started_at": "2026-11-07T09:38:00+00:00"}),
                                             encoding="utf-8")
    log = r3.token_log(r3._log_runs({}, {}))
    assert log == [(datetime(2026, 11, 7, 9, 38, tzinfo=timezone.utc), r3.ALLOWANCE)]


def test_the_round2_record_is_counted_call_by_call_when_it_exists(round_dir):
    import json
    from src.config import EVAL_DIR
    for name in (p6.ATTEMPT_FILE, p6.RECORD_FILE):
        (round_dir / name).write_text((EVAL_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")
    log = r3.token_log(r3._log_runs({}, {}))
    record = json.loads((EVAL_DIR / p6.RECORD_FILE).read_text(encoding="utf-8"))
    calls = sum(len(e["result"]["usage"]["per_call"]) for e in record["records"])
    assert len(log) == calls and sum(n for _, n in log) == 27_626


# ---------- real tokens over the allowance ----------

def test_real_tokens_over_the_allowance_stop_further_calls(monkeypatch):
    from src.summarizer import LLMUnavailable
    llm = FakeLLM(answer="x", tokens=46_000)
    guard = r3.AllowanceGuard(llm, allowance=45_000, output_limit=4096)
    assert guard("a") == "x" and guard.exceeded and guard.refused
    with pytest.raises(LLMUnavailable):
        guard("a")


def test_a_run_whose_real_tokens_pass_the_allowance_stops_the_round(round_dir, monkeypatch):
    use_llm(monkeypatch, FakeLLM(answer="not json", tokens=46_000))
    r3._pair_command(0)
    records, markers, stop = r3._load_round()
    only = next(iter(records.values()))
    assert only["allowance_exceeded"] and only["tokens"] > r3.ALLOWANCE and len(records) == 1
    state = r3.round_state(records, set(markers), stop)
    assert state["stopped_at"] == 1 and state["kept"] == []
    plan = json.loads((round_dir / r3.PLAN_FILE).read_text(encoding="utf-8"))
    baseline = json.loads((round_dir / p6.P4_RECORD).read_text(encoding="utf-8"))
    assert r3.record_problems(only["run"], only, markers[only["run"]], plan, baseline) == []


def test_an_empty_stats_object_is_shared_not_replaced():
    class Empty(FakeLLM):
        pass
    llm = Empty(answer="x")
    llm.stats = {}
    guard = r3.AllowanceGuard(llm)
    assert guard.stats is llm.stats


def test_the_plan_holds_every_frozen_setting_and_a_changed_one_is_found(monkeypatch):
    plan = r3.plan_record(200_000, "base", "abc")
    assert {"correction", "number_check", "generation", "model", "pacer_version", "output_limit"} <= set(plan["settings"])
    plan["settings"] = dict(plan["settings"], number_check="0")
    assert any("frozen settings" in f for f in r3.plan_problems("x", "x", plan, "base"))


def test_a_full_comparison_says_it_was_made_and_what_it_concludes():
    c = r3.choose({"4": measure(pairs=(2, 2)), "6": measure(pairs=(3, 3))}, {"4": full(), "6": full()}, [1, 2, 3])
    assert c["comparison_made"] and c["conclusion"] == "prompt 6 meets every condition"
    assert c["days_left_out_of_the_empty_section_comparison"] == []
    lost = r3.choose({"4": measure(), "6": measure()}, {"4": full(), "6": full()}, [1, 2, 3])
    assert lost["comparison_made"] and "does not meet every condition" in lost["conclusion"]


# ---------- review of 9 Nov: overruns, cut-short runs, marker times, inputs, the key ----------

class ScheduledLLM(FakeLLM):
    """Answers every call; the tokens reported for call n come from a schedule."""

    def __init__(self, schedule, answer="not json"):
        super().__init__(answer=answer)
        self.schedule = schedule

    def __call__(self, prompt):
        self.tokens = self.schedule[len(self.prompts)] if len(self.prompts) < len(self.schedule) else 150
        return super().__call__(prompt)


def test_an_overrun_on_the_last_call_of_a_run_stops_the_pair_before_the_other_run(round_dir, monkeypatch):
    llm = ScheduledLLM([150, 150, 150, 150, 150, 46_000])        # 3 failed days, 2 calls each; the 6th overruns
    use_llm(monkeypatch, llm)
    r3._pair_command(0)
    records, markers, stop = r3._load_round()
    assert len(records) == 1 and len(markers) == 1 and len(llm.prompts) == 6
    only = next(iter(records.values()))
    assert only["allowance_exceeded"] and [e["result"]["outcome"] for e in only["records"]] == ["failed"] * 3
    assert r3.round_state(records, set(markers), stop)["stopped_at"] == 1
    plan = json.loads((round_dir / r3.PLAN_FILE).read_text(encoding="utf-8"))
    baseline = json.loads((round_dir / p6.P4_RECORD).read_text(encoding="utf-8"))
    assert r3.record_problems(only["run"], only, markers[only["run"]], plan, baseline) == []


def test_an_overrun_on_the_first_day_leaves_a_valid_one_day_record(round_dir, monkeypatch):
    llm = ScheduledLLM([150, 46_000])                           # day 1 attempt 2 overruns; nothing more is sent
    use_llm(monkeypatch, llm)
    r3._pair_command(0)
    records, markers, _ = r3._load_round()
    only = next(iter(records.values()))
    assert [e["run"] for e in only["records"]] == ["2026-10-14"] and len(llm.prompts) == 2
    plan = json.loads((round_dir / r3.PLAN_FILE).read_text(encoding="utf-8"))
    baseline = json.loads((round_dir / p6.P4_RECORD).read_text(encoding="utf-8"))
    assert r3.record_problems(only["run"], only, markers[only["run"]], plan, baseline) == []


def test_a_short_record_is_accepted_only_when_the_run_was_cut_short(valid):
    record, marker, plan, baseline = valid
    short = json.loads(json.dumps(record))
    short["records"] = short["records"][:1]
    short["tokens"] = sum(n for _, n in r3.call_entries(short, short["clock_start"]))
    found = r3.record_problems("pair1_p6", short, marker, plan, baseline)
    assert any("the days are" in f for f in found)              # an ordinary incomplete run
    cut = dict(short, allowance_exceeded=True)
    cut["records"][0]["result"]["usage"]["per_call"].append({"sent_at": 99.0, "input": 45_000, "output": 10})
    cut["tokens"] = sum(n for _, n in r3.call_entries(cut, cut["clock_start"]))
    assert r3.record_problems("pair1_p6", cut, marker, plan, baseline) == []
    empty = dict(cut, records=[], tokens=0, allowance_exceeded=False)
    assert any("the days are" in f for f in r3.record_problems("pair1_p6", empty, marker, plan, baseline))


def test_a_run_without_a_record_counts_from_its_own_start_not_the_pair_clock(round_dir):
    marker = {"run": "pair1_p6", "clock_start": "2026-11-08T10:00:00+00:00",
              "started_at": "2026-11-08T10:07:30+00:00", "prompt_fingerprint": "x"}
    (round_dir / r3.marker_file(1, "6")).write_text(json.dumps(marker), encoding="utf-8")
    records, markers, _ = r3._load_round()
    log = r3.token_log(r3._log_runs(records, markers))
    assert log == [(datetime(2026, 11, 8, 10, 7, 30, tzinfo=timezone.utc), r3.ALLOWANCE)]


def test_the_plan_covers_the_evaluation_code_and_the_judging_data():
    assert {"src/model_compare.py", "src/evaluate.py", "src/summary_run.py", "src/config.py",
            "src/fresh_eval.py"} <= set(r3.CODE_FILES)
    plan = r3.plan_record(200_000, "base", "abc")
    assert set(plan["data_files"]) == set(r3.DATA_FILES) and all(plan["data_files"].values())
    changed = dict(plan, data_files=dict(plan["data_files"], **{"events_dev_tata_motors.csv": "0"}))
    assert any("events_dev_tata_motors.csv" in f for f in r3.plan_problems("x", "x", changed, "base"))


def key_case():
    records = complete(**{"pair1_p6": run(1, "6", {"2026-10-23": "failed"})})
    plan = r3.plan_record(200_000, "base", "abc")
    order = {p: plan["orders"]["full_item_order"][p] for p in r3.PROMPTS}
    b = r3.build_sheets(records, [1, 2, 3], order, DAYS, {}, EVENTS, LABELS, {}, p4_items())
    return records, plan, json.loads(json.dumps(b["key"]))


def test_a_key_made_from_the_saved_runs_passes():
    records, plan, key = key_case()
    assert r3.key_problems(key, records, set(records), None, plan, DAYS) == []


@pytest.mark.parametrize("change, expected", [
    (lambda k: k.update(kept=[1, 2]), "kept pairs"),
    (lambda k: k["full_item_evidence"]["6"].update({"2026-10-14": "pair1_p6"}), "full-item evidence"),
    (lambda k: k["letters"].pop(sorted(k["letters"])[0]), "summaries"),
    (lambda k: k["letters"].update({"A": k["letters"]["B"], "B": k["letters"]["A"]}), "letters"),
])
def test_an_edited_key_is_found_against_the_saved_runs(change, expected):
    records, plan, key = key_case()
    change(key)
    assert any(expected in f for f in r3.key_problems(key, records, set(records), None, plan, DAYS))


# ---------- review of 9 Nov, second part: the limit boundary, no retry time, every guide frozen ----------

def test_no_retry_time_is_given_when_the_stated_other_use_alone_blocks_the_pair():
    check = r3.budget_check([(NOW - timedelta(hours=1), 5_000)], 120_000, 90_000, 200_000, NOW)
    assert not check["fits"] and check["earliest_utc"] is None and not check["fits_later_from_the_log"]
    later = r3.budget_check([(NOW - timedelta(hours=1), 50_000)], 0, 90_000, 120_000, NOW)
    assert later["fits_later_from_the_log"] and later["earliest_utc"] is not None


def test_the_pair_command_says_no_time_can_be_worked_out(round_dir, monkeypatch):
    llm = FakeLLM(answer="not json")
    use_llm(monkeypatch, llm)
    with pytest.raises(r3.Round3Error, match="no time can be worked out") as error:
        r3._pair_command(150_000)
    assert "None" not in str(error.value) and llm.prompts == []


def test_the_plan_freezes_all_three_guides():
    plan = r3.plan_record(200_000, "base", "abc")
    assert set(plan["guides"]) == {sj.GUIDE_FILE, "SUMMARY_JUDGING_GUIDE_PART4.md", p6.GUIDE_FILE}
    assert all(plan["guides"].values())


@pytest.mark.parametrize("guide", [sj.GUIDE_FILE, "SUMMARY_JUDGING_GUIDE_PART4.md", p6.GUIDE_FILE])
@pytest.mark.parametrize("command", ["_sheet_command", "_lock_command", "_score_command"])
def test_a_guide_changed_after_the_plan_stops_the_sheet_lock_and_score(round_dir, monkeypatch, guide, command):
    stopped_round(monkeypatch)
    path = round_dir / guide
    path.write_text(path.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")
    with pytest.raises(r3.Round3Error, match="refused") as error:
        getattr(r3, command)()
    assert guide in str(error.value)
    assert not (round_dir / r3.SCORE_FILE).exists()
