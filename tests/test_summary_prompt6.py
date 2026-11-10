import pytest

from src import summarizer as sm
from src import summary_judging as sj
from src import summary_prompt6 as p6

V4 = sm.PROMPT_TEXTS["4"]


def rules_of(text):
    return [line for line in text.split("\n") if line[:1].isdigit() and line[1:3] in (". ", "0.")]


# ---------- the prompt ----------

def test_prompt6_adds_the_pair_rule_and_the_industry_sentence_and_keeps_everything_else():
    v6 = p6.prompt_v6(V4)
    old, new = rules_of(V4), rules_of(v6)
    assert len(old) == 9 and len(new) == 10
    assert new[3] == "4. " + p6.NEW_RULE
    assert new[4] == "5" + old[3][1:] + " " + p6.INDUSTRY_SENTENCE
    assert [r.split(". ", 1)[1] for r in new[:3]] == [r.split(". ", 1)[1] for r in old[:3]]
    assert [r.split(". ", 1)[1] for r in new[5:]] == [r.split(". ", 1)[1] for r in old[4:]]
    assert v6.split("Rules:")[0] == V4.split("Rules:")[0]
    assert v6.split("Answer with")[1] == V4.split("Answer with")[1]


def test_prompt6_refuses_a_prompt_without_nine_rules_or_without_the_industry_rule_fourth():
    with pytest.raises(p6.Prompt6Error):
        p6.prompt_v6(V4.replace("9. No investment", "No investment"))
    swapped = V4.replace('4. An "industry_context" point needs', '4. Xn "industry_context" point needs')
    with pytest.raises(p6.Prompt6Error):
        p6.prompt_v6(swapped)


def test_the_prompt_is_used_only_inside_the_block_and_restored_after_an_error():
    v6 = p6.prompt_v6(V4)
    with p6.prompt_in_use(v6):
        assert sm.RULES == v6
        assert sm.build_prompt([]).startswith(v6)
    assert sm.RULES == V4
    with pytest.raises(RuntimeError):
        with p6.prompt_in_use(v6):
            raise RuntimeError
    assert sm.RULES == V4


def test_the_frozen_prompt_text_is_unchanged():
    assert p6.prompt_fingerprint(V4) != p6.prompt_fingerprint(p6.prompt_v6(V4))
    assert sm.PROMPT_VERSION == "4" and sm.RULES == V4


# ---------- the run checks ----------

def plan_for(prompt, baseline_text):
    return {"prompt_fingerprint": p6.prompt_fingerprint(prompt),
            "baseline_fingerprint": sj._hash(baseline_text.replace("\r\n", "\n")),
            "code_fingerprint": p6.code_fingerprint()}


def test_run_problems_is_empty_when_everything_matches():
    v6 = p6.prompt_v6(V4)
    assert p6.run_problems("plan", "plan\r\n".rstrip("\r\n"), plan_for(v6, "base"), "base", v6, []) == []


@pytest.mark.parametrize("change, expected", [
    ({"existing": [p6.RECORD_FILE]}, "already exists"),
    ({"tag": None}, "differs from the tag"),
    ({"tag": "other"}, "differs from the tag"),
    ({"prompt": V4}, "prompt differs"),
    ({"baseline": "changed"}, "differs from the plan"),
])
def test_run_problems_finds_each_mismatch(change, expected):
    v6 = p6.prompt_v6(V4)
    args = {"plan_text": "plan", "tag": "plan", "plan": plan_for(v6, "base"), "baseline": "base", "prompt": v6,
            "existing": []}
    args.update(change)
    found = p6.run_problems(args["plan_text"], args["tag"], args["plan"], args["baseline"], args["prompt"],
                            args["existing"])
    assert any(expected in f for f in found)


def test_run_problems_finds_changed_code():
    v6 = p6.prompt_v6(V4)
    plan = dict(plan_for(v6, "base"), code_fingerprint="0" * 64)
    assert any("code differs" in f for f in p6.run_problems("p", "p", plan, "base", v6, []))


def test_same_articles_needs_the_same_keys_in_the_same_order():
    baseline = {"records": [{"run": "d1", "articles": [{"key": "a"}, {"key": "b"}]}]}
    assert p6.same_articles(baseline, {"d1": [{"key": "a"}, {"key": "b"}]}) == []
    assert p6.same_articles(baseline, {"d1": [{"key": "b"}, {"key": "a"}]})
    assert p6.same_articles(baseline, {})


# ---------- fake records ----------

def entry(run, points, industry=()):
    return {"run": run, "design": "direct",
            "articles": [{"id": 1, "key": f"{run}-a", "title": "A"}, {"id": 2, "key": f"{run}-b", "title": "B"}],
            "result": {"status": "ok", "summary": {
                "key_points": [{"text": t, "citations": [1]} for t in points],
                "industry_context": [{"text": t, "citations": [2]} for t in industry]}}}


def records():
    v4 = {"records": [entry("2026-10-14", ["v4 a"]), entry("2026-10-23", ["v4 b"], ["v4 sector"]),
                      entry("2026-10-24", ["v4 c"])]}
    v6 = {"records": [entry("2026-10-14", ["v6 a"]), entry("2026-10-23", ["v6 b"]),
                      entry("2026-10-24", ["v6 c"], ["v6 sector", "v6 market"])]}
    return {"4": v4, "6": v6}


# ---------- sheet 1 ----------

def test_sheet1_has_a_pair_row_per_paired_summary_and_an_overreach_row_per_industry_point():
    one = p6.sheet1(records(), {})
    assert len(one["pairs"]) == 4 and len(one["overreach"]) == 3
    assert len(one["key"]["letters"]) == 6
    runs = {(v["version"], v["run"]) for v in one["key"]["letters"].values()}
    assert len(runs) == 6
    paired = {one["key"]["letters"][r["summary"]]["run"] for r in one["pairs"]}
    assert paired == set(p6.PAIRS)


def test_sheet1_hides_the_versions():
    one = p6.sheet1(records(), {})
    shown = str(one["pairs"]) + str(one["overreach"]) + one["page"]
    for word in ("version", "prompt 4", "prompt 6", "v4 ", "v6 "):
        assert word not in shown.replace("v4 sector", "").replace("v6 sector", "").replace(
            "v6 market", "").replace("v4 a", "").replace("v4 b", "").replace("v4 c", "").replace(
            "v6 a", "").replace("v6 b", "").replace("v6 c", "")
    assert set(one["key"]["letters"]) == set("ABCDEF")


def test_sheet1_is_the_same_for_the_same_seed_and_shuffles_the_letters():
    a, b = p6.sheet1(records(), {}), p6.sheet1(records(), {})
    assert a["key"] == b["key"] and a["pairs"] == b["pairs"]
    order = [v["version"] for _, v in sorted(a["key"]["letters"].items())]
    assert order != ["4", "4", "4", "6", "6", "6"]


def filled(one, outlook="yes", apart="yes", over="no"):
    pairs = [dict(r, outlook_stated=outlook, kept_apart=apart) for r in one["pairs"]]
    overreach = [dict(r, overreach=over, note="claim" if over == "yes" else "") for r in one["overreach"]]
    return pairs, overreach


def test_sheet1_problems_finds_blanks_changed_rows_and_a_yes_without_a_note():
    one = p6.sheet1(records(), {})
    pairs, overreach = filled(one)
    assert p6.sheet1_problems(pairs, overreach, one["key"]) == []
    assert p6.sheet1_problems(one["pairs"], overreach, one["key"])
    changed = [dict(overreach[0], text="other")] + overreach[1:]
    assert any("changed" in f for f in p6.sheet1_problems(pairs, changed, one["key"]))
    no_note = [dict(overreach[0], overreach="yes", note="")] + overreach[1:]
    assert any("note" in f for f in p6.sheet1_problems(pairs, no_note, one["key"]))
    assert p6.sheet1_problems(pairs[1:], overreach, one["key"])


# ---------- sheet 2 decoys ----------

def p4_items(n=12):
    return [{"row": i, "kind": "key point", "text": f"locked {i}", "evidence": "e", "judgement": "supported",
             "contradicted": "no", "central": "yes", "note": ""} for i in range(1, n + 1)]


def new_sheets():
    items = [{"row": i, "kind": "key point", "text": f"new {i}", "evidence": "e", "judgement": "", "contradicted": "",
              "central": "", "note": ""} for i in (1, 2, 3)]
    key = {"items": {str(i): {"design": "direct", "run": "2026-10-14", "section": "key_points", "index": i - 1,
                              "ids": [1], "repaired": False, "fingerprint": sj._hash("key point", f"new {i}", "e")}
                     for i in (1, 2, 3)}, "carried": [], "summaries": {}, "events": {}}
    return {"items": items, "events": [], "key": key, "page": ""}


def test_decoys_are_unmarked_on_the_sheet_and_marked_in_the_key():
    two = p6.add_decoys(new_sheets(), p4_items())
    assert len(two["items"]) == 13
    assert all(set(r) == set(sj.ITEM_COLUMNS) and r["judgement"] == "" for r in two["items"])
    assert [r["row"] for r in two["items"]] == list(range(1, 14))
    decoys = [e for e in two["key"]["items"].values() if e.get("decoy")]
    assert len(decoys) == 10 and all(e["locked_judgement"] == "supported" for e in decoys)
    assert sj.item_problems([dict(r, judgement="supported", contradicted="no", central="")
                             for r in two["items"]], two["key"]["items"]) == []


def test_the_decoy_choice_is_fixed_by_its_seed():
    a, b = p6.add_decoys(new_sheets(), p4_items()), p6.add_decoys(new_sheets(), p4_items())
    assert a == b


def test_scored_items_drops_the_decoys_and_keeps_the_new_items_with_their_judgements():
    two = p6.add_decoys(new_sheets(), p4_items())
    judged = [dict(r, judgement="partly_supported" if r["text"].startswith("new") else "supported")
              for r in two["items"]]
    rows, key = p6.scored_items(judged, two["key"])
    assert sorted(r["text"] for r in rows) == ["new 1", "new 2", "new 3"]
    assert all(r["judgement"] == "partly_supported" for r in rows)
    assert [r["row"] for r in rows] == [1, 2, 3] and set(key["items"]) == {"1", "2", "3"}
    assert all(not e.get("decoy") for e in key["items"].values())


def test_decoy_agreement_counts_matches_with_the_locked_judgements():
    two = p6.add_decoys(new_sheets(), p4_items())
    first = True
    judged = []
    for r in two["items"]:
        e = two["key"]["items"][str(r["row"])]
        value = "supported"
        if e.get("decoy") and first:
            value, first = "not_supported", False
        judged.append(dict(r, judgement=value, contradicted="no"))
    agree = p6.decoy_agreement(judged, two["key"])
    assert agree["decoys"] == 10 and agree["same_judgement"] == 9 and agree["same_contradicted"] == 10


# ---------- the choice ----------

def p6_score(industry_not=0, contradicted=0, supported=32, total=40, events=None, empty=(), without=(), core_not=0):
    events = events or {"2026-10-14": (3, 3), "2026-10-23": (4, 4), "2026-10-24": (4, 4)}
    return {"total": {"industry": {"not_supported": industry_not, "contradicted": 0},
                      "core": {"contradicted": contradicted, "not_supported": core_not}},
            "per_run": {r: {"events": {"covered": c, "total": t}} for r, (c, t) in events.items()},
            "key_points_and_briefs": {"supported": supported, "total": total},
            "empty_section_with_evidence": list(empty), "points_without_evidence": list(without)}


P4_COVERAGE = {"2026-10-14": (3, 3), "2026-10-23": (4, 4), "2026-10-24": (4, 4)}
P4_EMPTY = 1                        # the locked prompt 4 score: 24 Oct has evidence and an empty section


def pairs(v4, v6):
    return {"4": dict(zip(sorted(p6.PAIRS), v4)), "6": dict(zip(sorted(p6.PAIRS), v6))}


def test_prompt6_is_chosen_when_its_pair_score_is_higher_and_every_guardrail_holds():
    c = p6.choose(pairs((1, 1), (2, 1)), {"4": 1, "6": 1}, p6_score(), P4_COVERAGE, [], P4_EMPTY)
    assert c["prompt"] == "6" and all(c["guardrails"].values())


@pytest.mark.parametrize("v4, v6", [((1, 1), (1, 1)), ((2, 0), (1, 2)), ((2, 2), (2, 2)), ((1, 1), (0, 2))])
def test_a_tie_or_a_lower_pair_keeps_prompt4(v4, v6):
    assert p6.choose(pairs(v4, v6), {"4": 0, "6": 0}, p6_score(), P4_COVERAGE, [], P4_EMPTY)["prompt"] == "4"


@pytest.mark.parametrize("overreach, score, guardrail", [
    ({"4": 0, "6": 0}, p6_score(industry_not=1), "no_unsupported_industry_point"),
    ({"4": 0, "6": 0}, p6_score(empty=["2026-10-23", "2026-10-24"]),
     "empty_sections_with_evidence_not_more_than_prompt4"),
    ({"4": 0, "6": 0}, p6_score(without=["2026-10-24"]), "no_industry_points_without_evidence"),
    ({"4": 0, "6": 1}, p6_score(), "overreach_not_higher"),
    ({"4": 0, "6": 0}, p6_score(contradicted=1), "no_contradicted_item"),
    ({"4": 0, "6": 0}, p6_score(core_not=1), "no_unsupported_key_point_or_brief"),
    ({"4": 0, "6": 0}, p6_score(supported=29, total=39), "supported_share_at_least_0_75"),
    ({"4": 0, "6": 0}, p6_score(events={"2026-10-14": (2, 3), "2026-10-23": (4, 4), "2026-10-24": (4, 4)}),
     "event_coverage_not_lower"),
    ({"4": 0, "6": 0}, p6_score(events={"2026-10-14": (3, 3), "2026-10-23": (4, 4), "2026-10-24": (5, 5)}),
     "event_coverage_not_lower"),
])
def test_each_guardrail_alone_keeps_prompt4(overreach, score, guardrail):
    c = p6.choose(pairs((0, 0), (2, 2)), overreach, score, P4_COVERAGE, [], P4_EMPTY)
    assert c["prompt"] == "4" and not c["guardrails"][guardrail]
    assert sum(not v for v in c["guardrails"].values()) == 1


def test_the_share_is_compared_unrounded_at_0_75():
    assert p6.choose(pairs((0, 0), (2, 2)), {"4": 0, "6": 0}, p6_score(supported=30, total=40), P4_COVERAGE, [], P4_EMPTY)[
        "guardrails"]["supported_share_at_least_0_75"]
    assert not p6.choose(pairs((0, 0), (2, 2)), {"4": 0, "6": 0}, p6_score(supported=2999, total=4000),
                         P4_COVERAGE, [], P4_EMPTY)["guardrails"]["supported_share_at_least_0_75"]


def test_pair_scores_and_overreach_counts_read_the_hidden_letters():
    one = p6.sheet1(records(), {})
    p, o = filled(one, outlook="yes", apart="no", over="yes")
    scores = p6.pair_scores(p, one["key"])
    assert scores == {"4": {r: 1 for r in p6.PAIRS}, "6": {r: 1 for r in p6.PAIRS}}
    assert p6.overreach_counts(o, one["key"]) == {"4": 1, "6": 2}


# ---------- the lock ----------

def test_lock_changes_finds_any_change_after_the_lock():
    sheets = ([{"row": 1, "a": "x"}], [], [], [])
    lock = {"sheets": [sj.sheet_fingerprint(s) for s in sheets], "key": sj._hash({"k": 1}), "code": "c",
            "inputs": {"plan": "1"}}
    assert p6.lock_changes(lock, sheets, {"k": 1}, {"plan": "1"}, "c") == []
    assert p6.lock_changes(lock, ([{"row": 1, "a": "y"}], [], [], []), {"k": 1}, {"plan": "1"}, "c")
    assert p6.lock_changes(lock, sheets, {"k": 2}, {"plan": "1"}, "c")
    assert p6.lock_changes(lock, sheets, {"k": 1}, {"plan": "2"}, "c")
    assert p6.lock_changes(lock, sheets, {"k": 1}, {"plan": "1"}, "d")


def test_lock_record_refuses_incomplete_sheets():
    one = p6.sheet1(records(), {})
    key = {"sheet1": one["key"], "sheet2": {"items": {}, "events": {}}}
    with pytest.raises(p6.Prompt6Error):
        p6.lock_record((one["pairs"], one["overreach"], [], []), key, {}, "c")
    p, o = filled(one)
    assert p6.lock_record((p, o, [], []), key, {}, "c")["code"] == "c"


def test_run_problems_finds_a_changed_guide():
    v6 = p6.prompt_v6(V4)
    plan = dict(plan_for(v6, "base"), guide_fingerprint="0" * 64)
    assert any(p6.GUIDE_FILE in f for f in p6.run_problems("p", "p", plan, "base", v6, []))
    text = (p6.EVAL_DIR / p6.GUIDE_FILE).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert p6.run_problems("p", "p", dict(plan, guide_fingerprint=sj._hash(text)), "base", v6, []) == []


# ---------- a failed or missing day ----------

DAYS = ["2026-10-14", "2026-10-23", "2026-10-24"]


def failed(record, run):
    out = {"records": [dict(e) for e in record["records"]]}
    for e in out["records"]:
        if e["run"] == run:
            e["result"] = {"status": "failed", "summary": None}
    return out


def test_complete_records_have_no_missing_summary():
    assert p6.missing_summaries(records(), DAYS) == []


def test_a_failed_a_missing_a_doubled_or_an_extra_day_is_listed():
    r = records()
    assert p6.missing_summaries({"4": r["4"], "6": failed(r["6"], "2026-10-23")}, DAYS) == \
        ["prompt 6, 2026-10-23: no usable summary"]
    gone = {"records": [e for e in r["6"]["records"] if e["run"] != "2026-10-14"]}
    assert p6.missing_summaries({"4": r["4"], "6": gone}, DAYS) == ["prompt 6, 2026-10-14: 0 records instead of 1"]
    doubled = {"records": r["6"]["records"] + r["6"]["records"][:1]}
    assert "2 records" in p6.missing_summaries({"4": r["4"], "6": doubled}, DAYS)[0]
    empty = {"records": [dict(e, result={"status": "ok", "summary": {}}) if e["run"] == "2026-10-24" else e
                         for e in r["6"]["records"]]}
    assert p6.missing_summaries({"4": r["4"], "6": empty}, DAYS) == ["prompt 6, 2026-10-24: no usable summary"]
    extra = {"records": r["6"]["records"] + [entry("2026-10-30", ["x"])]}
    assert p6.missing_summaries({"4": r["4"], "6": extra}, DAYS) == ["prompt 6, 2026-10-30: not an expected day"]
    assert p6.missing_summaries({"4": failed(r["4"], "2026-10-14"), "6": r["6"]}, DAYS)


def test_sheet1_refuses_to_build_without_every_planned_pair_row():
    r = records()
    with pytest.raises(p6.Prompt6Error):
        p6.sheet1({"4": r["4"], "6": failed(r["6"], "2026-10-24")}, {})
    assert p6.sheet1(r, {})["key"]["planned_pairs"] == sorted(f"{v}:{d}" for v, d in p6.expected_pairs())


def test_a_missing_pair_row_in_the_key_is_found_at_the_lock():
    one = p6.sheet1(records(), {})
    p, o = filled(one)
    key = dict(one["key"], pairs={n: e for n, e in one["key"]["pairs"].items() if n != "1"})
    found = p6.sheet1_problems([r for r in p if str(r["row"]) != "1"], o, key)
    assert any("planned" in f for f in found)
    assert p6.sheet1_problems(p, o, dict(one["key"], planned_pairs=[]))


@pytest.mark.parametrize("missing", [None, ["prompt 6, 2026-10-23: no usable summary"]])
def test_a_failed_day_never_selects_prompt6_even_with_the_best_scores(missing):
    c = p6.choose(pairs((0, 0), (2, 2)), {"4": 1, "6": 0}, p6_score(), P4_COVERAGE, missing, P4_EMPTY)
    assert c["prompt"] == "4" and not c["guardrails"]["every_expected_day_has_a_usable_summary"]


def test_a_missing_pair_score_or_day_in_the_score_never_selects_prompt6():
    short = {"4": dict(zip(sorted(p6.PAIRS), (0, 0))), "6": {sorted(p6.PAIRS)[0]: 2}}
    assert p6.choose(short, {"4": 0, "6": 0}, p6_score(), P4_COVERAGE, [], P4_EMPTY)["prompt"] == "4"
    two_days = p6_score(events={"2026-10-23": (4, 4), "2026-10-24": (4, 4)})
    assert p6.choose(pairs((0, 0), (2, 2)), {"4": 0, "6": 0}, two_days, P4_COVERAGE, [], P4_EMPTY)["prompt"] == "4"


def test_the_failed_day_result_keeps_prompt4_and_names_what_is_missing():
    result = p6.failed_result(["prompt 6, 2026-10-23: no usable summary"])
    assert result["choice"]["prompt"] == "4" and result["choice"]["missing"]
    assert "not repeated" in result["rule"]["failed_day"]


# ---------- the industry section, compared with prompt 4 ----------

def best(score, p4_empty=P4_EMPTY):
    return p6.choose(pairs((0, 0), (2, 2)), {"4": 0, "6": 0}, score, P4_COVERAGE, [], p4_empty)


def test_one_empty_day_with_evidence_like_prompt4_passes():
    c = best(p6_score(empty=["2026-10-24"]))
    assert c["guardrails"]["empty_sections_with_evidence_not_more_than_prompt4"] and c["prompt"] == "6"


def test_more_empty_days_with_evidence_than_prompt4_fails():
    c = best(p6_score(empty=["2026-10-23", "2026-10-24"]))
    assert not c["guardrails"]["empty_sections_with_evidence_not_more_than_prompt4"] and c["prompt"] == "4"
    assert best(p6_score(empty=["2026-10-24"]), p4_empty=0)["prompt"] == "4"


def test_the_count_comes_from_the_baseline_not_a_fixed_number():
    two = p6_score(empty=["2026-10-23", "2026-10-24"])
    assert best(two, p4_empty=2)["guardrails"]["empty_sections_with_evidence_not_more_than_prompt4"]
    assert best(p6_score(), p4_empty=None)["prompt"] == "4"


def test_industry_points_without_evidence_fail_even_with_fewer_empty_days():
    c = best(p6_score(empty=[], without=["2026-10-14"]))
    assert c["guardrails"]["empty_sections_with_evidence_not_more_than_prompt4"]
    assert not c["guardrails"]["no_industry_points_without_evidence"] and c["prompt"] == "4"


def test_a_contradicted_item_still_keeps_prompt4_on_its_own():
    c = best(p6_score(contradicted=1, empty=["2026-10-24"]))
    assert not c["guardrails"]["no_contradicted_item"] and c["prompt"] == "4"
    assert sum(not v for v in c["guardrails"].values()) == 1
