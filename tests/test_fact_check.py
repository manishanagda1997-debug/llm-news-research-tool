"""
Tests for the hand check of important facts (Stage 4.1b; no network, no model calls).

Which input should the summary model read for a long article? The check
compares three inputs on the 17 relevant dev articles longer than 3,500
characters (shorter ones are given whole by every method):

    short     the NewsAPI short text (title, description, first 200 characters)
    passage   the relevance passage (src/relevance_check.py)
    evidence  the summary evidence (src/summary_evidence.py)

    facts    the labeller reads each full article and lists its important
             facts (seven categories), marking the central ones, before
             seeing any input; the list is fingerprinted when the inputs
             are made, so it cannot quietly change afterwards
    inputs   each article's three inputs appear as X, Y and Z, shuffled per
             article by a fixed seed; no file stores which is which
    marks    each fact against each input: kept yes / partly / no, and
             distorted yes / no (the meaning changes, for example "not
             approved" becomes "approved")
    scope    every fact is `company` (Tata Motors, its businesses and deals) or
             `context` (industry, market, competitors); scope and central are
             separate: a context fact can be central
    kept     only `yes` counts as kept; `partly` and `no` are counted on their
             own, for central, company and context facts
    choice   an input that distorts a central fact is out; then the highest
             share of central facts kept, then of company facts, then fewer
             other distortions, then the highest share of context facts; on a
             tie, the relevance passage (it already exists)
"""

import pytest

from src import fact_check as fc


# ---------- the sample ----------

def _row(article_id, label="direct", basis="full text"):
    return {"article_id": article_id, "label": label, "basis": basis, "title": f"T {article_id}"}


def test_the_sample_is_every_relevant_full_text_article_over_the_limit():
    rows = [_row("A1"), _row("A2", "indirect"), _row("A3", "not"), _row("A4", basis="short text (x)"), _row("A5")]
    lengths = {"A1": 5000, "A2": 3600, "A3": 9000, "A4": 9000, "A5": 3500}
    assert [r["article_id"] for r in fc.sample(rows, lengths)] == ["A1", "A2"]


# ---------- step A: facts ----------

def test_the_fact_sheet_has_empty_slots_for_every_article():
    rows = fc.fact_rows([_row("A1"), _row("A2")])
    assert list(rows[0]) == fc.FACT_COLUMNS == ["article_id", "title", "fact_no", "category", "fact", "central",
                                                "scope", "note"]
    assert len(rows) == 2 * fc.SLOTS and all(r["fact"] == "" for r in rows)


def _fact(article_id, n, fact="Tata Motors raises its offer.", category="main development", central="yes",
          scope="company"):
    return {"article_id": article_id, "title": "", "fact_no": str(n), "category": category, "fact": fact,
            "central": central, "scope": scope, "note": ""}


def test_a_complete_fact_list_passes():
    facts = [_fact("A1", 1), _fact("A1", 2, central="no", category="quantities"), _fact("A2", 1),
             {**_fact("A2", 2), "fact": "", "category": "", "central": "", "scope": ""}]   # an empty slot is fine
    assert fc.fact_problems(facts, ["A1", "A2"]) == []


@pytest.mark.parametrize("change, problem", [
    ({"category": "gossip"}, "category"),
    ({"central": "maybe"}, "central"),
    ({"scope": "industry"}, "scope"),
    ({"scope": ""}, "scope"),
])
def test_a_wrong_value_is_named(change, problem):
    facts = [{**_fact("A1", 1), **change}]
    assert any(problem in p for p in fc.fact_problems(facts, ["A1"]))


def test_every_article_needs_a_fact_and_a_central_fact():
    facts = [_fact("A1", 1, central="no")]
    problems = fc.fact_problems(facts, ["A1", "A2"])
    assert any("A1" in p and "central" in p for p in problems) and any("A2" in p for p in problems)


def test_the_seven_categories_are_the_agreed_ones():
    assert fc.CATEGORIES == ("main development", "who is involved", "quantities", "status and timing", "cause",
                             "impact and risks", "qualifications")


def test_the_fact_list_fingerprint_changes_when_a_fact_changes():
    facts = [_fact("A1", 1), _fact("A1", 2, central="no")]
    changed = [_fact("A1", 1), _fact("A1", 2, central="yes")]
    rescoped = [_fact("A1", 1), _fact("A1", 2, central="no", scope="context")]
    assert fc.facts_fingerprint(facts) == fc.facts_fingerprint(list(reversed(facts)))
    assert fc.facts_fingerprint(facts) != fc.facts_fingerprint(changed)
    assert fc.facts_fingerprint(facts) != fc.facts_fingerprint(rescoped)


# ---------- step B: blind inputs ----------

def test_the_xyz_order_is_fixed_per_article_and_differs_between_articles():
    orders = {a: fc.xyz_order(a) for a in [f"A{i}" for i in range(20)]}
    assert all(sorted(o.values()) == sorted(fc.METHODS) for o in orders.values())
    assert orders["A3"] == fc.xyz_order("A3")
    assert len({tuple(o.items()) for o in orders.values()}) > 1


def test_the_marks_sheet_has_every_fact_against_every_input():
    facts = [_fact("A1", 1), _fact("A1", 2, central="no")]
    rows = fc.mark_rows(facts)
    assert list(rows[0]) == fc.MARK_COLUMNS == ["article_id", "fact_no", "fact", "input", "kept", "distorted", "note"]
    assert [(r["fact_no"], r["input"]) for r in rows] == [("1", "X"), ("1", "Y"), ("1", "Z"),
                                                         ("2", "X"), ("2", "Y"), ("2", "Z")]


def test_the_inputs_page_never_names_a_method():
    page = fc.inputs_page([{"article_id": "A1", "title": "T", "facts": [_fact("A1", 1)],
                            "inputs": {"X": "text one", "Y": "text two", "Z": "text three"}}]).lower()
    for word in ("short", "passage", "evidence", "newsapi", "relevance", "summary"):
        assert word not in page


# ---------- step C: marks and the choice ----------

def _mark(article_id, n, letter, kept="yes", distorted="no"):
    return {"article_id": article_id, "fact_no": str(n), "fact": "", "input": letter, "kept": kept,
            "distorted": distorted, "note": ""}


def test_marks_need_valid_values_and_a_distortion_needs_a_kept_fact():
    assert fc.mark_problems([_mark("A1", 1, "X")]) == []
    assert fc.mark_problems([_mark("A1", 1, "X", kept="maybe")])
    assert fc.mark_problems([_mark("A1", 1, "X", distorted="")])
    assert fc.mark_problems([_mark("A1", 1, "X", kept="no", distorted="yes")])


def test_a_context_fact_can_be_central_and_a_company_fact_supporting():
    facts = [_fact("A1", 1, central="yes", scope="context"), _fact("A1", 2, central="no", scope="company")]
    assert fc.fact_problems(facts, ["A1"]) == []


def _scored():
    """A1: fact 1 central company, fact 2 supporting company, fact 3 central context, fact 4 supporting context."""
    facts = [_fact("A1", 1), _fact("A1", 2, central="no"), _fact("A1", 3, scope="context"),
             _fact("A1", 4, central="no", scope="context")]
    letter = {method: l for l, method in fc.xyz_order("A1").items()}
    plan = {"evidence": [("yes", "no"), ("yes", "no"), ("yes", "no"), ("partly", "no")],
            "passage": [("yes", "no"), ("partly", "no"), ("no", "no"), ("no", "no")],
            "short": [("yes", "yes"), ("no", "no"), ("partly", "yes"), ("no", "no")]}
    marks = [_mark("A1", n, letter[method], kept, distorted)
             for method, rows in plan.items() for n, (kept, distorted) in enumerate(rows, start=1)]
    return fc.score(facts, marks)


def test_only_yes_counts_as_kept_and_partly_is_counted_on_its_own():
    scores = _scored()
    assert scores["evidence"]["context"] == {"facts": 2, "yes": 1, "partly": 1, "no": 0, "distorted": 0, "kept": 0.5}
    assert scores["passage"]["company"] == {"facts": 2, "yes": 1, "partly": 1, "no": 0, "distorted": 0, "kept": 0.5}


def test_central_company_and_context_facts_are_scored_apart():
    s = _scored()["evidence"]
    assert s["central"]["facts"] == 2 and s["central"]["kept"] == 1.0              # facts 1 and 3
    assert s["company"]["facts"] == 2 and s["company"]["kept"] == 1.0              # facts 1 and 2
    assert s["context"]["facts"] == 2 and s["all"]["facts"] == 4


def test_distortions_are_counted_for_central_facts_and_for_the_others():
    s = _scored()["short"]
    assert s["central_distortions"] == 2 and s["other_distortions"] == 0
    assert s["central"]["distorted"] == 2


def test_a_group_with_no_facts_has_no_share():
    facts = [_fact("A1", 1)]
    letter = {method: l for l, method in fc.xyz_order("A1").items()}
    scores = fc.score(facts, [_mark("A1", 1, letter[m]) for m in fc.METHODS])
    assert scores["passage"]["context"] == {"facts": 0, "yes": 0, "partly": 0, "no": 0, "distorted": 0, "kept": None}


def _s(central, company, context=None, cd=0, od=0):
    return {"central": {"kept": central}, "company": {"kept": company}, "context": {"kept": context},
            "central_distortions": cd, "other_distortions": od}


def test_the_choice_rule_in_the_agreed_order():
    # 1. a central distortion puts an input out
    assert fc.choose_input({"short": _s(1, 1, cd=1), "passage": _s(0.8, 0.7), "evidence": _s(0.9, 0.6)}) == "evidence"
    # 2. central facts kept
    assert fc.choose_input({"short": _s(0.5, 1), "passage": _s(0.8, 1), "evidence": _s(0.9, 0.2)}) == "evidence"
    # 3. company facts kept
    assert fc.choose_input({"short": _s(0.5, 1), "passage": _s(0.9, 0.7), "evidence": _s(0.9, 0.8)}) == "evidence"
    # 4. fewer other distortions comes before context
    assert fc.choose_input({"short": _s(0.5, 1), "passage": _s(0.9, 0.8, 0.2, od=0),
                            "evidence": _s(0.9, 0.8, 1.0, od=1)}) == "passage"
    # 5. context facts kept
    assert fc.choose_input({"short": _s(0.5, 1), "passage": _s(0.9, 0.8, 0.5),
                            "evidence": _s(0.9, 0.8, 0.9)}) == "evidence"
    # 6. a genuine tie: the relevance passage
    assert fc.choose_input({"short": _s(0.5, 1), "passage": _s(0.9, 0.8, 0.5),
                            "evidence": _s(0.9, 0.8, 0.5)}) == "passage"
    # every input distorts a central fact: none is chosen
    assert fc.choose_input({m: _s(1, 1, cd=1) for m in fc.METHODS}) is None


def test_no_context_facts_ties_on_context():
    assert fc.choose_input({"short": _s(0.5, 1), "passage": _s(0.9, 0.8), "evidence": _s(0.9, 0.8)}) == "passage"

def test_the_marks_must_cover_every_fact_against_every_input():
    facts = [_fact("A1", 1)]
    with pytest.raises(fc.FactCheckError):
        fc.score(facts, [_mark("A1", 1, "X"), _mark("A1", 1, "Y")])


def test_the_folders_resolve():
    eval_dir, fulltext_dir = fc._paths()
    assert eval_dir.name == "eval" and fulltext_dir.name == "fulltext"


# ---------- articles whose saved text is another story ----------

def test_an_article_whose_saved_text_fails_the_wrong_page_check_leaves_the_sample():
    rows = [_row("A1"), _row("A2")]
    lengths = {"A1": 5000, "A2": 5000}
    assert [r["article_id"] for r in fc.sample(rows, lengths, wrong_page={"A2"})] == ["A1"]


def test_facts_of_articles_outside_the_sample_are_ignored_not_deleted():
    facts = [_fact("A1", 1), _fact("A2", 1)]
    kept = fc.facts_for(facts, ["A1"])
    assert [f["article_id"] for f in kept] == ["A1"] and len(facts) == 2
    assert [r["article_id"] for r in fc.mark_rows(kept)] == ["A1"] * 3


# ---------- Step A: the summary passage version 2 against the facts ----------

def test_marks_for_the_new_passage_cover_only_the_facts_of_changed_articles():
    facts = [_fact("A1", 1), _fact("A1", 2, central="no"), _fact("A2", 1)]
    rows = fc.passage2_mark_rows(facts, changed={"A1"})
    assert [(r["article_id"], r["fact_no"], r["input"]) for r in rows] == [("A1", "1", "P"), ("A1", "2", "P")]
    assert list(rows[0]) == fc.MARK_COLUMNS


def _old_marks(article_id, values):
    """The passage marks of the first check, placed under the passage's letter."""
    letter = {method: l for l, method in fc.xyz_order(article_id).items()}
    rows = []
    for n, kept in enumerate(values, start=1):
        for method in fc.METHODS:
            rows.append(_mark(article_id, n, letter[method], kept if method == "passage" else "no"))
    return rows


def test_the_new_passage_is_accepted_when_it_gains_and_loses_nothing():
    facts = [_fact("A1", 1), _fact("A1", 2, central="no"), _fact("A2", 1)]
    old = _old_marks("A1", ["no", "yes"]) + _old_marks("A2", ["yes"])
    new = [_mark("A1", 1, "P", "yes"), _mark("A1", 2, "P", "yes")]
    added = [_added("A1")]
    result = fc.passage2_compare(facts, old, new, changed={"A1"}, lengths={"A1": 3900, "A2": 3000}, added=added)
    assert result["old"]["central"]["yes"] == 1 and result["new"]["central"]["yes"] == 2
    assert result["gained"] == [["A1", "1"]] and result["lost"] == [] and result["accepted"] is True


@pytest.mark.parametrize("new_values, distorted, length, reason", [
    (["yes", "partly"], "no", 3400, "lost"),          # a fact kept in full before is now partly kept
    (["yes", "yes"], "yes", 3400, "distortions"),     # a new distortion
    (["yes", "yes"], "no", 4300, "length"),           # over the hard cap
])
def test_the_new_passage_is_refused_on_any_loss_distortion_or_overlong_passage(new_values, distorted, length, reason):
    facts = [_fact("A1", 1), _fact("A1", 2, central="no")]
    old = _old_marks("A1", ["no", "yes"])
    new = [_mark("A1", 1, "P", new_values[0], distorted), _mark("A1", 2, "P", new_values[1])]
    result = fc.passage2_compare(facts, old, new, changed={"A1"}, lengths={"A1": length}, added=[_added("A1")])
    assert result["accepted"] is False and reason in result["refused_because"]


def _added(article_id, attribution="yes", meaning="yes"):
    return {"article_id": article_id, "added_text": "“…”", "attribution_correct": attribution,
            "meaning_preserved": meaning, "note": ""}


@pytest.mark.parametrize("attribution, meaning", [("no", "yes"), ("yes", "no"), ("", "yes")])
def test_the_added_text_must_keep_attribution_and_meaning(attribution, meaning):
    facts = [_fact("A1", 1)]
    old = _old_marks("A1", ["no"])
    new = [_mark("A1", 1, "P", "yes")]
    result_or_error = None
    try:
        result_or_error = fc.passage2_compare(facts, old, new, changed={"A1"}, lengths={"A1": 3900},
                                              added=[_added("A1", attribution, meaning)])
    except fc.FactCheckError:
        assert attribution == ""                        # an empty answer is a mark to fix
        return
    assert result_or_error["accepted"] is False and "added text" in result_or_error["refused_because"]


def test_the_added_text_sheet_has_one_row_per_added_paragraph():
    rows = fc.added_rows([{"article_id": "A1", "added": ["“One.”", "“Two.”"]}, {"article_id": "A2", "added": []}])
    assert [(r["article_id"], r["added_text"]) for r in rows] == [("A1", "“One.”"), ("A1", "“Two.”")]
    assert list(rows[0]) == fc.ADDED_COLUMNS


def test_the_inputs_page_shows_whatever_inputs_an_article_has():
    page = fc.inputs_page([{"article_id": "A1", "title": "T", "facts": [_fact("A1", 1)],
                            "inputs": {"P": "new passage text", "Added text (also inside P)": "“a quote”"}}])
    assert "new passage text" in page and "Added text (also inside P)" in page and "a quote" in page


def test_the_passage2_files_are_written_only_after_everything_is_built(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fc, "_write_csv", lambda path, rows, columns: calls.append(path.name))
    monkeypatch.setattr(fc, "_write_new", lambda path, text: calls.append(path.name))
    files = fc.passage2_files([{"article_id": "A1", "title": "T", "old": "o", "new": "n", "added": ["“q”"],
                                "not_fitted": 0}], [_fact("A1", 1)], "fp")
    assert calls == [] and set(files) == {fc.PASSAGE2_MARKS_FILE, fc.ADDED_FILE, fc.PASSAGE2_PAGE,
                                          fc.PASSAGE2_RECORD_FILE}
