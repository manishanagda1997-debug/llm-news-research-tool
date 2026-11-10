"""
Tests for the summary pipeline version 2 (Stage 4.2, no network, no model calls).

Both designs read the same articles: the top k by O1+S1, each as its summary passage (version 2). Both
produce the same full output (article_briefs, key_points, industry_context, not_used) and pass the same
checks (src/summarizer.py): structure, citations, accounting and number consistency.

    direct       one call reads all k passages and writes the whole summary
    brief_first  one call per article writes its brief; one more call combines the briefs into the key
                 points, the industry context and the articles not used; the saved briefs and the combined
                 points are assembled into the full output and checked as one

Provider failures (a refused request, a rate or daily limit) are recorded by kind, apart from quality.
"""

import json

import pytest

from src import summary_v2 as sv
from src.summarizer import LLMUnavailable


def _article(i, passage):
    return {"id": i, "key": f"A{i}", "title": f"Tata Motors news {i}", "source": "S", "published_at": "",
            "url": f"https://x.com/{i}", "description": passage, "content": ""}


ARTICLES = [_article(1, "Tata Motors raised its Iveco offer to 14.40 euro per share."),
            _article(2, "Car registrations fell 22 per cent this Navratri.")]


class ScriptedLLM:
    """Answers brief prompts and combine prompts from fixed scripts."""
    model_name, cache_tag, provider = "fake", "", "fake"

    def __init__(self, combine=None, brief=None, fail_on=None):
        self.prompts = []
        self.combine = combine or {"key_points": [{"text": "Tata Motors raised its offer to 14.40 euro.",
                                                   "citations": [1]}],
                                   "industry_context": [{"text": "Registrations fell 22 per cent.",
                                                         "citations": [2]}],
                                   "not_used": []}
        self.brief = brief or (lambda article_id: {"brief": f"Short brief for story {article_id}.", "enough_text": True})
        self.fail_on = fail_on

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self.fail_on is not None and len(self.prompts) == self.fail_on:
            error = LLMUnavailable("refused")
            error.kind = "too_large"
            raise error
        if prompt.startswith(sv.COMBINE_RULES[:40]):
            return json.dumps(self.combine)
        article_id = int(prompt.split('<article id="')[1].split('"')[0])
        return json.dumps(self.brief(article_id))


def test_brief_first_makes_one_call_per_article_and_one_to_combine():
    llm = ScriptedLLM()
    result = sv.summarize_brief_first(ARTICLES, llm)
    assert len(llm.prompts) == len(ARTICLES) + 1
    assert result["status"] == "ok"


def test_brief_first_assembles_the_same_full_output_as_the_direct_design():
    result = sv.summarize_brief_first(ARTICLES, ScriptedLLM())
    summary = result["summary"]
    assert set(summary) >= {"article_briefs", "key_points", "industry_context", "not_used"}
    assert [b["id"] for b in summary["article_briefs"]] == [1, 2]
    assert summary["article_briefs"][0]["brief"] == "Short brief for story 1."


def test_each_brief_prompt_holds_only_its_own_article():
    llm = ScriptedLLM()
    sv.summarize_brief_first(ARTICLES, llm)
    assert "14.40" in llm.prompts[0] and "Navratri" not in llm.prompts[0]


def test_a_combined_point_with_an_unconfirmed_number_is_removed_by_the_same_rule_as_direct():
    combine = {"key_points": [{"text": "Tata Motors raised its offer to 14.40 euro.", "citations": [1]},
                              {"text": "Tata Motors sold 99,999 cars.", "citations": [1]}],
               "industry_context": [], "not_used": [{"id": 2, "reason": "no point"}]}
    result = sv.summarize_brief_first(ARTICLES, ScriptedLLM(combine=combine))
    assert result["status"] == "partial"
    assert [p["text"] for p in result["summary"]["key_points"]] == ["Tata Motors raised its offer to 14.40 euro."]
    assert any("99,999" in r["text"] for r in result["removed"])


def test_a_provider_failure_gives_no_summary_and_its_kind_is_recorded():
    llm = sv.Recorder(ScriptedLLM(fail_on=2))
    result = sv.summarize_brief_first(ARTICLES, llm)
    assert result["status"] == "unavailable" and llm.failures == ["too_large"]


def test_the_direct_design_uses_the_existing_summary_step_on_the_same_articles(monkeypatch):
    seen = {}
    monkeypatch.setattr(sv, "summarize", lambda articles, llm, correction: seen.update(correction=correction) or
                        seen.setdefault("articles", articles) and
                        {"status": "ok", "summary": {}, "removed": [], "attempts": 1})
    sv.run_design("direct", ARTICLES, ScriptedLLM())
    assert seen["articles"] == ARTICLES and seen["correction"] == sv.CORRECTION


def test_the_recorder_counts_calls_and_characters():
    llm = sv.Recorder(lambda prompt: "{}")
    llm("abcd")
    llm("ef")
    assert llm.calls == 2 and llm.prompt_chars == 6 and llm.failures == []


def test_each_design_has_its_own_cache_key():
    assert sv.cache_key(ARTICLES, "m", "direct", 5) != sv.cache_key(ARTICLES, "m", "brief_first", 5)


def test_the_articles_are_the_top_k_by_o1_s1_as_summary_passages(monkeypatch):
    candidates = [{"key": f"K{i}", "title": f"Tata Motors item {i}", "description": "d", "source": "S",
                   "url": f"https://x.com/{i}", "text": "", "mention_level": "title", "check": None}
                  for i in range(8)]
    companies = set()
    monkeypatch.setattr(sv, "_passage", lambda c, aliases, company: companies.add(company) or f"passage of {c['key']}")
    articles = sv.summary_articles(candidates, 5, aliases=None)
    assert companies == {"Tata Motors"}                                             # Stage 4 stays on Tata Motors
    assert [a["id"] for a in articles] == [1, 2, 3, 4, 5]
    assert [a["key"] for a in articles] == ["K0", "K1", "K2", "K3", "K4"]          # O1 keeps NewsAPI order on a tie
    assert articles[0]["description"] == "passage of K0" and articles[0]["content"] == ""


# ---------- numbers: the independent check (regression cases from the pilot) ----------

from src.summarizer import unconfirmed_numbers, PROMPT_VERSION, RULES, repair_points, summarize as plain_summarize

PILOT_SOURCE = ("JM Financial estimates volumes at 1,35,114 units. MOFSL projects net operating income of "
                "Rs 24,638.1 crore; JM estimates Rs 24,218.4 crore. Maruti sold 74,387 cars. Tata Motors fell "
                "16.4% to 24,843 units.")


@pytest.mark.parametrize("claim", [
    "volumes of about 1.35 million units",          # Indian grouping turned into millions
    "revenue of Rs 24,200-24,600 crore",            # a rounded range
    "revenue of Rs 24,218 crore",                   # a dropped decimal
    "revenue of Rs 24.2 billion",                   # a converted unit
])
def test_the_number_check_refuses_changed_numbers(claim):
    assert unconfirmed_numbers(claim, PILOT_SOURCE)


@pytest.mark.parametrize("claim", ["volumes of 1,35,114 units", "revenue of Rs 24,218.4-24,638.1 crore"])
def test_the_number_check_accepts_numbers_copied_exactly(claim):
    assert unconfirmed_numbers(claim, PILOT_SOURCE) == []


def test_a_known_limit_the_right_number_for_the_wrong_company_passes_the_check():
    """The check proves only that a number appears in the cited text. Company, metric, period and direction
    are judged by a person (SUMMARY_JUDGING_GUIDE.md, Part 2, Numbers)."""
    assert unconfirmed_numbers("Maruti fell 16.4% to 24,843 units", PILOT_SOURCE) == []


def test_the_prompts_carry_the_same_general_number_rule_and_a_new_version():
    assert PROMPT_VERSION == "4"
    for prompt in (RULES, sv.BRIEF_RULES, sv.COMBINE_RULES):
        assert "digit grouping" in prompt and "both ends of a range" in prompt and "Never round" in prompt
    assert "1,35,114" not in RULES                  # no figure from one article in the instructions


# ---------- the targeted repair of a point with an unconfirmed number ----------

def _output(text):
    return {"article_briefs": [{"id": 1, "brief": "Tata Motors raised its offer.", "enough_text": True},
                               {"id": 2, "brief": "Registrations fell.", "enough_text": True}],
            "key_points": [{"text": text, "citations": [1]}], "industry_context": [],
            "not_used": [{"id": 2, "reason": "r"}]}


def test_a_successful_repair_keeps_the_story_with_the_exact_number():
    output = _output("Tata Motors raised its offer to 14 euro per share.")
    report = {"structural": [], "numbers": [{"section": "key_points", "index": 0, "numbers": ["14 euro"]}]}
    repaired, repairs = repair_points(output, report, ARTICLES,
                                       lambda prompt: json.dumps({"text": "Tata Motors raised its offer to 14.40 euro per share."}))
    assert repaired["key_points"][0]["text"] == "Tata Motors raised its offer to 14.40 euro per share."
    assert repaired["key_points"][0]["citations"] == [1] and repairs[0]["ok"] is True


def test_a_failed_repair_leaves_the_point_to_be_removed_by_the_usual_rule():
    output = _output("Tata Motors raised its offer to 14 euro per share.")
    report = {"structural": [], "numbers": [{"section": "key_points", "index": 0, "numbers": ["14 euro"]}]}
    repaired, repairs = repair_points(output, report, ARTICLES,
                                       lambda prompt: json.dumps({"text": "Tata Motors raised its offer to 15 euro."}))
    assert repaired["key_points"][0]["text"] == "Tata Motors raised its offer to 14 euro per share."
    assert repairs[0]["ok"] is False


def test_the_repair_prompt_holds_the_point_the_numbers_and_only_the_cited_article():
    seen = []
    output = _output("Tata Motors raised its offer to 14 euro per share.")
    report = {"structural": [], "numbers": [{"section": "key_points", "index": 0, "numbers": ["14 euro"]}]}
    repair_points(output, report, ARTICLES, lambda prompt: seen.append(prompt) or json.dumps({"text": "x"}))
    assert "14 euro" in seen[0] and "14.40 euro per share" in seen[0] and "Navratri" not in seen[0]


def test_the_whole_flow_repairs_instead_of_removing_when_the_repair_succeeds():
    answers = iter([
        json.dumps(_output("Tata Motors raised its offer to 14 euro per share.")),     # attempt 1
        json.dumps(_output("Tata Motors raised its offer to 14 euro per share.")),     # attempt 2, same mistake
        json.dumps({"text": "Tata Motors raised its offer to 14.40 euro per share."}),  # the repair
    ])
    result = plain_summarize(ARTICLES, lambda prompt: next(answers))
    assert result["status"] == "ok" and result["removed"] == []
    assert result["repairs"][0]["ok"] is True


# ---------- diagnostics for answers that are not valid JSON ----------
#
# The development runner checks every answer it records. A broken answer gets a description in the run
# record (length, error, position, fences, text around the JSON, finish reason) and its excerpt goes to an
# ignored local file, named by a fingerprint that the record keeps. Excerpts never go into Git.

def test_a_broken_answer_is_described_without_its_text():
    text = '{"key_points": [{"text": "He said "we will grow" today", "citations": [1]}]}'
    report = sv.bad_json_report(text)
    assert report["length"] == len(text)
    assert report["error"] and isinstance(report["position"], int)
    assert report["starts_with_brace"] and report["ends_with_brace"]
    assert not report["fenced"] and not report["text_before_json"] and not report["text_after_json"]
    assert "we will grow" not in json.dumps(report)


def test_a_cut_off_answer_is_seen_as_not_ending_with_a_brace():
    report = sv.bad_json_report('{"key_points": [{"text": "Tata Motors')
    assert report["starts_with_brace"] and not report["ends_with_brace"]


def test_text_around_the_json_is_seen():
    report = sv.bad_json_report('Here is the summary:\n{"a": 1} Hope this helps')
    assert report["text_before_json"] and report["text_after_json"]


def test_a_valid_answer_gives_no_report():
    assert sv.bad_json_report('{"a": 1}') is None
    assert sv.bad_json_report('```json\n{"a": 1}\n```') is None


def test_the_excerpt_is_the_text_around_the_error():
    text = "x" * 1000 + "ERROR" + "y" * 1000
    excerpt = sv.excerpt(text, 1000, width=50)
    assert "ERROR" in excerpt and len(excerpt) <= 105


def test_the_diagnostic_folder_is_ignored_by_git():
    from pathlib import Path
    assert sv.DIAGNOSTIC_DIR.name == "diagnostics"
    assert "data/diagnostics/" in Path(".gitignore").read_text(encoding="utf-8").splitlines()


class StatsLLM:
    """A fake model with the same stats the real one keeps."""

    model_name, cache_tag, provider, generation = "fake", "", "groq", {}

    def __init__(self, *answers):
        self.answers = list(answers)
        self.stats = {"per_call": [], "requests": []}

    def __call__(self, prompt):
        answer = self.answers.pop(0)
        self.stats["requests"].append({"estimated": 10, "prompt_chars": len(prompt), "outcome": "answered"})
        self.stats["per_call"].append({"input": 10, "output": 5, "reasoning": 1, "cached": 0,
                                       "estimated": 10, "prompt_chars": len(prompt), "finish_reason": "stop"})
        return answer if isinstance(answer, str) else json.dumps(answer)


def test_the_runner_records_broken_answers_and_keeps_excerpts_out_of_the_record(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "DIAGNOSTIC_DIR", tmp_path)
    broken = '{"key_points": [{"text": "He said "growth" today"}]}'
    llm = StatsLLM(broken, broken)
    result = sv._one("direct", ARTICLES, llm, sv.SummaryCache(tmp_path / "cache"), 3)
    assert result["status"] == "unavailable"
    assert [b["call"] for b in result["bad_answers"]] == [1, 2]
    assert all(b["finish_reason"] == "stop" for b in result["bad_answers"])
    files = [b["excerpt_file"] for b in result["bad_answers"]]
    assert all((tmp_path / name).exists() for name in files)
    assert "growth" not in json.dumps(result)
    assert result["usage"]["requests"] == [{"estimated": 10, "prompt_chars": r["prompt_chars"], "outcome": "answered"}
                                           for r in llm.stats["requests"]]
    assert result["usage"]["finish_reasons"] == ["stop", "stop"]


def test_the_recorder_marks_a_preflight_refusal_apart_from_a_provider_refusal():
    from src.llm import RequestTooLarge

    def refuse(stage):
        def llm(prompt):
            error = RequestTooLarge("too big")
            error.kind, error.stage = "too_large", stage
            raise error
        return llm

    for stage, expected in (("preflight", "preflight_too_large"), ("provider", "too_large")):
        recorder = sv.Recorder(refuse(stage))
        with pytest.raises(LLMUnavailable):
            recorder("x")
        assert recorder.failures == [expected]
        said = "refused before sending: " if stage == "preflight" else "provider said: "
        assert recorder.messages == [said + "too big"]           # the project's own refusal is never the provider's


def test_each_prompt_version_has_its_own_record_file():
    from src.summarizer import PROMPT_VERSION
    assert sv.record_file() == f"summary_v2_dev_k5_c{sv.SUMMARY_VERSION}_p{PROMPT_VERSION}.json"


# ---------- source naming: the independent check ----------
#
# A point names its sources only through its citation ids. Prose such as "brief 2", "sources 1 and 3",
# "the first article" or "[2]" means nothing to a reader and is flagged. A number counts as a source position
# only when it is one of the summary's own article ids, so "Article 370" passes. Ordinal words count only
# before brief, article, excerpt or passage, so "its first electric truck" passes.

from src.summarizer import REPAIR_RULES, source_references, validate

FIVE = {1, 2, 3, 4, 5}


@pytest.mark.parametrize("text, found", [
    ("Brief 2 states the deal value.", ["Brief 2"]),
    ("The first article reports a signing on Monday.", ["first article"]),
    ("Sources 1 and 3 disagree on the size.", ["Sources 1 and 3"]),
    ("Briefs 1 and 2 give 3.91 billion euro.", ["Briefs 1 and 2"]),
    ("The deal closes in October [2].", ["[2]"]),
    ("Another brief cites a lower value.", ["Another brief"]),
    ("The latter article gives the date.", ["latter article"]),
    ("Article 4 says the offer rose.", ["Article 4"]),
])
def test_source_positions_in_prose_are_flagged(text, found):
    assert source_references(text, FIVE) == found


@pytest.mark.parametrize("text", [
    "Article 370 applies to the region.",
    "Its first electric truck launches next year.",
    "Shares rose after a brief 2% dip.",
    "Another source close to the deal says talks continue.",
    "Article 9 of the agreement covers exports.",             # 9 is not one of the five article ids
    "Q2 profit rose 12% and the FY27 target stays.",
    "The second quarter shows higher volumes.",
])
def test_legitimate_wording_is_not_flagged(text):
    assert source_references(text, FIVE) == []


def test_the_article_ids_come_from_the_actual_articles_not_a_fixed_range():
    twelve = set(range(1, 13))
    assert source_references("Article 12 gives the date.", twelve) == ["Article 12"]
    assert source_references("Article 12 gives the date.", FIVE) == []


def _two_point_output(text):
    return {"article_briefs": [{"id": 1, "brief": "Tata Motors raised its offer.", "enough_text": True},
                               {"id": 2, "brief": "Registrations fell.", "enough_text": True}],
            "key_points": [{"text": "Tata Motors raised its offer to 14.40 euro per share.", "citations": [1]},
                           {"text": text, "citations": [1, 2]}],
            "industry_context": [], "not_used": []}


def test_structured_citations_are_valid_and_source_wording_is_reported_apart_from_numbers():
    clean = validate(_two_point_output("Registrations fell 22 per cent while the offer rose."), ARTICLES)
    assert clean["sources"] == [] and clean["numbers"] == []
    report = validate(_two_point_output("Briefs 1 and 2 give 14.40 euro and 22 per cent."), ARTICLES)
    assert report["sources"] == [{"section": "key_points", "index": 1, "phrases": ["Briefs 1 and 2"]}]
    assert report["numbers"] == []           # the digits 1 and 2 of the reference are not wrong numbers
    assert report["structural"] == []


def test_a_wrong_number_beside_source_wording_is_still_reported_as_a_number():
    report = validate(_two_point_output("Brief 2 says registrations fell 30 per cent."), ARTICLES)
    assert report["sources"][0]["phrases"] == ["Brief 2"]
    assert report["numbers"] == [{"section": "key_points", "index": 1, "numbers": ["30 %"]}]


def test_every_prompt_carries_the_source_rule_and_the_disagreement_rule():
    for prompt in (RULES, sv.BRIEF_RULES, sv.COMBINE_RULES, REPAIR_RULES):
        assert "Never refer to briefs or articles by number or position" in prompt
    for prompt in (RULES, sv.COMBINE_RULES):
        assert "same company, measure, period and stage" in prompt


# ---------- source wording: correction, repair, then removal of that point only ----------

def _report_for(output):
    return validate(output, ARTICLES)


def test_a_repair_removes_source_wording_and_keeps_the_point():
    output = _two_point_output("Brief 2 says registrations fell 22 per cent this Navratri.")
    fixed = "Registrations fell 22 per cent this Navratri."
    repaired, repairs = repair_points(output, _report_for(output), ARTICLES, lambda p: json.dumps({"text": fixed}))
    assert repaired["key_points"][1] == {"text": fixed, "citations": [1, 2]}
    assert repairs[0]["ok"] is True and repairs[0]["references"] == ["Brief 2"]


def test_a_repair_that_keeps_source_wording_is_refused():
    output = _two_point_output("Brief 2 says registrations fell 22 per cent this Navratri.")
    still = "The second article says registrations fell 22 per cent this Navratri."
    repaired, repairs = repair_points(output, _report_for(output), ARTICLES, lambda p: json.dumps({"text": still}))
    assert repaired["key_points"][1]["text"].startswith("Brief 2") and repairs[0]["ok"] is False


def test_the_repair_prompt_names_the_wording_to_remove():
    seen = []
    output = _two_point_output("Brief 2 says registrations fell 22 per cent this Navratri.")
    repair_points(output, _report_for(output), ARTICLES, lambda p: seen.append(p) or json.dumps({"text": "x"}))
    assert "Brief 2" in seen[0] and REPAIR_RULES in seen[0]


def test_a_point_that_keeps_source_wording_is_removed_with_its_own_reason():
    bad = _two_point_output("Brief 2 says registrations fell 22 per cent this Navratri.")
    answers = iter([json.dumps(bad), json.dumps(bad), json.dumps({"text": bad["key_points"][1]["text"]})])
    result = plain_summarize(ARTICLES, lambda prompt: next(answers))
    assert result["status"] == "partial"
    assert result["removed"] == [{"section": "key_points", "text": bad["key_points"][1]["text"],
                                  "reason": "source_reference"}]
    assert "numerical" not in result["message"]
    reasons = {item["id"]: item["reason"] for item in result["summary"]["not_used"]}
    assert "numerical" not in reasons[2]                  # article 2 lost its only point to a wording problem
    assert [p["text"] for p in result["summary"]["key_points"]] == [bad["key_points"][0]["text"]]


def test_the_correction_attempt_is_told_about_source_wording():
    bad = _two_point_output("Brief 2 says registrations fell 22 per cent this Navratri.")
    good = _two_point_output("Registrations fell 22 per cent this Navratri.")
    prompts, answers = [], iter([json.dumps(bad), json.dumps(good)])
    result = plain_summarize(ARTICLES, lambda prompt: prompts.append(prompt) or next(answers))
    assert result["status"] == "ok" and result["repairs"] == []
    assert "Brief 2" in prompts[1] and "citations" in prompts[1]


def test_emptying_the_key_points_by_removal_gives_no_summary():
    bad = {"article_briefs": [{"id": 1, "brief": "Tata Motors raised its offer.", "enough_text": True},
                              {"id": 2, "brief": "Registrations fell.", "enough_text": True}],
           "key_points": [{"text": "Brief 1 says the offer rose to 14.40 euro per share.", "citations": [1]}],
           "industry_context": [{"text": "Registrations fell 22 per cent this Navratri.", "citations": [2]}],
           "not_used": []}
    answers = iter([json.dumps(bad), json.dumps(bad), json.dumps({"text": bad["key_points"][0]["text"]})])
    result = plain_summarize(ARTICLES, lambda prompt: next(answers))
    assert result["status"] == "unavailable"
    assert result["removed"][0]["reason"] == "source_reference"     # the loss stays on record


def test_emptying_the_industry_context_by_removal_keeps_a_partial_summary():
    bad = {"article_briefs": [{"id": 1, "brief": "Tata Motors raised its offer.", "enough_text": True},
                              {"id": 2, "brief": "Registrations fell.", "enough_text": True}],
           "key_points": [{"text": "Tata Motors raised its offer to 14.40 euro per share.", "citations": [1]}],
           "industry_context": [{"text": "Article 2 says registrations fell 22 per cent.", "citations": [2]}],
           "not_used": []}
    answers = iter([json.dumps(bad), json.dumps(bad), json.dumps({"text": bad["industry_context"][0]["text"]})])
    result = plain_summarize(ARTICLES, lambda prompt: next(answers))
    assert result["status"] == "partial" and result["summary"]["industry_context"] == []


def test_the_brief_first_design_repairs_source_wording_in_a_combined_point():
    combine = {"key_points": [{"text": "Tata Motors raised its offer to 14.40 euro.", "citations": [1]},
                              {"text": "Brief 2 says registrations fell 22 per cent.", "citations": [2]}],
               "industry_context": [], "not_used": []}
    llm = ScriptedLLM(combine=combine)
    fixed = {"text": "Registrations fell 22 per cent."}
    original = llm.__call__

    def answer(prompt):
        return json.dumps(fixed) if prompt.startswith("You correct one sentence") else original(prompt)

    result = sv.summarize_brief_first(ARTICLES, answer)
    assert result["status"] == "ok"
    assert [p["text"] for p in result["summary"]["key_points"]][1] == "Registrations fell 22 per cent."


# ---------- the smoke test: it checks the pipeline and never feeds the comparison ----------

def test_the_smoke_test_never_reads_or_saves_summaries(tmp_path):
    cache = sv.NoCache()
    llm = ScriptedLLM()
    llm.stats = {"per_call": [], "requests": []}
    result = sv._one("brief_first", ARTICLES, llm, cache, 2)
    assert result["status"] == "ok"
    assert cache.load("anything") is None and list(tmp_path.iterdir()) == []


def test_the_smoke_checklist_reports_each_check():
    good = {"status": "ok", "summary": {"key_points": [{"text": "Tata Motors raised its offer.", "citations": [1]}],
                                        "industry_context": [], "article_briefs": [], "not_used": []},
            "bad_answers": [], "usage": {"requests": [{"outcome": "answered"}]}, "repairs": []}
    checks = sv.smoke_checks(good, ARTICLES)
    assert all(checks.values())
    bad = {**good, "summary": {**good["summary"],
                               "key_points": [{"text": "Brief 1 says the offer rose.", "citations": [1]}]}}
    assert sv.smoke_checks(bad, ARTICLES)["no_source_wording"] is False


# ---------- the comparison mode: fresh runs, one record each ----------

def test_a_record_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "EVAL_DIR", tmp_path)
    first = sv.free_name("summary_v2_dev_k5_p3.json")
    assert first == "summary_v2_dev_k5_p3.json"
    (tmp_path / first).write_text("{}", encoding="utf-8")
    second = sv.free_name("summary_v2_dev_k5_p3.json")
    assert second == "summary_v2_dev_k5_p3_run2.json"
    (tmp_path / second).write_text("{}", encoding="utf-8")
    assert sv.free_name("summary_v2_dev_k5_p3.json") == "summary_v2_dev_k5_p3_run3.json"


def test_the_record_keeps_every_call_in_detail():
    llm = StatsLLM(json.dumps({"brief": "Short brief.", "enough_text": True}),
                   json.dumps({"brief": "Short brief.", "enough_text": True}),
                   json.dumps({"key_points": [{"text": "Tata Motors raised its offer to 14.40 euro.", "citations": [1]}],
                               "industry_context": [{"text": "Registrations fell 22 per cent.", "citations": [2]}],
                               "not_used": []}))
    result = sv._one("brief_first", ARTICLES, llm, sv.NoCache(), 2)
    assert result["usage"]["per_call"] == llm.stats["per_call"]


# ---------- the explicit output limit for summaries (configuration version 3) ----------

def test_summaries_use_the_stage3_settings_plus_an_output_limit(monkeypatch):
    from src.model_compare import CONFIGS
    seen = {}
    monkeypatch.setattr("src.llm.make_llm", lambda provider, model=None, generation=None, **kw:
                        seen.update(provider=provider, model=model, generation=generation) or "llm")
    assert sv.summary_llm() == "llm"
    assert seen["generation"] == {**CONFIGS[sv.MODEL_SETTING]["generation"], "max_completion_tokens": 4096}
    assert seen["model"] == CONFIGS[sv.MODEL_SETTING]["model"]


def test_the_stage3_setting_itself_is_unchanged():
    from src.model_compare import CONFIGS
    assert "max_completion_tokens" not in CONFIGS[sv.MODEL_SETTING]["generation"]


def test_the_configuration_version_and_the_settings_are_part_of_the_summary_key():
    assert sv.OUTPUT_LIMIT == 4096
    a = sv.cache_key(ARTICLES, "m", "direct", 2, tag="groq|{}")
    b = sv.cache_key(ARTICLES, "m", "direct", 2, tag='groq|{"max_completion_tokens": 4096}')
    assert a != b


# ---------- the attempt log ----------

def test_every_attempt_is_logged_with_its_problems_and_no_text():
    bad = _output("Tata Motors raised its offer to 14 euro per share.")
    good = _output("Tata Motors raised its offer to 14.40 euro per share.")
    answers = iter([json.dumps(bad), json.dumps(good)])
    result = plain_summarize(ARTICLES, lambda prompt: next(answers))
    log = result["attempt_log"]
    assert [a["attempt"] for a in log] == [1, 2]
    assert log[0]["valid_json"] is True and log[0]["numbers"] == [{"section": "key_points", "index": 0}]
    assert log[1]["numbers"] == [] and log[1]["structural"] == [] and log[1]["sources"] == []
    assert "Tata Motors raised" not in json.dumps(log)


def test_a_broken_first_answer_is_logged_as_not_valid_json():
    good = _output("Tata Motors raised its offer to 14.40 euro per share.")
    answers = iter(["{broken", json.dumps(good)])
    log = plain_summarize(ARTICLES, lambda prompt: next(answers))["attempt_log"]
    assert log[0] == {"attempt": 1, "valid_json": False, "structural": ["malformed_json"], "numbers": [],
                      "sources": []}


def test_the_brief_first_design_logs_its_combine_attempts():
    result = sv.summarize_brief_first(ARTICLES, ScriptedLLM())
    assert result["attempt_log"][0]["attempt"] == 1 and result["attempt_log"][0]["valid_json"] is True


# ---------- step 2: targeted correction (configuration version 4) ----------
#
# A first answer with valid JSON and no structural problem goes straight to targeted repair of its failing
# points; broken JSON or a structural problem still gets the full correction. Each failing point gets up to two
# repair tries; the second try repairs the first try's result and is told its new problems. A point that still
# fails is removed; valid points stay exactly as they are; citations never change.

from src.summarizer import REPAIR_RULES as _RULES_FOR_REPAIR


SIX = [_article(i, f"Tata Motors fact {i}: figure {i}0 crore.") for i in range(1, 7)]


def _first_answer():
    """A valid first answer: two correct points and four points with a number not in their article."""
    points = [{"text": f"Tata Motors fact {i}: figure {i}0 crore.", "citations": [i]} for i in (1, 2)]
    points += [{"text": f"Tata Motors fact {i}: figure {i}5 crore.", "citations": [i]} for i in (3, 4, 5, 6)]
    return {"article_briefs": [{"id": i, "brief": f"Tata Motors fact {i}.", "enough_text": True} for i in range(1, 7)],
            "key_points": points, "industry_context": [], "not_used": []}


class PathLLM:
    """Fake model: the first full prompt gets the first answer; a later full prompt (a full correction) gets an
    answer that also changes a correct point; repair prompts get the exact sentence for their article."""

    def __init__(self, repair_answer=None):
        self.prompts, self.full = [], 0
        self.repair_answer = repair_answer

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if prompt.startswith(_RULES_FOR_REPAIR[:30]):
            if self.repair_answer:
                return json.dumps(self.repair_answer(prompt))
            number = int(prompt.split('<article id="')[1].split('"')[0])
            return json.dumps({"text": f"Tata Motors fact {number}: figure {number}0 crore."})
        self.full += 1
        if self.full == 1:
            return json.dumps(_first_answer())
        changed = _first_answer()
        changed["key_points"][0]["text"] = "Tata Motors fact 1: figure 99 crore."      # a new mistake
        return json.dumps(changed)


def test_the_same_first_answer_old_path_rewrites_everything_new_path_repairs_only_the_failing_points():
    old, new = PathLLM(), PathLLM()
    old_result = plain_summarize(SIX, old, correction="full")
    new_result = plain_summarize(SIX, new, correction="targeted")
    assert old.full == 2 and new.full == 1                      # no full rewrite on the new path
    assert len(new.prompts) == 1 + 4                            # one summary request, four small repairs
    assert new_result["status"] == "ok"
    first = _first_answer()["key_points"]
    assert new_result["summary"]["key_points"][:2] == first[:2] # correct points untouched
    assert all(p["citations"] == first[i]["citations"] for i, p in enumerate(new_result["summary"]["key_points"]))
    assert any(r["index"] == 0 for r in old_result["repairs"])  # the rewrite broke a correct point
    assert not any(r["index"] == 0 for r in new_result["repairs"])


def test_broken_json_still_gets_the_full_correction_on_the_new_path():
    answers = iter(["{broken", json.dumps(_output("Tata Motors raised its offer to 14.40 euro per share."))])
    prompts = []
    result = plain_summarize(ARTICLES, lambda p: prompts.append(p) or next(answers), correction="targeted")
    assert result["status"] == "ok" and len(prompts) == 2
    assert result["attempt_log"][0]["valid_json"] is False


def test_a_structural_problem_still_gets_the_full_correction_on_the_new_path():
    missing = _output("Tata Motors raised its offer to 14.40 euro per share.")
    del missing["not_used"]
    answers = iter([json.dumps(missing), json.dumps(_output("Tata Motors raised its offer to 14.40 euro per share."))])
    result = plain_summarize(ARTICLES, lambda p: next(answers), correction="targeted")
    assert result["status"] == "ok" and result["attempt_log"][0]["structural"] == ["missing_field"]


def test_the_second_repair_try_repairs_the_first_tries_result_and_is_told_its_problems():
    tries = []

    def answer(prompt):
        tries.append(prompt)
        return {"text": "Tata Motors fact 3: figure 77 crore."} if len(tries) == 1 else \
               {"text": "Tata Motors fact 3: figure 30 crore."}

    output = _first_answer()
    report = sv.validate(output, SIX)
    one = {"structural": [], "numbers": [p for p in report["numbers"] if p["index"] == 2], "sources": []}
    repaired, repairs = repair_points(output, one, SIX, PathLLM(answer), tries=2)
    assert repaired["key_points"][2]["text"] == "Tata Motors fact 3: figure 30 crore."
    assert repairs[0]["ok"] is True and repairs[0]["requests"] == 2 and repairs[0]["responses"] == 2
    assert repairs[0]["stopped"] is None
    assert "figure 77 crore" in tries[1] and "77" in tries[1].split("Numbers not found:")[1]


def test_a_point_that_fails_both_tries_is_removed_and_no_third_try_is_made():
    llm = PathLLM(lambda prompt: {"text": "Tata Motors fact 3: figure 88 crore."})
    first = _first_answer()
    first["key_points"] = first["key_points"][:3]
    first["not_used"] = [{"id": i, "reason": "Not needed."} for i in (4, 5, 6)]     # every article accounted for
    answers = iter([json.dumps(first)])

    def model(prompt):
        return next(answers) if not prompt.startswith(_RULES_FOR_REPAIR[:30]) else llm(prompt)

    result = plain_summarize(SIX, model, correction="targeted")
    assert len(llm.prompts) == 2                                # two tries, never a third
    assert [r["reason"] for r in result["removed"]] == ["numbers"]
    assert result["status"] == "partial" and len(result["summary"]["key_points"]) == 2


def test_the_full_path_keeps_one_repair_try_as_before():
    _, repairs = repair_points(_first_answer(), sv.validate(_first_answer(), SIX), SIX,
                               PathLLM(lambda p: {"text": "still 1 wrong 5 number"}))
    assert all(r["requests"] == 1 and r["responses"] == 1 for r in repairs)


def test_a_refused_repair_counts_as_a_request_with_no_answer_and_stops_the_repairs():
    def refuse(prompt):
        error = LLMUnavailable("Rate limit reached ... tokens per day (TPD)")
        error.kind = "daily_limit"
        raise error

    _, repairs = repair_points(_first_answer(), sv.validate(_first_answer(), SIX), SIX, refuse, tries=2)
    assert len(repairs) == 1                                    # no request for the other failing points
    assert repairs[0]["requests"] == 1 and repairs[0]["responses"] == 0
    assert repairs[0]["stopped"] == "daily_limit" and repairs[0]["ok"] is False and repairs[0]["after"] is None


def test_an_answer_that_is_not_json_still_counts_as_a_response():
    _, repairs = repair_points(_first_answer(), sv.validate(_first_answer(), SIX), SIX,
                               lambda prompt: "not json", tries=2)
    assert all(r["requests"] == 2 and r["responses"] == 2 and r["stopped"] is None for r in repairs)


def test_a_point_is_checked_against_its_own_cited_article_only():
    # "about 20 per cent" is in article 2, but the point cites article 1, which says 20.4%
    articles = [_article(1, "Three funds hold about 20.4% of Iveco's shares."),
                _article(2, "Three funds together hold about 20 per cent of Iveco.")]
    output = {"article_briefs": [{"id": 1, "brief": "Three funds hold 20.4% of Iveco.", "enough_text": True},
                                 {"id": 2, "brief": "Three funds hold about 20% of Iveco.", "enough_text": True}],
              "key_points": [{"text": "Three funds hold about 20% of Iveco.", "citations": [1]}],
              "industry_context": [], "not_used": []}
    report = sv.validate(output, articles)
    assert [(p["section"], p["index"], p["numbers"]) for p in report["numbers"]] == [("key_points", 0, ["20%"])]


def test_the_brief_first_design_follows_the_same_policy():
    combine = {"key_points": [{"text": "Tata Motors raised its offer to 14 euro.", "citations": [1]}],
               "industry_context": [{"text": "Registrations fell 22 per cent.", "citations": [2]}], "not_used": []}
    llm = ScriptedLLM(combine=combine)
    original = llm.__call__

    def model(prompt):
        if prompt.startswith(_RULES_FOR_REPAIR[:30]):
            return json.dumps({"text": "Tata Motors raised its offer to 14.40 euro."})
        return original(prompt)

    result = sv.summarize_brief_first(ARTICLES, model, correction="targeted")
    combine_prompts = [p for p in llm.prompts if p.startswith(sv.COMBINE_RULES[:40])]
    assert len(combine_prompts) == 1 and result["status"] == "ok"
    assert result["repairs"][0]["ok"] is True


def test_each_correction_policy_and_number_check_has_its_own_configuration_version_and_files():
    # 3 and 4 are the records made with number check version 1; they keep their numbers and files
    assert sv.config_version("full", "1", "2") == "3" and sv.config_version("targeted", "1", "2") == "4"
    assert sv.config_version("targeted", "2", "2") == "5" and sv.config_version("full", "2", "2") == "6"
    assert sv.config_version("targeted", "2", "3") == "7" and sv.config_version("full", "2", "3") == "8"
    assert sv.config_version("targeted") == "7"                 # the current number check and passage by default
    assert sv.CORRECTION == "targeted" and sv.SUMMARY_VERSION == "7"
    assert sv.cache_key(ARTICLES, "m", "direct", 2, correction="full") != \
        sv.cache_key(ARTICLES, "m", "direct", 2, correction="targeted")
    assert sv.record_file() == f"summary_v2_dev_k5_c7_p{PROMPT_VERSION}.json"
    assert sv.diagnosis_file() == f"summary_v2_diagnosis_c7_p{PROMPT_VERSION}.json"
    assert sv.diagnosis_file("full") == f"summary_v2_diagnosis_c8_p{PROMPT_VERSION}.json"


# ---------- the provider's wait time after a refusal ----------

DAILY = ("RateLimitError: Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` in organization "
         "`org_01abcDEF23` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199100, "
         "Requested 6400. Please try again in 2h3m4.5s.")


def test_the_wait_is_read_from_the_providers_message():
    assert sv.wait_seconds(DAILY) == 2 * 3600 + 3 * 60 + 4.5
    assert sv.wait_seconds("no wait here") is None


def test_the_shown_message_hides_the_organisation():
    shown = sv.safe_message(DAILY)
    assert "org_01abcDEF23" not in shown and "try again in 2h3m4.5s" in shown


def test_the_record_keeps_the_kind_and_the_wait_but_never_the_message(tmp_path):
    def refuse(prompt):
        error = LLMUnavailable(DAILY)
        error.kind = "daily_limit"
        raise error
    refuse.stats = {"per_call": [], "requests": []}
    refuse.model_name, refuse.cache_tag = "fake", ""
    result = sv._one("direct", ARTICLES, refuse, sv.NoCache(), 2)
    messages = sv.take_messages(result)
    assert result["failure_waits"] == [{"kind": "daily_limit", "wait_seconds": 7384.5}]
    assert "org_" not in json.dumps(result) and "Rate limit reached" not in json.dumps(result)
    assert messages and "org_[hidden]" in messages[0] and messages[0].startswith("provider said: ")



def test_the_summary_articles_use_passage_version_3():
    from src import summary_passage as sp
    candidate = {"key": "A1", "title": "Tata Motors completes Iveco settlement", "url": "u",
                 "description": "", "text": "Tata Motors completes the Iveco settlement.\nContacts\nE-Mail: a@b.com"}
    assert sv.passage_articles([candidate], None)[0]["description"] == \
        sp.build(candidate["title"], candidate["text"], "Tata Motors", None)["text"]



def test_the_smoke_record_names_its_configuration_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "EVAL_DIR", tmp_path)
    assert sv.smoke_file() == f"summary_v2_smoke_c{sv.SUMMARY_VERSION}_p{PROMPT_VERSION}.json"
    (tmp_path / sv.smoke_file()).write_text("{}", encoding="utf-8")
    assert sv.free_name(sv.smoke_file()) == f"summary_v2_smoke_c{sv.SUMMARY_VERSION}_p{PROMPT_VERSION}_run2.json"


# ---------- prompt version 4: the industry-context rule ----------

import re  # noqa: E402

from src.summarizer import INDUSTRY_RULE  # noqa: E402


def test_the_industry_rule_is_in_both_designs():
    assert PROMPT_VERSION == "4"
    for prompt in (RULES, sv.COMBINE_RULES):
        assert INDUSTRY_RULE in prompt
    assert INDUSTRY_RULE not in sv.BRIEF_RULES          # a brief covers one article and has no sections


def test_the_industry_rule_needs_wider_evidence_excludes_deal_parties_and_allows_an_empty_section():
    assert "reports on the wider industry, the market or other companies" in INDUSTRY_RULE
    assert "A fact about the company alone" in INDUSTRY_RULE and "never stated as a trend" in INDUSTRY_RULE
    assert "only as a party to a deal, partnership or transaction" in INDUSTRY_RULE
    assert '"industry_context" is an empty list' in INDUSTRY_RULE
    assert "Tata" not in INDUSTRY_RULE and "Iveco" not in INDUSTRY_RULE   # no company from the test data


def test_the_rules_stay_numbered_one_by_one():
    for prompt in (RULES, sv.COMBINE_RULES):
        numbers = [int(n) for n in re.findall(r"^(\d+)\. ", prompt, re.M)]
        assert numbers == list(range(1, len(numbers) + 1)) and len(numbers) >= 8


def test_the_record_files_name_k_the_configuration_and_the_active_prompt_version():
    assert sv.record_file() == f"summary_v2_dev_k5_c{sv.SUMMARY_VERSION}_p4.json"
    assert sv.record_file(10) == f"summary_v2_dev_k10_c{sv.SUMMARY_VERSION}_p4.json"
    assert sv.smoke_file() == f"summary_v2_smoke_c{sv.SUMMARY_VERSION}_p4.json"
    assert sv.diagnosis_file() == f"summary_v2_diagnosis_c{sv.SUMMARY_VERSION}_p4.json"


def test_an_earlier_prompt_version_answer_is_never_reused(monkeypatch):
    articles = [{"id": 1, "key": "k", "title": "t", "description": "d", "content": ""}]
    now = sv.cache_key(articles, "m", "direct", 5)
    for earlier in ("3", "5"):
        monkeypatch.setattr(sv, "PROMPT_VERSION", earlier)
        assert sv.cache_key(articles, "m", "direct", 5) != now


def test_an_empty_industry_section_passes_the_checks():
    from src.summarizer import validate
    output = _output("Tata Motors raised its offer.")
    report = validate(output, [{"id": 1, "title": "t", "description": "Tata Motors raised its offer.", "content": ""},
                               {"id": 2, "title": "t", "description": "Registrations fell.", "content": ""}])
    assert report == {"structural": [], "numbers": [], "sources": []}


# ---------- the rerun of one design (prompt version 4: direct only) ----------

def test_the_run_command_can_run_one_design_and_names_it_in_the_record(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sv, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(sv, "_dev_candidates", lambda: ({"r1": ["c"], "r2": ["c"]}, {}))
    monkeypatch.setattr(sv, "summary_articles", lambda candidates, k, aliases: ARTICLES)
    monkeypatch.setattr(sv, "summary_llm", lambda: type("L", (), {"model_name": "m"})())
    monkeypatch.setattr(sv, "_one", lambda design, articles, llm, cache, k: calls.append((design, type(cache).__name__))
                        or {"status": "ok", "summary": {}, "usage": {"calls": 1}})
    sv._run_command(("direct",))
    assert calls == [("direct", "NoCache"), ("direct", "NoCache")]          # one fresh try per run, no saved answer
    record = json.loads((tmp_path / sv.record_file()).read_text(encoding="utf-8"))
    assert record["designs"] == ["direct"] and record["prompt_version"] == "4"
    assert [r["design"] for r in record["records"]] == ["direct", "direct"]


def test_the_run_command_without_a_design_still_runs_both(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sv, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(sv, "_dev_candidates", lambda: ({"r1": ["c"]}, {}))
    monkeypatch.setattr(sv, "summary_articles", lambda candidates, k, aliases: ARTICLES)
    monkeypatch.setattr(sv, "summary_llm", lambda: type("L", (), {"model_name": "m"})())
    monkeypatch.setattr(sv, "_one", lambda design, articles, llm, cache, k: calls.append(design)
                        or {"status": "ok", "summary": {}, "usage": {"calls": 1}})
    sv._run_command()
    assert calls == ["direct", "brief_first"]
    record = json.loads((tmp_path / sv.record_file()).read_text(encoding="utf-8"))
    assert record["designs"] == ["direct", "brief_first"]


# ---------- prompt version 5: what an industry point must include ----------

from src.summarizer import INDUSTRY_INCLUDE  # noqa: E402


def test_prompt_version_5_keeps_the_version_4_rule_and_adds_what_to_include_in_both_designs():
    from src.summarizer import INDUSTRY_RULES, PROMPT_TEXTS
    assert INDUSTRY_RULES["5"] == INDUSTRY_RULES["4"] + " " + INDUSTRY_INCLUDE
    assert "never stated as a trend" in INDUSTRY_RULES["4"] and "only as a party to a deal" in INDUSTRY_RULES["4"]
    for prompt in (PROMPT_TEXTS["5"], sv.COMBINE_TEXTS["5"]):
        assert INDUSTRY_RULES["5"] in prompt and INDUSTRY_INCLUDE in prompt


def test_the_include_sentence_names_same_sector_companies_and_excludes_other_sectors_in_lists():
    assert "another company operating in that same sector" in INDUSTRY_INCLUDE
    assert "include that fact as an \"industry_context\" point and cite the article" in INDUSTRY_INCLUDE
    assert "a rival's sales, a sector-wide figure, or a supplier's results" in INDUSTRY_INCLUDE
    assert "companies from other sectors" in INDUSTRY_INCLUDE and "stocks in news" in INDUSTRY_INCLUDE
    assert "results roundup" in INDUSTRY_INCLUDE
    for name in ("Tata", "JK Tyre", "Mahindra", "Iveco"):
        assert name not in INDUSTRY_INCLUDE                     # nothing from the development data


# ---------- prompt version 4 selected; version 5 kept ----------

def test_version_4_is_active_in_both_designs_and_holds_no_include_sentence():
    from src.summarizer import INDUSTRY_RULES, PROMPT_TEXTS
    assert RULES == PROMPT_TEXTS["4"] and sv.COMBINE_RULES == sv.COMBINE_TEXTS["4"]
    assert INDUSTRY_RULE == INDUSTRY_RULES["4"]
    for prompt in (RULES, sv.COMBINE_RULES):
        assert INDUSTRY_RULES["4"] in prompt and INDUSTRY_INCLUDE not in prompt


def test_version_4_is_the_version_5_text_without_exactly_the_one_added_sentence():
    from src.summarizer import PROMPT_TEXTS
    for v4, v5 in ((PROMPT_TEXTS["4"], PROMPT_TEXTS["5"]), (sv.COMBINE_TEXTS["4"], sv.COMBINE_TEXTS["5"])):
        assert len(v5) - len(v4) == len(" " + INDUSTRY_INCLUDE)
        assert v5.replace(" " + INDUSTRY_INCLUDE, "") == v4


def test_removing_the_sentence_refuses_a_text_without_it_or_with_it_twice():
    from src.summarizer import without_include
    with pytest.raises(ValueError):
        without_include("no sentence here")
    with pytest.raises(ValueError):
        without_include(("x " + INDUSTRY_INCLUDE) * 2)


def test_the_run_command_takes_k_and_names_it_in_the_record(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(sv, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(sv, "_dev_candidates", lambda: ({"r1": ["c"]}, {}))
    monkeypatch.setattr(sv, "summary_articles", lambda candidates, k, aliases: seen.append(("articles", k)) or ARTICLES)
    monkeypatch.setattr(sv, "summary_llm", lambda: type("L", (), {"model_name": "m"})())
    monkeypatch.setattr(sv, "_one", lambda design, articles, llm, cache, k: seen.append((design, k))
                        or {"status": "ok", "summary": {}, "usage": {"calls": 1}})
    sv._run_command(("direct",), 10)
    assert seen == [("articles", 10), ("direct", 10)]
    record = json.loads((tmp_path / "summary_v2_dev_k10_c7_p4.json").read_text(encoding="utf-8"))
    assert record["k"] == 10 and record["designs"] == ["direct"]
    assert not (tmp_path / "summary_v2_dev_k5_c7_p4.json").exists()


def test_the_run_command_accepts_only_the_tested_k_values():
    assert sv.RUN_KS == ("5", "10")
