"""
Tests for version 2 of the stage 2 relevance check (fake model only, no API calls).

Passage selection, version 2:
    - when the whole article fits in the limit, the model sees all of it
    - otherwise parts are chosen by priority: paragraphs naming the company or
      a subsidiary, then paragraphs with business content (figures, results,
      sales, agreements, rules, prices) even without the company's name, then
      the opening paragraphs, then one neighbour on each side of a mention
    - long paragraphs are trimmed around the important part; the title counts
      towards the limit; the original order is kept, "[…]" marks each gap
    - the parts left out because of the limit are counted

Model answer, version 2: a label, a certainty (sure / unsure) and a reason.
A missing or invalid certainty is a problem in the answer, like a missing label.

Decision, kept separate from the label:
    direct or indirect -> kept
    not + sure         -> dropped
    not + unsure       -> borderline (kept, marked)
    fallback           -> fallback (kept: the stage 1 decision stands)
"""

import json

import pytest

from src import relevance_check as rc
from src.config import DATA_DIR
from src.query import load_aliases
from src.summarizer import LLMUnavailable

COMPANY = "Tata Motors"


# ---------- passages ----------

def test_the_limit_is_3500_characters():
    assert rc.MAX_PASSAGE_CHARS == 3500


def test_a_short_article_is_sent_whole():
    text = "\n".join(["Car registrations fell 22% during Navratri.", "Filler about the weather.",
                      "Tata Motors fell 16.4% to 24,843 units.", "Analysts expect Diwali to help."])
    passage = rc.build_passage("Navratri sales", text, COMPANY, aliases=None)
    assert passage["complete"] is True and "[…]" not in passage["text"]
    assert passage["text"] == "Navratri sales\n" + text
    assert passage["omitted_mentions"] == 0 and passage["omitted_business"] == 0


# A live results blog shaped like the Q2 case: a long opening list, many results of
# other companies, and the company's own result deep inside.
BLOG_OPENING = ("Q2 results today live updates of " + ", ".join(f"Company{n}" for n in range(150))
                + ", Tata Motors and others.")
BLOG_SECOND = "Investors will watch margins and guidance closely this season. " * 11
OTHERS = [f"- October 23, 18:{n:02d} Company{n} Q2 net profit rises {n}% to ₹{100 + n} crore; revenue up {n}%."
          for n in range(60)]
TATA_RESULT = "- October 23, 18:14 Tata Motors Q2 consolidated profit rebounds to ₹1,257 crore; revenue rises 36.3%"
TATA_LATER = "- October 23, 12:28 Tata Motors shares decline nearly 4% ahead of Q2 results."
BLOG = "\n".join([BLOG_OPENING, BLOG_SECOND, *OTHERS[:6], TATA_RESULT, *OTHERS[6:40], TATA_LATER, *OTHERS[40:]])


def test_a_long_blog_keeps_the_company_result_deep_inside():
    passage = rc.build_passage("Q2 Results Today Live", BLOG, COMPANY, aliases=None)
    text = passage["text"]
    assert len(text) <= rc.MAX_PASSAGE_CHARS
    assert "₹1,257 crore" in text and "shares decline nearly 4%" in text
    assert passage["omitted_mentions"] == 0 and passage["complete"] is False
    assert "[…]" in text


def test_mentions_come_before_business_paragraphs_about_other_companies():
    passage = rc.build_passage("Q2 Results Today Live", BLOG, COMPANY, aliases=None, max_chars=1200)
    assert "₹1,257 crore" in passage["text"]
    assert passage["omitted_business"] > 0


def test_the_original_order_is_kept():
    text = rc.build_passage("Q2 Results Today Live", BLOG, COMPANY, aliases=None)["text"]
    assert text.index("Tata Motors and others") < text.index("₹1,257 crore") < text.index("shares decline nearly 4%")


INDUSTRY = "\n".join(
    ["Car registrations fell 22% during the Navratri festival, data from dealers showed."]
    + ["The event venue was decorated with lights and flowers for the evening." for _ in range(30)]
    + ["The government notified new emission rules for commercial vehicles from April, raising costs."]
    + ["Visitors enjoyed music and food stalls late into the night." for _ in range(30)]
    + ["Tata Motors said its truck orders stayed firm through the month."]
)


def test_business_paragraphs_without_the_company_name_are_chosen():
    passage = rc.build_passage("Festive season update", INDUSTRY, COMPANY, aliases=None, max_chars=1500)
    text = passage["text"]
    assert "new emission rules for commercial vehicles" in text
    assert "truck orders stayed firm" in text
    assert text.count("decorated with lights") <= 2                 # filler only as a neighbour, if at all


def test_a_paragraph_longer_than_the_limit_is_trimmed_around_the_mention():
    long_paragraph = ("Background words about the wider economy. " * 60
                      + "Tata Motors signed a definitive agreement with Stellantis. "
                      + "More background about other sectors. " * 60)
    passage = rc.build_passage("Stocks to watch", long_paragraph, COMPANY, aliases=None, max_chars=800)
    assert len(passage["text"]) <= 800
    assert "Tata Motors signed a definitive agreement with Stellantis." in passage["text"]
    assert "…" in passage["text"]


def test_the_title_counts_towards_the_limit():
    title = "A very long headline " * 20
    passage = rc.build_passage(title, BLOG, COMPANY, aliases=None, max_chars=1500)
    assert len(passage["text"]) <= 1500 and passage["text"].startswith(title.strip())


def test_a_mention_in_an_opening_paragraph_appears_once():
    text = "\n".join(["Tata Motors raised its Iveco offer to 14.40 euro.", "Background " * 200,
                      *["Filler paragraph about other things. " * 5 for _ in range(20)]])
    passage = rc.build_passage("Iveco", text, COMPANY, aliases=None, max_chars=1000)
    assert passage["text"].count("Tata Motors raised its Iveco offer") == 1


def test_omitted_mentions_are_counted():
    mentions = [f"Tata Motors item {n}: " + "detail words here " * 30 for n in range(10)]
    passage = rc.build_passage("Many mentions", "\n".join(mentions), COMPANY, aliases=None, max_chars=1500)
    included = sum(1 for n in range(10) if f"Tata Motors item {n}:" in passage["text"])
    assert 0 < included < 10
    assert passage["omitted_mentions"] == 10 - included


def test_select_passage_returns_the_text_only():
    assert rc.select_passage("Q2 Results Today Live", BLOG, COMPANY, aliases=None) == \
        rc.build_passage("Q2 Results Today Live", BLOG, COMPANY, aliases=None)["text"]


# ---------- prompt version 2 ----------

ITEMS = [{"id": "A1", "title": "Navratri sales", "passage": "Tata Motors fell 16.4%."},
         {"id": "A2", "title": "Nazara founder steps down", "passage": "He earlier worked at Tata Motors."}]


@pytest.mark.parametrize("rule", [
    "sector",                       # sector-wide share moves
    "market report",                # general market reports and round-ups
    "round-up",                     # company news inside a market round-up
    "policy",                       # industry and manufacturing policy
    "past",                         # historical references are passing mentions
    "results calendar",             # calendars versus actual results
    "subsidiar",                    # subsidiaries count as the company
    "Jaguar Land Rover",
    "quote",                        # quotes about the company
])
def test_prompt_holds_the_version_3_rules(rule):
    scope = rc.company_scope(COMPANY, load_aliases(DATA_DIR / "company_aliases.json"))
    assert rule.lower() in rc.build_prompt(ITEMS, COMPANY, scope=scope).lower()


def test_company_scope_comes_from_the_alias_file_not_from_the_prompt_rules():
    assert "Jaguar Land Rover" not in rc.RULES                      # the rules work for any company
    scope = rc.company_scope(COMPANY, load_aliases(DATA_DIR / "company_aliases.json"))
    assert "Tata Motors Passenger Vehicles Limited" in scope and "Trucks, buses" in scope
    assert "subsidiaries" in rc.company_scope("Reliance Industries", None)


def test_a_changed_company_scope_is_asked_again(tmp_path):
    cache = rc.CheckCache(tmp_path)
    rc.classify(ITEMS, FakeLLM(_answer(("A1", "indirect", "sure", "s"), ("A2", "not", "sure", "f"))), COMPANY,
                cache=cache, scope="scope one")
    llm = FakeLLM(_answer(("A1", "indirect", "sure", "s"), ("A2", "not", "sure", "f")))
    rc.classify(ITEMS, llm, COMPANY, cache=cache, scope="scope two")
    assert len(llm.prompts) == 1


def test_prompt_asks_for_a_certainty_and_never_tells_the_model_to_prefer_a_label_when_unsure():
    prompt = rc.build_prompt(ITEMS, COMPANY)
    assert '"certainty": "sure|unsure"' in prompt
    assert "when unsure" not in prompt.lower() and "if unsure" not in prompt.lower()


# ---------- reading the answer ----------

def _answer(*results):
    return json.dumps({"results": [{"id": i, "label": l, "certainty": c, "reason": r} for i, l, c, r in results]})


def test_certainty_is_read():
    parsed, problems = rc.parse_answer(_answer(("A1", "indirect", "Unsure", "sales"), ("A2", "not", "sure", "job")),
                                       ["A1", "A2"])
    assert problems == []
    assert parsed["A1"] == {"label": "indirect", "certainty": "unsure", "reason": "sales"}
    assert parsed["A2"]["certainty"] == "sure"


@pytest.mark.parametrize("certainty", [None, "", "maybe", "high"])
def test_a_missing_or_invalid_certainty_is_a_problem(certainty):
    item = {"id": "A1", "label": "not", "reason": "x"}
    if certainty is not None:
        item["certainty"] = certainty
    parsed, problems = rc.parse_answer(json.dumps({"results": [item]}), ["A1"])
    assert parsed == {} and [p["kind"] for p in problems] == ["invalid_certainty"]


class FakeLLM:
    model_name = "fake-model"

    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_an_invalid_certainty_is_retried_once_then_falls_back():
    bad = json.dumps({"results": [{"id": "A1", "label": "not", "reason": "x"},
                                  {"id": "A2", "label": "not", "certainty": "sure", "reason": "y"}]})
    bad_again = json.dumps({"results": [{"id": "A1", "label": "not", "certainty": "very", "reason": "x"}]})
    llm = FakeLLM(bad, bad_again)
    result = rc.classify(ITEMS, llm, COMPANY)
    assert len(llm.prompts) == 2 and "invalid_certainty" in llm.prompts[1]
    assert result["A2"] == {"label": "not", "certainty": "sure", "reason": "y", "source": "model"}
    assert result["A1"]["source"] == "fallback" and result["A1"]["certainty"] is None


def test_a_fallback_has_no_label_and_no_certainty():
    result = rc.classify(ITEMS, FakeLLM(LLMUnavailable("quota")), COMPANY)
    assert result["A1"] == {"label": None, "certainty": None, "reason": "", "source": "fallback",
                            "error": "LLMUnavailable", "kind": "unknown"}


def test_the_same_article_id_twice_is_an_error():
    with pytest.raises(ValueError):
        rc.classify([ITEMS[0], dict(ITEMS[0])], FakeLLM(), COMPANY)


# ---------- decision ----------

@pytest.mark.parametrize("label, certainty, decision", [
    ("direct", "sure", "kept"), ("direct", "unsure", "kept"),
    ("indirect", "sure", "kept"), ("indirect", "unsure", "kept"),
    ("not", "sure", "dropped"), ("not", "unsure", "borderline"),
])
def test_decision_from_label_and_certainty(label, certainty, decision):
    result = {"label": label, "certainty": certainty, "reason": "r", "source": "model"}
    assert rc.decide(result) == decision
    assert rc.keep_after_check(result) is (decision != "dropped")


def test_a_fallback_keeps_the_stage_1_decision():
    result = {"label": None, "certainty": None, "reason": "", "source": "fallback", "error": "LLMUnavailable", "kind": "unknown"}
    assert rc.decide(result) == "fallback" and rc.keep_after_check(result) is True


def test_a_model_answer_without_certainty_is_never_treated_as_sure():
    with pytest.raises(ValueError):
        rc.decide({"label": "not", "reason": "r", "source": "model"})


# ---------- progress and saved answers ----------

def test_progress_is_reported_after_each_batch():
    items = [{"id": f"A{n}", "title": "t", "passage": "p"} for n in range(5)]
    llm = FakeLLM(_answer(*[(f"A{n}", "not", "sure", "r") for n in range(2)]),
                  _answer(*[(f"A{n}", "not", "sure", "r") for n in range(2, 4)]),
                  _answer(("A4", "not", "sure", "r")))
    calls = []
    rc.classify(items, llm, COMPANY, batch_size=2, progress=lambda *args: calls.append(args))
    assert [(done, total) for done, total, _ in calls] == [(1, 3), (2, 3), (3, 3)]
    assert all(seconds >= 0 for _, _, seconds in calls)


def test_answers_from_another_prompt_version_are_not_reused(tmp_path, monkeypatch):
    cache = rc.CheckCache(tmp_path)
    monkeypatch.setattr(rc, "PROMPT_VERSION", "1")
    rc.classify(ITEMS, FakeLLM(_answer(("A1", "indirect", "sure", "s"), ("A2", "not", "sure", "f"))), COMPANY,
                cache=cache)
    monkeypatch.setattr(rc, "PROMPT_VERSION", "2")
    llm = FakeLLM(_answer(("A1", "indirect", "sure", "s"), ("A2", "not", "sure", "f")))
    rc.classify(ITEMS, llm, COMPANY, cache=cache)
    assert len(llm.prompts) == 1


# ---------- the summary passage (version 2): a quote that continues into the next paragraph ----------

def _long_article():
    filler = [f"Maruti Suzuki retail sales rose {n} per cent this festive season, with demand from rural markets "
              f"and strong bookings across its range of cars and SUVs, the company said in a statement today. "
              f"Dealers reported long waiting periods for its most popular models in many cities this month."
              for n in range(10, 22)]
    speaker = ("Similarly, Amit Kamat, Chief Commercial Officer, Tata Motors Passenger Vehicles, said that there "
               "was a robust demand across its range of cars and SUVs.")
    quote = ("“The response during Navratri has been particularly encouraging, with over 50,000 vehicles "
             "delivered during this festive period,” Kamat added.")
    return "\n".join(filler[:6] + [speaker, quote] + filler[6:])


@pytest.mark.parametrize("paragraph, following, expected", [
    ("Amit Kamat of Tata Motors said that demand was robust.", "“Over 50,000 vehicles were delivered,” he added.", True),
    ("Tata Motors gave this statement:", "\"We expect growth to continue.\"", True),
    ("Tata Motors launched a new SUV on Monday.", "“Sales are strong,” said a Maruti official.", False),   # no reporting verb
    ("Amit Kamat of Tata Motors said that demand was robust.", "Maruti Suzuki sales rose 5 per cent.", False),  # no quote
    ("Amit Kamat of Tata Motors said that demand was robust.", "", False),
])
def test_a_paragraph_continues_a_quote_only_after_a_reporting_verb_and_with_an_opening_quote(paragraph, following,
                                                                                             expected):
    assert rc.continues_quote(paragraph, following) is expected


def test_the_summary_passage_keeps_the_continued_quote_within_the_hard_cap():
    text = _long_article()
    passage = rc.build_summary_passage("Car sales decline this Navratri", text, "Tata Motors", aliases=None)
    assert "50,000" in passage["text"] and len(passage["text"]) <= rc.SUMMARY_MAX_CHARS == 4200
    assert passage["added"] == 1 and passage["not_fitted"] == 0


def test_the_summary_passage_keeps_every_line_of_the_relevance_passage():
    text = _long_article()
    old = rc.build_passage("Car sales decline this Navratri", text, "Tata Motors", aliases=None)["text"]
    new = rc.build_summary_passage("Car sales decline this Navratri", text, "Tata Motors", aliases=None)["text"]
    text_lines = [line for line in old.split("\n") if line != rc.GAP]          # gap marks are not article text
    assert [line for line in new.split("\n") if line in text_lines] == text_lines


def test_the_added_paragraph_is_whole_and_never_cut():
    text = _long_article()
    quote = [p for p in text.split("\n") if p.startswith("“")][0]
    new = rc.build_summary_passage("Car sales decline this Navratri", text, "Tata Motors", aliases=None)["text"]
    assert quote in new.split("\n")


def test_a_continuation_that_does_not_fit_is_recorded_not_cut():
    long_quote = "“" + " ".join(["The festive response is strong across our range and our dealers are busy."] * 14) + "”"
    text = _long_article().replace([p for p in _long_article().split("\n") if p.startswith("“")][0], long_quote)
    old = rc.build_passage("Car sales decline", text, "Tata Motors", aliases=None)["text"]
    new = rc.build_summary_passage("Car sales decline", text, "Tata Motors", aliases=None)
    assert new["text"] == old and new["added"] == 0 and new["not_fitted"] == 1


def test_the_relevance_passage_is_unchanged():
    text = _long_article()
    old = rc.build_passage("Car sales decline this Navratri", text, "Tata Motors", aliases=None)
    assert "50,000" not in old["text"]                                   # the frozen version still behaves as before
    assert rc.SUMMARY_PASSAGE_VERSION == "2"


def test_a_next_paragraph_that_is_not_a_quote_is_not_added():
    text = _long_article().replace("“The response during Navratri", "The response during Navratri")
    old = rc.build_passage("Car sales decline", text, "Tata Motors", aliases=None)
    new = rc.build_summary_passage("Car sales decline", text, "Tata Motors", aliases=None)
    assert new["text"] == old["text"] and new["added"] == 0
