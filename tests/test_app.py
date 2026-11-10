"""
Tests for app.py with Streamlit's AppTest: the whole page runs with the fake pipeline of tests/test_app_flow.py,
so no NewsAPI or model call is made. The order of drawing during a search is tested on src/app_flow.py itself
(AppTest shows only the end of a run).
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src import app_flow as af  # noqa: E402
from tests.test_app_flow import FakePipeline  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(af, "default_pipeline", FakePipeline)
    monkeypatch.setattr(af, "summary_is_frozen", lambda: False)
    monkeypatch.setattr(af, "summary_design", lambda: "direct")
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def _text(at):
    return " ".join(str(e.value) for e in list(at.markdown) + list(at.caption) + list(at.info) + list(at.warning)
                    + list(at.error) + list(at.success) + list(at.subheader))


def _search(at, company="Tata Motors"):
    at.text_input(key="company").input(company)
    at.button[0].click().run()          # the form's Get News button
    return at


def test_the_page_has_the_query_input_the_get_news_button_and_asks_for_a_company(app):
    assert not app.exception
    assert app.text_input(key="company").label == "Company"
    assert any(b.label == "Get News" for b in app.button)
    assert "one month" in app.select_slider(key="days").help
    assert "Enter a company and press" in _text(app)


def test_get_news_shows_the_final_default_list_with_dates_badges_and_the_summary_button(app):
    at = _search(app)
    assert not at.exception
    text = _text(at)
    assert "Story k2" in text and "Story k1" in text and "Story k3" in text
    assert "Articles selected for the summary: the top 2 of the tested default ranking" in text
    assert "Sources used in this summary" not in text
    assert [m.label for m in at.metric] == ["Articles found", "Kept after cleaning", "For the summary",
                                           "Full text", "Short text only"]
    assert [m.value for m in at.metric] == ["3", "3", "2", "0", "3"]
    assert any("More articles (1)" in e.label for e in at.expander)
    assert "Short text only: the page could not be loaded" in text and "Date unavailable" in text
    assert at.button(key="summarize").label == "Summarize the top 2 articles"
    assert "the tested design is not frozen yet" in text
    assert af.SOURCE_NOTE in text


def test_the_summary_links_its_citations_to_the_cards(app):
    at = _search(app)
    at.button(key="summarize").click().run()
    text = _text(at)
    assert "Summary ready · Automated checks passed" in text
    assert "relies on short article descriptions" in text
    assert "Sources used in this summary: the top 2 of the tested default ranking" in text
    assert "[[1]](#card-k2)" in text and "[[2]](#card-k1)" in text
    assert "Key findings" in text and "Industry context" in text and "About the company" not in text


def test_after_the_experimental_order_the_summary_belongs_to_the_earlier_selection(app):
    at = _search(app)
    at.button(key="summarize").click().run()
    assert at.button(key="experimental").disabled            # needs the review first
    at.button(key="review").click().run()
    assert "Model: not" in _text(at)
    at.button(key="experimental").click().run()
    text = _text(at)
    assert "not the tested default" in text and "belongs to the earlier selection" in text
    assert "(#card-" not in text                             # citations no longer linked
    at.button(key="default").click().run()
    assert "belongs to the earlier selection" not in _text(at)


def test_a_failed_search_shows_its_error_and_no_list(monkeypatch):
    monkeypatch.setattr(af, "default_pipeline", lambda: FakePipeline(found_status="failed"))
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at = _search(at)
    assert "NewsAPI did not answer." in _text(at) and "Top 2 articles" not in _text(at)


def test_the_about_tab_marks_each_stage_by_its_saved_record(app):
    """The marks follow the project's records: a plain list item when a stage's record exists, a pending mark when
    it does not (the pending case on missing records is tested in test_app_flow.py). No success mark, because some
    results report a failed test."""
    from src.config import EVAL_DIR
    text = _text(app)
    assert "Tested results" in text and "✅" not in text
    for stage in af.about_status(EVAL_DIR):
        mark = "-" if stage["status"] == "done" else "⏳"
        assert f"{mark} **{stage['title']}:**" in text



class DollarPipeline(FakePipeline):
    def summarize(self, result, company):
        outcome = super().summarize(result, company)
        outcome["summary"]["key_points"][0]["text"] = "The deal is valued at $4.4 billion, up from $3.8 billion."
        return outcome


def test_dollar_amounts_are_shown_as_written_not_as_a_formula(monkeypatch):
    monkeypatch.setattr(af, "default_pipeline", DollarPipeline)
    monkeypatch.setattr(af, "summary_is_frozen", lambda: False)
    monkeypatch.setattr(af, "summary_design", lambda: "direct")
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at = _search(at)
    at.button(key="summarize").click().run()
    assert "valued at \\$4.4 billion, up from \\$3.8 billion" in _text(at)


def test_the_model_buttons_say_they_use_the_free_allowance(app):
    at = _search(app)
    assert "free" in at.button(key="summarize").help and "free daily allowance" in at.button(key="review").help


class ManyPipeline(FakePipeline):
    """2 selected and 25 more articles."""

    def final(self, found, enriched, company):
        from tests.test_app_flow import _card
        cards = [_card(f"k{i}") for i in range(1, 28)]
        return {"company": company, "k": 2, "ordering": "O1+S1", "ranked": cards, "selected": cards[:2],
                "additional": cards[2:], "state": "final", "label": "Default"}


def test_the_more_articles_are_shown_ten_at_a_time(monkeypatch):
    monkeypatch.setattr(af, "default_pipeline", ManyPipeline)
    monkeypatch.setattr(af, "summary_is_frozen", lambda: False)
    monkeypatch.setattr(af, "summary_design", lambda: "direct")
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at = _search(at)
    pager = at.radio[0]
    assert list(pager.options) == ["3–12", "13–22", "23–27"]
    assert "Story k12" in _text(at) and "Story k13" not in _text(at)
    pager.set_value("13–22").run()
    text = _text(at)
    assert "Story k13" in text and "Story k22" in text and "Story k12" not in text and "Story k23" not in text


class NoIndustryPipeline(FakePipeline):
    def summarize(self, result, company):
        outcome = super().summarize(result, company)
        outcome["summary"]["industry_context"] = []
        return outcome


def test_an_empty_industry_section_keeps_its_heading_and_says_why(monkeypatch):
    monkeypatch.setattr(af, "default_pipeline", NoIndustryPipeline)
    monkeypatch.setattr(af, "summary_is_frozen", lambda: False)
    monkeypatch.setattr(af, "summary_design", lambda: "direct")
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at = _search(at)
    at.button(key="summarize").click().run()
    text = _text(at)
    assert "Industry context" in text and af.NO_INDUSTRY in text


def test_a_section_with_points_shows_no_empty_note(app):
    at = _search(app)
    at.button(key="summarize").click().run()
    assert af.NO_INDUSTRY not in _text(at)


# ---------- the ranking rule the screen describes is the one the pipeline runs ----------

FIVE_MENTION_RULE = "together with articles that name it at least five times in their text"
OLD_RULE = "come first, then in the description"


def test_the_about_tab_describes_the_o1m5_rule(app):
    text = _text(app)
    assert "How it works" in text and FIVE_MENTION_RULE in text and OLD_RULE not in text


def test_the_list_caption_describes_the_o1m5_rule(app):
    at = _search(app)
    text = _text(at)
    assert FIVE_MENTION_RULE in text and OLD_RULE not in text


def test_the_described_rule_matches_the_pipeline_default():
    from src import research as rs
    from src.ranking_variants import SETTINGS
    assert rs.DEFAULT_ORDER == "O1M5" and SETTINGS[rs.DEFAULT_ORDER] == ("mentions", 5)
    assert FIVE_MENTION_RULE in Path(APP).read_text(encoding="utf-8")
