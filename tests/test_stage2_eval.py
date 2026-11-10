"""
Tests for scoring stage 2, version 2 (no network, no model calls).

Acceptance thresholds, fixed before the development re-run (broad labels):
    recall_kept - relevant articles kept after stage 2 / relevant articles stage 1 keeps   >= 43/45
    stories     - relevant stories covered after stage 2 / all relevant stories             >= 41/43
    fp_removed  - stage 1 false positives dropped by stage 2 / all stage 1 false positives  >= 2/3
The required count is the denominator times the threshold, rounded up to a
whole article or story. With fewer than 20 articles, 20 stories or 8 false
positives, a check is "too_few" (too few to judge), never "pass".
Borderline articles (not + unsure) count as kept.

The 7 relevant articles that version 1 dropped and the 18 false positives
stage 1 kept are tracked by article id in every run.
"""

from fractions import Fraction

import pytest

from src import fulltext_eval as fe
from src.fulltext import article_id


# ---------- thresholds ----------

def test_thresholds_are_exact_fractions_with_minimum_sample_sizes():
    assert fe.THRESHOLDS == {"recall_kept": (Fraction(43, 45), 20),
                             "stories": (Fraction(41, 43), 20),
                             "fp_removed": (Fraction(2, 3), 8)}


@pytest.mark.parametrize("total, threshold, required", [
    (45, Fraction(43, 45), 43),
    (43, Fraction(41, 43), 41),
    (18, Fraction(2, 3), 12),
    (30, Fraction(43, 45), 29),         # 28.67 -> 29
    (9, Fraction(2, 3), 6),             # exactly 6, not rounded up to 7
])
def test_required_count_rounds_up_to_a_whole_article(total, threshold, required):
    assert fe.required_count(total, threshold) == required


def test_judge_passes_fails_or_says_too_few():
    assert fe.judge(43, 45, Fraction(43, 45), 20) == {"status": "pass", "count": 43, "total": 45, "required": 43}
    assert fe.judge(42, 45, Fraction(43, 45), 20)["status"] == "fail"
    assert fe.judge(19, 19, Fraction(43, 45), 20)["status"] == "too_few"
    assert fe.judge(7, 7, Fraction(2, 3), 8)["status"] == "too_few"


# ---------- scoring with certainty and borderline articles ----------

def _raw(n, title, description, url=None):
    return {"source": {"id": None, "name": "News"}, "author": "A", "title": title, "description": description,
            "content": "", "url": url or f"https://x.com/{n}", "urlToImage": None,
            "publishedAt": "2026-10-20T10:00:00Z"}


IVECO = _raw(1, "Tata Motors raises Iveco offer", "Tata Motors raised its offer to 14.40 euro.")
NAZARA = _raw(2, "Nazara founder steps down", "He leaves after 27 years at the company he started.")
NAZARA_FULL = ("Nazara founder Nitish Mittersain steps down after 27 years at the company he started.\n"
               "He earlier worked at Tata Motors, Wipro and GE.")
SENSEX = _raw(3, "Sensex falls 700 points on crude worries", "Markets fell for a third day.")
SENSEX_FULL = "Markets fell for a third day.\nTata Motors PV fell 2.42 per cent to 279.90 among the losers."
ARTICLES = [IVECO, NAZARA, SENSEX]
LABELS = {IVECO["url"]: "direct", NAZARA["url"]: "not", SENSEX["url"]: "not"}
FULL = {NAZARA["url"]: NAZARA_FULL, SENSEX["url"]: SENSEX_FULL}


def _model(label, certainty, reason="r"):
    return {"label": label, "certainty": certainty, "reason": reason, "source": "model"}


CHECKS = {IVECO["url"]: _model("direct", "sure"),
          NAZARA["url"]: _model("not", "sure", "former job"),
          SENSEX["url"]: _model("not", "unsure", "bare share price")}


def test_a_borderline_article_counts_as_kept():
    result = fe.compare(ARTICLES, LABELS, FULL, "Tata Motors", aliases=None, checks=CHECKS)
    checked = result["full+check"]
    assert (checked["tp"], checked["fp"]) == (1, 1)                  # the Sensex report came back as borderline
    assert checked["borderline"] == ["Sensex falls 700 points on crude worries"]
    assert result["decision_counts"] == {"kept": 1, "dropped": 1, "borderline": 1, "fallback": 0}


def test_acceptance_uses_the_three_checks_and_reports_small_samples():
    result = fe.compare(ARTICLES, LABELS, FULL, "Tata Motors", aliases=None, checks=CHECKS)
    acceptance = result["acceptance"]
    assert set(acceptance) == {"recall_kept", "stories", "fp_removed"}
    assert acceptance["recall_kept"] == {"status": "too_few", "count": 1, "total": 1, "required": 1}
    assert acceptance["fp_removed"] == {"status": "too_few", "count": 1, "total": 2, "required": 2}


def test_strict_scores_have_no_acceptance_checks():
    result = fe.compare(ARTICLES, LABELS, FULL, "Tata Motors", aliases=None, checks=CHECKS, strict=True)
    assert "acceptance" not in result


def test_kept_article_ids_are_listed_for_each_text():
    result = fe.compare(ARTICLES, LABELS, FULL, "Tata Motors", aliases=None, checks=CHECKS)
    assert result["full"]["kept_ids"] == {article_id(a["url"]) for a in ARTICLES}
    assert result["full+check"]["kept_ids"] == {article_id(IVECO["url"]), article_id(SENSEX["url"])}


def test_copies_with_different_model_labels_are_reported():
    copy = _raw(4, "Tata Motors raises Iveco offer", "A copy of the same story.", url="https://y.com/4")
    labels = {**LABELS, copy["url"]: "direct"}
    checks = {**CHECKS, copy["url"]: _model("not", "sure")}
    result = fe.compare(ARTICLES + [copy], labels, FULL, "Tata Motors", aliases=None, checks=checks)
    assert result["copy_disagreements"] == [["Tata Motors raises Iveco offer", "Tata Motors raises Iveco offer"]]


def test_records_hold_the_raw_answer_and_the_final_decision():
    checks = {**CHECKS, NAZARA["url"]: {**_model("not", "sure", "former job"),
                                        "omitted_mentions": 0, "omitted_business": 2}}
    result = fe.compare(ARTICLES, LABELS, FULL, "Tata Motors", aliases=None, checks=checks)
    records = {r["article_id"]: r for r in result["records"]}
    nazara = records[article_id(NAZARA["url"])]
    assert nazara == {"article_id": article_id(NAZARA["url"]), "title": "Nazara founder steps down", "label": "not",
                      "text_source": "full", "fallback_reasons": [], "model_label": "not", "certainty": "sure",
                      "reason": "former job", "decision": "dropped", "omitted_mentions": 0, "omitted_business": 2}
    assert records[article_id(IVECO["url"])]["text_source"] == "short"
    assert len(records) == 3                                          # every article stage 1 keeps


def test_the_same_article_under_two_link_forms_has_one_id_and_one_record():
    variant = dict(IVECO, url="https://www.x.com/1/?utm_source=feed")
    result = fe.compare(ARTICLES + [variant], LABELS, FULL, "Tata Motors", aliases=None, checks=CHECKS)
    assert result["articles"] == 3
    assert [r["article_id"] for r in result["records"]].count(article_id(IVECO["url"])) == 1


# ---------- tracked articles from the version 1 run ----------

def test_tracked_articles_are_the_7_misses_and_18_false_positives_of_version_1():
    tracked = fe.load_tracked()
    assert len(set(tracked["false_negatives"])) == 7 == len(tracked["false_negatives"])
    assert len(set(tracked["false_positives"])) == 18 == len(tracked["false_positives"])
    assert not set(tracked["false_negatives"]) & set(tracked["false_positives"])
    assert "A79f20394bc" in tracked["false_negatives"]               # the Q2 live blog


def test_tracked_status_shows_kept_or_dropped():
    tracked = {"false_negatives": ["A1"], "false_positives": ["A2"]}
    status = fe.tracked_status(tracked, kept_ids={"A1"}, known_ids={"A1", "A2"})
    assert status == {"false_negatives": {"A1": "kept"}, "false_positives": {"A2": "dropped"}}


def test_a_tracked_article_missing_from_the_run_is_an_error():
    tracked = {"false_negatives": ["A1"], "false_positives": ["A9"]}
    with pytest.raises(fe.FulltextEvalError):
        fe.tracked_status(tracked, kept_ids=set(), known_ids={"A1"})
