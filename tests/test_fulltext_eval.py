"""
Tests for measuring the relevance filter on full text (no network).

The same filter (query rules, verified aliases, junk removal, sale-listing
check) decides each labelled article twice:
    short - on the text NewsAPI gives (title, description, first 200 characters)
    full  - on the title, description and the cleaned full text, when the
            page is fetched; otherwise on the short text
Both are scored against the same labels (guide version 3), so the
difference comes from the text alone.
"""

import pytest

from src import fulltext_eval as fe


def _raw(n, title, description, content="", url=None):
    return {"source": {"id": None, "name": "News"}, "author": "A", "title": title, "description": description,
            "content": content, "url": url or f"https://x.com/{n}", "urlToImage": None,
            "publishedAt": "2026-10-20T10:00:00Z"}


NAVRATRI = _raw(1, "Car registrations plunge 22% this Navratri", "Registrations fell sharply during the festival.")
NAVRATRI_FULL = ("Registrations fell 22%. Maruti sold 74,387 cars. Tata Motors fell 16.4% to 24,843 units. "
                 "Dealers say the festival has one day fewer this year.")
IVECO = _raw(2, "Tata Motors raises Iveco offer", "Tata Motors raised its offer to 14.40 euro.")
CRUDE = _raw(3, "Crude oil rises above $114", "Brent rose on supply fears.")
LISTING = _raw(4, "39k-Mile 2002 Jaguar XJ8 at No Reserve", "This 2002 Jaguar XJ8 sedan now has 39k miles.",
               url="https://bringatrailer.com/listing/2002-jaguar-xj8/")


# ---------- one decision ----------

def test_short_text_misses_what_full_text_finds():
    assert fe.keeps(NAVRATRI, None, "Tata Motors", aliases=None) is False
    assert fe.keeps(NAVRATRI, NAVRATRI_FULL, "Tata Motors", aliases=None) is True


def test_without_full_text_the_decision_uses_the_short_text():
    assert fe.keeps(IVECO, None, "Tata Motors", aliases=None) is True
    assert fe.keeps(CRUDE, None, "Tata Motors", aliases=None) is False


def test_a_sale_listing_is_dropped_even_with_full_text():
    assert fe.keeps(LISTING, "Jaguar Land Rover dealer sold this car. " * 5, "Tata Motors", aliases=None) is False


def test_the_raw_article_is_not_changed():
    article = dict(NAVRATRI)
    fe.keeps(article, NAVRATRI_FULL, "Tata Motors", aliases=None)
    assert article == NAVRATRI


# ---------- scores ----------

ARTICLES = [NAVRATRI, IVECO, CRUDE, LISTING]
LABELS = {"https://x.com/1": "indirect", "https://x.com/2": "direct", "https://x.com/3": "not",
          "https://bringatrailer.com/listing/2002-jaguar-xj8/": "not"}
FULL = {"https://x.com/1": NAVRATRI_FULL}


def test_scores_compare_short_and_full_text_on_the_same_labels():
    result = fe.compare(ARTICLES, LABELS, FULL, "Tata Motors", aliases=None)
    short, full = result["short"], result["full"]
    assert (short["tp"], short["fp"], short["fn"], short["tn"]) == (1, 0, 1, 2)
    assert (full["tp"], full["fp"], full["fn"], full["tn"]) == (2, 0, 0, 2)
    assert full["recall"] == pytest.approx(1.0) and short["recall"] == pytest.approx(0.5)


def test_scores_list_the_mistakes_by_title():
    result = fe.compare(ARTICLES, LABELS, FULL, "Tata Motors", aliases=None)
    assert result["short"]["missed"] == ["Car registrations plunge 22% this Navratri"]
    assert result["full"]["missed"] == [] and result["full"]["wrongly_kept"] == []


def test_strict_scores_count_direct_only():
    result = fe.compare(ARTICLES, LABELS, FULL, "Tata Motors", aliases=None, strict=True)
    # Navratri is indirect: under strict scoring, keeping it is a false positive
    assert (result["full"]["tp"], result["full"]["fp"]) == (1, 1)


def test_scores_say_how_many_decisions_used_full_text():
    result = fe.compare(ARTICLES, LABELS, FULL, "Tata Motors", aliases=None)
    assert result["full_text_used"] == 1 and result["articles"] == 4


def test_scores_stop_when_an_article_has_no_label():
    with pytest.raises(fe.FulltextEvalError):
        fe.compare(ARTICLES + [_raw(9, "Other", "Other")], LABELS, FULL, "Tata Motors", aliases=None)


def test_each_article_is_counted_once_even_if_it_appears_twice():
    result = fe.compare(ARTICLES + [dict(IVECO)], LABELS, FULL, "Tata Motors", aliases=None)
    assert result["articles"] == 4


# ---------- stage 2 and story-level coverage ----------

def test_stage_two_is_scored_with_the_same_definitions_and_counts_fallbacks():
    checks = {"https://x.com/1": {"label": "indirect", "certainty": "sure", "source": "model"},
              "https://x.com/2": {"label": None, "source": "fallback"}}
    result = fe.compare(ARTICLES, LABELS, FULL, "Tata Motors", aliases=None, checks=checks)
    checked = result["full+check"]
    assert (checked["tp"], checked["fp"], checked["fn"], checked["tn"]) == (2, 0, 0, 2)
    assert result["check_counts"] == {"model": 1, "fallback": 1, "not_checked": 0}


def test_stage_two_can_drop_a_wrongly_kept_article():
    nazara = _raw(5, "Nazara founder steps down", "He leaves after 27 years.")
    labels = {**LABELS, "https://x.com/5": "not"}
    full = {**FULL, "https://x.com/5": "The board names an interim chief while it looks for a successor. He earlier worked at Tata Motors, Wipro and GE."}
    checks = {"https://x.com/1": {"label": "indirect", "certainty": "sure", "source": "model"},
              "https://x.com/2": {"label": "direct", "certainty": "sure", "source": "model"},
              "https://x.com/5": {"label": "not", "certainty": "sure", "source": "model"}}
    result = fe.compare(ARTICLES + [nazara], labels, full, "Tata Motors", aliases=None, checks=checks)
    assert result["full"]["fp"] == 1 and result["full+check"]["fp"] == 0
    assert result["full+check"]["wrongly_dropped_by_check"] == []


def test_story_coverage_counts_a_story_once_when_any_copy_is_kept():
    copy = _raw(6, "Car registrations plunge 22% this Navratri festival", "A copy.", url="https://y.com/6")
    labels = {**LABELS, "https://y.com/6": "indirect"}
    result = fe.compare(ARTICLES + [copy], labels, FULL, "Tata Motors", aliases=None)
    assert result["full"]["fn"] == 1                         # article level: the copy itself is missed
    assert result["full"]["stories_relevant"] == 2 and result["full"]["stories_covered"] == 2


# ---------- Stage 1 on each label version, reproducible ----------

def test_the_label_file_is_chosen_from_the_two_known_versions():
    assert fe.label_versions() == ("v3_dev_tata_motors.csv", "v3_dev_tata_motors_checked.csv")


def test_the_stage1_record_states_the_label_file_the_denominators_and_the_definitions():
    result = {"full": {"tp": 45, "fp": 18, "fn": 2, "precision": 45 / 63, "recall": 45 / 47, "f1": 0.818,
                       "stories_relevant": 42, "stories_covered": 42}}
    record = fe.stage1_record(result, "v3_dev_tata_motors_checked.csv")
    assert record["label_file"] == "v3_dev_tata_motors_checked.csv" and record["guide_version"] == "3"
    assert record["relevant_articles"] == 47 and record["relevant_kept"] == 45 and record["relevant_missed"] == 2
    assert record["kept_not_relevant"] == 18 and record["kept"] == 63
    assert record["recall"] == 45 / 47 and record["precision"] == 45 / 63
    assert record["stories_relevant"] == 42 and record["stories_covered"] == 42
    assert "relevant" in record["definitions"]["recall"] and "keeps" in record["definitions"]["precision"]


def test_the_inputs_read_the_named_label_file(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(fe, "load_label_set", lambda paths: seen.extend(paths) or {})
    monkeypatch.setattr(fe, "RAW_DIR", tmp_path)
    fe._inputs("v3_dev_tata_motors_checked.csv")
    assert [p.name for p in seen] == ["v3_dev_tata_motors_checked.csv"]
