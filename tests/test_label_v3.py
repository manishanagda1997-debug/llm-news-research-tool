"""
Tests for the version 2 -> version 3 label comparison (labels on full text).

compare  - joins the two label sets by article, reports agreement, Cohen's
           kappa and the moves (for example not -> indirect), and lists the
           changed articles for a written reason
finalize - one version 3 label file for every labelled development article:
              full text           the 71 articles labelled on full text
              short text          articles without full text keep their
                                  version 2 label, with the reason written
                                  (page blocked by robots.txt, or a sale
                                  listing that is never fetched)
           every changed label needs a reason from the guide
"""

import pytest

from src import label_v3 as lv

V2 = {"https://x.com/1": "direct", "https://x.com/2": "not", "https://x.com/3": "indirect",
      "https://x.com/4": "not", "https://x.com/5": "not", "https://x.com/6": "indirect"}
V3 = {"https://x.com/1": "direct", "https://x.com/2": "indirect", "https://x.com/3": "indirect",
      "https://x.com/4": "not"}
TITLES = {url: f"Title {url[-1]}" for url in V2}


# ---------- compare ----------

def test_compare_counts_agreement_and_moves_on_the_articles_with_full_text():
    result = lv.compare_versions(V2, V3)
    assert result["articles"] == 4
    assert result["agreement"] == pytest.approx(0.75)
    assert result["moves"] == {("not", "indirect"): 1}
    assert result["changes"] == [("https://x.com/2", "not", "indirect")]
    assert -1 <= result["kappa"] <= 1


def test_compare_stops_when_a_full_text_article_has_no_version_2_label():
    with pytest.raises(lv.LabelV3Error):
        lv.compare_versions(V2, {**V3, "https://x.com/9": "not"})


def test_compare_matches_web_addresses_after_tidying():
    v3 = {"https://www.x.com/1/?utm_source=feed": "direct"}
    assert lv.compare_versions(V2, v3)["articles"] == 1


def test_difference_rows_list_only_changed_articles_with_an_empty_reason():
    rows = lv.difference_rows(V2, V3, TITLES)
    assert rows == [{"article_id": lv.article_id("https://x.com/2"), "title": "Title 2", "url": "https://x.com/2",
                     "label_v2": "not", "label_v3": "indirect", "reason": ""}]


# ---------- finalize ----------

V3_ROWS = [{"url": url, "label": label, "unsure": "yes" if url.endswith("3") else "", "note": f"note {url[-1]}"}
           for url, label in V3.items()]
REASONS = {"https://x.com/2": "Full text gives Tata Motors' sales figures (condition 4)."}


def test_final_file_covers_every_labelled_article_with_its_basis():
    rows = lv.final_rows(V2, V3_ROWS, TITLES, blocked={"https://x.com/5"}, reasons=REASONS,
                         listings={"https://x.com/6"})
    by_url = {row["url"]: row for row in rows}
    assert set(by_url) == set(V2)
    assert by_url["https://x.com/1"]["basis"] == "full text"
    assert by_url["https://x.com/5"]["basis"] == "short text (page blocked by robots.txt)"
    assert by_url["https://x.com/6"]["basis"] == "short text (sale listing, not fetched)"
    assert by_url["https://x.com/5"]["label"] == "not" and by_url["https://x.com/6"]["label"] == "indirect"


def test_final_file_records_version_2_label_change_and_reason():
    rows = {row["url"]: row for row in lv.final_rows(V2, V3_ROWS, TITLES, blocked={"https://x.com/5"},
                                                         reasons=REASONS, listings={"https://x.com/6"})}
    changed = rows["https://x.com/2"]
    assert (changed["label_v2"], changed["label"], changed["changed"]) == ("not", "indirect", "yes")
    assert changed["reason"] == REASONS["https://x.com/2"]
    assert rows["https://x.com/1"]["changed"] == "" and rows["https://x.com/1"]["reason"] == ""
    assert rows["https://x.com/3"]["unsure"] == "yes" and rows["https://x.com/3"]["note"] == "note 3"
    assert all(row["guide_version"] == "3" for row in rows.values())


def test_final_file_columns():
    rows = lv.final_rows(V2, V3_ROWS, TITLES, blocked={"https://x.com/5"}, reasons=REASONS,
                         listings={"https://x.com/6"})
    assert list(rows[0].keys()) == lv.FINAL_COLUMNS


def test_every_changed_label_needs_a_reason():
    with pytest.raises(lv.LabelV3Error):
        lv.final_rows(V2, V3_ROWS, TITLES, blocked={"https://x.com/5"}, reasons={}, listings={"https://x.com/6"})
    with pytest.raises(lv.LabelV3Error):
        lv.final_rows(V2, V3_ROWS, TITLES, blocked={"https://x.com/5"}, reasons={"https://x.com/2": "  "},
                      listings={"https://x.com/6"})


def test_final_file_can_be_read_by_the_evaluation_code(tmp_path):
    from src.evaluate import load_label_set
    rows = lv.final_rows(V2, V3_ROWS, TITLES, blocked={"https://x.com/5"}, reasons=REASONS,
                         listings={"https://x.com/6"})
    path = lv.write_final(rows, tmp_path, "v3_dev_tata_motors")
    labels = load_label_set([path])
    assert labels["https://x.com/2"] == "indirect" and labels["https://x.com/6"] == "indirect"
    assert len(labels) == len(V2)


# ---------- every article without full text needs a known reason ----------

def test_a_copy_of_a_fetched_article_is_marked_as_a_copy():
    v2 = {**V2, "https://y.com/1": "direct"}
    rows = {r["url"]: r for r in lv.final_rows(v2, V3_ROWS, {**TITLES, "https://y.com/1": "Title 1 copy"},
                                               blocked={"https://x.com/5"}, reasons=REASONS,
                                               listings={"https://x.com/6"},
                                               copies={"https://y.com/1": "https://x.com/1"})}
    assert rows["https://y.com/1"]["basis"] == "short text (copy of a fetched article)"
    assert rows["https://y.com/1"]["label"] == "direct"


def test_an_article_without_full_text_and_without_a_known_reason_stops_the_run():
    with pytest.raises(lv.LabelV3Error):
        lv.final_rows(V2, V3_ROWS, TITLES, blocked={"https://x.com/5"}, reasons=REASONS)   # x.com/6 unexplained


def test_a_copy_must_have_the_same_label_as_its_fetched_article():
    v2 = {**V2, "https://y.com/2": "not"}                 # x.com/2 is now indirect on full text
    with pytest.raises(lv.LabelV3Error):
        lv.final_rows(v2, V3_ROWS, TITLES, blocked={"https://x.com/5"}, reasons=REASONS,
                      listings={"https://x.com/6"}, copies={"https://y.com/2": "https://x.com/2"})


def test_find_copies_uses_the_duplicate_rules_of_cleaning():
    fetched = {"https://a.com/1": "Tata Motors raises Iveco takeover offer price to 14.40 euro per share"}
    others = {"https://b.com/9": "Tata Motors raises Iveco takeover offer to 14.40 euro per share",
              "https://c.com/7": "Car registrations plunge 22% this Navratri despite strong deliveries"}
    assert lv.find_copies(others, fetched) == {"https://b.com/9": "https://a.com/1"}
