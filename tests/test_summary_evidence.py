"""
Tests for the summary evidence (Stage 4.1; no network, no model calls).

The summary model reads up to 3,500 characters of each relevant article. When
the cleaned full text fits, it reads the whole article. When it does not,
the selection keeps what an analyst needs to understand what happens and why
it matters:

    whole sentences  a sentence is kept whole or left out, never cut, so a
                     negation, a condition or an attribution ("not yet
                     approved", "subject to", "the company denies") survives
    the lead         the opening paragraph comes first: it usually states the
                     main development
    signals          sentences on status and timing (approved, delayed,
                     denied, expected), causes, impact and risks, quantities,
                     and the company come next; a table-like paragraph full of
                     numbers and names ranks low
    qualifications   a kept sentence brings the next one when that one starts
                     with "However", "But", "Although" and similar words
    order            kept sentences stay in article order; a gap mark shows
                     where text is left out

The signals only order the sentences; the final choice of input method rests
on a hand check of important facts on a sample of long dev articles.
"""

import pytest

from src import summary_evidence as se

COMPANY = "Tata Motors"


def _long(*paragraphs, filler=40):
    """An article over the limit: the given paragraphs, then plain filler paragraphs."""
    filler_text = [f"Filler sentence number {i} says nothing in particular about anything at all here." for i in
                   range(filler)]
    return "\n".join(list(paragraphs) + [" ".join(filler_text[i:i + 4]) for i in range(0, filler, 4)])


# ---------- sentences ----------

@pytest.mark.parametrize("text, count", [
    ("Tata Motors rose 3.5%. Shares closed higher.", 2),
    ("The deal is worth Rs. 100 crore. It closes soon.", 2),
    ("Mr. Balaji joins the board. He leads JLR.", 2),
    ("Tata Motors is No. 2 in India. Mahindra is third.", 2),
    ("Sales in the U.S. rise. Europe is flat.", 2),
    ("It said: \"We are not selling.\" The stock fell.", 2),
    ("Revenue was €21 bn. vs last year? Unclear! Fine.", 3),
])
def test_sentences_split_at_real_sentence_ends_only(text, count):
    assert len(se.split_sentences(text)) == count


def test_sentence_splitting_loses_no_text():
    text = "Tata Motors rose 3.5%. Mr. Balaji said the No. 2 spot matters. Rs. 100 crore is spent."
    assert " ".join(se.split_sentences(text)) == text


# ---------- short articles ----------

def test_an_article_that_fits_is_given_whole():
    result = se.build_summary_evidence("Title", "One paragraph.\nTwo paragraph.", COMPANY, None)
    assert result == {"text": "Title\nOne paragraph.\nTwo paragraph.", "complete": True,
                      "sentences_kept": 2, "sentences_total": 2}


# ---------- long articles ----------

def _no_cut(result):
    """No text line carries a cut mark (the gap mark between left-out text is not a cut)."""
    return all(se.CUT not in line for line in result["text"].split("\n") if line != se.GAP)


def _kept_sentences(result):
    return [s for part in result["text"].split("\n")[1:] if part != se.GAP for s in se.split_sentences(part)]


def test_every_kept_sentence_is_a_whole_sentence_of_the_article():
    text = _long("Tata Motors raises its Iveco offer to 14.40 euro per share. The offer closes on 26 October.",
                 "The deal is not yet approved by the Dutch regulator. Approval is expected in November.",
                 filler=80)
    result = se.build_summary_evidence("Tata Motors raises Iveco offer", text, COMPANY, None)
    originals = {s for p in text.split("\n") for s in se.split_sentences(p)}
    assert result["complete"] is False
    assert all(s in originals for s in _kept_sentences(result))
    assert _no_cut(result)


def test_a_negation_or_condition_is_never_cut_away():
    text = _long("Tata Motors says the merger is subject to shareholder approval.",
                 "The deal is not yet approved by the Dutch regulator, and it may still be blocked.")
    result = se.build_summary_evidence("Iveco deal", text, COMPANY, None)
    assert "The deal is not yet approved by the Dutch regulator, and it may still be blocked." in result["text"]
    assert "Tata Motors says the merger is subject to shareholder approval." in result["text"]


def test_the_opening_paragraph_comes_first():
    text = _long("Tata Motors signs a manufacturing partnership with Stellantis in India.")
    result = se.build_summary_evidence("Partnership", text, COMPANY, None)
    assert result["text"].split("\n")[1] == "Tata Motors signs a manufacturing partnership with Stellantis in India."


def test_a_qualification_after_a_kept_sentence_comes_with_it():
    body = ["Opening paragraph about the market in general terms only, with no company named."]
    body += [f"Filler sentence number {i} says nothing in particular about anything at all here." for i in range(8)]
    body += ["Tata Motors expects the plant to start in March. However, the start depends on a state permit."]
    text = "\n".join(body) + "\n" + _long(filler=40)
    result = se.build_summary_evidence("Plant", text, COMPANY, None)
    assert "Tata Motors expects the plant to start in March." in result["text"]
    assert "However, the start depends on a state permit." in result["text"]


def test_status_and_development_sentences_beat_a_stock_table():
    table = ("Tata Motors 712.5 1.2% Maruti 12,400 0.8% M&M 3,105 -0.4% Hero 5,210 0.3% Bajaj 9,870 1.1% "
             "Eicher 5,640 -0.2% TVS 2,710 0.9% Ashok Leyland 241 1.4% Escorts 3,880 0.1% Sona 690 -1.0%")
    status = "Tata Motors withdraws its margin guidance for the year after weak demand in Europe."
    text = _long("A general opening line about the day in the markets.", table, filler=36) + "\n" + status
    result = se.build_summary_evidence("Markets", text, COMPANY, None, max_chars=600)
    assert status in result["text"] and table not in result["text"]


def test_the_limit_is_respected_and_the_order_kept_with_gap_marks():
    text = _long("Tata Motors raises its offer.", filler=80) + "\nTata Motors says the offer is final."
    result = se.build_summary_evidence("Offer", text, COMPANY, None, max_chars=900)
    assert len(result["text"]) <= 900
    assert result["text"].index("raises its offer") < result["text"].index("offer is final")
    assert se.GAP in result["text"]


def test_the_selection_is_the_same_every_time():
    text = _long("Tata Motors raises its offer.", "Shareholders must accept by 26 October.", filler=60)
    first = se.build_summary_evidence("Offer", text, COMPANY, None)
    assert first == se.build_summary_evidence("Offer", text, COMPANY, None)


def test_a_sentence_longer_than_the_room_left_is_skipped_not_cut():
    giant = "Tata Motors " + " ".join(["announces a very long list of details"] * 80) + "."
    text = "Tata Motors opens a plant.\n" + giant + "\n" + _long(filler=8)
    result = se.build_summary_evidence("Plant", text, COMPANY, None, max_chars=500)
    assert giant not in result["text"] and _no_cut(result)


# ---------- the audit of the relevance passage ----------

def test_cut_sentences_in_a_passage_are_counted():
    full = "Tata Motors raises its offer. The deal is not yet approved by the regulator in the Netherlands."
    passage = "Title\nTata Motors raises its offer. The deal is not yet approved…"
    assert se.cut_sentences(passage, full) == 1
    assert se.cut_sentences("Title\n" + full, full) == 0
