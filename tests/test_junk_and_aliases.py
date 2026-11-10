"""
Tests for Proposal 1 (agreed): page-junk removal and verified company aliases.

Page junk: Times of India article content can end with the website text
"Follow Us On Social Media" followed by a copyright line or other
headlines. That tail is cut, the original is kept, and the cut is logged.

Aliases: optional and off by default. A search for "Tata Motors" may also
accept the verified names of the two demerged companies (TMPV, TMCV).

Run with:  pytest
"""

import json
from pathlib import Path

import pytest

import src.preprocess as preprocess
import src.query as query
from src.preprocess import clean_articles

ALIAS_FILE = Path(__file__).resolve().parent.parent / "data" / "company_aliases.json"

TOI_JUNK_TAIL = (
    "Follow Us On Social Media Tata Motors pips Mahindra to be No. 2 car co in April-September"
    "Katyayani's strength on Shashthi, and why it was never the same as rage"
)
CRIME_CONTENT = "A sessions court sentenced a caretaker and her boyfriend to life terms. " + TOI_JUNK_TAIL


def make_article(title="Court sentences caretaker", description="A Mumbai court gave its verdict.",
                 url="https://example.com/a", content=None, source="The Times of India"):
    return {
        "source": {"id": None, "name": source},
        "author": "Reporter",
        "title": title,
        "description": description,
        "url": url,
        "urlToImage": None,
        "publishedAt": "2026-10-16T08:00:00Z",
        "content": content,
    }


# ---------- Page junk: the cleaning function ----------

def test_junk_tail_after_marker_is_cut():
    cleaned, removed = preprocess.remove_page_junk(CRIME_CONTENT)
    assert cleaned == "A sessions court sentenced a caretaker and her boyfriend to life terms."
    assert removed.startswith("Follow Us On Social Media")
    assert "Tata Motors" in removed


def test_copyright_tail_is_cut():
    text = ("Shares rose 3% on Monday. Follow Us On Social Media Copyright © 2026 "
            "Bennett, Coleman & Co. Ltd. All rights reserved.")
    cleaned, removed = preprocess.remove_page_junk(text)
    assert cleaned == "Shares rose 3% on Monday."
    assert "Copyright" in removed


def test_marker_matches_in_any_letter_case():
    cleaned, removed = preprocess.remove_page_junk("Real text. FOLLOW US ON SOCIAL MEDIA junk here")
    assert cleaned == "Real text."
    assert removed


def test_real_sentence_about_social_media_is_kept():
    text = ("The company's controversial rebrand prompted a social media backlash. "
            "The automotive manufacturer said sales would recover.")
    cleaned, removed = preprocess.remove_page_junk(text)
    assert cleaned == text
    assert removed == ""


def test_marker_inside_quotation_marks_is_kept():
    text = 'The campaign asked buyers to "Follow Us On Social Media" for festive offers, and sales rose.'
    cleaned, removed = preprocess.remove_page_junk(text)
    assert cleaned == text
    assert removed == ""


def test_text_without_marker_is_unchanged():
    cleaned, removed = preprocess.remove_page_junk("Tata Motors raised its offer for Iveco.")
    assert cleaned == "Tata Motors raised its offer for Iveco."
    assert removed == ""


# ---------- Page junk: inside clean_articles ----------

def test_crime_story_with_junk_tail_is_no_longer_kept_for_tata_motors():
    articles, stats = clean_articles([make_article(content=CRIME_CONTENT)], query="Tata Motors")
    assert articles == []
    assert stats["query_not_in_visible_text"] == 1
    assert stats["junk_text_removed"] == 1


def test_original_content_and_junk_record_are_kept():
    raw = make_article(title="Tata Motors shares rise", content="Tata Motors shares rose. " + TOI_JUNK_TAIL)
    articles, stats = clean_articles([raw], query="Tata Motors")
    article = articles[0]
    assert article["content"] == "Tata Motors shares rose."
    assert article["content_original"].endswith("never the same as rage")
    assert article["junk_removed"] is True
    assert article["junk_removed_chars"] == len(TOI_JUNK_TAIL)
    assert stats["junk_text_removed"] == 1


def test_article_without_junk_is_marked_unchanged():
    raw = make_article(title="Tata Motors shares rise", content="Tata Motors shares rose on Monday.")
    articles, stats = clean_articles([raw], query="Tata Motors")
    assert articles[0]["junk_removed"] is False
    assert articles[0]["junk_removed_chars"] == 0
    assert stats["junk_text_removed"] == 0


# ---------- Alias file ----------

def test_alias_file_keeps_companies_separate_and_has_sources():
    data = json.loads(ALIAS_FILE.read_text(encoding="utf-8"))
    identities = data["identities"]
    assert {"tata_motors_pv", "tata_motors_cv"} <= set(identities)
    assert "TMPV" in identities["tata_motors_pv"]["names"]
    assert "TMCV" in identities["tata_motors_cv"]["names"]
    assert "TMPV" not in identities["tata_motors_cv"]["names"]
    for identity in identities.values():
        assert identity["sources"], "every identity needs at least one source link"
    group = data["search_groups"]["tata motors"]
    assert group["identities"][:2] == ["tata_motors_pv", "tata_motors_cv"]
    all_names = {name.lower() for identity in identities.values() for name in identity["names"]}
    assert "tata" not in all_names


# ---------- Aliases: query rules ----------

@pytest.fixture
def aliases():
    return query.load_aliases(ALIAS_FILE)


def test_alias_names_for_tata_motors(aliases):
    names = query.alias_names("Tata Motors", aliases)
    assert {"tata motors passenger vehicles", "tmpv", "tata motors commercial vehicles", "tmcv"} <= set(names)
    assert "tata" not in names


def test_unknown_company_has_no_aliases(aliases):
    assert query.alias_names("Infosys", aliases) == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Lupin, SRF, TMPV, Maruti, NTPC among 74 BSE 500 stocks hit 52-week lows", True),
        ("TMCV shares rise after truck sales data", True),
        ("Tata Motors raises Iveco offer", True),
        ("Tata Steel signs a deal in the UK", False),
        ("Code XTMPVX appears in a product list", False),
    ],
)
def test_alias_matching_on_tata_motors(aliases, text, expected):
    rule = query.parse_query("Tata Motors", aliases=aliases)
    assert query.matches(text, rule) is expected


def test_aliases_off_by_default_keeps_old_behaviour():
    rule = query.parse_query("Tata Motors")
    assert query.matches("Lupin, SRF, TMPV, Maruti among stocks at lows", rule) is False


def test_exclusion_still_works_with_aliases(aliases):
    rule = query.parse_query('"Tata Motors" -TMPV', aliases=aliases)
    assert rule.supported
    assert query.matches("Lupin, SRF, TMPV, Maruti among stocks at lows", rule) is False
    assert query.matches("TMCV truck sales rise", rule) is True
    assert query.matches("Tata Motors raises Iveco offer", rule) is True


def test_aliases_do_not_change_other_queries(aliases):
    rule = query.parse_query("Microsoft OR Google", aliases=aliases)
    assert query.matches("TMPV shares fall", rule) is False


# ---------- Aliases: inside clean_articles ----------

def test_clean_articles_with_aliases_keeps_ticker_article(aliases):
    raw = [make_article(title="Lupin, SRF, TMPV, Maruti, NTPC among 74 BSE 500 stocks hit 52-week lows",
                        description="Several large stocks fell to 52-week lows.", url="https://b.com/1")]
    without, _ = clean_articles(raw, query="Tata Motors")
    with_aliases, stats = clean_articles(raw, query="Tata Motors", aliases=aliases)
    assert without == []
    assert len(with_aliases) == 1
    assert stats["aliases_used"] is True
