"""
Tests for src/summary_passage.py: summary passage version 3 is version 2 (the relevance passage plus whole
continued quotes, within the hard cap) built on the text after press-release boilerplate is removed, so the freed
room goes to the article's own paragraphs. The relevance passage of Stages 2 and 3 is not changed.
"""

from src import summary_passage as sp
from src.relevance_check import SUMMARY_MAX_CHARS, build_passage, build_summary_passage

# As in a real press release: the company is named at the start and in the boilerplate, while most news
# paragraphs name only the other company (here Iveco), so version 2 spends its room on the boilerplate mentions.
NEWS = (["Tata Motors completes the settlement of its tender offer for Iveco Group, the company says today."]
        + [f"Iveco update {n}: the board changes and the committees take their new form, step {n} of the plan."
           + " More detail follows." * 8 for n in range(1, 16)])
TAIL = (["Contacts", "Tata Motors Media Contacts", "E-Mail: press@example.com", "About Tata Motors",
         "Tata Motors is part of the Tata Group and makes trucks and buses in India and abroad. " * 8]
        + ["General restrictions"] + ["This press release about Tata Motors is not an offer to buy shares. " * 9] * 4)
TEXT = "\n".join(NEWS + TAIL)
NOTICE = "NOT FOR RELEASE, PUBLICATION OR DISTRIBUTION IN ANY JURISDICTION WHERE THIS WOULD BREAK THE LAW"


def _build(text, short_text=""):
    return sp.build("Tata Motors completes Iveco settlement", text, "Tata Motors", None, short_text=short_text)


def test_the_version_is_three():
    assert sp.PASSAGE_VERSION == "3"


def test_boilerplate_is_removed_and_the_room_goes_to_news():
    old = build_summary_passage("Tata Motors completes Iveco settlement", TEXT, "Tata Motors", None)["text"]
    new = _build(TEXT)
    assert "General restrictions" not in new["text"] and "E-Mail" not in new["text"]
    assert "E-Mail: press@example.com" in old and "General restrictions" in old
    news_old = sum(1 for n in range(1, 16) if f"Iveco update {n}:" in old)
    news_new = sum(1 for n in range(1, 16) if f"Iveco update {n}:" in new["text"])
    assert news_new > news_old                              # the freed room holds more of the article's news
    assert len(new["text"]) >= len(old) - 200               # the room is used, not left empty
    assert len(new["text"]) <= SUMMARY_MAX_CHARS
    assert new["removed_chars"] > 0 and [r["kind"] for r in new["removed"]] == ["tail"]


def test_an_article_without_boilerplate_gets_exactly_the_version_2_passage():
    plain = "\n".join(NEWS)
    v2 = build_summary_passage("Tata Motors completes Iveco settlement", plain, "Tata Motors", None)
    v3 = _build(plain)
    assert v3["text"] == v2["text"] and v3["removed"] == []


def test_a_notice_in_the_short_text_is_cut_when_there_is_no_full_text():
    result = _build("", short_text=NOTICE + " Mumbai – Tata Motors completes the Iveco settlement.")
    assert "NOT FOR RELEASE" not in result["text"] and "Tata Motors completes the Iveco settlement." in result["text"]


def test_the_relevance_passage_is_unchanged():
    assert "General restrictions" in build_passage("Tata Motors completes Iveco settlement", "\n".join(NEWS[:1] + TAIL),
                                                   "Tata Motors", None)["text"]



def test_the_fill_goes_in_article_order_and_never_past_the_limit():
    from src.relevance_check import MAX_PASSAGE_CHARS
    new = _build(TEXT)
    numbers = [n for n in range(1, 16) if f"Iveco update {n}:" in new["text"]]
    assert numbers == list(range(1, len(numbers) + 1))      # the first news paragraphs, in order
    assert len(new["text"]) <= MAX_PASSAGE_CHARS


def test_news_lines_of_version_2_missing_in_version_3_are_found():
    v2 = "Title\nFirst news line.\n…middle of a long news paragraph…\nContacts\nE-Mail: a@b.com"
    boilerplate = {"Contacts", "E-Mail: a@b.com"}
    assert sp.news_lines_lost(v2, "Title\nFirst news line.\nmiddle of a long news paragraph and more", boilerplate) == []
    assert sp.news_lines_lost(v2, "Title\nmiddle of a long news paragraph", boilerplate) == ["First news line."]
