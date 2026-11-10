"""
Tests for src/boilerplate.py: press-release boilerplate removed from the summary input (summary passage
version 3). The rules work on the structure of the text, never on single words:

    leading notice  in the first two paragraphs only: mostly capital letters and a distribution notice
                    ("NOT FOR RELEASE", "NOT FOR DISTRIBUTION", "FOR IMMEDIATE RELEASE")
    boilerplate tail  from a boilerplate heading (contacts, media or investor relations, "About <Name>",
                    general restrictions, disclaimer, forward-looking statements, comments) to the end, and only
                    when everything after it is boilerplate too: more such headings, contact lines, or text under
                    an About, legal or footer heading; a news paragraph under a contacts heading stops the rule
"""

import pytest

from src import boilerplate as bp

NEWS = [
    "Mumbai, 30 October 2026. Tata Motors completes the settlement of its tender offer for Iveco Group.",
    "The offeror now holds 89.42% of Iveco's shares. The acceptance period reopens from 2 to 6 November.",
    "Iveco's board now includes N. Chandrasekaran as chairperson and Olof Persson as chief executive.",
]
NOTICE = ("NOT FOR RELEASE, DISSEMINATION, PUBLICATION OR DISTRIBUTION, IN WHOLE OR IN PART, DIRECTLY OR "
          "INDIRECTLY IN, INTO OR FROM ANY JURISDICTION WHERE TO DO SO WOULD CONSTITUTE A VIOLATION")
CONTACTS = ["Contacts", "Tata Motors Media Contacts", "Corporate Communications: +91 22 6665 8282",
            "E-Mail: press@example.com", "Jane Doe, Manager", "Phone: + 39 340 7701750",
            "Tata Motors Investor Relations", "E-mail: investors@example.com"]
ABOUT = ["About Tata Motors (Formerly TML Commercial Vehicles Ltd):",
         "Part of the USD 180 billion Tata Group, Tata Motors is India's largest commercial vehicle maker, with a "
         "presence across trucks, buses and small commercial vehicles in many markets.",
         "About Iveco Group",
         "Iveco Group N.V. is a global automotive leader in commercial and specialty vehicles and powertrains."]
LEGAL = ["General restrictions",
         "The offer referred to in this press release is promoted by the offeror for all issued common shares.",
         "This press release contains forward-looking information and statements about the offer."]
FOOTER = ["Comments", "Postmedia is committed to maintaining a lively but civil forum for discussion.",
          "Sign in or Sign up to Comment"]


def _clean(paragraphs):
    return bp.clean("\n".join(paragraphs))


def test_a_press_release_keeps_only_its_news():
    result = _clean([NOTICE] + NEWS + CONTACTS + ABOUT + LEGAL + FOOTER)
    assert result["text"].split("\n") == NEWS
    kinds = [r["kind"] for r in result["removed"]]
    assert kinds == ["leading_notice", "tail"]
    assert result["removed_chars"] == sum(len(p) for p in [NOTICE] + CONTACTS + ABOUT + LEGAL + FOOTER)


def test_an_article_without_boilerplate_is_unchanged():
    result = _clean(NEWS)
    assert result["text"] == "\n".join(NEWS) and result["removed"] == [] and result["removed_chars"] == 0


def test_a_news_paragraph_with_the_word_investor_or_offer_or_contact_stays():
    words = ["Tata Motors' investor relations team says the offer gets regulatory approval in Brazil.",
             "Analysts contact the company about the offer; the investor call is on 5 November."]
    result = _clean(NEWS + words)
    assert result["text"].split("\n") == NEWS + words


def test_a_news_paragraph_after_a_contacts_heading_stops_the_rule():
    late_news = "Separately, Tata Motors says its Lucknow plant restarts production after a two-week halt."
    result = _clean(NEWS + ["Contacts", "E-Mail: press@example.com", late_news])
    assert result["text"].split("\n") == NEWS + ["Contacts", "E-Mail: press@example.com", late_news]


def test_an_about_section_followed_by_more_news_is_kept():
    middle = ["About Tata Motors", "Tata Motors is India's largest commercial vehicle maker."]
    more = [f"In other news, item {n} on the company's new electric trucks for Europe in 2027." for n in range(4)]
    later = ["Contacts", "E-Mail: press@example.com"]
    result = _clean(NEWS + middle + more + later)
    assert result["text"].split("\n") == NEWS + middle + more            # only the contacts tail goes


def test_an_about_heading_needs_a_name():
    lines = ["About the deal", "The deal values Iveco at 3.8 billion euro and needs approval in Brazil."]
    assert _clean(NEWS + lines)["text"].split("\n") == NEWS + lines


def test_the_notice_is_removed_only_at_the_start():
    late = ("NOT FOR DISTRIBUTION IN THE UNITED STATES, SAYS THE OFFER DOCUMENT QUOTED BY THE NEWSPAPER IN ITS "
            "REPORT ON THE DEAL")
    assert _clean(NEWS + [late])["text"].split("\n") == NEWS + [late]
    assert _clean(["FOR IMMEDIATE RELEASE"] + NEWS)["text"].split("\n") == NEWS


def test_a_lowercase_paragraph_that_mentions_release_is_not_a_notice():
    first = "Not for release until the board meets, the results are now public after a leak, the paper says."
    assert _clean([first] + NEWS)["text"].split("\n") == [first] + NEWS


def test_the_news_part_is_never_emptied():
    only = ["About Tata Motors", "Tata Motors is India's largest commercial vehicle maker."]
    assert _clean(only)["text"].split("\n") == only


def test_a_notice_at_the_start_of_a_short_text_is_cut_and_its_news_kept():
    short = NOTICE + " OF THE RELEVANT LAWS OF SUCH JURISDICTION Mumbai, India/Turin, Italy – Tata Motors completes"
    result = bp.clean(short)
    assert result["text"] == "Mumbai, India/Turin, Italy – Tata Motors completes"
    assert result["removed"][0]["kind"] == "leading_notice"
