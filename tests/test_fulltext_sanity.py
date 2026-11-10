"""
Tests for the full-text sanity check (no network).

Sometimes the fetched page is not the article: an aggregator page can show
other headlines, a market summary or navigation instead. Using that text
would make the relevance check judge the wrong article.

The check compares the fetched text with what NewsAPI says about the
article, using several signals:
    description_missing - the description's opening sentence is not in the text
    title_low           - few of the title's content words appear in the text
    company_missing     - the title or description names the company, the text does not
                          (only checked when the title or description names it)
    looks_like_list     - most lines are short, like a list of headlines or links

Fallback to the short NewsAPI text needs the content signals together:
    description known:   description_missing and title_low
    description unknown: title_low and company_missing and looks_like_list
A missing company name alone never causes a fallback, because a real
industry article can discuss the sector without naming the company.
"""

from src import fulltext as ft
from src import fulltext_eval as fe

AGGREGATOR_PAGE = "\n".join([
    "Crude climbed to its highest level since April as investors weighed the risk of renewed fighting.",
    "Market Summary",
    "Gold slips as dollar firms",
    "Bitcoin tops $98,000 again",
    "Fed minutes show split on rates",
    "Oil majors rally in early trade",
    "Asian shares mixed ahead of data",
    "Trending now",
    "Top stories",
])
JAGUAR_TITLE = "Jaguar unveils $130,500 electric vehicle two years after controversial rebrand"
JAGUAR_DESCRIPTION = ("Luxury carmaker Jaguar has launched its new all-electric vehicle, two years after the famed "
                      "British company's controversial rebrand prompted a social media backlash…")

IVECO_BODY = "\n".join([
    "Tata Motors raised its offer for Iveco to 14.40 euro per share, as key investors held out for a better price.",
    "The company said the revised offer values the truck maker at about 3.9 billion euro.",
    "Several large shareholders had said the earlier price was too low for the business.",
    "The offer period was extended by two weeks to give investors time to decide.",
])


def _check(title, description, text, company="Tata Motors"):
    return ft.check_full_text(title, description, text, company, aliases=None)


# ---------- wrong page: fall back to the short text ----------

def test_an_aggregator_page_falls_back_with_its_reasons():
    check = _check(JAGUAR_TITLE, JAGUAR_DESCRIPTION, AGGREGATOR_PAGE, company="Jaguar")
    assert check["use_full_text"] is False
    assert "description_missing" in check["reasons"] and "title_low" in check["reasons"]
    assert check["signals"]["company_missing"] is True


def test_without_a_description_a_list_page_falls_back_only_when_all_other_signals_agree():
    check = _check(JAGUAR_TITLE, "", AGGREGATOR_PAGE, company="Jaguar")
    assert check["signals"]["description_missing"] is None          # unknown, not a mismatch
    assert check["use_full_text"] is False
    assert set(check["reasons"]) == {"title_low", "company_missing", "looks_like_list"}


# ---------- real articles keep their full text ----------

def test_a_real_article_with_a_differently_worded_headline_keeps_its_full_text():
    check = _check("A comeback bid that keeps getting richer",
                   "Tata Motors raised its offer for Iveco to 14.40 euro per share.", IVECO_BODY)
    assert check["signals"]["title_low"] is True
    assert check["use_full_text"] is True


def test_an_empty_description_is_unknown_not_a_mismatch():
    check = _check("A comeback bid that keeps getting richer", "", IVECO_BODY)
    assert check["signals"]["description_missing"] is None
    assert check["use_full_text"] is True


def test_a_description_cut_off_mid_sentence_is_still_found():
    check = _check("Tata Motors raises Iveco offer",
                   "Tata Motors raised its offer for Iveco to 14.40 euro per share, as key inves… [+2140 chars]",
                   IVECO_BODY)
    assert check["signals"]["description_missing"] is False
    assert check["use_full_text"] is True


def test_a_separately_written_description_does_not_reject_a_matching_body():
    check = _check("Tata Motors raises Iveco offer to 14.40 euro per share",
                   "Analysts weigh in on what the sweetened deal means for minority holders and the Agnelli family.",
                   IVECO_BODY)
    assert check["signals"]["description_missing"] is True
    assert check["use_full_text"] is True


INDUSTRY_BODY = "\n".join([
    "Car registrations fell 22% during the Navratri festival, data from dealers showed.",
    "The fall reflects a high base after tax cuts lifted sales last year.",
    "Two-wheeler sales also declined, though rural demand held up better.",
    "Dealers said waiting periods for CNG models stayed long in large cities.",
])


def test_an_industry_article_that_never_names_the_company_keeps_its_full_text():
    check = _check("Car registrations plunge 22% this Navratri",
                   "Car registrations fell 22% during the Navratri festival.", INDUSTRY_BODY)
    assert check["signals"]["company_missing"] is None               # not checked: nothing names the company
    assert check["use_full_text"] is True


def test_company_named_only_in_the_description_does_not_reject_an_industry_body():
    check = _check("Car registrations plunge 22% this Navratri",
                   "Car registrations fell 22% during Navratri, with Tata Motors among the hardest hit.",
                   INDUSTRY_BODY)
    assert check["signals"]["company_missing"] is True
    assert check["use_full_text"] is True


def test_the_check_reports_every_signal():
    check = _check(JAGUAR_TITLE, JAGUAR_DESCRIPTION, AGGREGATOR_PAGE, company="Jaguar")
    assert set(check["signals"]) == {"description_missing", "title_low", "title_overlap",
                                     "company_missing", "looks_like_list", "teaser"}
    assert 0.0 <= check["signals"]["title_overlap"] <= 1.0


# ---------- used before stage 1 and the passage ----------

def _raw(n, title, description, url=None):
    return {"source": {"id": None, "name": "News"}, "author": "A", "title": title, "description": description,
            "content": "", "url": url or f"https://x.com/{n}", "urlToImage": None,
            "publishedAt": "2026-10-20T10:00:00Z"}


def test_usable_full_texts_drop_a_wrong_page_and_record_why():
    wrong = _raw(1, JAGUAR_TITLE, JAGUAR_DESCRIPTION, url="https://biztoc.com/x/1")
    right = _raw(2, "Tata Motors raises Iveco offer", "Tata Motors raised its offer for Iveco to 14.40 euro per share.")
    texts, fallbacks = fe.usable_full_texts([wrong, right], {wrong["url"]: AGGREGATOR_PAGE, right["url"]: IVECO_BODY},
                                            "Jaguar", aliases=None)
    assert wrong["url"] not in texts and texts[right["url"]] == IVECO_BODY
    assert set(fallbacks) == {wrong["url"]}
    assert "description_missing" in fallbacks[wrong["url"]]


def test_scores_count_full_text_fallbacks():
    wrong = _raw(1, JAGUAR_TITLE, JAGUAR_DESCRIPTION, url="https://biztoc.com/x/1")
    result = fe.compare([wrong], {wrong["url"]: "direct"}, {wrong["url"]: AGGREGATOR_PAGE}, "Jaguar", aliases=None)
    assert result["full_text_used"] == 0
    assert result["full_text_fallbacks"] == {JAGUAR_TITLE: ["description_missing", "title_low", "company_missing",
                                                            "looks_like_list"]}


# ---------- a teaser page: less text than the short NewsAPI text ----------
#
# A fetched page can hold only a teaser (an aggregator link page) or a login
# box. When the page holds less text than the short NewsAPI text (title,
# description, first 200 characters of the content), the short text is used.

TEASER_TITLE = "Tata Motors clears key hurdle in Iveco takeover as shareholders back buyout"
TEASER_DESCRIPTION = ("The majority of Iveco shareholders tendered their shares for Tata Motors' proposed "
                      "takeover offer, valuing the Italian manufacturer at around $4.36 billion.")
TEASER_CONTENT = TEASER_DESCRIPTION + " This story appeared on wsj.com, 2026-10-27 08:12:00. [+120 chars]"
TEASER_PAGE = "\n".join([TEASER_TITLE, "wsj.com/business/deals/tata-motors-iveco", TEASER_DESCRIPTION])


def test_the_short_newsapi_text_is_the_title_description_and_first_200_characters():
    short = ft.short_newsapi_text("T", "D", "x" * 300)
    assert short == "T\nD\n" + "x" * 200


def test_a_page_with_less_text_than_the_short_newsapi_text_is_a_teaser():
    check = ft.check_full_text(TEASER_TITLE, TEASER_DESCRIPTION, TEASER_PAGE, "Tata Motors", aliases=None,
                               content=TEASER_CONTENT)
    assert check["use_full_text"] is False and check["reasons"] == ["teaser"]
    assert check["signals"]["teaser"] is True


def test_a_real_article_with_more_text_than_the_short_newsapi_text_is_kept():
    title = "Tata Motors raises Iveco offer to 14.40 euro as key investors hold out"
    description = "Tata Motors raised its offer for Iveco to 14.40 euro per share, as key investors held out."
    check = ft.check_full_text(title, description, IVECO_BODY, "Tata Motors", aliases=None,
                               content=description)
    assert check["use_full_text"] is True and check["signals"]["teaser"] is False


def test_usable_full_texts_drop_a_teaser_using_the_newsapi_content():
    teaser = {"url": "https://biztoc.com/x/1", "title": TEASER_TITLE, "description": TEASER_DESCRIPTION,
              "content": TEASER_CONTENT}
    texts, fallbacks = fe.usable_full_texts([teaser], {teaser["url"]: TEASER_PAGE}, "Tata Motors", None)
    assert texts == {} and fallbacks == {teaser["url"]: ["teaser"]}
