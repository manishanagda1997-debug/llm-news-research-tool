"""
Sale listings: auctions and adverts for second-hand vehicles.

The labelling guide labels a private sale or auction of an old vehicle `not`:
the company earns nothing from it, and it says nothing about the business
today. Two separate checks find such articles, so each can be measured:

    pattern - at least two of three groups of signals in the visible text:
              1. vehicle history: mileage written like "39k miles" or
                 "has 52,000 miles", "original owner", "the seller",
                 "selling dealer", "initially sold by"
              2. sale words: "no reserve", "auction", "for sale",
                 "reserve price", "bidding"
              3. a title that starts with a model year ("1967 Jaguar 420",
                 "39k-Mile 2002 Jaguar XJ8")
              One group alone is never enough: a recall of "2024 Range Rover"
              models or "JLR to auction prototypes" is company news.
    site    - the article comes from a website in LISTING_SITES

A website is listed only after checking that its articles in the labelled
data are sale listings, not news.
"""

import re
from urllib.parse import urlsplit

LISTING_SITES = {
    "bringatrailer.com": (
        "Online auction site for collector and second-hand cars. All 22 of its articles "
        "in the development data are auction listings, labelled `not`."
    ),
}

VEHICLE_HISTORY = [
    re.compile(r"\b\d+(?:\.\d+)?k[- ](?:mile|kilometer|kilometre)s?\b"),
    re.compile(r"\b(?:has|with|shows|showing)\s+[\d,]+\s+(?:miles|kilometers|kilometres)\b"),
    re.compile(r"\b(?:original|second|previous|one)[- ]owner\b"),
    re.compile(r"\b(?:the seller|selling dealer|initially sold by|first sold by)\b"),
]
SALE_WORDS = re.compile(r"\b(?:no reserve|auction\w*|for sale|reserve price|bidding)\b")
TITLE_STARTS_WITH_YEAR = re.compile(r"^\s*(?:[\w.,-]+\s+)?(?:19|20)\d{2}\s+[A-Z]")


def _host_matches(url: str, site: str) -> bool:
    host = (urlsplit(url or "").hostname or "").lower()
    return host == site or host.endswith("." + site)


def listing_signals(article: dict) -> dict[str, bool]:
    """Which of the two checks finds that the article is a sale listing."""
    title = article.get("title") or ""
    text = " ".join(article.get(field) or "" for field in ("title", "description", "content")).lower()
    groups = [
        any(pattern.search(text) for pattern in VEHICLE_HISTORY),
        bool(SALE_WORDS.search(text)),
        bool(TITLE_STARTS_WITH_YEAR.search(title)),
    ]
    return {
        "pattern": sum(groups) >= 2,
        "site": any(_host_matches(article.get("url"), site) for site in LISTING_SITES),
    }


LISTING_MODES = ("off", "patterns", "sites", "both")


def is_sale_listing(article: dict, mode: str) -> bool:
    if mode not in LISTING_MODES:
        raise ValueError(f"drop_listings must be one of {LISTING_MODES}.")
    if mode == "off":
        return False
    signals = listing_signals(article)
    if mode == "patterns":
        return signals["pattern"]
    if mode == "sites":
        return signals["site"]
    return signals["pattern"] or signals["site"]
