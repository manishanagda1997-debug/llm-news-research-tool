"""
Tests, group 2: sale listings (auctions and second-hand car adverts).

The labelling guide labels a private sale or auction of an old vehicle `not`.
Two checks find them, and each is measured on its own:

    patterns - general words of a sale listing; at least two of three groups:
               1. vehicle history (mileage, "original owner", "the seller",
                  "selling dealer", "initially sold by")
               2. sale words ("no reserve", "auction", "for sale", "reserve price")
               3. a title that starts with a model year ("1967 Jaguar 420",
                  "39k-Mile 2002 Jaguar XJ8")
    sites    - a short documented list of auction and listing websites

Real company news (recalls, dealer openings, reviews of current models,
takeover bids, market reports) must not be caught.
"""

import json
from pathlib import Path

import pytest

from src import listings
from src.preprocess import clean_articles


def _article(title, description="", content="", url="https://news.example.com/story"):
    return {"source": {"id": None, "name": "News"}, "author": "A", "title": title,
            "description": description, "content": content, "url": url,
            "urlToImage": None, "publishedAt": "2026-10-20T10:00:00Z"}


# ---------- patterns: sale listings are caught ----------

@pytest.mark.parametrize("title, description", [
    ("39k-Mile 2002 Jaguar XJ8 at No Reserve",
     "This 2002 Jaguar XJ8 sedan now has 39k miles and is finished in Topaz Metallic."),
    ("1993 Jaguar XJS Convertible",
     "This 1993 Jaguar XJS convertible was purchased by the seller in 2023."),
    ("2017 Land Rover Range Rover at No Reserve",
     "This 2017 Land Rover Range Rover was initially sold by a dealer in Florida."),
    ("Classic 1965 Mustang offered for sale",
     "The car has had one original owner and shows 52,000 miles."),
])
def test_patterns_catch_sale_listings(title, description):
    assert listings.listing_signals(_article(title, description))["pattern"] is True


# ---------- patterns: real news is not caught ----------

@pytest.mark.parametrize("title, description", [
    ("2024 Range Rover recalled over seatbelt fault",
     "Jaguar Land Rover is recalling vehicles built between March and June."),
    ("Jaguar Land Rover India expands retail presence in Delhi, Gurugram",
     "The company has opened a new dealership and relocated to a larger facility."),
    ("Better than the V8: Range Rover Electric tested",
     "After four years of engineering, the luxury SUV enters the EV era."),
    ("Tata Motors raises Iveco takeover offer to 14.40 euro per share",
     "The revised offer follows delays; shareholders can tender their shares."),
    ("Used Truck Market to Reach USD 70.74 Billion by 2031",
     "Digital auctions and fleet renewal reshape the secondary market, says a report."),
    ("JLR to auction historic prototypes for charity",
     "Jaguar Land Rover said the money will support engineering scholarships."),
])
def test_patterns_do_not_catch_company_news(title, description):
    assert listings.listing_signals(_article(title, description))["pattern"] is False


def test_one_signal_group_alone_is_not_enough():
    only_year = _article("2026 Jaguar Type 01 revealed", "The new electric GT goes on sale next year.")
    only_history = _article("Range Rover owners report fewer faults", "The original owner survey covers 5,000 cars.")
    assert listings.listing_signals(only_year)["pattern"] is False
    assert listings.listing_signals(only_history)["pattern"] is False


# ---------- sites ----------

def test_site_check_uses_the_web_address():
    listing = _article("Jaguar XKE Series I Roadster", url="https://bringatrailer.com/listing/1965-jaguar-xke-12/")
    assert listings.listing_signals(listing)["site"] is True
    assert listings.listing_signals(_article("Jaguar XKE"))["site"] is False


def test_site_check_matches_subdomains_but_not_look_alikes():
    assert listings.listing_signals(_article("x", url="https://www.bringatrailer.com/a"))["site"] is True
    assert listings.listing_signals(_article("x", url="https://notbringatrailer.com/a"))["site"] is False


def test_every_listed_site_has_a_written_reason():
    assert listings.LISTING_SITES
    assert all(reason.strip() for reason in listings.LISTING_SITES.values())


# ---------- the check inside cleaning ----------

LISTING = _article("39k-Mile 2002 Jaguar XJ8 at No Reserve",
                   "This 2002 Jaguar XJ8 sedan now has 39k miles.",
                   url="https://bringatrailer.com/listing/2002-jaguar-xj8/")
NEWS = _article("Jaguar Land Rover posts quarterly profit", "Jaguar Land Rover reported higher sales.",
                url="https://news.example.com/jlr-profit")


@pytest.mark.parametrize("mode, kept", [
    ("off", 2), ("patterns", 1), ("sites", 1), ("both", 1),
])
def test_cleaning_can_drop_listings_in_each_mode(mode, kept):
    articles, stats = clean_articles([LISTING, NEWS], max_articles=10, drop_listings=mode)
    assert len(articles) == kept
    assert stats["sale_listing"] == 2 - kept


def test_cleaning_is_off_by_default():
    articles, stats = clean_articles([LISTING, NEWS], max_articles=10)
    assert len(articles) == 2 and stats["sale_listing"] == 0


def test_cleaning_refuses_an_unknown_mode():
    with pytest.raises(ValueError):
        clean_articles([NEWS], max_articles=10, drop_listings="all")


# ---------- regression on the labelled development data ----------

EVAL = Path(__file__).resolve().parent.parent / "data" / "eval"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def _labelled_articles():
    from src.evaluate import load_label_set
    from src.preprocess import normalize_url
    files = [EVAL / "v2_dev_tata_motors.csv"] + sorted(EVAL.glob("dev_*_retrieval_*.csv"))
    if not all(path.exists() for path in files) or not list(RAW.glob("*.json")):
        pytest.skip("labelled development data not present")
    labels = load_label_set(files)
    articles = {}
    for path in RAW.glob("*.json"):
        for article in json.loads(path.read_text(encoding="utf-8")).get("articles", []):
            if article.get("url") and normalize_url(article["url"]) in labels:
                articles[normalize_url(article["url"])] = article
    return articles, labels


@pytest.mark.parametrize("check", ["pattern", "site"])
def test_no_relevant_labelled_article_is_caught(check):
    articles, labels = _labelled_articles()
    caught = [url for url, article in articles.items()
              if listings.listing_signals(article)[check] and labels[url] != "not"]
    assert caught == []
