"""
Step 7 - Cleaning: turn raw NewsAPI articles into clean, numbered articles.

Rules, in order:
    1. skip articles with broken data (wrong types) and count them
    2. drop articles NewsAPI marks as removed ("[Removed]")
    3. drop articles with no title, no description or no link
    4. clean the text: HTML tags, HTML codes like &amp;, extra spaces,
       and the "... [+1234 chars]" ending NewsAPI adds to content.
       Page junk is cut from the content: some Times of India contents end
       with the website text "Follow Us On Social Media" followed by a
       copyright line or other headlines. Only this exact, observed marker
       is used, it is not cut when it sits inside quotation marks, and the
       original content is kept next to the cleaned one.
    5. drop articles whose visible text (title, description and the first
       200 characters of content) does not follow the query rules in
       src/query.py. NewsAPI also matches the full article, which the
       free plan does not return, so such articles give the summary
       nothing to work with. Queries the rules cannot read safely
       (brackets, AND mixed with OR ...) skip this rule only.
    6. drop duplicates, in three separate counts:
         - same link (after removing www., tracking codes and a final /)
         - same headline
         - nearly the same headline (90% alike, e.g. "$130,500" and
           "$130,000" versions of one wire story)
       The copy with more text is kept, in the slot of the first copy.
       Every decision is written to a log for checking.
    7. order the articles: NewsAPI's own order, or (ranking="mention") articles
       naming the company in the title first, then in the description, then
       only in the content; NewsAPI's order is kept inside each level
    8. keep the first N articles (the limit comes after the ordering)
    9. number them 1, 2, 3 ... so the summary can cite [1], [2]

clean_articles() also returns counts of what each rule drops and why.
"""

import html
import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from src.config import PROCESSED_DIR
from src.listings import LISTING_MODES, is_sale_listing
from src.query import matched_identities, matches, parse_query
from src.utils import unique_path

REMOVED_MARK = "[Removed]"
TRUNCATION_TAIL = re.compile(r"\s*(…|\.\.\.)?\s*\[\+\d+\s*chars?\]\s*$")
HTML_TAG = re.compile(r"<[^>]+>")
SPACES = re.compile(r"\s+")

NEAR_DUPLICATE_RATIO = 0.9      # titles at least 90% alike count as the same story (to be checked in the notebook)
NEAR_DUPLICATE_MIN_LENGTH = 30  # short titles are skipped, they look alike too easily
TRACKING_PARAMS = ("utm_", "fbclid", "gclid")
TEXT_FIELDS = ("title", "description", "url", "content", "author", "publishedAt", "urlToImage")

# Exact website text seen at the start of page junk in NewsAPI content (lower case).
# Add a marker only after finding it in real data.
JUNK_MARKERS = ("follow us on social media",)
OPENING_QUOTES = ('"', "“", "'", "‘")
RANKINGS = ("newsapi", "mention")
MENTION_ORDER = {"title": 0, "description": 1, "content": 2, None: 3}


def clean_text(text: str | None) -> str:
    """Remove HTML, HTML codes, the [+N chars] tail and extra spaces."""
    if not text:
        return ""
    text = html.unescape(text)
    text = HTML_TAG.sub(" ", text)
    text = TRUNCATION_TAIL.sub("", text)
    text = SPACES.sub(" ", text)
    return text.strip()


def remove_page_junk(text: str) -> tuple[str, str]:
    """
    Cut website junk from the end of a content text.

    Returns (cleaned text, removed text). The removed text is "" when
    nothing is cut. A marker right after an opening quote mark is part of
    the article (someone is being quoted), so it is left alone.
    """
    lower = text.lower()
    for marker in JUNK_MARKERS:
        start = 0
        while (position := lower.find(marker, start)) != -1:
            if text[:position].rstrip().endswith(OPENING_QUOTES):
                start = position + len(marker)
                continue
            return text[:position].rstrip(), text[position:]
    return text, ""


def normalize_url(url: str) -> str:
    """
    One form for links that point to the same page:
        https://www.Example.com/news/a/?utm_source=x&id=5  ->  https://example.com/news/a?id=5
    Only safe changes are made: lower-case host, no "www.", no tracking
    codes, no final "/", no "#part". Real parameters like ?id=5 stay.
    """
    parts = urlsplit(url.strip())
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    params = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
              if not key.lower().startswith(TRACKING_PARAMS)]
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), host, path, urlencode(params), ""))


def _title_key(title: str) -> str:
    """Simple form of a title for duplicate checks: lower case, letters and digits only.
    Also drops a ' - Source Name' ending, which some sites add to the same headline."""
    title = re.sub(r"\s[-|]\s[^-|]{2,40}$", "", title)
    return re.sub(r"[^a-z0-9]", "", title.lower())


def _has_valid_types(raw) -> bool:
    """Each field must be text or empty, and source must be a dictionary or empty."""
    if not isinstance(raw, dict):
        return False
    for name in TEXT_FIELDS:
        if raw.get(name) is not None and not isinstance(raw.get(name), str):
            return False
    source = raw.get("source")
    if source is not None and not isinstance(source, dict):
        return False
    if isinstance(source, dict) and source.get("name") is not None and not isinstance(source.get("name"), str):
        return False
    return True


def _text_amount(article: dict) -> int:
    return len(article["description"]) + len(article["content"])


def _log_entry(kind: str, kept: dict, removed: dict, score: float) -> dict:
    def short(article):
        return {"title": article["title"], "url": article["url"], "source": article["source"]}

    return {"type": kind, "score": round(score, 3), "kept": short(kept), "removed": short(removed)}


def clean_articles(
    raw_articles: list[dict],
    max_articles: int = 10,
    query: str | None = None,
    exact_phrase: bool = True,
    aliases: dict | None = None,
    remove_junk: bool = True,
    ranking: str = "newsapi",
    drop_listings: str = "off",
) -> tuple[list[dict], dict]:
    """
    Clean a list of raw NewsAPI articles.

    query        - when given, the visible text must follow the query rules
    exact_phrase - same meaning as in fetch_news (plain words as one phrase)
    aliases      - loaded alias file (src.query.load_aliases), or None
    remove_junk  - cut page junk from the content (True normally; False only
                   to measure the effect of this rule)
    ranking      - "newsapi" keeps NewsAPI's order; "mention" puts title
                   mentions first, then description, then content
    drop_listings - remove sale listings (auctions, second-hand adverts):
                   "off", "patterns", "sites" or "both" (see src/listings.py)

    Returns:
        articles - list of clean articles, each with an "id" (1, 2, 3 ...)
        stats    - how many articles each rule drops, whether the
                   query check ran, and a log of every duplicate decision
    """
    if not isinstance(raw_articles, list):
        raise ValueError("raw_articles must be a list of articles.")
    if not isinstance(max_articles, int) or isinstance(max_articles, bool) or max_articles < 1:
        raise ValueError("max_articles must be a whole number of 1 or more.")
    if ranking not in RANKINGS:
        raise ValueError(f"ranking must be one of {RANKINGS}.")
    if drop_listings not in LISTING_MODES:
        raise ValueError(f"drop_listings must be one of {LISTING_MODES}.")

    stats = {
        "received": len(raw_articles),
        "invalid_data": 0,
        "removed_by_source": 0,
        "missing_fields": 0,
        "sale_listing": 0,
        "query_not_in_visible_text": 0,
        "duplicate_url": 0,
        "duplicate_title": 0,
        "near_duplicate_title": 0,
        "over_limit": 0,
        "kept": 0,
        "junk_text_removed": 0,
        "aliases_used": False,
        "relevance_check": "off",
        "relevance_note": "",
        "duplicate_log": [],
    }

    rule = None
    if query:
        rule = parse_query(query, exact_phrase=exact_phrase, aliases=aliases)
        if rule.supported:
            stats["relevance_check"] = "applied"
            stats["aliases_used"] = rule.aliases_used
        else:
            stats["relevance_check"] = "skipped"
            stats["relevance_note"] = rule.reason
            rule = None

    articles: list[dict] = []
    slot_by_url: dict[str, int] = {}
    slot_by_title: dict[str, int] = {}

    for raw in raw_articles:
        # Rule 1: broken data
        if not _has_valid_types(raw):
            stats["invalid_data"] += 1
            continue

        title = clean_text(raw.get("title"))
        description = clean_text(raw.get("description"))
        url = (raw.get("url") or "").strip()

        # Rule 2: removed articles
        if title == REMOVED_MARK or description == REMOVED_MARK or "removed.com" in url:
            stats["removed_by_source"] += 1
            continue

        # Rule 3: must have title, description and link
        if not title or not description or not url:
            stats["missing_fields"] += 1
            continue

        content_original = clean_text(raw.get("content"))
        content, junk = remove_page_junk(content_original) if remove_junk else (content_original, "")
        if junk:
            stats["junk_text_removed"] += 1
        source = raw.get("source") or {}
        article = {
            "title": title,
            "description": description,
            # content is often just the start of the description again
            "content": "" if content.startswith(description[:50]) else content,
            "content_original": content_original,
            "junk_removed": bool(junk),
            "junk_removed_chars": len(junk),
            "source": (source.get("name") or "Unknown source").strip(),
            "author": clean_text(raw.get("author")) or None,
            "url": url,
            "image_url": raw.get("urlToImage"),
            "published_at": raw.get("publishedAt") or "",
        }

        # Rule 4: sale listings (auctions, second-hand adverts)
        if is_sale_listing(article, drop_listings):
            stats["sale_listing"] += 1
            continue

        # Rule 5: query rules on the visible text
        if rule and not matches(f"{title} {description} {content}", rule):
            stats["query_not_in_visible_text"] += 1
            continue

        # Rule 6: duplicates
        url_key = normalize_url(url)
        title_key = _title_key(title)
        duplicate_of, kind, score = None, None, 0.0

        if url_key in slot_by_url:
            duplicate_of, kind, score = slot_by_url[url_key], "duplicate_url", 1.0
        elif title_key in slot_by_title:
            duplicate_of, kind, score = slot_by_title[title_key], "duplicate_title", 1.0
        elif len(title_key) >= NEAR_DUPLICATE_MIN_LENGTH:
            for seen_key, slot in slot_by_title.items():
                if len(seen_key) < NEAR_DUPLICATE_MIN_LENGTH:
                    continue
                ratio = SequenceMatcher(None, title_key, seen_key).ratio()
                if ratio >= NEAR_DUPLICATE_RATIO and ratio > score:
                    duplicate_of, kind, score = slot, "near_duplicate_title", ratio

        if duplicate_of is not None:
            stats[kind] += 1
            kept = articles[duplicate_of]
            if _text_amount(article) > _text_amount(kept):
                # keep the copy with more text, in the first copy's place
                stats["duplicate_log"].append(_log_entry(kind, article, kept, score))
                articles[duplicate_of] = article
            else:
                stats["duplicate_log"].append(_log_entry(kind, kept, article, score))
            # remember this copy's link and title too, so a third copy is also caught
            slot_by_url[url_key] = duplicate_of
            slot_by_title.setdefault(title_key, duplicate_of)
            continue

        slot_by_url[url_key] = len(articles)
        slot_by_title[title_key] = len(articles)
        articles.append(article)

    # Rule 7: order
    for article in articles:
        article["mention_level"] = _mention_level(article, rule)
        article["matched_identities"] = (
            matched_identities(f"{article['title']} {article['description']} {article['content']}",
                               rule.alias_group, aliases)
            if rule is not None and rule.aliases_used else []
        )
    if ranking == "mention":
        articles.sort(key=lambda article: MENTION_ORDER[article["mention_level"]])   # sort keeps ties in order

    # Rule 8: limit, after ordering
    stats["over_limit"] = max(len(articles) - max_articles, 0)
    articles = articles[:max_articles]

    # Rule 9: number for citations
    articles = [{"id": number, **article} for number, article in enumerate(articles, start=1)]
    stats["kept"] = len(articles)
    return articles, stats


def _mention_level(article: dict, rule) -> str | None:
    """Where the query (or a verified alias) first appears: title, description or content."""
    if rule is None:
        return None
    for field_name in ("title", "description"):
        if matches(article[field_name], rule):
            return field_name
    return "content"


def articles_to_text(articles: list[dict]) -> str:
    """
    Build the article block the LLM reads. Each article starts with its
    number, so the LLM can cite it as [1], [2] ...
    """
    blocks = []
    for article in articles:
        lines = [
            f"[{article['id']}] {article['title']}",
            f"Source: {article['source']} | Published: {article['published_at'][:10]}",
            f"Summary: {article['description']}",
        ]
        if article.get("content"):
            lines.append(f"Extract: {article['content']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def save_processed(articles: list[dict], stats: dict, query: str, folder: Path = PROCESSED_DIR) -> Path:
    """Save the clean articles and the cleaning counts as one JSON file."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:40] or "query"
    path = unique_path(folder, f"{stamp}_{slug}")
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"query": query, "stats": stats, "articles": articles}, file, ensure_ascii=False, indent=2)
    return path
