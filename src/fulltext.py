"""
Full article text from the article's own web page.

NewsAPI's free plan gives only the first 200 characters of each article. This
step fetches the article page itself:
    - only when the site's robots.txt allows it (robots.txt is the file in
      which a website says which pages automatic programs may fetch); a site
      without the file allows everything; a file that cannot be read means
      "do not fetch"
    - once per page: every result, success or failure, is kept in a local
      cache, so a site is never asked twice
    - slowly: one page at a time, with a pause between requests
Page junk is removed before the text is taken out of the page:
    - boxes marked as related stories, "also read", "read more",
      recommendations or newsletters (by their class or id in the page code),
      unless the box holds most of the page's text (then it is the article
      itself, whatever its name)
    - short lines and boxes that start with a junk label such as
      "Also Read:" or "Read more:" (the label must be followed by a colon, a
      bar or a dash, so a sentence like "Analysts also read the results" stays)
Then trafilatura takes out the main text, and the same label rule is applied
to any line left over. Everything removed is recorded. A page with less than
MIN_CHARS characters of text is treated as a paywall or a teaser.

The cache (data/fulltext/) is never put in Git: the text belongs to the
publisher and is used only to judge relevance and to summarise.

Commands (no API keys):
    python -m src.fulltext check     fetch and clean the labelled articles, report what full text adds
    python -m src.fulltext verify    paragraph check: real paragraphs kept, in their original order
    python -m src.fulltext sheet     files for full-text labelling: a reading page and a label CSV
"""

import csv
import hashlib
import html as html_module
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from src.config import DATA_DIR, EVAL_DIR, RAW_DIR
from src.evaluate import is_relevant, load_label_set
from src.preprocess import clean_articles, normalize_url
from src.query import load_aliases, matches, parse_query
from src.utils import unique_path

USER_AGENT_NAME = "NewsResearchTool"
USER_AGENT = f"{USER_AGENT_NAME}/0.1 (research project; reads one page at a time)"
FULLTEXT_DIR = DATA_DIR / "fulltext"
MIN_CHARS = 400
DELAY_SECONDS = 1.0
EXTRACTION_METHOD = "page-junk removal v3 (real-paragraph guard, whole junk lines) + trafilatura + junk-line rule"

JUNK_CLASS = re.compile(r"related|also[-_]?read|read[-_]?more|recommend|more[-_]?stories|trending|"
                        r"newsletter|subscribe|outbrain|taboola", re.I)
JUNK_LABEL = re.compile(r"^\s*(also read|read more|read also|must read|you may also like|"
                        r"related (?:stories|news|articles|coverage)|recommended(?: for you| stories| articles)?)"
                        r"\s*(?:[:|\u2013\u2014-]|$)\s*", re.I)
MAX_JUNK_LINE = 300                 # a junk label line is short; long paragraphs are never removed by label
WRAPPER_SHARE = 0.5                 # a box with more than half of the page's text is the article itself
NEVER_REMOVED = {"html", "body", "article", "main"}
BOXES = {"div", "aside", "section", "ul", "ol", "li", "nav", "footer", "header", "figure", "form",
         "table", "p", "h1", "h2", "h3", "h4", "h5", "h6"}   # only boxes are removed by their name
REAL_PARAGRAPH_CHARS = 200          # a real paragraph: at least this long ...
MAX_LINK_SHARE = 0.5                # ... and less than half of its text inside links


def http_get(url: str, timeout: int = 10) -> tuple[int, str]:
    import requests
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    return response.status_code, response.text


def extract_text(html: str) -> str:
    import trafilatura
    return trafilatura.extract(html, include_comments=False, include_tables=False) or ""


# ---------- page junk ----------

def _normal_space(text: str) -> str:
    return " ".join(text.split())


def strip_page_junk(html: str) -> tuple[str, list[str]]:
    """Remove related-story boxes and junk-label lines from the page code. Returns (page, removed texts)."""
    if not html or not html.strip():
        return html, []
    try:
        import lxml.html
        root = lxml.html.fromstring(html)
    except Exception:
        return html, []
    total = len(_normal_space(root.text_content())) or 1

    def has_real_paragraph(element) -> bool:
        for paragraph in [element] + list(element.iterdescendants("p")):
            if not isinstance(paragraph.tag, str) or paragraph.tag.lower() != "p":
                continue
            text = _normal_space(paragraph.text_content())
            linked = sum(len(_normal_space(link.text_content())) for link in paragraph.iterdescendants("a"))
            if len(text) >= REAL_PARAGRAPH_CHARS and linked < MAX_LINK_SHARE * len(text):
                return True
        return False

    def is_junk(element) -> bool:
        if not isinstance(element.tag, str) or element.tag.lower() in NEVER_REMOVED:
            return False
        text = _normal_space(element.text_content())
        names = f"{element.get('class', '')} {element.get('id', '')}"
        if (element.tag.lower() in BOXES and JUNK_CLASS.search(names)
                and len(text) <= total * WRAPPER_SHARE and not has_real_paragraph(element)):
            return True
        return bool(JUNK_LABEL.match(text)) and len(text) <= MAX_JUNK_LINE

    def is_short_link_box(element) -> bool:
        if element is None or not isinstance(element.tag, str):
            return False
        text = _normal_space(element.text_content())
        linked = sum(len(_normal_space(link.text_content())) for link in element.iter("a"))
        return 0 < len(text) <= MAX_JUNK_LINE and linked >= MAX_LINK_SHARE * len(text)

    removed_elements = []
    for element in root.iter():
        if element in removed_elements or any(parent in removed_elements for parent in element.iterancestors()):
            continue
        if is_junk(element):
            removed_elements.append(element)
            label_only = JUNK_LABEL.fullmatch(_normal_space(element.text_content()) or "")
            following = element.getnext()
            if label_only and is_short_link_box(following):
                removed_elements.append(following)        # the headline the label introduces
    removed = [_normal_space(element.text_content()) for element in removed_elements]
    for element in removed_elements:
        element.drop_tree()                      # keeps the text that follows the element
    return lxml.html.tostring(root, encoding="unicode"), [text for text in removed if text]


def strip_text_junk(text: str) -> tuple[str, list[str]]:
    """Remove short lines that start with a junk label. Returns (text, removed lines)."""
    kept, removed = [], []
    for line in text.split("\n"):
        if JUNK_LABEL.match(line) and len(line.strip()) <= MAX_JUNK_LINE:
            removed.append(line.strip())
        else:
            kept.append(line)
    return "\n".join(kept), removed


# ---------- robots.txt ----------

class RobotsCheck:
    """Reads each site's robots.txt once and answers whether a page may be fetched."""

    def __init__(self, http_get=http_get):
        self.http_get = http_get
        self.sites: dict[str, RobotFileParser | bool] = {}

    def _rules(self, site: str):
        if site not in self.sites:
            try:
                status, text = self.http_get(f"{site}/robots.txt", timeout=10)
            except Exception:
                self.sites[site] = False                 # cannot be read: do not fetch
                return self.sites[site]
            if status in (404, 410):
                self.sites[site] = True                  # no file: everything allowed
            elif status != 200:
                self.sites[site] = False
            else:
                parser = RobotFileParser()
                parser.parse(text.splitlines())
                self.sites[site] = parser
        return self.sites[site]

    def allows(self, url: str) -> bool:
        parts = urlsplit(url)
        rules = self._rules(f"{parts.scheme}://{parts.netloc}")
        if isinstance(rules, bool):
            return rules
        return rules.can_fetch(USER_AGENT_NAME, url)


# ---------- fetching ----------

class FullTextFetcher:
    def __init__(self, http_get=http_get, extract=extract_text, cache_dir: Path = FULLTEXT_DIR,
                 delay_seconds: float = DELAY_SECONDS, min_chars: int = MIN_CHARS, robots: RobotsCheck | None = None):
        self.http_get, self.extract = http_get, extract
        self.cache_dir = Path(cache_dir)
        self.delay_seconds, self.min_chars = delay_seconds, min_chars
        self.robots = robots or RobotsCheck(http_get)

    def _cache_path(self, url: str, suffix: str = ".json") -> Path:
        return self.cache_dir / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}{suffix}"

    def _save(self, result: dict, html: str | None = None) -> dict:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(result["url"]).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        if html is not None:
            self._cache_path(result["url"], ".html").write_text(html, encoding="utf-8")
        return result

    def cached_html(self, url: str) -> str | None:
        path = self._cache_path(url, ".html")
        return path.read_text(encoding="utf-8") if path.exists() else None

    def fetch(self, url: str) -> dict:
        cached_path = self._cache_path(url)
        old = json.loads(cached_path.read_text(encoding="utf-8")) if cached_path.exists() else None
        if old is not None and old.get("extraction_method") == EXTRACTION_METHOD:
            return old
        if old is not None and old.get("extraction_method") and old.get("status") in ("ok", "too_short"):
            html = self.cached_html(url)
            if html is not None:
                # an older cleaning version: clean again from the saved page, without the web
                result = self._process(self._empty(url) | {"http_status": old.get("http_status")}, html)
                if old.get("status") == "ok" and result["status"] != "ok" and old.get("text"):
                    return self._save({**old, "extraction_method": "trafilatura (no page cleaning)",
                                       "html_cleaning": "unavailable", "refetch_status": result["status"]})
                return self._save(result)
        if old is not None and old.get("extraction_method") and old.get("status") not in ("ok", "too_short"):
            return old          # blocked, http error or network error: not asked again

        result = self._fetch_new(url)
        if old is not None and old.get("status") == "ok" and result["status"] != "ok":
            # the page can no longer be fetched: keep the usable old text and mark it as not cleaned
            return self._save({**old, "extraction_method": "trafilatura (no page cleaning)",
                               "html_cleaning": "unavailable", "refetch_status": result["status"]})
        return result

    @staticmethod
    def _empty(url: str) -> dict:
        return {"url": url, "status": "", "text": "", "chars": 0, "http_status": None,
                "extraction_method": EXTRACTION_METHOD, "html_cleaning": "not_done",
                "original_length": 0, "cleaned_length": 0, "boilerplate_removed": False,
                "removed_texts": [], "content_quality": ""}

    def _process(self, result: dict, html: str) -> dict:
        """Clean the page, take out the text and fill in the record."""
        original = (self.extract(html) or "").strip()
        cleaned_html, removed = strip_page_junk(html)
        text, removed_lines = strip_text_junk((self.extract(cleaned_html) or "").strip())
        text = text.strip()
        removed += removed_lines
        quality = "ok" if len(text) >= self.min_chars else "too_short"
        result.update(status=quality, html_cleaning="done", original_length=len(original),
                      cleaned_length=len(text), boilerplate_removed=bool(removed), removed_texts=removed,
                      content_quality=quality, chars=len(text), text=text if quality == "ok" else "")
        return result

    def _fetch_new(self, url: str) -> dict:
        result = self._empty(url)
        if not self.robots.allows(url):
            return self._save({**result, "status": "blocked_by_robots"})
        try:
            status, html = self.http_get(url, timeout=15)
        except Exception as error:
            return self._save({**result, "status": "error", "error": f"{type(error).__name__}: {error}"})
        finally:
            time.sleep(self.delay_seconds)
        result["http_status"] = status
        if status != 200:
            return self._save({**result, "status": "http_error"})

        return self._save(self._process(result, html), html)


# ---------- paragraph check ----------

REAL_TEXT_CHARS = 80        # a missing paragraph this long is probably real text, not a junk label


def _paragraphs(text: str) -> list[str]:
    return [line.strip() for line in (text or "").split("\n") if line.strip()]


CHANGED_SIMILARITY = 0.6     # a paragraph this similar to a missing one is the same paragraph, changed


def _cut_words(before: str, after: str) -> tuple[str, str]:
    """Words taken out of, and words put into, a paragraph."""
    from difflib import SequenceMatcher
    old, new = before.split(), after.split()
    cut, inserted = [], []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag in ("delete", "replace"):
            cut.append(" ".join(old[i1:i2]))
        if tag in ("insert", "replace"):
            inserted.append(" ".join(new[j1:j2]))
    return " … ".join(cut), " … ".join(inserted)


def paragraph_check(original: str, cleaned: str) -> dict:
    """Compare the text before and after cleaning, paragraph by paragraph."""
    from difflib import SequenceMatcher
    before, after = _paragraphs(original), _paragraphs(cleaned)
    missing = [p for p in before if p not in after]
    added = [p for p in after if p not in before]
    changed = []
    for paragraph in list(added):
        best, best_ratio = None, 0.0
        for old in missing:
            ratio = SequenceMatcher(None, old.split(), paragraph.split(), autojunk=False).ratio()
            if ratio > best_ratio:
                best, best_ratio = old, ratio
        if best is not None and best_ratio >= CHANGED_SIMILARITY:
            cut, inserted = _cut_words(best, paragraph)
            changed.append({"before": best, "after": paragraph, "cut": cut, "inserted": inserted})
            missing.remove(best)
            added.remove(paragraph)
    position, order_kept = -1, True
    for paragraph in after:
        try:
            found = before.index(paragraph, position + 1)
        except ValueError:
            if paragraph in before:
                order_kept = False
            continue
        position = found
    return {
        "kept": sum(1 for p in after if p in before),
        "missing": missing,
        "missing_long": [p for p in missing if len(p) >= REAL_TEXT_CHARS],
        "order_kept": order_kept,
        "added": added,
        "changed": changed,
    }


# ---------- files for full-text labelling ----------

# ---------- is the fetched text the article? ----------
#
# An aggregator page can show other headlines or a market summary instead of
# the article. The fetched text is compared with what NewsAPI says about the
# article. Signals (True / False, or None when they cannot be checked):
#     description_missing - the description's opening sentence is not in the text
#     title_low           - few of the title's content words are in the text
#     company_missing     - the title or description names the company, the text does not
#     looks_like_list     - most lines are short, like headlines or links
# The short text replaces the full text only when the content signals agree:
#     description known:   description_missing and title_low
#     description unknown: title_low and company_missing and looks_like_list
# A missing company name alone is never enough: an industry article can
# discuss the sector without naming the company.
#
# A page can also hold only a teaser (an aggregator link page) or a login box.
#     teaser              - the page holds less text than the short NewsAPI text
# A teaser page always falls back to the short text.

TITLE_OVERLAP_LOW = 0.3             # less than this share of title words in the text is low
DESCRIPTION_FOUND = 0.6             # at least this share of the description's words: found
MIN_DESCRIPTION_WORDS = 4           # fewer content words: the description cannot be checked
SHORT_LINE_CHARS = 80
LIST_SHORT_SHARE = 0.6              # more than this share of short lines looks like a list
SIGNAL_ORDER = ("description_missing", "title_low", "company_missing", "looks_like_list")
SHORT_CONTENT_CHARS = 200           # the short NewsAPI text keeps this much of the content
STOP_WORDS = {"the", "and", "for", "with", "from", "that", "this", "its", "his", "her", "their", "has", "have",
              "had", "was", "were", "are", "but", "not", "after", "before", "during", "among", "into", "over",
              "about", "than", "per", "who", "what", "when", "will", "would", "can", "could", "new", "says",
              "said", "all", "out", "off", "one", "two", "now", "how", "why", "more", "most", "you", "our"}


def _content_words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [w for w in words if (len(w) >= 3 or w.isdigit()) and w not in STOP_WORDS]


def _share_found(words: list[str], text_words: set[str]) -> float:
    return sum(1 for w in words if w in text_words) / len(words) if words else 0.0


def _description_opening(description: str) -> list[str]:
    """Content words of the description's first sentence; a cut-off ending ("…", "[+123 chars]") is removed."""
    text = re.sub(r"\[\+\d+ chars\]\s*$", "", (description or "").strip())
    cut_off = bool(re.search(r"(…|\.\.\.)\s*$", text))
    text = re.sub(r"(…|\.\.\.)\s*$", "", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    first = _content_words(sentences[0])
    if cut_off and len(sentences) == 1:
        first = first[:-1]                          # the sentence is cut off: its last word may be cut in half
    if len(first) >= MIN_DESCRIPTION_WORDS:
        return first
    words = _content_words(text)
    return words[:-1] if cut_off else words


def short_newsapi_text(title: str, description: str, content: str) -> str:
    """The short NewsAPI text: title, description and the first 200 characters of the content."""
    return "\n".join([title or "", description or "", (content or "")[:SHORT_CONTENT_CHARS]])


def check_full_text(title: str, description: str, full_text: str, company: str, aliases: dict | None,
                    content: str = "") -> dict:
    text_words = set(_content_words(full_text))
    title_words = _content_words(title)
    overlap = _share_found(title_words, text_words)

    opening = _description_opening(description)
    description_missing = (None if len(opening) < MIN_DESCRIPTION_WORDS
                           else _share_found(opening, text_words) < DESCRIPTION_FOUND)

    rule = parse_query(company, aliases=aliases)
    named_outside = rule.supported and matches(f"{title or ''} {description or ''}", rule)
    company_missing = (not matches(full_text or "", rule)) if named_outside else None

    lines = [line.strip() for line in (full_text or "").split("\n") if line.strip()]
    short = sum(1 for line in lines if len(line) < SHORT_LINE_CHARS)
    looks_like_list = bool(lines) and short / len(lines) > LIST_SHORT_SHARE

    signals = {"description_missing": description_missing, "title_low": overlap < TITLE_OVERLAP_LOW,
               "title_overlap": round(overlap, 3), "company_missing": company_missing,
               "looks_like_list": looks_like_list}
    if description_missing is None:
        wrong_page = signals["title_low"] and company_missing is True and looks_like_list
    else:
        wrong_page = description_missing and signals["title_low"]
    teaser = len((full_text or "").strip()) < len(short_newsapi_text(title, description, content).strip())
    signals["teaser"] = teaser
    reasons = (["teaser"] if teaser else []) + ([name for name in SIGNAL_ORDER if signals[name] is True]
                                                if wrong_page else [])
    return {"use_full_text": not (wrong_page or teaser), "reasons": reasons, "signals": signals}


LABEL_COLUMNS = ["row", "article_id", "title", "url", "label", "unsure", "note"]
SEED = 42


def article_id(url: str) -> str:
    """A stable id for an article: the same tidy web address always gives the same id."""
    return "A" + hashlib.sha1(normalize_url(url).encode("utf-8")).hexdigest()[:10]


def labelling_items(meta: list[dict], results: dict[str, dict], seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Rows for the label CSV and items for the reading page, in the same shuffled order."""
    usable = [m for m in meta if results.get(m["url"], {}).get("status") == "ok"]
    random.Random(seed).shuffle(usable)
    csv_rows, page_items = [], []
    for number, m in enumerate(usable, start=1):
        result = results[m["url"]]
        identifier = article_id(m["url"])
        csv_rows.append({"row": number, "article_id": identifier, "title": m["title"], "url": m["url"],
                         "label": "", "unsure": "", "note": ""})
        page_items.append({
            "row": number, "article_id": identifier, "url": m["url"], "title": m["title"],
            "source": m.get("source", ""), "published": m.get("published", ""),
            "status": "complete" if result.get("html_cleaning") == "done"
                      else "older text, page cleaning unavailable",
            "paragraphs": _paragraphs(result.get("text", "")),
        })
    return csv_rows, page_items


def labelling_page(items: list[dict]) -> str:
    e = html_module.escape
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Full-text labelling</title>",
             "<style>body{font-family:Georgia,serif;max-width:46rem;margin:2rem auto;padding:0 1rem;line-height:1.55}"
             "section{border-top:2px solid #444;padding-top:1rem;margin-top:2.5rem}"
             ".meta{font-family:Arial,sans-serif;font-size:.85rem;color:#555}"
             ".status{font-family:Arial,sans-serif;font-size:.85rem;font-weight:bold}</style></head><body>",
             "<h1>Full-text labelling</h1><p class='meta'>Read each article here. Write the label in the CSV row "
             "with the same row number and article id. Follow LABELLING_GUIDE.md (version 3).</p>"]
    for item in items:
        parts.append(
            '<section id="' + e(item["article_id"]) + '">'
            f"<h2>Row {item['row']} — {e(item['title'])}</h2>"
            f"<p class='meta'>Article id: {e(item['article_id'])} | {e(item['source'])} | {e(item['published'])}</p>"
            f"<p class='status'>Text: {e(item['status'])}</p>"
            + "".join(f"<p>{e(paragraph)}</p>" for paragraph in item["paragraphs"])
            + "</section>")
    parts.append("</body></html>")
    return "".join(parts)


def write_labelling_files(csv_rows: list[dict], items: list[dict], eval_dir: Path, page_dir: Path,
                          stamp: str) -> tuple[Path, Path]:
    csv_path = unique_path(Path(eval_dir), f"fulltext_labels_v3_{stamp}", suffix=".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)
    page_path = unique_path(Path(page_dir), f"labelling_view_{stamp}", suffix=".html")
    page_path.write_text(labelling_page(items), encoding="utf-8")
    return csv_path, page_path


# ---------- the feasibility report ----------

def feasibility_report(articles: list[dict], labels: dict[str, str], results: dict[str, dict],
                       missed: set[str], company: str, aliases: dict | None) -> dict:
    labels = {normalize_url(url): label for url, label in labels.items()}
    results = {normalize_url(url): result for url, result in results.items()}
    missed = {normalize_url(url) for url in missed}
    rule = parse_query(company, aliases=aliases)

    def names_company(key):
        result = results.get(key, {})
        return result.get("status") == "ok" and matches(result.get("text", ""), rule)

    keys = [normalize_url(a["url"]) for a in articles]
    relevant = [k for k in keys if is_relevant(labels[k])]
    not_relevant = [k for k in keys if not is_relevant(labels[k])]
    missed_keys = [k for k in keys if k in missed]
    fetched = lambda group: sum(1 for k in group if results.get(k, {}).get("status") == "ok")
    return {
        "status_counts": dict(Counter(results[k]["status"] for k in keys if k in results)),
        "relevant": {"total": len(relevant), "fetched": fetched(relevant)},
        "missed": {"total": len(missed_keys), "fetched": fetched(missed_keys),
                   "names_company_in_full_text": sum(1 for k in missed_keys if names_company(k))},
        "not_relevant": {"total": len(not_relevant), "fetched": fetched(not_relevant),
                         "names_company_in_full_text": sum(1 for k in not_relevant if names_company(k))},
        "missed_details": [{"url": a["url"], "title": a["title"],
                            "status": results.get(normalize_url(a["url"]), {}).get("status", ""),
                            "names_company": names_company(normalize_url(a["url"]))}
                           for a in articles if normalize_url(a["url"]) in missed],
    }


# ---------- running it ----------

def _labelled_articles(labels: dict[str, str], aliases: dict | None, company: str) -> tuple[list[dict], set[str]]:
    """Every labelled article once (sale listings left out), and the relevant ones the filter missed."""
    articles, covered = {}, set()
    for path in sorted(RAW_DIR.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8")).get("articles", [])
        kept, stats = clean_articles(raw, max_articles=max(len(raw), 1), query=company, aliases=aliases,
                                     remove_junk=True, drop_listings="both")
        kept_keys = {normalize_url(a["url"]) for a in kept}
        covered |= kept_keys
        for entry in stats["duplicate_log"]:
            if normalize_url(entry["kept"]["url"]) in kept_keys:
                covered.add(normalize_url(entry["removed"]["url"]))
        listings, _ = clean_articles(raw, max_articles=max(len(raw), 1), drop_listings="off", remove_junk=True)
        not_listing, _ = clean_articles(raw, max_articles=max(len(raw), 1), drop_listings="both", remove_junk=True)
        allowed = {normalize_url(a["url"]) for a in not_listing}
        for article in raw:
            key = normalize_url(article.get("url") or "")
            if key in labels and key in allowed and key not in articles:
                articles[key] = {"url": article["url"], "title": " ".join((article.get("title") or "").split()),
                                 "source": (article.get("source") or {}).get("name") or "",
                                 "published": article.get("publishedAt") or ""}
    missed = {k for k in articles if is_relevant(labels[k]) and k not in covered}
    return list(articles.values()), missed


def _check_command(company: str = "Tata Motors") -> None:
    from src.retrieval_check import score_label_files
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    labels = load_label_set(score_label_files() + sorted(EVAL_DIR.glob("dev_*_fresh_*.csv")))
    labels = {normalize_url(url): label for url, label in labels.items()}
    articles, missed = _labelled_articles(labels, aliases, company)
    fetcher = FullTextFetcher()
    results = {}
    for number, article in enumerate(articles, start=1):
        results[article["url"]] = fetcher.fetch(article["url"])
        print(f"{number:>3}/{len(articles)}  {results[article['url']]['status']:<18} {article['title'][:70]}")
    report = feasibility_report(articles, labels, results, missed, company, aliases)
    cleaned = [r for r in results.values() if r.get("boilerplate_removed")]
    print(f"\nPage junk removed from {len(cleaned)} of {len(results)} articles. Removed text, for checking:")
    for result in cleaned:
        title = next(a["title"] for a in articles if a["url"] == result["url"])
        print(f"   {title[:60]}")
        for text in result["removed_texts"]:
            print(f"      - {text[:110]}")
    unavailable = [r for r in results.values() if r.get("html_cleaning") == "unavailable"]
    if unavailable:
        print(f"Page cleaning unavailable (old text kept): {len(unavailable)} articles")
    print(f"\nFetch results: {report['status_counts']}")
    print(f"Relevant articles: {report['relevant']['fetched']} of {report['relevant']['total']} fetched")
    print(f"Relevant articles the filter missed: {report['missed']['total']}, fetched {report['missed']['fetched']}, "
          f"full text names the company: {report['missed']['names_company_in_full_text']}")
    print(f"Not relevant articles: {report['not_relevant']['total']}, fetched {report['not_relevant']['fetched']}, "
          f"full text names the company: {report['not_relevant']['names_company_in_full_text']} "
          f"(possible new wrong keeps)")
    print("\nMissed articles:")
    for item in report["missed_details"]:
        print(f"   {item['status']:<18} names company: {'yes' if item['names_company'] else 'no ':<4} {item['title'][:70]}")
    out = EVAL_DIR / "fulltext_feasibility.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved: {out}")


def _cached_results(articles: list[dict]) -> tuple[FullTextFetcher, dict[str, dict]]:
    fetcher = FullTextFetcher()
    return fetcher, {a["url"]: fetcher.fetch(a["url"]) for a in articles}


def _verify_command(company: str = "Tata Motors") -> None:
    from src.retrieval_check import score_label_files
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    labels = load_label_set(score_label_files() + sorted(EVAL_DIR.glob("dev_*_fresh_*.csv")))
    labels = {normalize_url(url): label for url, label in labels.items()}
    articles, _ = _labelled_articles(labels, aliases, company)
    fetcher, results = _cached_results(articles)
    problems = 0
    for article in articles:
        result, page = results[article["url"]], fetcher.cached_html(article["url"])
        if result.get("status") != "ok" or page is None or result.get("html_cleaning") != "done":
            continue
        check = paragraph_check(extract_text(page), result["text"])
        real_cuts = [c for c in check["changed"] if c["inserted"] or not JUNK_LABEL.match(c["cut"])]
        long_lines = [p for p in check["missing_long"] if not JUNK_LABEL.match(p)]
        flagged = long_lines or check["added"] or real_cuts or not check["order_kept"]
        problems += bool(flagged)
        print(f"{'CHECK' if flagged else 'ok   '} kept {check['kept']:>3} | removed {len(check['missing']):>2} | "
              f"order {'kept' if check['order_kept'] else 'CHANGED'} | {article['title'][:60]}")
        for paragraph in check["missing_long"]:
            kind = "junk line" if JUNK_LABEL.match(paragraph) else "REMOVED LONG PARAGRAPH"
            print(f"        {kind}: {paragraph[:120]}")
        for change in check["changed"]:
            print(f"        words cut from a paragraph: {change['cut'][:120]}"
                  + (f"  | WORDS ADDED: {change['inserted'][:80]}" if change["inserted"] else ""))
        for paragraph in check["added"]:
            print(f"        text not in the original: {paragraph[:120]}")
    print(f"\nArticles to look at: {problems}")


def _sheet_command(company: str = "Tata Motors") -> None:
    from datetime import datetime
    from src.retrieval_check import score_label_files
    aliases = load_aliases(DATA_DIR / "company_aliases.json")
    labels = load_label_set(score_label_files() + sorted(EVAL_DIR.glob("dev_*_fresh_*.csv")))
    labels = {normalize_url(url): label for url, label in labels.items()}
    articles, _ = _labelled_articles(labels, aliases, company)
    _, results = _cached_results(articles)
    csv_rows, items = labelling_items(articles, results)
    csv_path, page_path = write_labelling_files(csv_rows, items, EVAL_DIR, FULLTEXT_DIR,
                                                datetime.now().strftime("%Y%m%d"))
    print(f"{len(csv_rows)} articles to label")
    print(f"Label sheet:  {csv_path}")
    print(f"Reading page: {page_path}  (open it in your browser)")


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        _check_command()
    elif sys.argv[1:] == ["verify"]:
        _verify_command()
    elif sys.argv[1:] == ["sheet"]:
        _sheet_command()
    else:
        print(__doc__)
