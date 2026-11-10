"""
What the Streamlit app does, step by step. app.py only draws; every step and rule is here, tested with a fake
screen and a fake pipeline (tests/test_app_flow.py), and the pipeline itself is src/research.py.

A context is one search: company, window and time, with its lists, the full-text results, the optional model
review and the summary. Get News with the same company and window within FRESH_MINUTES shows the same context
again (NewsAPI's free plan allows 100 requests a day, and each search uses 2); another company or window is a
new context, so nothing from one search is shown with another.

The order on screen: the provisional list first, then the progress of the full-text loading, then the final
default list. A summary keeps the fingerprint of the exact selection it is made from: after a change of order
it is shown as belonging to the earlier selection, and its article numbers are no longer linked to the cards.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

FRESH_MINUTES = 30
DAY_CHOICES = (3, 7, 14, 28)        # NewsAPI's free plan searches at most one month back; 3 because the newest day
DEFAULT_DAYS = 7                    # is nearly empty (articles appear about a day late); 7 is the tested window
SOURCE_NOTE = ("NewsAPI's free plan shows articles about one day after publication and searches at most one month "
               "back, so the newest day holds few or no articles.")
TESTED_COMPANIES = ("tata motors",)
UNTESTED_NOTE = ("The ranking is tested on Tata Motors. For other companies it runs the same way, but its results "
                 "are not measured.")
SHORT_TEXT_REASONS = {"ok": "the page's text did not pass the check",
                      "not_fetched_in_time": "the page was not loaded within the time limit",
                      "blocked_by_robots": "the publisher does not allow automatic reading"}
COULD_NOT_LOAD = "the page could not be loaded"
TEASER = "the page shows only a teaser of the article"
WRONG_PAGE = "the link leads to a page that does not hold this article"


class FlowError(Exception):
    pass


# ---------- one search ----------

def context_key(company: str, days: int) -> str:
    return f"{' '.join(company.split()).lower()}|{days}"


def active(state: dict) -> dict | None:
    return state.get("contexts", {}).get(state.get("active"))


def md(text) -> str:
    """Text from the model or a publisher, shown as written: $ would start a formula, * _ [ ] ` # change format."""
    return re.sub(r"([\\$*_\[\]`#<>|~])", r"\\\1", str(text))


def get_news(company: str, days: int, screen, pipeline, state: dict, now: datetime | None = None,
             refresh: bool = False) -> dict | None:
    now = now or datetime.now(timezone.utc)
    company = " ".join(company.split())
    if not company:
        screen.notice("warning", "Enter a company name.")
        return None
    key = context_key(company, days)
    contexts = state.setdefault("contexts", {})
    known = contexts.get(key)
    if known and not refresh and (now - known["searched_at"]).total_seconds() < FRESH_MINUTES * 60:
        state["active"] = key
        screen.notice("info", f"Showing the search made at {known['searched_at']:%H:%M} UTC "
                              f"(searches are reused for {FRESH_MINUTES} minutes).")
        screen.final(known)
        return known

    found = pipeline.search(company, days)
    if found.get("status") != "ok":
        screen.notice("error", found.get("error") or "The search did not return articles.")
        return None
    for note in found.get("notes", []):
        screen.notice("info", note)
    if company.lower() not in TESTED_COMPANIES:
        screen.notice("info", UNTESTED_NOTE)
    first = pipeline.provisional(found, company)
    screen.provisional(first)
    enriched = pipeline.enrich(found, screen.progress)
    last = pipeline.final(found, enriched, company)
    context = {"key": key, "company": company, "days": days, "searched_at": now, "found": found,
               "first": first, "result": last, "default_result": last, "enriched": enriched,
               "changes": pipeline.changes(first, last), "summary": None}
    contexts[key] = context
    state["active"] = key
    screen.final(context)
    return context


# ---------- the summary and its selection ----------

def selection_fingerprint(result: dict) -> str:
    text = json.dumps([result.get("ordering"), [c["key"] for c in result["selected"]]])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_summary(context: dict, pipeline) -> dict:
    outcome = pipeline.summarize(context["result"], context["company"])
    context["summary"] = {"fingerprint": selection_fingerprint(context["result"]), "outcome": outcome,
                          "ordering": context["result"].get("ordering"),
                          "made_at": datetime.now(timezone.utc)}
    return context["summary"]


def summary_view(context: dict) -> dict:
    summary = context.get("summary")
    if not summary:
        return {"state": "none", "outcome": None}
    same = summary["fingerprint"] == selection_fingerprint(context["result"])
    return {"state": "current" if same else "earlier_selection", "outcome": summary["outcome"]}


def citation_cards(outcome: dict, result: dict) -> dict[int, str]:
    """{article number in the summary: card key}, only when the summary's articles are the selected ones."""
    selected = result["selected"]
    urls = [h.get("url") for h in outcome.get("headlines") or []]
    if urls != [c.get("url") for c in selected]:
        return {}
    return {number: card["key"] for number, card in enumerate(selected, start=1)}


# ---------- review and the experimental order ----------

def run_review(context: dict, pipeline) -> dict:
    context["result"] = pipeline.review(context["result"], context["company"])
    return context["result"]


def experimental(context: dict, pipeline) -> dict:
    if not context["result"].get("reviewed"):
        raise FlowError("Run the model review first: the experimental order uses its answers.")
    context["result"] = pipeline.apply(context["result"])
    return context["result"]


def default(context: dict, pipeline) -> dict:
    if context["result"].get("default_order"):
        context["result"] = pipeline.back(context["result"])
    return context["result"]


# ---------- what the summary section shows ----------

STATUS_LINES = {"ready": ("Summary ready · Automated checks passed", "success"),
                "ready_with_removed_points": ("Summary ready · Automated checks passed after removing some points",
                                              "warning"),
                "model_unavailable": ("No summary: the model is not available right now", "error"),
                "failed_checks": ("No summary: the answer does not pass the checks", "error")}
NO_FINDINGS = ("No summary: no key finding passes the checks", "error")


def summary_status_line(outcome: dict) -> tuple[str, str]:
    """The status shown above the summary; "ready" only when at least one key finding is there."""
    status = outcome.get("summary_status")
    if status in ("ready", "ready_with_removed_points") and not (outcome.get("summary") or {}).get("key_points"):
        return NO_FINDINGS
    return STATUS_LINES.get(status, STATUS_LINES["failed_checks"])


NO_INDUSTRY = "The selected articles do not report on the wider industry."


def industry_note(outcome: dict) -> str:
    """The note under an empty industry section (prompt version 4 leaves it empty when no article reports on the
    wider industry); none when the section has points or when the checks removed industry points."""
    summary = outcome.get("summary") or {}
    if summary.get("industry_context"):
        return ""
    if any(item.get("section") == "industry_context" for item in outcome.get("removed") or []):
        return ""
    return NO_INDUSTRY


def evidence_note(context: dict) -> str | None:
    """A note when the summary's evidence is limited to the short descriptions."""
    selected = context["result"]["selected"]
    short = sum(1 for card in selected if not card.get("text"))
    if not short:
        return None
    if short == len(selected):
        return ("This summary relies on short article descriptions: the full text of the selected articles is not "
                "available.")
    verb = "relies" if short == 1 else "rely"
    return f"{short} of the {len(selected)} selected articles {verb} on {'its' if short == 1 else 'their'} short description only."


def not_used(outcome: dict) -> list[dict]:
    """The selected articles the summary leaves out, with the model's reason."""
    titles = {n: h.get("title", "") for n, h in enumerate(outcome.get("headlines") or [], start=1)}
    return [{"number": item["id"], "title": titles.get(item["id"], ""), "reason": item.get("reason", "")}
            for item in (outcome.get("summary") or {}).get("not_used", [])]


def overview(context: dict) -> dict:
    ranked = context["result"]["ranked"]
    full = sum(1 for card in ranked if card.get("text"))
    return {"found": len(context["found"].get("articles", [])), "kept": len(ranked),
            "selected": len(context["result"]["selected"]), "full_text": full, "short_text": len(ranked) - full}


def window_text(found: dict, searched_at: datetime) -> str:
    try:
        start = datetime.fromisoformat(str((found.get("window") or {})["from"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return "dates not given"
    same_year = start.year == searched_at.year
    return (f"{start.day} {start:%b}{'' if same_year else f' {start.year}'} to "
            f"{searched_at.day} {searched_at:%b %Y}")


# ---------- what each card shows ----------

def evidence(card: dict, enriched: dict) -> dict:
    if card.get("text"):
        return {"kind": "full_text", "reason": None}
    status = (enriched.get("status") or {}).get(card.get("url"))
    reasons = (enriched.get("check_reasons") or {}).get(card.get("url")) or []
    if status == "ok" and reasons:
        return {"kind": "short_text", "reason": TEASER if "teaser" in reasons else WRONG_PAGE}
    return {"kind": "short_text", "reason": SHORT_TEXT_REASONS.get(status, COULD_NOT_LOAD)}


AGGREGATORS = {"biztoc.com": "Biztoc"}     # sites that post a short entry pointing to the original article
BARE_ADDRESS = re.compile(r"^(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+/\S+$", re.I)


def _aggregator(url: str) -> str | None:
    host = (url or "").split("/")[2].lower() if (url or "").count("/") >= 2 else ""
    host = host[4:] if host.startswith("www.") else host
    return AGGREGATORS.get(host)


def original_link(url: str, page_text: str) -> str | None:
    """The original article's address in an aggregator's entry (headline, then the bare address); None when the
    page is not an aggregator's or holds no such address. Nothing is guessed."""
    if not _aggregator(url):
        return None
    lines = [line.strip() for line in (page_text or "").split("\n") if line.strip()]
    if len(lines) >= 2 and BARE_ADDRESS.match(lines[1]):
        return "https://" + lines[1]
    return None


def web_link(url) -> bool:
    """True only for an http or https address: a title is never linked to any other kind (for example javascript:)."""
    return isinstance(url, str) and url.strip().lower().startswith(("http://", "https://"))


def card_link(card: dict, enriched: dict) -> dict:
    """Where the card's title links: the original article for an aggregator's entry when its address is known.
    An address that is not an http or https link is never linked (url None)."""
    if not web_link(card.get("url")):
        return {"url": None, "via": None, "note": "The article's address is not a web link, so the title is not linked."}
    via = _aggregator(card.get("url"))
    if not via:
        return {"url": card.get("url"), "via": None, "note": None}
    original = original_link(card["url"], (enriched.get("texts") or {}).get(card["url"], ""))
    if original:
        return {"url": original, "via": via, "note": None}
    return {"url": card.get("url"), "via": via, "note": f"{via} is an aggregator; its link may not open the article."}


def take_pending(state) -> str | None:
    """The model step a button asks for: it runs once, in view, on the next drawing of the page."""
    return state.pop("pending", None)


def pages(total: int, size: int) -> list[tuple[int, int]]:
    return [(start, min(start + size, total)) for start in range(0, total, size)]


def readable_date(stamp: str | None) -> str | None:
    """"2026-11-04T10:05:00Z" -> "4 Nov 2026, 10:05 UTC"; None when there is no valid date."""
    try:
        when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    return f"{when.day} {when:%b %Y, %H:%M} UTC"


def published_dates(found: dict) -> dict[str, str]:
    """{link: readable date} from the search results (the ranked cards do not carry the date)."""
    dates = {}
    for article in found.get("articles", []):
        readable = readable_date(article.get("publishedAt"))
        if readable:
            dates[article.get("url")] = readable
    return dates


# ---------- the about page ----------

def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        return None


def _transfer_text(transfer: dict) -> str:
    conditions = transfer["ranking"].get("conditions", {})
    met, failed = sum(bool(v) for v in conditions.values()), [k for k, v in conditions.items() if not v]
    said = {"no_direct_event_lost": "a direct event is lost", "event_recall_not_lower": "event recall is lower",
            "precision_not_lower": "precision is lower"}
    ranking = (f"ranking: {met} of {len(conditions)} conditions met"
               + (f" ({'; '.join(said.get(name, name.replace('_', ' ')) for name in failed)})" if failed else ""))
    summary = "summary pass levels met" if transfer["summary"]["passed"] else "summary pass levels not met"
    return f"{ranking[0].upper()}{ranking[1:]}; {summary}."


WEEK2_FOLDERS = {"Tata Motors": "week2_tata_2026-10-31_11-06",
                 "Reliance Industries": "week2_reliance_2026-11-02_11-08"}


def _times(count: int, total: int) -> str:
    return "both times" if count == total == 2 else f"in {count} of {total} weeks"


def _ranking_update_text(variants: dict, week2_rankings: list[dict | None]) -> str:
    chosen, passing = variants["choice"]["chosen"], variants["choice"].get("passing", [])
    which = "the only variant" if passing == [chosen] else f"one of {len(passing)} variants"
    text = f"Default {chosen}+S1: {which} that passes on all three development days."
    if any(record is None for record in week2_rankings):
        return text + " The fresh-week test is pending."
    total = len(week2_rankings)
    beats = sum(bool(r["second"]["passed"]) for r in week2_rankings)
    fails = sum(not r["primary"]["passed"] for r in week2_rankings)
    return (text + f" On two fresh weeks it beats O1+S1 {_times(beats, total)} and fails against the retrieval "
                   f"order {_times(fails, total)}.")


def _summary_design_text(freeze: dict, round3: dict | None) -> str:
    ks = ", ".join(map(str, freeze["ks"]))
    text = f"{freeze['design'].capitalize()} summary design frozen for k = {ks}"
    if round3 is None:
        return text + "; the three-run comparison of the prompts (Round 3) is pending."
    chosen = round3["choice"]["prompt"]
    if str(chosen) == str(freeze.get("settings", {}).get("prompt_version")):
        return text + f"; Prompt {chosen} kept after the three-run comparison with Prompt 6 (Round 3)."
    return text + f"; Round 3 chose Prompt {chosen}."


def _week2_summary_text(scores: dict[str, dict]) -> str:
    parts = []
    for company, score in scores.items():
        values = score["summary"]["values"]
        if score["summary"]["passed"]:
            parts.append(f"{company} passed")
        elif not score["summary"]["checks"].get("event_coverage", True):
            parts.append(f"{company} not passed (events {values['events']['covered']} of "
                         f"{values['events']['total']})")
        else:
            parts.append(f"{company} not passed")
    return "Summary test: " + "; ".join(parts) + "."


def about_status(eval_dir) -> list[dict]:
    """Each stage's result as the project records hold it; a missing record is pending."""
    eval_dir = Path(eval_dir)
    fresh = _json(eval_dir / "fresh_2026-10-24_30" / "fresh_ranking_record.json")
    freeze = _json(eval_dir / "summary_freeze.json")
    transfer = _json(eval_dir / "transfer_reliance_2026-10-26_11-01" / "transfer_score.json")
    variants = _json(eval_dir / "ranking_variants_dev_record.json")
    round3 = _json(eval_dir / "summary_v2_r3_judging_score.json")
    week2_rankings = [_json(eval_dir / folder / "week2_ranking.json") for folder in WEEK2_FOLDERS.values()]
    week2_scores = {company: _json(eval_dir / folder / "week2_summary_score.json")
                    for company, folder in WEEK2_FOLDERS.items()}
    stages = [{"stage": "ranking", "title": "Ranking (Stage 3)",
               "status": "done" if fresh else "pending",
               "result": (f"{fresh['default']} chosen and confirmed on a fresh week labelled before any model ran; "
                          "model-based orders remain experimental.") if fresh else "Not scored yet."},
              {"stage": "ranking_update", "title": "Ranking update (second development cycle)",
               "status": "done" if variants else "pending",
               "result": _ranking_update_text(variants, week2_rankings) if variants else "Not decided yet."},
              {"stage": "summary", "title": "Summary design (Stage 4)",
               "status": "done" if freeze else "pending",
               "result": (_summary_design_text(freeze, round3) if freeze
                          else "The design comparison and its blind judging are not finished yet.")}]
    scored = transfer and transfer.get("summary", {}).get("status") == "scored"
    stages.append({"stage": "transfer", "title": "Second company: Reliance Industries (Stage 7)",
                   "status": "done" if scored else "pending",
                   "result": (_transfer_text(transfer) if scored
                              else "The test runs after the summary design is frozen.")})
    week2_done = all(score and score.get("summary", {}).get("status") == "scored" for score in week2_scores.values())
    stages.append({"stage": "week2", "title": "Two fresh weeks (Stage 7, week 2)",
                   "status": "done" if week2_done else "pending",
                   "result": _week2_summary_text(week2_scores) if week2_done else "Not scored yet."})
    return stages


# ---------- the real pipeline ----------

class Pipeline:
    """src/research.py with the alias file, the page fetcher, the summary model and the saved summaries."""

    def __init__(self):
        from src.config import DATA_DIR
        from src.query import load_aliases
        self.aliases = load_aliases(DATA_DIR / "company_aliases.json")

    def summary_model(self):
        """Made once per session, so the pacer keeps what it learns about the provider's limits."""
        if getattr(self, "_summary", None) is None:
            self._summary = _summary_model()
        return self._summary

    def review_model(self):
        if getattr(self, "_review", None) is None:
            self._review = _review_model()
        return self._review

    def search(self, company, days):
        from src import research
        return research.search(company, self.aliases, days)

    def provisional(self, found, company):
        from src import research
        return research.provisional(found, company, self.aliases)

    def enrich(self, found, progress):
        from src import research
        from src.fulltext import FullTextFetcher
        from src.safe_web import safe_http_get
        pages, total = FullTextFetcher(http_get=safe_http_get), len(found["articles"])   # public addresses only

        class Counting:
            done = 0

            def fetch(self, url):
                result = pages.fetch(url)
                Counting.done += 1
                progress(Counting.done, total)
                return result

        return research.enrich(found, Counting())

    def final(self, found, enriched, company):
        """The final list; the reasons a fetched page is not used are kept for the cards."""
        from src import research
        from src.fulltext_eval import usable_full_texts
        _, rejected = usable_full_texts(found["articles"], enriched["texts"], company, self.aliases)
        enriched["check_reasons"] = rejected
        return research.final(found, enriched, company, self.aliases)

    def changes(self, first, last):
        from src import research
        return research.list_changes(first, last)

    def summarize(self, result, company):
        from src import research
        from src.summarizer import SummaryCache
        return research.summarize_selection(result, company, self.aliases, self.summary_model(), summary_design(),
                                            SummaryCache(Path("data") / "app_summaries"))

    def review(self, result, company):
        from src import research
        from src.relevance_check import CheckCache
        return research.review(result, company, self.aliases, self.review_model(), CheckCache(Path("data") / "app_checks"))

    def apply(self, result):
        from src import research
        return research.apply_updates(result)

    def back(self, result):
        from src import research
        return research.back_to_default(result)


def _summary_model():
    from src.summary_v2 import summary_llm
    return summary_llm()


def _review_model():
    """The Stage 3 setting of the relevance check."""
    from src.llm import make_llm
    from src.model_compare import CONFIGS
    from src.summary_v2 import MODEL_SETTING
    config = CONFIGS[MODEL_SETTING]
    return make_llm(config["provider"], model=config["model"], generation=config["generation"])


def default_pipeline():
    return Pipeline()


def summary_design() -> str:
    """The frozen Stage 4 design when the freeze is in place; until then the development setting."""
    from src.summary_freeze import FreezeError, frozen_summary
    try:
        return frozen_summary(5)["design"]
    except FreezeError:
        return "direct"


def summary_is_frozen() -> bool:
    from src.summary_freeze import FreezeError, frozen_summary
    try:
        frozen_summary(5)
        return True
    except FreezeError:
        return False
