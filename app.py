"""
News Research Tool: a company in, its ranked news and a checked summary out.

Run:  streamlit run app.py
Every step and rule is in src/app_flow.py and src/research.py; this file only draws.
"""

import os

import streamlit as st

# On a hosted app the keys come from Streamlit's secrets; locally they come from the environment.
try:
    for _name in ("NEWSAPI_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY"):
        if _name in st.secrets and not os.getenv(_name):
            os.environ[_name] = str(st.secrets[_name])
except Exception:
    pass

from src import app_flow as af  # noqa: E402  (after the keys are in the environment)

st.set_page_config(page_title="News Research Tool", page_icon="📰", layout="centered")
st.markdown("""<style>
.badge {display:inline-block; padding:1px 8px; border-radius:10px; font-size:0.78rem; margin-right:6px;}
.badge-full {background:#E3F1E8; color:#1E5631;}
.badge-short {background:#FFF4DB; color:#7A5300;}
.badge-model {background:#E8EEF7; color:#1F4E79;}
.badge-exp {background:#FBE7E7; color:#8A1C1C;}
.badge-success {background:#E3F1E8; color:#1E5631;}
.badge-warning {background:#FFF4DB; color:#7A5300;}
.badge-error {background:#FBE7E7; color:#8A1C1C;}
.meta {color:#5A6270; font-size:0.85rem;}
.card-title {font-size:1.05rem; font-weight:600; line-height:1.35;}
div[data-testid="stVerticalBlock"] {gap:0.6rem;}
@media print {div[data-testid="stVerticalBlockBorderWrapper"] {break-inside:avoid; page-break-inside:avoid;}}
h3 {padding-top:0.6rem; padding-bottom:0.2rem;}
</style>""", unsafe_allow_html=True)

REMOVAL_REASONS = {"numbers": "a number not found in the articles it cites",
                   "source_reference": "it names its source by position instead of citing it"}

flow = st.session_state.setdefault("flow", {})
if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = af.default_pipeline()
pipeline = st.session_state["pipeline"]


class Screen:
    """What af.get_news draws while it works: the provisional list, then the loading progress."""

    def __init__(self):
        self.area = st.empty()
        self.bar = None

    def notice(self, kind, text):
        {"error": st.error, "warning": st.warning}.get(kind, st.toast)(text)

    def provisional(self, result):
        with self.area.container(border=True):
            st.markdown(f"**{result['label']}**")
            for number, card in enumerate(result["ranked"], start=1):
                st.markdown(f"{number}. {_escape(card['title'])}  <span class='meta'>{_escape(card.get('source', ''))}</span>",
                            unsafe_allow_html=True)

    def progress(self, done, total):
        text = f"Loading the full text of each article: {done} of {total}"
        if self.bar is None:
            self.bar = st.progress(0.0, text=text)
        self.bar.progress(done / total if total else 1.0, text=text)

    def final(self, context):
        self.area.empty()
        if self.bar is not None:
            self.bar.empty()


def _ask(step):
    st.session_state["pending"] = step


def _act(action, *args):
    try:
        action(*args)
    except af.FlowError as error:
        st.session_state["message"] = str(error)
    except Exception as error:                      # shown in plain words, never with keys or paths
        from src.research import safe_text
        st.session_state["message"] = "Something went wrong: " + safe_text(error)


def _badges(card, context):
    found = af.evidence(card, context["enriched"])
    html = ("<span class='badge badge-full'>Full text used</span>" if found["kind"] == "full_text"
            else f"<span class='badge badge-short'>Short text only: {found['reason']}</span>")
    check = card.get("check") or {}
    if check.get("source") == "model":
        html += f"<span class='badge badge-model'>Model: {check.get('label')}</span>"
    return html


def _card(number, card, context, dates):
    link = af.card_link(card, context["enriched"])
    title = (f"<a href='{_escape(link['url'])}' target='_blank'>{_escape(card['title'])}</a>" if link["url"]
             else _escape(card["title"]))                   # only an http or https address is ever linked
    with st.container(border=True):
        st.markdown(f"<div id='card-{card['key']}'></div><div class='card-title'>{number}. {title}</div>",
                    unsafe_allow_html=True)
        source = _escape(card.get("source") or "Publisher not given")
        if link["via"] and not link["note"]:
            source = f"{_escape(link['url'].split('/')[2])}, via {_escape(link['via'])} (aggregator)"
        st.markdown(f"<span class='meta'>{source} · {dates.get(card['url'], 'Date unavailable')}</span><br>"
                    f"{_badges(card, context)}", unsafe_allow_html=True)
        if link["note"]:
            st.caption(link["note"])
        short = (card.get("description") or "").strip()
        if short:
            st.caption(af.md(short[:300] + ("…" if len(short) > 300 else "")))
        check = card.get("check") or {}
        if check.get("source") == "model" and check.get("reason"):
            st.caption(f"Model's reason: {af.md(check['reason'])}")


def _escape(text):
    """For text inside HTML: escaped, and $ written as an entity so it never starts a formula."""
    import html
    return html.escape(str(text)).replace("$", "&#36;")


def _points(points, links):
    for point in points:
        cited = " ".join(f"[[{n}]](#card-{links[n]})" if n in links else f"[{n}]" for n in point.get("citations", []))
        st.markdown(f"- {af.md(point['text'])} {cited}")


def _summary(context):
    view = af.summary_view(context)
    st.markdown("### News summary")
    if "frozen" not in st.session_state:                 # read once per session (it checks Git and the files)
        st.session_state["frozen"] = af.summary_is_frozen()
        st.session_state["design"] = af.summary_design()
    top = len(context["result"]["selected"])
    st.button(f"Summarize the top {top} articles", key="summarize", type="primary", on_click=_ask, args=("summary",),
              help="Calls the language model on the free plan, which allows a limited number of tokens a day. "
                   "A summary of 5 articles uses about 10,000 to 20,000 of them.")
    if not st.session_state["frozen"]:
        st.caption(f"Summary design: {st.session_state['design']} (development setting; the tested design is not "
                   "frozen yet).")
    if st.session_state.get("pending") == "summary":
        af.take_pending(st.session_state)
        with st.spinner(f"Writing the summary from {top} articles. This takes about half a minute to two minutes."):
            _act(af.make_summary, context, pipeline)
        view = af.summary_view(context)
        if st.session_state.get("message"):
            st.warning(st.session_state.pop("message"))
    if view["state"] == "none":
        st.caption("The summary uses the selected articles below. It calls the language model, so it runs only "
                   "when asked.")
        return
    outcome = view["outcome"]
    if view["state"] == "earlier_selection":
        st.warning("This summary belongs to the earlier selection of articles. Summarize again for the current one.")
    label, kind = af.summary_status_line(outcome)
    st.markdown(f"<span class='badge badge-{kind}'>{label}</span>", unsafe_allow_html=True)
    if kind != "error":
        st.caption("The automated checks confirm each number against the articles it cites and that every point "
                   "cites its sources. They do not prove that every statement is correct; open the cited articles "
                   "to check.")
        if af.evidence_note(context):
            st.caption(af.evidence_note(context))
    if outcome.get("summary_status") == "model_unavailable" and outcome.get("wait_seconds"):
        st.caption(f"The provider asks to wait about {round(outcome['wait_seconds'] / 60) or 1} minute(s).")
    if outcome.get("message"):
        st.caption(af.md(outcome["message"]))
    summary = outcome.get("summary") or {}
    if kind == "error":
        return
    links = af.citation_cards(outcome, context["result"]) if view["state"] == "current" else {}
    st.markdown("**Key findings**")
    _points(summary["key_points"], links)
    st.markdown("**Industry context**")
    if summary.get("industry_context"):
        _points(summary["industry_context"], links)
    elif af.industry_note(outcome):
        st.caption(af.industry_note(outcome))
    titles = {n: h.get("title", "") for n, h in enumerate(outcome.get("headlines") or [], start=1)}
    briefs = [b for b in summary.get("article_briefs", []) if not b.get("hidden") and b.get("brief")]
    if briefs:
        with st.expander(f"Article briefs ({len(briefs)})"):
            for brief in briefs:
                number = brief["id"]
                target = f"(#card-{links[number]})" if number in links else ""
                ref = f"[[{number}]]{target}" if target else f"[{number}]"
                st.markdown(f"{ref} **{af.md(titles.get(number, ''))}**  \n{af.md(brief['brief'])}")
    unused = af.not_used(outcome)
    if unused:
        with st.expander(f"Articles not used ({len(unused)})"):
            for item in unused:
                st.markdown(f"- [{item['number']}] **{af.md(item['title'])}**: {af.md(item['reason'])}")
    if outcome.get("removed"):
        with st.expander(f"Removed points ({len(outcome['removed'])})"):
            for item in outcome["removed"]:
                reason = REMOVAL_REASONS.get(item.get("reason"), item.get("reason"))
                st.markdown(f"- {af.md(item['text'])}  \n<span class='meta'>Reason: {_escape(reason)}</span>",
                            unsafe_allow_html=True)


def _results(context):
    result, found = context["result"], context["found"]
    dates = af.published_dates(found)
    st.markdown(f"### {af.md(context['company'])}")
    latest = af.readable_date(found.get("latest_published")) or "not given"
    st.markdown(f"<span class='meta'>News from {af.window_text(found, context['searched_at'])} · searched at "
                f"{context['searched_at']:%H:%M} UTC · latest article {latest}</span>", unsafe_allow_html=True)
    st.caption(af.SOURCE_NOTE)
    counts = af.overview(context)
    columns = st.columns(5)
    for column, (label, value) in zip(columns, [("Articles found", counts["found"]), ("Kept after cleaning", counts["kept"]),
                                                ("For the summary", counts["selected"]), ("Full text", counts["full_text"]),
                                                ("Short text only", counts["short_text"])]):
        column.metric(label, value)
    with st.expander("Search details"):
        searches = ", ".join(f"search {name}: {n}" for name, n in found.get("counts", {}).items() if n is not None)
        st.markdown(f"Search A is the company name; search B, its subsidiaries' names. Articles per search: {searches}. "
                    f"Cleaning removes duplicates, sale listings and pages that do not name the company.")
        for note in found.get("notes", []):
            st.markdown(f"- {af.md(note)}")
        if context["company"].lower() not in af.TESTED_COMPANIES:
            st.markdown(f"- {af.UNTESTED_NOTE}")
        changes = context["changes"]
        if changes["added"] or changes["removed"]:
            st.markdown(f"The full text added {len(changes['added'])} and removed {len(changes['removed'])} "
                        "article(s) compared with the provisional list.")
        for title in result.get("not_in_time_titles", []):
            st.markdown(f"- Not loaded in time: {af.md(title)}")

    _summary(context)

    st.markdown("### Articles")
    experimental_on = result.get("ordering") == "O3+S1"
    if experimental_on:
        st.warning("Experimental order: the articles are ordered by the model's labels. This is not the tested "
                   "default. Use Back to default to return to the original order.")
    columns = st.columns(2)
    columns[0].button("Model review (optional)", key="review", disabled=bool(result.get("reviewed")),
                      on_click=_ask, args=("review",),
                      help="Adds the model's label and reason to each article. The order stays the same. It reads "
                           "every article, so it uses more of the free daily allowance than a summary.")
    if experimental_on:
        columns[1].button("Back to default", key="default", on_click=_act, args=(af.default, context, pipeline))
    else:
        columns[1].button("Experimental order", key="experimental", disabled=not result.get("reviewed"),
                          on_click=_act, args=(af.experimental, context, pipeline),
                          help="Needs the model review first. Not the tested default.")
    if st.session_state.get("pending") == "review":
        af.take_pending(st.session_state)
        with st.spinner(f"The model reads all {len(result['ranked'])} articles. This takes a few minutes."):
            _act(af.run_review, context, pipeline)
        st.rerun()
    top = len(result["selected"])
    if experimental_on:
        st.markdown(f"**Top {top} articles in the experimental order**")
    else:
        heading = ("Sources used in this summary" if af.summary_view(context)["state"] == "current"
                   else "Articles selected for the summary")
        st.markdown(f"**{heading}: the top {top} of the tested default ranking**")
        st.caption("Articles that name the company in the title come first, together with articles that name it at "
                   "least five times in their text; then articles that name it in the description, then only in "
                   "the text. These are the articles the summary uses.")
    for number, card in enumerate(result["selected"], start=1):
        _card(number, card, context, dates)
    if result["additional"]:
        with st.expander(f"More articles ({len(result['additional'])}), lower in the ranking"):
            ranges = af.pages(len(result["additional"]), 10)
            labels = [f"{top + a + 1}–{top + b}" for a, b in ranges]
            chosen = st.radio("Articles", labels, horizontal=True, key=f"more_page_{context['key']}",
                              label_visibility="collapsed") if len(ranges) > 1 else labels[0]
            a, b = ranges[labels.index(chosen)]
            for number, card in enumerate(result["additional"][a:b], start=top + a + 1):
                _card(number, card, context, dates)


def _about():
    from src.config import EVAL_DIR
    st.subheader("How it works")
    st.markdown(
        "1. **Search:** NewsAPI, for the company name and, when the company's subsidiaries are known, for their "
        "names too.\n"
        "2. **Clean and rank:** duplicates, sale listings and unrelated pages are removed; articles that name the "
        "company in the title come first, together with articles that name it at least five times in their text; "
        "then articles that name it in the description, then only in the text.\n"
        "3. **Full text:** each publisher's page is read where it allows it (robots.txt respected); the text is "
        "used only when it matches the article.\n"
        "4. **Summary:** the language model summarizes the selected articles; every number in a point must "
        "appear in the articles it cites, and a point that fails is repaired or removed, with its reason shown.")
    st.subheader("Tested results")
    for stage in af.about_status(EVAL_DIR):
        mark = "-" if stage["status"] == "done" else "⏳"
        st.markdown(f"{mark} **{stage['title']}:** {stage['result']}")


st.title("📰 News Research Tool")
st.caption("Company news for equity research: ranked articles and a summary checked against its sources.")
research_tab, about_tab = st.tabs(["Research", "About"])

with research_tab:
    with st.form("search"):
        company = st.text_input("Company", key="company", placeholder="for example, Tata Motors")
        days = st.select_slider("Days of news", options=af.DAY_CHOICES, value=af.DEFAULT_DAYS, key="days",
                                help="NewsAPI's free plan searches at most one month back and shows articles about "
                                     "one day after publication, so the shortest choice is 3 days. The ranking is "
                                     "tested on 7 days of news.")
        refresh = st.checkbox("Fresh search", key="refresh",
                              help=f"The same search within {af.FRESH_MINUTES} minutes shows its saved results; "
                                   "tick this to search NewsAPI again.")
        submitted = st.form_submit_button("Get News", type="primary")
    if submitted:
        screen = Screen()
        try:
            af.get_news(company, days, screen, pipeline, flow, refresh=refresh)
        except Exception as error:
            from src.research import safe_text
            screen.final(None)
            st.error("The search stopped: " + safe_text(error))
    if st.session_state.get("message"):
        st.warning(st.session_state.pop("message"))
    context = af.active(flow)
    if context:
        _results(context)
    elif not submitted:
        st.info("Enter a company and press **Get News**.")

with about_tab:
    _about()
