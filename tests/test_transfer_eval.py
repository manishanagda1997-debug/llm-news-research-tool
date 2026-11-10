"""
Tests for src/transfer_eval.py: the Reliance transfer test, up to the sealed labels (no model calls).

    protocol  fixed before any fetch: company, window, both search strings (search B built from the alias
              file), fingerprints of the alias file and the guide, the sample rule, the measure definitions,
              the pass levels and how the result is read
    fetch     refused before the freeze tag exists, when the alias file or a search string differs from the
              protocol, or before the window's day of delay has passed
    sample    A then B, each article once by tidy link; outside the window or without a date: excluded and
              logged; a count at each step; no top-up and no replacement
    seal      after labelling and event grouping: every label valid, every relevant article with an event;
              then the fingerprints are saved
"""

import json
from datetime import datetime, timezone

import pytest

from src import transfer_eval as te

ALIASES = json.load(open("data/company_aliases.json", encoding="utf-8"))


# ---------- the company and the searches ----------

def test_the_alias_file_holds_reliance_with_sources_and_subsidiaries():
    group = ALIASES["search_groups"]["reliance industries"]["identities"]
    assert "reliance_industries" in group
    for identity in group:
        entry = ALIASES["identities"][identity]
        assert entry["names"] and entry["sources"]
        if identity != "reliance_industries":
            assert entry["parent"] == "reliance_industries"


def test_separate_companies_with_a_similar_name_are_not_in_the_group():
    names = " ".join(n for i in ALIASES["search_groups"]["reliance industries"]["identities"]
                     for n in ALIASES["identities"][i]["names"]).lower()
    for separate in ("jio financial", "reliance power", "reliance infrastructure", "reliance communications"):
        assert separate not in names


def test_the_tata_group_is_unchanged_by_the_reliance_entry():
    from src.relevance_check import company_scope
    assert "Reliance" not in company_scope("Tata Motors", ALIASES)


def test_search_a_is_the_exact_company_name_and_search_b_comes_from_the_alias_file():
    searches = te.search_strings(ALIASES)
    assert searches["A"] == '"Reliance Industries"'
    for identity in ALIASES["search_groups"]["reliance industries"]["identities"]:
        entry = ALIASES["identities"][identity]
        if "parent" in entry:
            for name in entry["names"]:
                assert f'"{name}"' in searches["B"]
    assert "Reliance Industries" not in searches["B"]


# ---------- the protocol ----------

def _protocol():
    return te.protocol_record(ALIASES, alias_text="aliases", guide_text="guide", source_commit="abc123")


def test_the_protocol_fixes_the_window_the_searches_and_the_fingerprints():
    p = _protocol()
    assert p["company"] == "Reliance Industries"
    assert p["window"] == ["2026-10-26T00:00:00Z", "2026-11-01T23:59:59Z"]
    assert p["searches"] == te.search_strings(ALIASES)
    assert p["search_settings"]["page_size"] == 30 and p["search_settings"]["sort_by"] == "relevancy"
    assert p["alias_fingerprint"] and p["guide_fingerprint"] and p["source_commit"] == "abc123"


def test_the_protocol_states_every_definition_and_pass_level_before_any_data():
    p = _protocol()
    for name in ("relevant_event", "direct_event", "event_recall", "precision", "no_direct_event_lost",
                 "model_coverage", "model_time", "summary_event_coverage", "contradicted_rate", "produced"):
        assert p["definitions"][name]
    ranking = p["pass_levels"]["ranking"]
    assert ranking["primary_k"] == 10 and ranking["configuration"] == "O1+S1" and ranking["baseline"] == "O1"
    assert ranking["min_model_coverage"] == 0.9 and ranking["max_model_seconds"] == 180
    summary = p["pass_levels"]["summary"]
    assert summary["min_event_coverage"] == 0.8 and summary["max_contradicted"] == 0.05
    for note in ("candidate_pool", "one_summary", "no_tuning", "fixes"):
        assert p["reading"][note]


def test_the_sample_rule_allows_no_top_up_or_replacement():
    rule = _protocol()["sample_rule"]
    assert "no top-up" in rule and "no replacement" in rule


# ---------- the fetch guard ----------

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


def test_fetch_is_refused_without_the_freeze_tag():
    p = _protocol()
    with pytest.raises(te.TransferError, match="tag"):
        te.fetch_problems(p, ALIASES, "aliases", NOW, tag_protocol=None, protocol_text="x")


def test_fetch_is_refused_when_the_tagged_protocol_differs():
    with pytest.raises(te.TransferError, match="tag"):
        te.fetch_problems(_protocol(), ALIASES, "aliases", NOW, tag_protocol="old", protocol_text="new")


def test_fetch_is_refused_when_the_alias_file_changed():
    with pytest.raises(te.TransferError, match="alias"):
        te.fetch_problems(_protocol(), ALIASES, "changed aliases", NOW, tag_protocol="x", protocol_text="x")


def test_fetch_is_refused_when_a_search_string_differs():
    p = _protocol()
    p["searches"]["B"] = '"Jio Platforms"'
    with pytest.raises(te.TransferError, match="search"):
        te.fetch_problems(p, ALIASES, "aliases", NOW, tag_protocol="x", protocol_text="x")


def test_fetch_is_refused_too_early():
    early = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(te.TransferError, match="early"):
        te.fetch_problems(_protocol(), ALIASES, "aliases", early, tag_protocol="x", protocol_text="x")


def test_fetch_is_allowed_when_everything_matches():
    te.fetch_problems(_protocol(), ALIASES, "aliases", NOW, tag_protocol="x", protocol_text="x")


# ---------- the sample ----------

def _article(url, published="2026-10-28T10:00:00Z", title=None):
    return {"url": url, "publishedAt": published, "title": title or url, "description": "", "content": ""}


def test_the_sample_merges_a_then_b_once_each_and_counts_every_step():
    a = [_article("https://x.com/1"), _article("https://x.com/2"), _article("https://x.com/3", "2026-10-20T00:00:00Z")]
    b = [_article("https://x.com/2/"), _article("https://x.com/4", ""), _article("https://x.com/5")]
    result = te.sample({"A": {"articles": a}, "B": {"articles": b}})
    assert [x["url"] for x in result["kept"]] == ["https://x.com/1", "https://x.com/2", "https://x.com/5"]
    assert result["counts"] == {"search_A": 3, "search_B": 3, "after_merge": 5, "outside_window": 1,
                                "no_date": 1, "sample": 3}
    assert {e["reason"] for e in result["excluded"]} == {"outside_window", "no_date"}


def test_a_small_sample_is_kept_as_it_is():
    result = te.sample({"A": {"articles": [_article("https://x.com/1")]}, "B": {"articles": []}})
    assert result["counts"]["sample"] == 1 and len(result["kept"]) == 1


# ---------- the seal ----------

def test_the_seal_refuses_missing_or_invalid_labels():
    labels = {"A1": "direct", "A2": ""}
    problems = te.seal_problems(labels, {"A1": "ev"})
    assert any("A2" in p for p in problems)


def test_the_seal_refuses_a_relevant_article_without_an_event():
    problems = te.seal_problems({"A1": "direct", "A2": "indirect", "A3": "not"}, {"A1": "ev"})
    assert any("A2" in p for p in problems) and not any("A3" in p for p in problems)


def test_the_seal_accepts_complete_labels_and_events():
    assert te.seal_problems({"A1": "direct", "A2": "not"}, {"A1": "ev"}) == []


def test_a_new_state_carries_this_runs_id(tmp_path):
    assert te.load_state(tmp_path)["run_id"] == te.RUN_ID


def test_the_tagged_protocol_is_looked_up_at_its_path_in_the_repository():
    from src.config import ROOT_DIR
    assert (te.RUN_DIR / te.PROTOCOL_FILE).relative_to(ROOT_DIR).as_posix() == \
        f"data/eval/{te.RUN_ID}/{te.PROTOCOL_FILE}"


# ---------- the amendment: the retrieval-order baseline ----------
#
# The frozen protocol compares O1+S1 with "O1" at k = 10. S1 takes the first k of the ordering, so the two
# lists are the same by definition and the conditions pass on any data. Amendment 1, recorded before any
# Reliance ranking, model or summary run, replaces that baseline with the retrieval order: search A's articles
# as returned, then search B's, through the frozen Stage 1 rules, surviving articles in their relative order.
# All data below is made up; no Reliance label or event is used.

import builtins
from pathlib import Path

from src import ranking as rk
from src.ranking_eval import stage1_candidates

LEVEL = {"title": "title", "desc": "description", "body": "content"}


def _candidate(key, where):
    return {"key": key, "url": f"https://news.example/{key}", "title": f"Story {key}", "description": "",
            "source": "S", "mention_level": LEVEL[where], "check": None, "text": ""}


# retrieval order: two body mentions first, then title mentions
POOL = [_candidate("k1", "body"), _candidate("k2", "body"), _candidate("k3", "title"), _candidate("k4", "title"),
        _candidate("k5", "desc"), _candidate("k6", "title")]


def test_the_old_baseline_is_the_default_itself():
    o1 = [c["key"] for c in rk.order(POOL, "O1")][:3]
    o1_s1 = [c["key"] for c in rk.select(rk.order(POOL, "O1"), 3, "S1")["selected"]]
    assert o1 == o1_s1                                  # the same list by definition: the old test cannot fail


def test_the_retrieval_order_baseline_can_differ_from_the_default():
    chosen = te.selections(POOL, [3])
    assert chosen["retrieval_order"]["3"] == ["k1", "k2", "k3"]
    assert chosen["O1+S1"]["3"] == ["k3", "k4", "k6"]


def test_stage1_keeps_the_retrieval_order_of_the_surviving_articles():
    def raw(n, title):
        return {"url": f"https://news.example/a{n}", "title": title, "description": f"Tata Motors item {n}.",
                "content": "", "source": {"name": f"S{n}"}, "publishedAt": "2026-10-27T10:00:00Z"}
    searches_a = [raw(1, "Steel prices move"), raw(2, "Tata Motors opens a plant"), raw(3, "Tata Motors sales")]
    searches_b = [raw(4, "Tata Motors truck order"), raw(2, "Tata Motors opens a plant")]   # a repeat of a2
    merged = te.merge_run([{"articles": searches_a}, {"articles": searches_b}])
    kept = stage1_candidates(merged, {}, "Tata Motors", None)
    urls = [c["url"] for c in kept]
    assert urls == sorted(urls, key=[a["url"] for a in merged].index)        # relative order unchanged
    assert len(urls) == len(set(urls))


def test_the_amendment_records_its_reason_and_keeps_the_protocol_conditions():
    protocol = {"pass_levels": te.PASS_LEVELS}
    record = te.amendment_record("protocol text", protocol, "abc123", datetime(2026, 11, 4, tzinfo=timezone.utc))
    assert record["baseline"]["name"] == "retrieval_order"
    assert record["conditions"] == te.PASS_LEVELS["ranking"]["conditions"]
    assert record["primary_k"] == 10 and record["reported_k"] == [5, 15]
    assert "logical flaw" in record["reason"] and "not blind" in record["independence"]
    assert record["protocol_fingerprint"] and record["source_commit"] == "abc123" and record["tag"] == te.AMEND_TAG
    assert record["created"].startswith("2026-11-04")


class _Trap:
    """Any read of labels.csv or events.csv during the block fails the test."""

    def __init__(self, monkeypatch):
        real_open, real_text, real_bytes = builtins.open, Path.read_text, Path.read_bytes
        sealed = (te.LABEL_FILE, te.EVENTS_FILE)

        def guarded(path):
            if Path(str(path)).name in sealed:
                raise AssertionError(f"{path} is read")

        monkeypatch.setattr(builtins, "open", lambda f, *a, **k: guarded(f) or real_open(f, *a, **k))
        monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: guarded(self) or real_text(self, *a, **k))
        monkeypatch.setattr(Path, "read_bytes", lambda self, *a, **k: guarded(self) or real_bytes(self, *a, **k))


def _sealed_folder(tmp_path):
    (tmp_path / te.LABEL_FILE).write_text("row,article_id,label\n", encoding="utf-8")
    (tmp_path / te.EVENTS_FILE).write_text("row,event,url\n", encoding="utf-8")
    return tmp_path


def test_the_amendment_is_written_without_reading_the_labels_or_events(tmp_path, monkeypatch):
    folder = _sealed_folder(tmp_path)
    (folder / te.PROTOCOL_FILE).write_text(json.dumps({"pass_levels": te.PASS_LEVELS}), encoding="utf-8")
    _Trap(monkeypatch)
    te.write_amendment(folder, "abc123", datetime(2026, 11, 4, tzinfo=timezone.utc))
    assert json.loads((folder / te.AMENDMENT_FILE).read_text(encoding="utf-8"))["baseline"]["name"] == \
        "retrieval_order"
    with pytest.raises(te.TransferError):
        te.write_amendment(folder, "abc123", datetime(2026, 11, 4, tzinfo=timezone.utc))   # never overwritten


# ---------- the ranking score ----------

PROTOCOL = {"pass_levels": te.PASS_LEVELS}
AMENDMENT = te.amendment_record("protocol text", PROTOCOL, "abc123", datetime(2026, 11, 4, tzinfo=timezone.utc))
KEYS = [f"k{i}" for i in range(1, 13)]


def _score(selected_default, selected_baseline, labels, events):
    chosen = {"retrieval_order": {"10": selected_baseline, "5": selected_baseline[:5], "15": selected_baseline},
              "O1+S1": {"10": selected_default, "5": selected_default[:5], "15": selected_default}}
    return te.ranking_score(chosen, KEYS, labels, events, PROTOCOL, AMENDMENT)


LABELS = {k: "not" for k in KEYS} | {"k1": "direct", "k2": "indirect", "k11": "direct"}
EVENTS = {"k1": "results", "k2": "plant", "k11": "deal"}


def test_the_default_passes_when_it_keeps_every_direct_event_and_is_not_lower():
    report = _score(["k1", "k2", "k11"] + KEYS[3:10], KEYS[:10], LABELS, EVENTS)
    assert report["passed"] is True
    assert set(report["conditions"]) == set(te.PASS_LEVELS["ranking"]["conditions"])
    assert report["baseline"] == "retrieval_order" and report["primary_k"] == 10


def test_the_default_fails_when_it_loses_a_direct_event_of_the_baseline():
    report = _score(KEYS[2:12], KEYS[:10], LABELS, EVENTS)         # drops k1 (direct event "results")
    assert report["passed"] is False and report["conditions"]["no_direct_event_lost"] is False
    assert report["events"]["direct_lost"] == ["results"]


def test_absolute_values_are_reported_at_every_k_without_a_level():
    report = _score(["k1", "k2", "k11"] + KEYS[3:10], KEYS[:10], LABELS, EVENTS)
    for name in ("retrieval_order", "O1+S1"):
        for k in ("5", "10", "15"):
            values = report["absolute"][name][k]
            assert set(values) == {"precision", "event_recall", "direct_event_recall", "selected"}
    assert "levels" not in report["absolute"]


# ---------- the run: guards, no labels, attempted once ----------

RUN_OK = dict(protocol_text="p", tag_protocol="p", amendment_text="a", tag_amendment="a", amendment2_text="b",
              tag_amendment2="b", seal_exists=True, freeze=[], existing=[], changed=[], confirmed=True,
              budget=[])


def test_the_run_is_refused_until_every_freeze_is_in_place():
    assert te.run_problems(**RUN_OK) == []
    assert te.run_problems(**dict(RUN_OK, tag_protocol=None))
    assert te.run_problems(**dict(RUN_OK, tag_protocol="older"))
    assert te.run_problems(**dict(RUN_OK, amendment_text=None))
    assert te.run_problems(**dict(RUN_OK, tag_amendment=None))
    assert te.run_problems(**dict(RUN_OK, seal_exists=False))
    assert te.run_problems(**dict(RUN_OK, freeze=["the requested k = 5 is not validated"])) == \
        ["Stage 4 freeze: the requested k = 5 is not validated"]


@pytest.mark.parametrize("change", [dict(amendment2_text=None), dict(tag_amendment2=None),
                                    dict(tag_amendment2="an older amendment 2")])
def test_the_run_is_refused_when_amendment_2_is_missing_or_differs_from_its_tag(change):
    found = te.run_problems(**dict(RUN_OK, **change))
    assert found == [f"{te.AMENDMENT2_FILE} is missing or differs from the tag {te.AMEND2_TAG}"]


def test_the_run_is_refused_without_the_users_24_hour_confirmation():
    found = te.run_problems(**dict(RUN_OK, confirmed=False))
    assert len(found) == 1 and te.CONFIRM_FLAG in found[0] and "24 hours" in found[0]
    assert "not a check made by the code" in te.CONFIRMATION


def test_the_run_is_refused_when_the_budget_report_is_missing_or_stale():
    assert te.run_problems(**dict(RUN_OK, budget=["the budget decision is stop"])) == \
        ["budget: the budget decision is stop"]


def test_the_run_is_refused_when_any_of_its_files_already_exists():
    found = te.run_problems(**dict(RUN_OK, existing=[te.ATTEMPT_FILE]))
    assert found == [f"{te.ATTEMPT_FILE} already exists: the run is attempted once only"]


def test_the_run_is_refused_when_a_protected_path_is_not_committed():
    status = "\n".join([" M src/summarizer.py", "?? data/eval/transfer_reliance_2026-10-26_11-01/new.csv",
                        " M data/eval/SUMMARY_JUDGING_GUIDE_PART6.md", " M data/eval/summary_freeze.json",
                        " M notebooks/04_stage4_summaries.ipynb", "?? data/eval/fresh_2026-10-24_30/x.csv"])
    changed = te.protected_changes(status)
    assert changed == ["data/eval/SUMMARY_JUDGING_GUIDE_PART6.md", "data/eval/summary_freeze.json",
                       "data/eval/transfer_reliance_2026-10-26_11-01/new.csv", "src/summarizer.py"]
    assert te.run_problems(**dict(RUN_OK, changed=changed)) == ["not committed: " + ", ".join(changed)]
    assert te.protected_changes(" M notebooks/04_stage4_summaries.ipynb\n?? scratch.txt") == []


def _prepare(chosen):
    return [{"id": n, "key": c["key"], "title": c["title"], "description": f"passage of {c['key']}", "content": ""}
            for n, c in enumerate(chosen, start=1)]


def _fake_relevance(candidates):
    answers = {c["key"]: {"label": "direct", "reason": "r", "source": "model"} for c in candidates}
    return answers, {"seconds_total": 12.0, "kind": "live", "fallbacks": 0}


def _fake_summary(articles, k):
    return {"status": "ok", "summary": {"key_points": []}, "design": "direct"}


def _run(folder, fulltext, **change):
    kwargs = dict(ks=[5, 10, 15], summary_k=5, design="direct", prepare=_prepare, relevance=_fake_relevance,
                  summarize=_fake_summary, fingerprints={"freeze": "f", "amendment2": "b"}, source_commit="c0ffee",
                  fulltext_dir=fulltext)
    return te.execute_run(folder, POOL, **dict(kwargs, **change))


@pytest.fixture
def places(tmp_path):
    folder, fulltext = tmp_path / "run", tmp_path / "fulltext"
    folder.mkdir()
    fulltext.mkdir()
    return folder, fulltext


def test_the_run_never_reads_the_labels_or_events(places, monkeypatch):
    folder, fulltext = places
    _sealed_folder(folder)
    _Trap(monkeypatch)
    record = _run(folder, fulltext)
    assert record["selections"]["O1+S1"]["5"] == ["k3", "k4", "k6", "k5", "k1"]
    assert record["selections"]["retrieval_order"]["5"] == ["k1", "k2", "k3", "k4", "k5"]
    assert record["relevance"]["coverage"] == 1.0 and record["relevance"]["seconds"] == 12.0


def test_the_summary_runs_on_the_defaults_top_k_and_is_saved_in_the_judging_shape(places):
    record = _run(*places)
    entry = record["summary_record"]["records"][0]
    assert entry["run"] == te.RUN_ID and entry["design"] == "direct" and entry["k"] == 5
    assert [a["key"] for a in entry["articles"]] == record["selections"]["O1+S1"]["5"]
    assert record["source_commit"] == "c0ffee" and record["cache"] == te.CACHE_NOTE
    assert record["fingerprints"]["amendment2"] == "b"


def test_the_marker_and_the_passages_exist_before_any_model_request(places):
    folder, fulltext = places
    paths, seen = te.run_paths(folder, fulltext), []

    def relevance(candidates):
        seen.append(("relevance", paths["attempt"].exists(), paths["passages"].exists(), paths["record"].exists()))
        return _fake_relevance(candidates)

    def summarize(articles, k):
        seen.append(("summary", paths["attempt"].exists(), paths["passages"].exists(), paths["record"].exists()))
        return _fake_summary(articles, k)

    _run(folder, fulltext, relevance=relevance, summarize=summarize)
    assert seen == [("summary", True, True, False), ("relevance", True, True, False)]   # the summary comes first


def test_the_relevance_pass_starts_after_a_full_minute_pause_that_follows_the_summary(places):
    order = []
    _run(*places, pause=lambda seconds: order.append(("pause", seconds)),
         summarize=lambda a, k: order.append("summary") or _fake_summary(a, k),
         relevance=lambda c: order.append("relevance") or _fake_relevance(c))
    assert order == ["summary", ("pause", te.RELEVANCE_PAUSE), "relevance"] and te.RELEVANCE_PAUSE > 60


def test_a_daily_limit_in_the_relevance_pass_leaves_the_summary_in_the_record(places):
    def daily_limit(candidates):
        answers = {c["key"]: {"label": None, "source": "fallback", "kind": "daily_limit"} for c in candidates}
        return answers, {"seconds_total": 1.0, "kind": "live", "fallbacks": len(candidates)}

    record = _run(*places, relevance=daily_limit)
    assert record["summary_record"]["records"][0]["result"]["status"] == "ok"
    assert record["relevance"]["coverage"] == 0.0


@pytest.mark.parametrize("stage", ["relevance", "summary"])
def test_a_failure_after_the_marker_stops_the_run_for_good(places, stage):
    folder, fulltext = places
    paths = te.run_paths(folder, fulltext)

    def broken(*args):
        raise RuntimeError("the connection drops")

    with pytest.raises(RuntimeError):                     # no automatic retry: the failure is raised as it is
        _run(folder, fulltext, **{"relevance" if stage == "relevance" else "summarize": broken})
    assert paths["attempt"].exists() and paths["passages"].exists() and not paths["record"].exists()
    sent = []
    with pytest.raises(te.TransferError, match="attempted once only"):
        _run(folder, fulltext, relevance=lambda c: sent.append(c) or _fake_relevance(c))
    assert sent == []                                     # the second attempt sends nothing


def test_the_run_is_made_once(places):
    _run(*places)
    with pytest.raises(te.TransferError, match="attempted once only"):
        _run(*places)


def test_the_marker_is_never_removed_and_records_the_commit(places):
    folder, fulltext = places
    _run(folder, fulltext)
    record = json.loads(te.run_paths(folder, fulltext)["record"].read_text(encoding="utf-8"))
    marker = json.loads(te.run_paths(folder, fulltext)["attempt"].read_text(encoding="utf-8"))
    assert marker["source_commit"] == "c0ffee" and marker["candidates"] == len(POOL)
    assert marker["confirmation"] == te.CONFIRMATION == record["confirmation"]
    assert marker["attempt_id"] == record["attempt_id"] and te.attempt_problems(marker, record) == []


def test_the_record_and_the_marker_must_come_from_the_same_attempt_and_commit():
    marker = {"attempt_id": "a1", "source_commit": "c1"}
    record = {"attempt_id": "a1", "source_commit": "c1", "attempt_file": te.ATTEMPT_FILE}
    assert te.attempt_problems(marker, record) == []
    assert te.attempt_problems(marker, dict(record, attempt_id="a2")) == \
        ["the run record and the attempt marker come from different attempts"]
    assert te.attempt_problems(marker, dict(record, source_commit="c2")) == \
        ["the run record and the attempt marker name different source commits"]
    assert te.attempt_problems(marker, dict(record, attempt_file="other.json")) == \
        ["the run record names another attempt marker"]
    assert len(te.attempt_problems({}, {})) == 3
    passages = {"attempt_id": "a1", "source_commit": "c1", "passages": []}
    assert te.attempt_problems(marker, record, passages) == []
    for other in (dict(passages, attempt_id="a2"), dict(passages, source_commit="c2"), [], {}):
        assert te.attempt_problems(marker, record, other) == \
            ["the saved passages come from another attempt or source commit than the run record"]


def test_the_exact_passages_are_saved_with_their_fingerprints(places):
    from src.summary_fresh import passage_entry
    folder, fulltext = places
    record = _run(folder, fulltext)
    saved_file = json.loads(te.run_paths(folder, fulltext)["passages"].read_text(encoding="utf-8"))
    assert saved_file["attempt_id"] == record["attempt_id"] and saved_file["source_commit"] == "c0ffee"
    saved = saved_file["passages"]
    chosen = record["selections"]["O1+S1"]["5"]
    assert [p["key"] for p in saved] == chosen
    assert saved[0]["description"] == f"passage of {chosen[0]}"
    assert [p["sha256"] for p in saved] == [passage_entry(p)["sha256"] for p in saved]
    assert record["passage_fingerprints"] == [{"id": p["id"], "key": p["key"], "sha256": p["sha256"]} for p in saved]
    assert record["passages_file"] == f"data/fulltext/{te.PASSAGES_FILE}"
    assert te.PASSAGES_FILE.startswith("transfer_passages_") and str(fulltext) in str(te.run_paths(folder, fulltext)["passages"])


# ---------- no answer cache in either model stage ----------

def test_the_relevance_pass_reads_and_writes_no_saved_answer(monkeypatch):
    from src import ranking_eval as re_
    from src import relevance_check as rc
    captured = {}

    def no_cache(*args, **kwargs):
        raise AssertionError("an answer cache is opened")

    def classify(items, llm, company, cache=None, **kwargs):
        captured["cache"] = cache
        return {item["id"]: {"label": "direct", "source": "model"} for item in items}

    monkeypatch.setattr(rc, "CheckCache", no_cache)
    monkeypatch.setattr(rc, "classify", classify)
    monkeypatch.setattr(re_, "query_items", lambda candidates, company, aliases: [{"id": c["key"]} for c in candidates])
    llm = type("Llm", (), {"provider": "groq", "model_name": "m", "cache_tag": "t", "stats": None})()
    answers, runtime = te.relevance_call(POOL, {}, llm)
    assert captured["cache"] is None and runtime["from_cache"] == 0 and runtime["kind"] == "live"
    assert len(answers) == len(POOL)


def test_the_summary_reads_and_writes_no_saved_answer(monkeypatch):
    from src import summary_v2 as sv
    captured = {}

    def one(design, articles, llm, cache, k, correction):
        captured.update(cache=cache, design=design, k=k, correction=correction)
        return {"status": "ok"}

    monkeypatch.setattr(sv, "_one", one)
    te.summary_call("direct", [], object(), 5, "targeted")
    assert isinstance(captured["cache"], sv.NoCache)
    assert captured["cache"].load("any key") is None
    assert (captured["design"], captured["k"], captured["correction"]) == ("direct", 5, "targeted")


# ---------- amendment 2 ----------

def _amended_folder(tmp_path):
    folder = _sealed_folder(tmp_path)
    (folder / te.PROTOCOL_FILE).write_text(json.dumps({"pass_levels": te.PASS_LEVELS}), encoding="utf-8")
    (folder / te.AMENDMENT_FILE).write_text(json.dumps({"amendment": 1}), encoding="utf-8")
    return folder


def test_amendment_2_states_why_when_and_that_no_pass_level_changes(tmp_path, monkeypatch):
    folder = _amended_folder(tmp_path)
    _Trap(monkeypatch)
    record = te.write_amendment2(folder, "abc123", datetime(2026, 11, 5, tzinfo=timezone.utc))
    assert record["amendment"] == 2 and record["tag"] == te.AMEND2_TAG and record["source_commit"] == "abc123"
    assert "industry-context rule" in record["reason"]
    assert "fresh-week result" in record["follows"]
    assert "before the Reliance run" in record["timing"] and "before its summary exists" in record["timing"]
    assert "PART6" in record["judging"] and "before the summary is read" in record["judging"]
    assert set(record["added"]) == {"industry_evidence", "industry_points", "industry_section"}
    assert "without a pass level" in record["level"]
    assert "summary pass levels" in record["unchanged"] and "ranking conditions" in record["unchanged"]
    assert record["protocol_fingerprint"] and record["amendment1_fingerprint"]
    assert te.PASS_LEVELS["summary"] == {"design": "the design frozen in Stage 4", "k": 5, "produced": True,
                                         "min_event_coverage": 0.8, "max_contradicted": 0.05,
                                         "no_central_contradiction": True}


def test_amendment_2_is_never_overwritten(tmp_path):
    folder = _amended_folder(tmp_path)
    te.write_amendment2(folder, "abc123", datetime(2026, 11, 5, tzinfo=timezone.utc))
    with pytest.raises(te.TransferError, match="not overwritten"):
        te.write_amendment2(folder, "abc123", datetime(2026, 11, 5, tzinfo=timezone.utc))


@pytest.mark.parametrize("name", [te.RUN_FILE, te.ATTEMPT_FILE])
def test_amendment_2_is_refused_once_the_run_has_started(tmp_path, name):
    folder = _amended_folder(tmp_path)
    (folder / name).write_text("{}", encoding="utf-8")
    with pytest.raises(te.TransferError, match="before the run"):
        te.write_amendment2(folder, "abc123", datetime(2026, 11, 5, tzinfo=timezone.utc))
    assert not (folder / te.AMENDMENT2_FILE).exists()


# ---------- the budget check ----------

def test_the_token_bound_covers_every_recorded_request():
    calls = [(2000, 10000), (5000, 20000), (500, 1000)]           # the short prompt carries fixed overhead
    bound = te.token_bound(calls)
    assert bound["rate"] == 0.25 and bound["base"] == 250.0
    assert all(tokens <= bound["base"] + bound["rate"] * chars for tokens, chars in calls)
    with pytest.raises(te.TransferError):
        te.token_bound([(500, 1000)])


def test_a_batch_too_large_splits_in_two_as_the_relevance_check_does():
    batch = [{"id": f"a{i}", "size": 10} for i in range(10)] + [{"id": "huge", "size": 1000}]
    sent, fallen = te.planned_requests(batch, lambda part: sum(i["size"] for i in part), lambda chars: chars <= 60)
    assert fallen == ["huge"] and sum(sent) == 100 and all(chars <= 60 for chars in sent)


BOUNDS = {"relevance": {"rate": 0.25, "base": 0.0, "requests": 1}, "summary": {"rate": 0.25, "base": 1000.0, "requests": 1}}


def test_the_budget_goes_ahead_only_when_every_request_and_the_day_fit():
    report = te.budget_report([20000, 20000], 16000, BOUNDS, 1500, 2)
    assert report["per_request"]["over"] == [] and report["decision"].startswith("go only if no Groq use")
    assert report["day"]["relevance"] == 2 * 2 * (5000 + 1500)
    assert report["day"]["summary"] == (2 + 2) * (5000 + 4096)
    assert "not a proven maximum" in report["assumptions"]["groq_extra"]
    assert "cannot see the Groq usage" in report["assumptions"]["no_groq_24h"]
    assert "not a guaranteed maximum" in report["assumptions"]["estimate"]
    assert report["day"]["limit"] == te.DAY_LIMIT == 200000


def test_the_budget_stops_when_one_request_would_pass_the_minute_limit():
    report = te.budget_report([30000], 16000, BOUNDS, 1500, 2)       # 7,500 + 551 > 8,000
    assert report["per_request"]["over"] and report["decision"] == "stop"


def test_the_budget_stops_when_the_day_total_is_over_its_share():
    report = te.budget_report([20000] * 16, 16000, BOUNDS, 1500, 4)
    assert report["day"]["total"] > report["day"]["limit"] == te.DAY_LIMIT * te.DAY_SHARE
    assert report["decision"] == "stop"


def _budget_folder(tmp_path, monkeypatch):
    folder = tmp_path / "run"
    folder.mkdir()
    for name in (te.PROTOCOL_FILE, te.AMENDMENT_FILE, te.AMENDMENT2_FILE, te.SEAL_FILE):
        (folder / name).write_text(name, encoding="utf-8")
    monkeypatch.setattr(te, "RUN_DIR", folder)
    monkeypatch.setattr(te, "run_paths", lambda *a: {"record": folder / te.RUN_FILE,
                                                     "attempt": folder / te.ATTEMPT_FILE,
                                                     "passages": tmp_path / te.PASSAGES_FILE})
    monkeypatch.setattr(te, "git", lambda *args: "" if args[0] == "status" else "c0ffee")
    pool = [dict(c, text=f"Reliance Industries story {c['key']}. " * 40, description="Reliance Industries")
            for c in POOL]
    monkeypatch.setattr(te, "_run_candidates", lambda aliases: pool)
    return folder


def test_the_budget_report_is_saved_once_with_no_model_request(tmp_path, monkeypatch, capsys):
    from src import llm as llm_module
    from src import relevance_check as rc
    from src import summary_v2 as sv

    def refuse(*args, **kwargs):
        raise AssertionError("a model or an answer cache is used")

    monkeypatch.setattr(llm_module, "make_llm", refuse)
    monkeypatch.setattr(sv, "summary_llm", refuse)
    monkeypatch.setattr(rc, "CheckCache", refuse)
    folder = _budget_folder(tmp_path, monkeypatch)
    te._budget_command()
    report = json.loads((folder / te.BUDGET_FILE).read_text(encoding="utf-8"))
    assert report["source_commit"] == "c0ffee" and report["model_requests"] == 0
    assert set(report["inputs"]) == {"code", "protocol", "amendment1", "amendment2", "seal", "freeze", "evidence"}
    assert report["code_files"] == list(te.BUDGET_CODE) and "src/relevance_check.py" in te.BUDGET_CODE
    assert report["plan"]["plan_fingerprint"] and report["plan"]["candidates"] == len(POOL)
    assert report["decision"] == report["report"]["decision"]
    assert "cannot verify the Groq usage" in report["confirmation_note"]
    assert "not a guaranteed maximum" in report["report"]["assumptions"]["estimate"]
    assert "Decision:" in capsys.readouterr().out
    with pytest.raises(te.TransferError, match="made once"):
        te._budget_command()                                       # never overwritten


@pytest.mark.parametrize("name", [te.RUN_FILE, te.ATTEMPT_FILE])
def test_the_budget_report_is_refused_once_the_run_has_started(tmp_path, monkeypatch, name):
    folder = _budget_folder(tmp_path, monkeypatch)
    (folder / name).write_text("{}", encoding="utf-8")
    with pytest.raises(te.TransferError, match="before the run"):
        te._budget_command()
    assert not (folder / te.BUDGET_FILE).exists()


def test_the_budget_report_is_refused_with_uncommitted_code(tmp_path, monkeypatch):
    folder = _budget_folder(tmp_path, monkeypatch)
    monkeypatch.setattr(te, "git", lambda *args: " M src/summarizer.py" if args[0] == "status" else "c0ffee")
    with pytest.raises(te.TransferError, match="Commit these first"):
        te._budget_command()
    assert not (folder / te.BUDGET_FILE).exists()


def test_the_plan_fingerprint_changes_with_any_prompt(monkeypatch):
    bounds = te._bounds()[0]
    pool = [dict(c, text=f"Reliance Industries story {c['key']}. " * 40, description="Reliance Industries")
            for c in POOL]
    first = te.budget_plan(pool, {}, bounds)
    assert te.budget_plan(pool, {}, bounds)["plan_fingerprint"] == first["plan_fingerprint"]
    edited = [dict(pool[0], text=pool[0]["text"] + " One more sentence about Reliance Industries.")] + pool[1:]
    assert te.budget_plan(edited, {}, bounds)["plan_fingerprint"] != first["plan_fingerprint"]


SAVED_BUDGET = {"decision": "go only if no Groq use in the 24 hours before the run is confirmed; otherwise stop",
                "inputs": {"code": "c", "protocol": "p", "seal": "s"}, "plan": {"plan_fingerprint": "f"}}


def test_the_run_needs_a_current_budget_report_that_allows_it():
    inputs = {"code": "c", "protocol": "p", "seal": "s"}
    assert te.budget_problems(SAVED_BUDGET, inputs, "f") == []
    assert "does not exist" in te.budget_problems(None, inputs, "f")[0]
    assert te.budget_problems(dict(SAVED_BUDGET, decision="stop"), inputs, "f") == ["the budget decision is stop"]
    assert te.budget_problems(SAVED_BUDGET, dict(inputs, code="new code"), "f") == \
        ["changed since the budget report: code; make and review a new report"]
    assert te.budget_problems(SAVED_BUDGET, inputs, "other prompts") == \
        ["the prompts of the run differ from the ones in the budget report"]


def test_planned_prompts_split_like_planned_requests():
    batch = [{"id": f"a{i}", "size": 10} for i in range(10)] + [{"id": "huge", "size": 1000}]
    build = lambda part: "x" * sum(i["size"] for i in part)
    texts, fallen = te.planned_prompts(batch, build, lambda chars: chars <= 60)
    sent, fallen_too = te.planned_requests(batch, lambda part: len(build(part)), lambda chars: chars <= 60)
    assert [len(x) for x in texts] == sent and fallen == fallen_too == ["huge"]


def test_the_run_prints_counts_and_the_summary_status_only(places):
    record = _run(*places)
    message = te.run_message(record)
    assert message.startswith(f"Candidates: {len(POOL)}") and "summary ok" in message
    assert not any(key in message for key in [c["key"] for c in POOL]) and "Story" not in message
    assert "O1+S1" not in message and "retrieval_order" not in message and "precision" not in message


# ---------- judging: the saved passages, the lock ----------

def _saved(record_fingerprints, texts):
    from src.summary_fresh import passage_entry
    return [passage_entry({"id": n, "key": key, "title": f"T{key}", "description": text, "content": ""})
            for n, (key, text) in enumerate(texts, start=1)]


def _file(saved, attempt="a1", commit="c1"):
    return {"attempt_id": attempt, "source_commit": commit, "passages": saved}


BUDGET_TEXT = json.dumps({"decision": "go only if ...", "report": {"day": {"total": 186976}}})


def _budget(folder):
    """A committed budget report in the folder and its fingerprint, as the run stores it."""
    from src.fresh_eval import file_fingerprint
    (folder / te.BUDGET_FILE).write_text(BUDGET_TEXT, encoding="utf-8")
    return {"budget": file_fingerprint(folder / te.BUDGET_FILE)}


def _judged_run(saved):
    return {"attempt_id": "a1", "source_commit": "c1", "attempt_file": te.ATTEMPT_FILE, "summary_record": {"records": [{"run": te.RUN_ID, "design": "direct", "k": 2,
                                            "articles": [{"id": p["id"], "key": p["key"], "title": p["title"]}
                                                         for p in saved], "result": {}}]},
            "passage_fingerprints": [{"id": p["id"], "key": p["key"], "sha256": p["sha256"]} for p in saved]}


def test_judging_reads_the_saved_passages_and_checks_their_fingerprints():
    saved = _saved(None, [("A1", "first text"), ("A2", "second text")])
    run = _judged_run(saved)
    assert te.saved_passages(run, _file(saved)) == {"A1": "first text", "A2": "second text"}
    changed = [dict(saved[0], description="edited text"), saved[1]]
    with pytest.raises(te.TransferError, match="differs from the one the model received"):
        te.saved_passages(run, _file(changed))
    with pytest.raises(te.TransferError, match="IDs or order"):
        te.saved_passages(run, _file(list(reversed(saved))))
    with pytest.raises(te.TransferError, match="another attempt"):
        te.saved_passages(run, _file(saved, attempt="a2"))
    with pytest.raises(te.TransferError, match="another source commit"):
        te.saved_passages(run, _file(saved, commit="c2"))
    with pytest.raises(te.TransferError, match="another attempt"):
        te.saved_passages(run, saved)                          # a bare list carries no attempt


def test_judging_never_rebuilds_the_passages(tmp_path, monkeypatch):
    from src import fulltext as ft
    folder, fulltext = tmp_path / "run", tmp_path / "fulltext"
    folder.mkdir()
    fulltext.mkdir()
    monkeypatch.setattr(te, "RUN_DIR", folder)
    monkeypatch.setattr(ft, "FULLTEXT_DIR", fulltext)
    monkeypatch.setattr(te, "_run_candidates", lambda *a: (_ for _ in ()).throw(AssertionError("Stage 1 rebuilt")))
    from src import summary_v2 as sv
    monkeypatch.setattr(sv, "passage_articles", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rebuilt")))
    saved = _saved(None, [("A1", "first text"), ("A2", "second text")])
    run = _judged_run(saved)
    entry = run["summary_record"]["records"][0]
    entry["result"] = {"status": "ok", "headlines": [{"id": 1, "title": "TA1", "url": "https://n.example/a1"},
                                                      {"id": 2, "title": "TA2", "url": "https://n.example/a2"}],
                       "summary": {"key_points": [{"text": "A point.", "citations": [1]}],
                                   "industry_context": [], "article_briefs": [{"id": 2, "brief": "A brief."}]},
                       "repairs": []}
    run.update(attempt_id="a1", source_commit="c1", attempt_file=te.ATTEMPT_FILE, fingerprints=_budget(folder))
    (folder / te.RUN_FILE).write_text(json.dumps(run), encoding="utf-8")
    (folder / te.ATTEMPT_FILE).write_text(json.dumps({"attempt_id": "a1", "source_commit": "c1"}), encoding="utf-8")
    (fulltext / te.PASSAGES_FILE).write_text(json.dumps(_file(saved)), encoding="utf-8")
    (folder / te.LABEL_FILE).write_text("row,article_id,title,url,label,unsure,note\n"
                                        "1,A1,TA1,https://n.example/a1,direct,,\n", encoding="utf-8")
    (folder / te.EVENTS_FILE).write_text("row,event,url\n1,results,https://n.example/a1\n", encoding="utf-8")
    from src.fresh_eval import file_fingerprint
    (folder / te.SEAL_FILE).write_text(json.dumps({"labels_sha256": file_fingerprint(folder / te.LABEL_FILE),
                                                   "events_sha256": file_fingerprint(folder / te.EVENTS_FILE)}),
                                       encoding="utf-8")
    te._judge_command()
    industry = (folder / te.JUDGING_INDUSTRY).read_text(encoding="utf-8")
    assert industry.startswith("row,summary,industry_evidence,note")
    page = (fulltext / te.JUDGING_PAGE).read_text(encoding="utf-8")
    assert page.index("1. Industry evidence") < page.index("2. Items") < page.index("3. Event coverage")
    assert "first text" in page and "second text" in page
    events = (folder / te.JUDGING_EVENTS).read_text(encoding="utf-8")
    assert "results" in events


def _judging_files(tmp_path):
    folder, fulltext = tmp_path / "run", tmp_path / "fulltext"
    folder.mkdir()
    fulltext.mkdir()
    for name in (te.PROTOCOL_FILE, te.AMENDMENT_FILE, te.AMENDMENT2_FILE, te.SEAL_FILE, te.LABEL_FILE,
                 te.EVENTS_FILE, te.RUN_FILE, te.ATTEMPT_FILE, te.BUDGET_FILE):
        (folder / name).write_text(name, encoding="utf-8")
    (fulltext / te.PASSAGES_FILE).write_text("passages", encoding="utf-8")
    return folder, fulltext


def test_the_lock_covers_every_input_of_the_judgements(tmp_path):
    folder, fulltext = _judging_files(tmp_path)
    inputs = te.judging_inputs(folder, fulltext)
    assert set(inputs) == {"guides", "protocol", "amendment1", "amendment2", "seal", "labels", "events", "freeze",
                           "run_record", "attempt", "passages", "budget"}
    assert len(inputs["guides"]) == 3 and set(te.JUDGING_GUIDES) >= {"SUMMARY_JUDGING_GUIDE.md",
                                                                     "SUMMARY_JUDGING_GUIDE_PART6.md"}
    assert "src/transfer_eval.py" in te.JUDGING_CODE and "src/summary_judging_p4.py" in te.JUDGING_CODE
    (fulltext / te.PASSAGES_FILE).unlink()
    with pytest.raises(te.TransferError, match="passages"):
        te.judging_inputs(folder, fulltext)
    (fulltext / te.PASSAGES_FILE).write_text("passages", encoding="utf-8")
    (folder / te.BUDGET_FILE).unlink()
    with pytest.raises(te.TransferError, match="budget"):
        te.judging_inputs(folder, fulltext)


ITEMS = [{"row": "1", "kind": "company point", "text": "t", "evidence": "A1", "judgement": "supported",
          "contradicted": "no", "central": "no", "note": ""}]
EVENT_ROWS = [{"row": "1", "summary": "A", "event": "results", "articles": "TA1", "covered": "yes", "note": ""}]
INDUSTRY = [{"row": "1", "summary": "A", "industry_evidence": "no", "note": "no article reports on the sector"}]


def _lock(monkeypatch, inputs):
    from src import summary_judging_p4 as p4
    monkeypatch.setattr(p4, "problems", lambda *a: [])
    return te.judging_lock(ITEMS, EVENT_ROWS, INDUSTRY, {"k": 1}, inputs)


def test_the_lock_refuses_incomplete_sheets(monkeypatch):
    from src import summary_judging_p4 as p4
    monkeypatch.setattr(p4, "problems", lambda *a: ["items row 1: judgement missing"])
    with pytest.raises(te.TransferError, match="not complete"):
        te.judging_lock(ITEMS, EVENT_ROWS, INDUSTRY, {"k": 1}, {})


@pytest.mark.parametrize("name", ["protocol", "amendment2", "seal", "labels", "freeze", "run_record", "attempt",
                                  "passages", "budget"])
def test_any_input_changed_after_the_lock_stops_the_score(monkeypatch, name):
    inputs = {n: "x" for n in ("protocol", "amendment2", "seal", "labels", "freeze", "run_record", "attempt",
                               "passages", "budget")}
    lock = _lock(monkeypatch, inputs)
    assert te.lock_changes(lock, ITEMS, EVENT_ROWS, INDUSTRY, {"k": 1}, inputs) == []
    assert te.lock_changes(lock, ITEMS, EVENT_ROWS, INDUSTRY, {"k": 1}, dict(inputs, **{name: "y"})) == \
        [f"{name} changed after the lock"]


def test_a_sheet_the_key_or_the_code_changed_after_the_lock_stops_the_score(monkeypatch):
    lock = _lock(monkeypatch, {})
    edited = [dict(ITEMS[0], judgement="partly_supported")]
    assert te.lock_changes(lock, edited, EVENT_ROWS, INDUSTRY, {"k": 1}, {}) == ["the items sheet changed after the lock"]
    assert te.lock_changes(lock, ITEMS, EVENT_ROWS, INDUSTRY, {"k": 2}, {}) == ["the key changed after the lock"]
    monkeypatch.setattr(te, "_code_fingerprint", lambda: "other code")
    assert te.lock_changes(lock, ITEMS, EVENT_ROWS, INDUSTRY, {"k": 1}, {}) == ["the judging code changed after the lock"]


def test_the_score_waits_for_the_lock_when_there_is_a_summary(tmp_path, monkeypatch):
    from src.fresh_eval import file_fingerprint
    folder = tmp_path
    monkeypatch.setattr(te, "RUN_DIR", folder)
    run = {"summary_record": {"records": [{"run": te.RUN_ID, "design": "direct", "articles": [],
                                           "result": {"status": "ok", "summary": {"key_points": [{"text": "p"}]}}}]},
           "attempt_id": "a1", "source_commit": "c1", "attempt_file": te.ATTEMPT_FILE,
           "fingerprints": _budget(folder)}
    (folder / te.RUN_FILE).write_text(json.dumps(run), encoding="utf-8")
    (folder / te.ATTEMPT_FILE).write_text(json.dumps({"attempt_id": "a1", "source_commit": "c1"}), encoding="utf-8")
    for name in (te.AMENDMENT_FILE, te.PROTOCOL_FILE):
        (folder / name).write_text("{}", encoding="utf-8")
    (folder / te.LABEL_FILE).write_text("row,article_id,label\n", encoding="utf-8")
    (folder / te.EVENTS_FILE).write_text("row,event,url\n", encoding="utf-8")
    (folder / te.SEAL_FILE).write_text(json.dumps({"labels_sha256": file_fingerprint(folder / te.LABEL_FILE),
                                                   "events_sha256": file_fingerprint(folder / te.EVENTS_FILE)}),
                                       encoding="utf-8")
    from src import fulltext as ft
    monkeypatch.setattr(ft, "FULLTEXT_DIR", folder)
    (folder / te.PASSAGES_FILE).write_text(json.dumps({"attempt_id": "a1", "source_commit": "c1", "passages": []}),
                                           encoding="utf-8")
    with pytest.raises(te.TransferError, match="not locked"):
        te._score_command()
    assert not (folder / te.SCORE_FILE).exists()


# ---------- the amendment 2 industry report ----------

def _industry_key(sections):
    return {"items": {str(n): {"section": s} for n, s in enumerate(sections, start=1)}}


def _industry_items(judgements):
    return [{"row": str(n), "judgement": j, "contradicted": "no"} for n, j in enumerate(judgements, start=1)]


def _industry_record(points):
    return {"records": [{"result": {"summary": {"industry_context": [{"text": "p"}] * points}}}]}


@pytest.mark.parametrize("evidence, points, judgements, failed", [
    ("yes", 1, ["supported"], []),
    ("yes", 1, ["not_supported"], ["no_unsupported_industry_point"]),
    ("yes", 0, [], ["industry_section_matches_evidence"]),
    ("no", 1, ["supported"], ["industry_section_matches_evidence"]),
    ("no", 0, [], []),
])
def test_the_industry_report_follows_part_6_without_a_level(evidence, points, judgements, failed):
    sections = ["industry_context"] * len(judgements) + ["key_points"]
    items = _industry_items(judgements + ["not_supported"])          # a company point never counts here
    report = te.industry_report(items, [{"industry_evidence": evidence}], _industry_key(sections),
                                _industry_record(points))
    assert [name for name, ok in report["checks"].items() if not ok] == failed
    assert report["industry_points"]["total"] == len(judgements) and "without a pass level" in report["level"]


# ---------- the score: seal, relevance report, summary pending until judged ----------

def test_the_score_is_refused_when_labels_or_events_differ_from_the_seal():
    seal = {"labels_sha256": "a", "events_sha256": "b"}
    assert te.seal_changes(seal, "a", "b") == []
    assert te.seal_changes(seal, "x", "b") == [te.LABEL_FILE]
    assert te.seal_changes(seal, "a", "y") == [te.EVENTS_FILE]


def test_the_relevance_check_is_reported_against_the_labels_without_a_level():
    answers = {"k1": {"label": "direct", "source": "model"}, "k2": {"label": "not", "source": "model"},
               "k3": {"label": "indirect", "source": "fallback"}}
    report = te.relevance_report(answers, {"k1": "direct", "k2": "indirect", "k3": "not"})
    assert report["answered"] == 2 and report["articles"] == 3
    assert report["same_label"] == 1 and report["relevant_precision"] == 1.0 and report["relevant_recall"] == 0.5
    assert "level" not in report


def test_the_summary_result_is_pending_until_the_judgements_are_locked_and_scored():
    assert te.summary_result(None, PROTOCOL) == {"status": "pending",
                                                 "reason": "the blind judgements are not locked and scored yet"}


@pytest.mark.parametrize("change, failed", [
    ({}, []),
    ({"coverage": 0.79}, ["event_coverage"]),
    ({"contradicted_share": 0.06}, ["contradicted"]),
    ({"central": 1}, ["no_central_contradiction"]),
    ({"produced": 0}, ["produced"]),
])
def test_the_summary_levels_come_from_the_protocol(change, failed):
    values = {"coverage": 0.8, "contradicted_share": 0.05, "central": 0, "produced": 1} | change
    judged = {"produced": {"count": values["produced"], "total": 1},
              "items": {"contradicted_share": values["contradicted_share"], "central_contradicted": values["central"]},
              "events": {"coverage": values["coverage"]}}
    result = te.summary_result(judged, PROTOCOL)
    assert result["status"] == "scored"
    assert [name for name, ok in result["checks"].items() if not ok] == failed
    assert result["passed"] is (not failed)


def test_unsure_labels_are_reported_on_their_own():
    rows = [{"article_id": "k1", "label": "direct", "unsure": "yes"}, {"article_id": "k2", "label": "not", "unsure": ""}]
    chosen = {"retrieval_order": {"10": ["k2"]}, "O1+S1": {"10": ["k1"]}}
    assert te.unsure_report(rows, chosen, 10) == [{"article_id": "k1", "label": "direct",
                                                   "retrieval_order": False, "O1+S1": True}]


@pytest.mark.parametrize("contradicted, total, covered, events, failed", [
    (1, 20, 4, 5, []),                                   # exactly 5% and exactly 80% pass
    (1, 19, 4, 5, ["contradicted"]),                     # 5.26% fails
    (0, 20, 3, 4, ["event_coverage"]),                   # 75% fails
])
def test_the_summary_levels_are_compared_unrounded(contradicted, total, covered, events, failed):
    judged = {"produced": {"count": 1, "total": 1},
              "items": {"contradicted": contradicted, "total": total, "contradicted_share": round(contradicted / total, 4),
                        "central_contradicted": 0},
              "events": {"covered": covered, "total": events, "coverage": round(covered / events, 4)}}
    assert [name for name, ok in te.summary_result(judged, PROTOCOL)["checks"].items() if not ok] == failed


def test_a_run_without_a_summary_fails_on_produced_and_nothing_is_judged():
    assert te.no_summary_result() == {"status": "scored", "checks": {"produced": False}, "passed": False,
                                      "values": None}


def test_judge_lock_and_score_work_together_on_the_saved_evidence(tmp_path, monkeypatch):
    """Made-up data end to end: sheets from the saved passages, judged, locked, scored once."""
    import csv
    from src import fulltext as ft
    from src.fresh_eval import file_fingerprint
    from src.fulltext import article_id
    k1, k2, k3 = (article_id(f"https://news.example/k{n}") for n in (1, 2, 3))
    folder, fulltext = tmp_path / "run", tmp_path / "fulltext"
    folder.mkdir()
    fulltext.mkdir()
    monkeypatch.setattr(te, "RUN_DIR", folder)
    monkeypatch.setattr(ft, "FULLTEXT_DIR", fulltext)
    saved = _saved(None, [(k1, "first text"), (k3, "second text")])
    run = _judged_run(saved)
    run["summary_record"]["records"][0]["result"] = {
        "status": "ok", "headlines": [{"id": 1, "title": "Tk1", "url": "https://news.example/k1"},
                                      {"id": 2, "title": "Tk3", "url": "https://news.example/k3"}],
        "summary": {"key_points": [{"text": "A point.", "citations": [1]}],
                    "industry_context": [{"text": "A sector fact.", "citations": [2]}],
                    "article_briefs": [{"id": 1, "brief": "Brief one."}]}, "repairs": []}
    run["candidates"] = [{"key": k} for k in (k1, k2, k3)]
    run["selections"] = {"retrieval_order": {str(k): [k1, k2, k3] for k in (5, 10, 15)},
                         "O1+S1": {str(k): [k1, k3, k2] for k in (5, 10, 15)}}
    run["relevance"] = {"answers": {k1: {"label": "direct", "source": "model"}}, "coverage": 1.0, "seconds": 3.0}
    run.update(attempt_id="a1", source_commit="c1", attempt_file=te.ATTEMPT_FILE, fingerprints=_budget(folder))
    files = {te.RUN_FILE: json.dumps(run), te.PROTOCOL_FILE: json.dumps(PROTOCOL),
             te.AMENDMENT_FILE: json.dumps(AMENDMENT), te.AMENDMENT2_FILE: "{}",
             te.ATTEMPT_FILE: json.dumps({"attempt_id": "a1", "source_commit": "c1"}),
             te.LABEL_FILE: f"row,article_id,title,url,label,unsure,note\n1,{k1},Tk1,https://news.example/k1,direct,,\n"
                            f"2,{k2},Tk2,https://news.example/k2,not,,\n3,{k3},Tk3,https://news.example/k3,not,,\n",
             te.EVENTS_FILE: "row,event,url\n1,results,https://news.example/k1\n"}
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    (fulltext / te.PASSAGES_FILE).write_text(json.dumps(_file(saved)), encoding="utf-8")
    (folder / te.SEAL_FILE).write_text(json.dumps({"labels_sha256": file_fingerprint(folder / te.LABEL_FILE),
                                                   "events_sha256": file_fingerprint(folder / te.EVENTS_FILE)}),
                                       encoding="utf-8")
    te._judge_command()

    def fill(name, values):
        path = folder / name
        with open(path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        columns = list(rows[0])
        for row in rows:
            row.update(values)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, columns)
            writer.writeheader()
            writer.writerows(rows)

    with pytest.raises(te.TransferError, match="not complete"):
        te._lock_command()
    (folder / te.ATTEMPT_FILE).write_text(json.dumps({"attempt_id": "other", "source_commit": "c1"}), encoding="utf-8")
    with pytest.raises(te.TransferError, match="different attempts"):
        te._lock_command()
    (folder / te.ATTEMPT_FILE).write_text(json.dumps({"attempt_id": "a1", "source_commit": "c1"}), encoding="utf-8")
    (fulltext / te.PASSAGES_FILE).write_text(json.dumps(_file(saved, attempt="other")), encoding="utf-8")
    with pytest.raises(te.TransferError, match="saved passages come from another attempt"):
        te._lock_command()
    (fulltext / te.PASSAGES_FILE).write_text(json.dumps(_file(saved)), encoding="utf-8")
    (folder / te.BUDGET_FILE).write_text(BUDGET_TEXT.replace("186976", "150000"), encoding="utf-8")   # edited and committed
    with pytest.raises(te.TransferError, match="differs from the report the run started from"):
        te._lock_command()
    (folder / te.BUDGET_FILE).write_text(BUDGET_TEXT, encoding="utf-8")
    fill(te.JUDGING_ITEMS, {"judgement": "supported", "contradicted": "no", "central": ""})
    fill(te.JUDGING_EVENTS, {"covered": "yes"})
    fill(te.JUDGING_INDUSTRY, {"industry_evidence": "yes", "note": "the second article reports a sector fact"})
    te._lock_command()
    lock = json.loads((folder / te.JUDGING_LOCK).read_text(encoding="utf-8"))
    assert lock["inputs"]["budget"] == run["fingerprints"]["budget"]
    (folder / te.BUDGET_FILE).write_text(BUDGET_TEXT.replace("186976", "150000"), encoding="utf-8")
    with pytest.raises(te.TransferError, match="differs from the report the run started from"):
        te._score_command()
    assert not (folder / te.SCORE_FILE).exists()
    (folder / te.BUDGET_FILE).write_text(BUDGET_TEXT, encoding="utf-8")
    te._score_command()
    score = json.loads((folder / te.SCORE_FILE).read_text(encoding="utf-8"))
    assert score["summary"]["passed"] is True and score["summary"]["checks"]["event_coverage"] is True
    assert score["industry"]["industry_evidence"] == "yes" and score["industry"]["industry_points"]["supported"] == 1
    assert all(score["industry"]["checks"].values())
    with pytest.raises(te.TransferError, match="not overwritten"):
        te._score_command()
    fill(te.JUDGING_ITEMS, {"judgement": "partly_supported"})
    (folder / te.SCORE_FILE).unlink()
    with pytest.raises(te.TransferError, match="items sheet changed after the lock"):
        te._score_command()



def test_the_budget_report_must_be_the_one_the_run_started_from():
    record = {"fingerprints": {"budget": "b1"}}
    assert te.budget_file_problems(record, "b1") == []
    assert te.budget_file_problems(record, None) == [f"{te.BUDGET_FILE} is missing"]
    assert te.budget_file_problems(record, "b2") == \
        [f"{te.BUDGET_FILE} differs from the report the run started from (modified or replaced)"]
    assert te.budget_file_problems({"fingerprints": {}}, "b1") == ["the run record holds no budget report fingerprint"]


def test_the_run_record_stores_the_fingerprint_of_the_exact_budget_report(tmp_path):
    from src.fresh_eval import file_fingerprint
    (tmp_path / te.SEAL_FILE).write_text("seal", encoding="utf-8")
    (tmp_path / te.BUDGET_FILE).write_text(BUDGET_TEXT, encoding="utf-8")
    prints = te.run_fingerprints("p", "a", "b", tmp_path)
    assert prints["budget"] == file_fingerprint(tmp_path / te.BUDGET_FILE)
    assert set(prints) == {"protocol", "amendment", "amendment2", "seal", "freeze", "budget"}
    (tmp_path / te.BUDGET_FILE).write_text(BUDGET_TEXT.replace("186976", "150000"), encoding="utf-8")
    assert te.run_fingerprints("p", "a", "b", tmp_path)["budget"] != prints["budget"]
    assert json.loads(BUDGET_TEXT)["report"]["day"]["total"] == 186976      # the figures themselves are not changed


@pytest.mark.parametrize("change", ["modified", "missing"])
def test_judging_is_refused_when_the_budget_report_changed_after_the_run(tmp_path, monkeypatch, change):
    from src import fulltext as ft
    from src.fresh_eval import file_fingerprint
    folder, fulltext = tmp_path / "run", tmp_path / "fulltext"
    folder.mkdir()
    fulltext.mkdir()
    monkeypatch.setattr(te, "RUN_DIR", folder)
    monkeypatch.setattr(ft, "FULLTEXT_DIR", fulltext)
    saved = _saved(None, [("A1", "first text")])
    run = dict(_judged_run(saved), fingerprints=_budget(folder))
    run["summary_record"]["records"][0]["result"] = {"status": "ok", "summary": {"key_points": [{"text": "p"}]}}
    (folder / te.RUN_FILE).write_text(json.dumps(run), encoding="utf-8")
    (folder / te.ATTEMPT_FILE).write_text(json.dumps({"attempt_id": "a1", "source_commit": "c1"}), encoding="utf-8")
    (fulltext / te.PASSAGES_FILE).write_text(json.dumps(_file(saved)), encoding="utf-8")
    (folder / te.LABEL_FILE).write_text("row,article_id,label\n", encoding="utf-8")
    (folder / te.EVENTS_FILE).write_text("row,event,url\n", encoding="utf-8")
    (folder / te.SEAL_FILE).write_text(json.dumps({"labels_sha256": file_fingerprint(folder / te.LABEL_FILE),
                                                   "events_sha256": file_fingerprint(folder / te.EVENTS_FILE)}),
                                       encoding="utf-8")
    if change == "modified":
        (folder / te.BUDGET_FILE).write_text(BUDGET_TEXT.replace("186976", "150000"), encoding="utf-8")
        expected = "differs from the report the run started from"
    else:
        (folder / te.BUDGET_FILE).unlink()
        expected = "is missing"
    with pytest.raises(te.TransferError, match=expected):
        te._judge_command()
    assert not (folder / te.JUDGING_ITEMS).exists()
