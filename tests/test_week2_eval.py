"""Tests for src/week2_eval.py: the week 2 protocol, data steps and ranking test (no model)."""

import json
from pathlib import Path
from datetime import datetime, timezone

import pytest

from src import week2_eval as w2
from src.config import EVAL_DIR
from src.query import load_aliases

ALIASES = load_aliases(w2.ALIAS_FILE)
UTC = timezone.utc


# ---------- the weeks, the windows and the searches ----------

def test_the_windows_end_at_the_next_midnight_and_the_fetch_waits_30_hours():
    tata, reliance = w2.week("tata"), w2.week("reliance")
    assert w2.window(tata) == (datetime(2026, 10, 31, tzinfo=UTC), datetime(2026, 11, 7, tzinfo=UTC))
    assert w2.window(reliance) == (datetime(2026, 11, 2, tzinfo=UTC), datetime(2026, 11, 9, tzinfo=UTC))
    assert w2.fetch_not_before(tata) == datetime(2026, 11, 8, 6, tzinfo=UTC)
    assert w2.fetch_not_before(reliance) == datetime(2026, 11, 10, 6, tzinfo=UTC)
    with pytest.raises(w2.Week2Error):
        w2.week("other")


def test_the_searches_are_the_same_as_week_1():
    tata_state = json.loads((EVAL_DIR / "fresh_2026-10-24_30" / "state.json").read_text(encoding="utf-8"))
    tata_week1 = {name: r["search_query"] for name, r in tata_state["stages"]["fetch"]["requests"].items()}
    assert w2.search_strings("Tata Motors", ALIASES) == tata_week1
    protocol = json.loads((EVAL_DIR / "transfer_reliance_2026-10-26_11-01" / "transfer_protocol.json")
                          .read_text(encoding="utf-8"))
    assert w2.search_strings("Reliance Industries", ALIASES) == protocol["searches"]
    assert w2.SEARCH_SETTINGS == protocol["search_settings"]


def test_the_protocol_fixes_both_comparisons_the_summary_rules_and_the_cutoffs():
    w = w2.week("tata")
    record = w2.protocol_record(w, ALIASES, "aliases", "guide", "abc")
    ranking = record["ranking"]
    assert ranking["primary"]["baseline"] == "retrieval_order" and ranking["primary"]["decides"] == "passed"
    assert ranking["second"]["baseline"] == "O1+S1" and ranking["second"]["decides"].startswith("nothing")
    assert ranking["primary"]["conditions"] == list(w2.CONDITIONS) == ranking["second"]["conditions"]
    summary = record["summary"]
    assert "not run" in summary["relevance_check"] and "2026-11-10T22:00:00+05:30" in summary["run_cutoff"]
    assert "2026-11-11T05:30:00+05:30" in summary["judging_cutoff"] and "no retry" in summary["tokens"]
    assert summary["pass_levels"] == {"produced": True, "min_event_coverage": 0.8, "max_contradicted": 0.05,
                                      "no_central_contradiction": True}
    assert record["fetch_not_before"] == "2026-11-08T06:00:00+00:00" and record["freeze_tag"] == "freeze-week2-tata"


def test_the_definitions_match_week_1_where_they_are_shared():
    from src.transfer_eval import DEFINITIONS as WEEK1
    for name in ("relevant_event", "direct_event", "event_recall", "precision", "summary_event_coverage",
                 "contradicted_rate", "produced"):
        assert w2.DEFINITIONS[name] == WEEK1[name]


# ---------- the fetch guards ----------

def _protocol(w):
    record = w2.protocol_record(w, ALIASES, w2.ALIAS_FILE.read_text(encoding="utf-8"),
                                w2.GUIDE_FILE.read_text(encoding="utf-8"), "abc")
    return record, json.dumps(record)


def test_the_fetch_is_refused_without_the_tag_too_early_or_with_changed_inputs():
    w = w2.week("reliance")
    protocol, text = _protocol(w)
    alias_text = w2.ALIAS_FILE.read_text(encoding="utf-8")
    later = datetime(2026, 11, 10, 6, tzinfo=UTC)
    assert w2.fetch_problems(w, protocol, ALIASES, alias_text, later, text, text) == []
    assert any("tag" in f for f in w2.fetch_problems(w, protocol, ALIASES, alias_text, later, None, text))
    assert any("tag" in f for f in w2.fetch_problems(w, protocol, ALIASES, alias_text, later, text + " ", text))
    early = datetime(2026, 11, 10, 5, 59, 59, tzinfo=UTC)
    assert any("too early" in f for f in w2.fetch_problems(w, protocol, ALIASES, alias_text, early, text, text))
    assert any("alias" in f for f in w2.fetch_problems(w, protocol, ALIASES, alias_text + "x", later, text, text))
    changed = dict(protocol, searches={"A": "x", "B": "y"})
    assert any("search" in f for f in w2.fetch_problems(w, changed, ALIASES, alias_text, later, text, text))


# ---------- the sample and the overlap ----------

def _a(i, title, published="2026-11-03T10:00:00Z", url=None):
    return {"url": url or f"https://news.example/{i}", "title": title, "description": f"text {i}",
            "publishedAt": published, "source": {"name": f"Site {i}"}}


def test_the_sample_excludes_earlier_links_dates_outside_and_missing_and_lists_near_titles():
    w = w2.week("tata")
    responses = {"A": {"articles": [_a(1, "Tata Motors opens a plant"),
                                    _a(2, "Old story", url="https://old.example/x"),
                                    _a(3, "Late story", published="2026-11-07T00:00:00Z"),
                                    _a(4, "Last second story", published="2026-11-06T23:59:59Z")]},
                 "B": {"articles": [_a(5, "No date", published=None),
                                    _a(6, "Jaguar Land Rover reveals the new Range Rover electric SUV today")]}}
    earlier = [{"url": "https://old.example/x", "title": "Old story", "publishedAt": "2026-10-20T10:00:00Z"},
               {"url": "https://old.example/y", "title": "Jaguar Land Rover reveals the new Range Rover electric SUV",
                "publishedAt": "2026-10-25T10:00:00Z"}]
    result = w2.sample(w, responses, earlier)
    kept = {a["title"] for a in result["kept"]}
    reasons = {e["title"]: e["reason"] for e in result["excluded"]}
    assert kept == {"Tata Motors opens a plant", "Last second story",
                    "Jaguar Land Rover reveals the new Range Rover electric SUV today"}
    assert reasons == {"Old story": "earlier_link", "Late story": "outside_window", "No date": "no_date"}
    assert [r["fresh_url"] for r in result["review"]] == ["https://news.example/6"]
    assert result["counts"]["kept_before_review"] == 3 and result["counts"]["to_review"] == 1


def test_the_earlier_data_is_each_companys_own():
    from tests.raw_data import require_raw_data
    require_raw_data()
    tata, reliance = w2.earlier_articles(w2.week("tata")), w2.earlier_articles(w2.week("reliance"))
    assert len(tata) > 200 and len(reliance) == 60
    assert "Tata Motors" in " ".join(a.get("title") or "" for a in tata)
    assert "Reliance" in " ".join(a.get("title") or "" for a in reliance)


def test_day_counts_list_every_day_of_the_window():
    counts = w2.day_counts(w2.week("tata"), [_a(1, "x", "2026-11-06T23:59:59Z"), _a(2, "y", "2026-11-07T00:00:00Z")])
    assert list(counts) == [f"2026-{d}" for d in ("10-31", "11-01", "11-02", "11-03", "11-04", "11-05", "11-06")]
    assert counts["2026-11-06"] == 1 and sum(counts.values()) == 1


# ---------- the seal ----------

def test_pairs_must_name_two_different_sealed_events():
    names = {"q2_preview", "q2_results"}
    assert w2.pairs_problems([], names) == []                       # no row: no pairs
    assert w2.pairs_problems([{"expectation_event": "Q2_Preview", "outcome_event": "q2_results"}], names) == []
    assert w2.pairs_problems([{"expectation_event": " q2_preview ", "outcome_event": "q2_results"}], names) == []
    assert w2.pairs_problems([{"expectation_event": "", "outcome_event": ""}], names)
    assert w2.pairs_problems([{"expectation_event": "q2_preview", "outcome_event": " "}], names)
    assert w2.pairs_problems([{"expectation_event": "", "outcome_event": "q2_results"}], names)
    assert w2.pairs_problems([{"expectation_event": "q2 preview", "outcome_event": "q2_results"}], names)
    assert w2.pairs_problems([{"expectation_event": "q2_preview", "outcome_event": "q2__results"}], names)
    assert w2.pairs_problems([{"expectation_event": "q2_preview", "outcome_event": "other"}], names)
    assert w2.pairs_problems([{"expectation_event": "q2_preview", "outcome_event": "q2_preview"}], names)


def test_a_change_after_the_seal_is_found(tmp_path):
    for name in (w2.LABEL_FILE, w2.EVENTS_FILE, w2.PAIRS_FILE):
        (tmp_path / name).write_text("a\n", encoding="utf-8")
    seal = w2.seal_record(tmp_path)
    assert w2.seal_changes(seal, tmp_path) == []
    (tmp_path / w2.PAIRS_FILE).write_text("b\n", encoding="utf-8")
    assert w2.seal_changes(seal, tmp_path) == [w2.PAIRS_FILE]


# ---------- the ranking test ----------

FIVE = "Tata Motors builds. Tata Motors hires. Tata Motors ships. Tata Motors grows. Tata Motors wins."


def candidates():
    """Retrieval order: 11 title articles, a description article and an untitled article with 5 names."""
    found = [{"key": f"t{i}", "title": f"Tata Motors story {i}", "mention_level": "title", "text": "Tata Motors."}
             for i in range(1, 10)]
    found.insert(2, {"key": "d1", "title": "Auto stocks", "mention_level": "description", "text": ""})
    found.insert(4, {"key": "u1", "title": "A carmaker's big export push", "mention_level": "body", "text": FIVE})
    found += [{"key": "t10", "title": "Tata Motors story 10", "mention_level": "title", "text": "x"}]
    return found


def labels_and_events():
    labels = {c["key"]: "direct" for c in candidates()}
    labels["t9"], labels["t10"] = "not", "not"
    events = {key: f"event_{key}" for key, label in labels.items() if label != "not"}
    return labels, events


def test_the_ranking_test_compares_o1m5_with_the_retrieval_order_first_and_o1_s1_second():
    labels, events = labels_and_events()
    result = w2.ranking_result(candidates(), "Tata Motors", ALIASES, labels, events)
    assert result["primary"]["baseline"] == "retrieval_order" and result["primary"]["configuration"] == "O1M5+S1"
    assert result["second"]["baseline"] == "O1+S1" and result["second"]["configuration"] == "O1M5+S1"
    assert result["passed"] == result["primary"]["passed"]
    assert result["old_default_against_retrieval"]["configuration"] == "O1+S1"
    orders = result["orders"]
    assert orders["O1M5+S1"].index("u1") < orders["O1+S1"].index("u1")
    untitled = {a["key"]: a for a in result["untitled_relevant"]}
    assert set(untitled) == {"d1", "u1"}
    assert untitled["u1"]["in_top_10"]["O1M5+S1"] and not untitled["u1"]["in_top_10"]["O1+S1"]


def test_the_second_comparison_never_changes_the_primary_result():
    labels, events = labels_and_events()
    result = w2.ranking_result(candidates(), "Tata Motors", ALIASES, labels, events)
    assert result["passed"] is result["primary"]["passed"] and "reported only" in result["second"]["decides"]


def test_the_default_in_the_test_is_the_pipelines_default():
    from src import research as rs
    labels, events = labels_and_events()
    ordered = w2.orders(candidates(), "Tata Motors", ALIASES)
    expected = rs.default_order([dict(c, check=None) for c in candidates()], "Tata Motors", ALIASES)
    assert [c["key"] for c in ordered["O1M5+S1"]] == [c["key"] for c in expected]
    assert w2.DEFAULT == rs.DEFAULT_ORDERING


# ---------- the commands, with a fake NewsAPI ----------

@pytest.fixture
def week_dir(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(w2, "EVAL_DIR", tmp_path / "eval")
    monkeypatch.setattr(w2, "RAW_DIR", raw)
    monkeypatch.setattr(w2, "earlier_articles", lambda w: [])
    w = w2.week("tata")
    w["run_dir"].mkdir(parents=True)
    protocol, text = _protocol(w)
    (w["run_dir"] / w2.PROTOCOL_FILE).write_text(text, encoding="utf-8")
    monkeypatch.setattr(w2, "_tag_text", lambda week: text)
    return w, raw


class FakeNews:
    def __init__(self):
        self.calls = []

    def __call__(self, company, aliases=None, part="all", **settings):
        self.calls.append((part, settings["from_time"], settings["to_time"]))
        return {"articles": [_a(len(self.calls), f"Tata Motors story {len(self.calls)}")], "request": {"q": company}}


def saver(raw):
    def save(response):
        path = raw / f"r{len(list(raw.iterdir()))}.json"
        path.write_text(json.dumps(response), encoding="utf-8")
        return path
    return save


def test_the_fetch_asks_the_window_records_counts_and_is_made_once(week_dir):
    w, raw = week_dir
    news = FakeNews()
    w2._fetch_command(w, news, saver(raw), datetime(2026, 11, 9, tzinfo=UTC))
    assert news.calls == [("all", "2026-10-31T00:00:00Z", "2026-11-06T23:59:59Z"),
                          ("subsidiaries", "2026-10-31T00:00:00Z", "2026-11-06T23:59:59Z")]
    stage = w2.load_state(w["run_dir"])["stages"]["fetch"]
    assert stage["done"] and set(stage["day_counts"]) == {"A", "B"} and stage["fetched_at"].startswith("2026-11-09")
    with pytest.raises(w2.Week2Error, match="mix two sets"):
        w2._fetch_command(w, news, saver(raw), datetime(2026, 11, 9, tzinfo=UTC))
    assert len(news.calls) == 2


def test_a_fetch_whose_first_request_failed_can_start_again_but_not_one_that_saved_a_response(week_dir):
    w, raw = week_dir

    class Failing(FakeNews):
        def __call__(self, company, **settings):
            raise ConnectionError("network")
    with pytest.raises(ConnectionError):
        w2._fetch_command(w, Failing(), saver(raw), datetime(2026, 11, 9, tzinfo=UTC))
    assert w2.load_state(w["run_dir"])["stages"]["fetch"]["raw_files"] == {}
    news = FakeNews()
    w2._fetch_command(w, news, saver(raw), datetime(2026, 11, 9, tzinfo=UTC))
    assert len(news.calls) == 2 and w2.load_state(w["run_dir"])["stages"]["fetch"]["attempt"] == 2

    class SecondFails(FakeNews):
        def __call__(self, company, **settings):
            if self.calls:
                raise ConnectionError("network")
            return super().__call__(company, **settings)
    other = w2.week("reliance")
    other["run_dir"] = w["run_dir"].parent / "half"
    other["run_dir"].mkdir()
    (other["run_dir"] / w2.PROTOCOL_FILE).write_text((w["run_dir"] / w2.PROTOCOL_FILE).read_text(encoding="utf-8"),
                                                     encoding="utf-8")
    w2.save_state(other["run_dir"], {"stages": {"fetch": {"done": False, "raw_files": {"A": "r0.json"}}}})
    with pytest.raises(w2.Week2Error, match="mix two sets"):
        w2._fetch_command(dict(w, run_dir=other["run_dir"]), FakeNews(), saver(raw), datetime(2026, 11, 9, tzinfo=UTC))


def test_empty_days_are_recorded(week_dir):
    w, raw = week_dir
    w2._fetch_command(w, FakeNews(), saver(raw), datetime(2026, 11, 9, tzinfo=UTC))
    empty = w2.load_state(w["run_dir"])["stages"]["fetch"]["empty_days"]
    assert "2026-11-06" in empty["A"] and "2026-11-03" not in empty["A"]


def test_the_protocol_states_the_whole_second_convention_and_the_request_uses_23_59_59():
    record = w2.protocol_record(w2.week("tata"), ALIASES, "a", "g", "abc")
    assert "whole seconds" in record["window_convention"] and record["window"][1].endswith("23:59:59Z")


def test_an_early_fetch_sends_nothing(week_dir):
    w, raw = week_dir
    news = FakeNews()
    with pytest.raises(w2.Week2Error, match="too early"):
        w2._fetch_command(w, news, saver(raw), datetime(2026, 11, 8, 5, tzinfo=UTC))
    assert news.calls == [] and not (w["run_dir"] / w2.STATE_FILE).exists()


def test_the_overlap_step_runs_once_and_the_rank_step_needs_the_seal(week_dir):
    w, raw = week_dir
    w2._fetch_command(w, FakeNews(), saver(raw), datetime(2026, 11, 9, tzinfo=UTC))
    w2._overlap_command(w)
    assert len(w2.kept_articles(w, w2.load_state(w["run_dir"]))) == 2
    with pytest.raises(w2.Week2Error, match="already"):
        w2._overlap_command(w)
    with pytest.raises(w2.Week2Error, match="not sealed"):
        w2._rank_command(w)


def test_the_commands_write_only_into_the_weeks_own_folder_and_the_raw_folder(week_dir):
    w, raw = week_dir
    root = w["run_dir"].parent.parent
    w2._fetch_command(w, FakeNews(), saver(raw), datetime(2026, 11, 9, tzinfo=UTC))
    w2._overlap_command(w)
    written = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert all(p.startswith(f"eval/{w['run_id']}/") or p.startswith("raw/") for p in written)


# ---------- the existing locks stay as they are ----------

def test_the_reliance_week_1_lock_still_matches_its_judging_code():
    from src import transfer_eval as te
    lock = json.loads((EVAL_DIR / te.RUN_ID / te.JUDGING_LOCK).read_text(encoding="utf-8"))
    assert lock["code_fingerprint"] == te._code_fingerprint()
    assert "src/week2_eval.py" not in te.JUDGING_CODE


def test_round_3_frozen_code_is_unchanged_and_does_not_include_this_module():
    from src import summary_prompt6 as p6
    from src import summary_round3 as r3
    plan = json.loads((EVAL_DIR / r3.PLAN_FILE).read_text(encoding="utf-8"))
    assert plan["code_fingerprint"] == p6.code_fingerprint(r3.CODE_FILES)
    assert "src/week2_eval.py" not in r3.CODE_FILES


def test_the_seal_needs_every_label_every_event_and_valid_pairs_and_runs_once(week_dir):
    import csv
    from src.fulltext import article_id
    w, raw = week_dir
    w2._fetch_command(w, FakeNews(), saver(raw), datetime(2026, 11, 9, tzinfo=UTC))
    w2._overlap_command(w)
    kept = w2.kept_articles(w, w2.load_state(w["run_dir"]))
    ids = [article_id(a["url"]) for a in kept]
    with open(w["run_dir"] / w2.LABEL_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["article_id", "label"])
        writer.writeheader()
        writer.writerows([{"article_id": ids[0], "label": "direct"}, {"article_id": ids[1], "label": "not"}])
    w2._events_command(w)
    assert (w["run_dir"] / w2.PAIRS_FILE).exists()
    with pytest.raises(w2.Week2Error, match="needs an event name"):
        w2._seal_command(w)
    rows = list(csv.DictReader(open(w["run_dir"] / w2.EVENTS_FILE, encoding="utf-8")))
    rows[0]["event"] = "plant_opening"
    with open(w["run_dir"] / w2.EVENTS_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(w["run_dir"] / w2.PAIRS_FILE, "a", newline="", encoding="utf-8") as handle:
        handle.write("plant_opening,unknown_event,\n")
    with pytest.raises(w2.Week2Error, match="not an event name"):
        w2._seal_command(w)
    with open(w["run_dir"] / w2.PAIRS_FILE, "w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=w2.PAIR_COLUMNS).writeheader()
    w2._seal_command(w)
    with pytest.raises(w2.Week2Error, match="already exists"):
        w2._seal_command(w)


# ---------- no earlier or locked file changes when every command runs ----------

def _snapshot(root):
    import hashlib
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_every_command_leaves_earlier_and_locked_files_unchanged(week_dir, monkeypatch):
    import csv
    import shutil
    from src.fulltext import article_id
    from src.ranking_eval import stage1_candidates
    from src import summary_round3 as r3
    w, raw = week_dir
    eval_dir = w["run_dir"].parent
    # copies of the earlier and locked records inside the test's own folders
    shutil.copytree(EVAL_DIR / "transfer_reliance_2026-10-26_11-01", eval_dir / "transfer_reliance_2026-10-26_11-01")
    (eval_dir / "fresh_2026-10-24_30").mkdir()
    shutil.copy(EVAL_DIR / "fresh_2026-10-24_30" / "state.json", eval_dir / "fresh_2026-10-24_30" / "state.json")
    for name in (r3.PLAN_FILE, "summary_v2_r3_pair1_p4.json", "summary_v2_r3_pair1_p6.json"):
        shutil.copy(EVAL_DIR / name, eval_dir / name)
    (raw / "earlier.json").write_text('{"articles": []}', encoding="utf-8")
    protected_root = w["run_dir"].parent.parent
    before = {k: v for k, v in _snapshot(protected_root).items() if not k.startswith(f"eval/{w['run_id']}/")}
    real = [EVAL_DIR / "transfer_reliance_2026-10-26_11-01", EVAL_DIR / "fresh_2026-10-24_30"]
    real_before = {str(d): _snapshot(d) for d in real}
    real_files = [EVAL_DIR / r3.PLAN_FILE, Path("src/transfer_eval.py"), Path("src/summary_round3.py")]
    real_files_before = {str(f): f.read_bytes() for f in real_files}

    w2._fetch_command(w, FakeNews(), saver(raw), datetime(2026, 11, 9, tzinfo=UTC))
    w2._overlap_command(w)
    kept = w2.kept_articles(w, w2.load_state(w["run_dir"]))
    with open(w["run_dir"] / w2.LABEL_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["article_id", "label"])
        writer.writeheader()
        writer.writerows([{"article_id": article_id(a["url"]), "label": "direct"} for a in kept])
    w2._events_command(w)
    rows = list(csv.DictReader(open(w["run_dir"] / w2.EVENTS_FILE, encoding="utf-8")))
    with open(w["run_dir"] / w2.EVENTS_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([dict(r, event=f"event_{n}") for n, r in enumerate(rows)])
    w2._seal_command(w)
    monkeypatch.setattr(w2, "_candidates",
                        lambda week, state, aliases: stage1_candidates(w2.kept_articles(week, state), {},
                                                                       week["company"], aliases))
    w2._rank_command(w)

    after = _snapshot(protected_root)
    new = {k for k in after if k not in before}
    assert {k: after.get(k) for k in before} == before                # nothing earlier changed or deleted
    assert all(k.startswith(f"eval/{w['run_id']}/") or k.startswith("raw/r") for k in new)
    saved = w2.load_state(w["run_dir"])["stages"]["fetch"]["raw_files"]
    assert {k for k in new if k.startswith("raw/")} == {f"raw/{name}" for name in saved.values()}
    assert {k.split("/")[-1] for k in new if k.startswith("eval/")} == {
        w2.PROTOCOL_FILE, w2.STATE_FILE, w2.LABEL_FILE, w2.EVENTS_FILE, w2.PAIRS_FILE, w2.SEAL_FILE, w2.RANK_FILE}
    assert {str(d): _snapshot(d) for d in real} == real_before        # the real records are untouched too
    assert {str(f): f.read_bytes() for f in real_files} == real_files_before
