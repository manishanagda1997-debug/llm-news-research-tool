"""
Tests for the fresh-week summary run (python -m src.summary_fresh; SUMMARY_JUDGING_GUIDE_PART6.md).
No model calls, no Git, no network.

    order       the record must not exist (checked before anything else, so no second run is possible);
                then the freeze must accept the code and settings; then Git must show no change or untracked
                file under src/, in the alias file, in the fresh week's labels and events, or in the guides
    one run     the frozen design and k, one live call path with no saved answers (summary and repair),
                written once; the record names the commit, the freeze record's fingerprint, every Stage 1
                candidate in its order and the five selected articles with their ranks
"""

import json

import pytest

from src import summary_fresh as fs
from src import summary_v2 as sv
from src.config import EVAL_DIR as REAL_EVAL_DIR
from src.fulltext import FULLTEXT_DIR as REAL_FULLTEXT_DIR


def _tree(folder):
    """{relative path: SHA-256 of the content} for every file under the folder, and {relative path: "folder"} for
    every folder under it (empty ones included), so an added or removed folder is seen too."""
    import hashlib
    if not folder.exists():
        return {}
    return {str(path.relative_to(folder)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "folder"
            for path in sorted(folder.rglob("*")) if path.is_file() or path.is_dir()}


def _real_folders():
    return {"data/eval": _tree(REAL_EVAL_DIR), "data/fulltext": _tree(REAL_FULLTEXT_DIR)}


@pytest.fixture(autouse=True, scope="module")
def no_test_writes_into_the_real_project_folders():
    """Every test of this file runs in a temporary folder: the project's own data/eval/ and data/fulltext/ trees
    must end exactly as they started (every file by its content, and every folder, empty ones included)."""
    before = _real_folders()
    yield
    after = _real_folders()
    for name in before:
        added, removed = sorted(set(after[name]) - set(before[name])), sorted(set(before[name]) - set(after[name]))
        changed = sorted(p for p in set(before[name]) & set(after[name]) if before[name][p] != after[name][p])
        assert not (added or removed or changed), f"{name}: added {added}, removed {removed}, changed {changed}"


def test_the_folder_snapshot_sees_nested_files_and_content_changes(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "deep.txt").write_text("one", encoding="utf-8")
    first = _tree(tmp_path)
    assert first["a/b/deep.txt"] != "folder" and first["a"] == first["a/b"] == "folder"
    (tmp_path / "a" / "b" / "deep.txt").write_text("two", encoding="utf-8")
    assert _tree(tmp_path) != first


def test_the_folder_snapshot_sees_an_empty_folder_added_or_removed(tmp_path):
    before = _tree(tmp_path)
    (tmp_path / "empty").mkdir()
    added = _tree(tmp_path)
    assert added == dict(before, empty="folder")
    (tmp_path / "empty").rmdir()
    assert _tree(tmp_path) == before


def test_only_the_protected_paths_count_as_changes():
    status = "\n".join([" M src/summarizer.py", "?? src/new_helper.py", " M data/company_aliases.json",
                        " M data/eval/fresh_2026-10-24_30/labels.csv", "?? data/eval/fresh_2026-10-24_30/events.csv",
                        " M data/eval/SUMMARY_JUDGING_GUIDE_PART6.md", " M README.md", "?? notes/todo.txt",
                        " M data/eval/fresh_2026-10-24_30/state.json"])
    assert fs.fresh_path_problems(status) == [
        "data/company_aliases.json", "data/eval/SUMMARY_JUDGING_GUIDE_PART6.md",
        "data/eval/fresh_2026-10-24_30/events.csv", "data/eval/fresh_2026-10-24_30/labels.csv",
        "src/new_helper.py", "src/summarizer.py"]
    assert fs.fresh_path_problems("") == []


def test_a_renamed_file_counts_by_its_new_path():
    assert fs.fresh_path_problems("R  old.txt -> src/moved.py") == ["src/moved.py"]


FREEZE = {"design": "direct", "ks": [5], "settings": {"prompt_version": "4", "passage_version": "3",
                                                       "pacer_version": "3"}}


class Calls:
    def __init__(self):
        self.order = []


def _setup(monkeypatch, tmp_path, calls, status="", freeze=FREEZE, freeze_error=None, crash=False):
    from src import summary_freeze as sf
    from src import fresh_eval as fe

    def frozen(k):
        calls.order.append(("freeze", k))
        if freeze_error:
            raise sf.FreezeError(freeze_error)
        return freeze

    def git(*args):
        calls.order.append(("git",) + args)
        return status if args[0] == "status" else "abc123"

    def one(design, articles, llm, cache, k):
        calls.order.append(("one", design, type(cache).__name__, k, [a["key"] for a in articles]))
        if crash:
            raise RuntimeError("the process stops after the request is sent")
        return {"status": "ok", "summary": {"key_points": []}, "usage": {"calls": 1}, "origin": "live"}

    monkeypatch.setattr(fs, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(sf, "frozen_summary", frozen)
    monkeypatch.setattr(sf, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(fe, "git", git)
    monkeypatch.setattr(fe, "load_state", lambda run_dir: {"stages": {}})
    monkeypatch.setattr(fe, "_candidates", lambda state, company, aliases: [{"key": f"K{i}"} for i in range(7)])
    monkeypatch.setattr(fe, "file_fingerprint", lambda path: "f" * 64)
    monkeypatch.setattr(fs, "_fresh_aliases", lambda: {})
    # article IDs deliberately differ from the ranking positions (10, 11, ...), so a rank cannot be taken from an ID
    monkeypatch.setattr(sv, "summary_articles", lambda candidates, k, aliases:
                        [{"id": 10 + i, "key": c["key"], "title": f"T{c['key']}", "description": f"Passage {c['key']}.",
                          "content": ""} for i, c in enumerate(list(reversed(candidates))[:k])])
    monkeypatch.setattr(fs, "FULLTEXT_DIR", tmp_path / "fulltext")
    (tmp_path / "fulltext").mkdir()
    monkeypatch.setattr(sv, "summary_llm", lambda: calls.order.append(("llm",)) or type("L", (), {"model_name": "m"})())
    monkeypatch.setattr(sv, "_one", one)
    (tmp_path / "summary_freeze.json").write_text("{}", encoding="utf-8")


def test_an_existing_record_stops_the_run_before_anything_else(monkeypatch, tmp_path):
    calls = Calls()
    _setup(monkeypatch, tmp_path, calls)
    (tmp_path / fs.FRESH_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(fs.FreshRunError, match="already exists"):
        fs.fresh_command()
    assert calls.order == []
    assert (tmp_path / fs.FRESH_FILE).read_text(encoding="utf-8") == "{}"


def test_a_freeze_problem_stops_the_run_before_git_and_the_model(monkeypatch, tmp_path):
    calls = Calls()
    _setup(monkeypatch, tmp_path, calls, freeze_error="the setting pacer_version differs from the freeze")
    with pytest.raises(fs.FreshRunError, match="pacer_version"):
        fs.fresh_command()
    assert calls.order == [("freeze", 5)] and not (tmp_path / fs.FRESH_FILE).exists()


def test_a_change_in_a_protected_path_stops_the_run_before_the_model(monkeypatch, tmp_path):
    calls = Calls()
    _setup(monkeypatch, tmp_path, calls, status=" M src/ranking.py\n M README.md")
    with pytest.raises(fs.FreshRunError, match="src/ranking.py"):
        fs.fresh_command()
    assert not any(c[0] in ("llm", "one") for c in calls.order) and not (tmp_path / fs.FRESH_FILE).exists()


@pytest.mark.parametrize("freeze", [dict(FREEZE, design="brief_first"), dict(FREEZE, ks=[10]),
                                    dict(FREEZE, settings=dict(FREEZE["settings"], prompt_version="5")),
                                    dict(FREEZE, settings=dict(FREEZE["settings"], passage_version="2")),
                                    dict(FREEZE, settings=dict(FREEZE["settings"], pacer_version="2"))])
def test_a_freeze_that_is_not_the_agreed_design_stops_the_run(monkeypatch, tmp_path, freeze):
    calls = Calls()
    _setup(monkeypatch, tmp_path, calls, freeze=freeze)
    with pytest.raises(fs.FreshRunError):
        fs.fresh_command()
    assert not any(c[0] == "one" for c in calls.order)


def test_one_live_run_with_the_frozen_design_and_a_complete_record(monkeypatch, tmp_path):
    calls = Calls()
    _setup(monkeypatch, tmp_path, calls, status=" M README.md")
    fs.fresh_command()
    ones = [c for c in calls.order if c[0] == "one"]
    assert ones == [("one", "direct", "NoCache", 5, ["K6", "K5", "K4", "K3", "K2"])]
    record = json.loads((tmp_path / fs.FRESH_FILE).read_text(encoding="utf-8"))
    assert record["run"] == "fresh_2026-10-24_30" and record["design"] == "direct" and record["k"] == 5
    assert record["source_commit"] == "abc123" and record["freeze_fingerprint"] == "f" * 64
    assert record["prompt_version"] == "4" and record["pacer_version"] == "3" and record["passage_version"] == "3"
    assert record["cache"] == "No answer cache is read or written; the run result is saved in this record."
    assert [c["key"] for c in record["candidates"]] == [f"K{i}" for i in range(7)]      # Stage 1 order
    assert record["candidates"][0]["position"] == 1
    assert [(s["rank"], s["id"], s["key"]) for s in record["selected"]] == \
        [(1, 10, "K6"), (2, 11, "K5"), (3, 12, "K4"), (4, 13, "K3"), (5, 14, "K2")]
    saved = json.loads((tmp_path / "fulltext" / fs.PASSAGES_FILE).read_text(encoding="utf-8"))
    assert [p["description"] for p in saved] == ["Passage K6.", "Passage K5.", "Passage K4.", "Passage K3.", "Passage K2."]
    assert [p["sha256"] for p in record["passage_fingerprints"]] == [p["sha256"] for p in saved]
    assert "Passage K6." not in json.dumps(record)                  # publisher text stays out of the record
    assert (tmp_path / fs.ATTEMPT_FILE).exists()
    assert [r["design"] for r in record["records"]] == ["direct"] and record["records"][0]["run"] == "fresh_2026-10-24_30"


def test_the_no_cache_never_reads_or_saves():
    cache = sv.NoCache()
    assert cache.load("any key") is None and cache.save("key", {"x": 1}) is None


def test_the_fresh_run_lives_outside_the_frozen_summary_code():
    for name in ("fresh_command", "_fresh_command", "FRESH_FILE", "FRESH_RUN", "FreshRunError"):
        assert not hasattr(sv, name)


def test_neither_the_summary_nor_a_repair_reads_or_saves_a_saved_answer(monkeypatch):
    """The real direct design with a repair: every saved-answer store raises if touched; only NoCache is given."""
    from src import summarizer as sm

    def touched(*args, **kwargs):
        raise AssertionError("a saved-answer store was used")

    monkeypatch.setattr(sm.SummaryCache, "load", touched)
    monkeypatch.setattr(sm.SummaryCache, "save", touched)
    articles = [{"id": 1, "key": "A1", "title": "T1", "source": "S", "published_at": "", "url": "https://x.com/1",
                 "description": "Tata Motors raised its Iveco offer to 14.40 euro per share.", "content": ""},
                {"id": 2, "key": "A2", "title": "T2", "source": "S", "published_at": "", "url": "https://x.com/2",
                 "description": "Car registrations fell 22 per cent this Navratri.", "content": ""}]
    answer = {"article_briefs": [{"id": 1, "brief": "Tata Motors raised its Iveco offer to 14.40 euro per share.",
                                  "enough_text": True},
                                 {"id": 2, "brief": "Car registrations fell 22 per cent this Navratri.",
                                  "enough_text": True}],
              "key_points": [{"text": "Tata Motors raised its offer to 14 euro per share.", "citations": [1]}],
              "industry_context": [], "not_used": [{"id": 2, "reason": "r"}]}

    class Live:
        model_name, cache_tag = "fake", ""

        def __init__(self):
            self.prompts = []

        def __call__(self, prompt):
            self.prompts.append(prompt)
            if prompt.startswith(sm.REPAIR_RULES[:30]):
                return json.dumps({"text": "Tata Motors raised its offer to 14.40 euro per share."})
            return json.dumps(answer)

    llm = Live()
    result = sv._one("direct", articles, llm, sv.NoCache(), 2)
    assert result["status"] in ("ok", "partial") and result.get("origin") != "saved"
    assert len(llm.prompts) == 2 and llm.prompts[1].startswith(sm.REPAIR_RULES[:30])      # one summary, one repair
    assert [r["ok"] for r in result["repairs"]] == [True]



# ---------- one attempt, even after an interruption ----------

def test_an_attempt_marker_stops_the_run_before_anything_else(monkeypatch, tmp_path):
    calls = Calls()
    _setup(monkeypatch, tmp_path, calls)
    (tmp_path / fs.ATTEMPT_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(fs.FreshRunError, match="attempted once only"):
        fs.fresh_command()
    assert calls.order == []


def test_an_interrupted_run_can_never_send_a_second_request(monkeypatch, tmp_path):
    calls = Calls()
    _setup(monkeypatch, tmp_path, calls, crash=True)
    with pytest.raises(RuntimeError):
        fs.fresh_command()                                       # stops after the call starts, before the record
    assert (tmp_path / fs.ATTEMPT_FILE).exists() and not (tmp_path / fs.FRESH_FILE).exists()
    with pytest.raises(fs.FreshRunError, match="attempted once only"):
        fs.fresh_command()
    assert len([c for c in calls.order if c[0] == "one"]) == 1


def test_the_marker_is_written_before_the_model_is_called(monkeypatch, tmp_path):
    calls = Calls()
    _setup(monkeypatch, tmp_path, calls)
    seen = []
    real_one = sv._one
    monkeypatch.setattr(sv, "_one", lambda *args: seen.append((tmp_path / fs.ATTEMPT_FILE).exists()) or real_one(*args))
    fs.fresh_command()
    assert seen == [True]


def test_the_passage_fingerprint_changes_with_any_change_of_the_text():
    article = {"id": 1, "key": "K", "title": "T", "description": "Passage.", "content": ""}
    assert fs.passage_entry(article)["sha256"] != fs.passage_entry(dict(article, description="Passage!"))["sha256"]
    assert fs.passage_entry(article)["sha256"] == fs.passage_entry(dict(article))["sha256"]



# ---------- one explicit set of paths; the marker first ----------

def test_the_run_has_one_explicit_set_of_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(fs, "EVAL_DIR", tmp_path / "eval")
    monkeypatch.setattr(fs, "FULLTEXT_DIR", tmp_path / "fulltext")
    assert fs.fresh_paths() == {"record": tmp_path / "eval" / "summary_v2_fresh_k5_c7_p4.json",
                                "attempt": tmp_path / "eval" / "summary_v2_fresh_k5_c7_p4_attempt.json",
                                "passages": tmp_path / "fulltext" / "summary_v2_fresh_k5_c7_p4_passages.json"}


def test_the_evidence_file_name_is_dedicated_to_this_run():
    assert fs.PASSAGES_FILE == "summary_v2_fresh_k5_c7_p4_passages.json"
    assert fs.PASSAGES_FILE != "summary_v2_fresh_passages.json"         # the old test file is never read or written


def test_an_existing_evidence_file_stops_the_run_before_anything_else(monkeypatch, tmp_path):
    calls = Calls()
    _setup(monkeypatch, tmp_path, calls)
    (tmp_path / "fulltext" / fs.PASSAGES_FILE).write_text("[]", encoding="utf-8")
    with pytest.raises(fs.FreshRunError, match="attempted once only"):
        fs.fresh_command()
    assert calls.order == [] and not (tmp_path / fs.ATTEMPT_FILE).exists()
    assert (tmp_path / "fulltext" / fs.PASSAGES_FILE).read_text(encoding="utf-8") == "[]"   # left as it is


def test_the_marker_is_written_before_the_evidence(monkeypatch, tmp_path):
    calls = Calls()
    _setup(monkeypatch, tmp_path, calls)
    import builtins
    real_open, order = builtins.open, []

    def watching(path, mode="r", *args, **kwargs):
        if "x" in mode:
            order.append(type(path)(path).name if hasattr(path, "name") else str(path))
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", watching)
    fs.fresh_command()
    assert order == [fs.ATTEMPT_FILE, fs.PASSAGES_FILE, fs.FRESH_FILE]


def test_a_failed_evidence_write_keeps_the_marker_and_sends_nothing(monkeypatch, tmp_path):
    calls = Calls()
    _setup(monkeypatch, tmp_path, calls)
    real_dumps = json.dumps

    def failing(value, *args, **kwargs):
        if isinstance(value, list) and value and "sha256" in value[0]:
            raise OSError("the disk is full")
        return real_dumps(value, *args, **kwargs)

    monkeypatch.setattr(fs.json, "dumps", failing)
    with pytest.raises(OSError):
        fs.fresh_command()
    assert (tmp_path / fs.ATTEMPT_FILE).exists()
    assert not any(c[0] in ("llm", "one") for c in calls.order)
    monkeypatch.setattr(fs.json, "dumps", real_dumps)
    with pytest.raises(fs.FreshRunError, match="attempted once only"):
        fs.fresh_command()                                         # the second invocation stops at once
    assert not any(c[0] in ("llm", "one") for c in calls.order)


def test_a_second_invocation_after_a_complete_run_sends_nothing(monkeypatch, tmp_path):
    calls = Calls()
    _setup(monkeypatch, tmp_path, calls)
    fs.fresh_command()
    with pytest.raises(fs.FreshRunError, match="attempted once only"):
        fs.fresh_command()
    assert len([c for c in calls.order if c[0] == "one"]) == 1



def test_a_failed_disk_write_of_the_evidence_keeps_the_marker_and_sends_nothing(monkeypatch, tmp_path):
    """The evidence file opens, then its write() fails as a full disk would."""
    import builtins
    calls = Calls()
    _setup(monkeypatch, tmp_path, calls)
    real_open = builtins.open

    class FailingWrite:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.handle.close()
            return False

        def write(self, text):
            raise OSError("No space left on device")

    def opening(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        if "x" in mode and str(path).endswith(fs.PASSAGES_FILE):
            return FailingWrite(handle)
        return handle

    monkeypatch.setattr(builtins, "open", opening)
    with pytest.raises(OSError, match="No space left"):
        fs.fresh_command()
    monkeypatch.setattr(builtins, "open", real_open)
    evidence = tmp_path / "fulltext" / fs.PASSAGES_FILE
    assert (tmp_path / fs.ATTEMPT_FILE).exists()
    assert not any(c[0] in ("llm", "one") for c in calls.order)
    left = evidence.read_bytes()                                      # what the failed write left (here: empty)
    with pytest.raises(fs.FreshRunError, match="attempted once only"):
        fs.fresh_command()
    assert not any(c[0] in ("llm", "one") for c in calls.order)
    assert evidence.read_bytes() == left                              # not overwritten by the second invocation
