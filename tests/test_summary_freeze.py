"""
Tests for src/summary_freeze.py: the record that fixes the Stage 4 summary configuration.

The freeze record names the design, the k values the development evidence validates, every setting the
summary depends on (model, generation, output limit, prompt, configuration, correction and repair tries,
number check, source check, passage), the fingerprints of the code and the requirements, the package
versions, the development evidence and the source commit. It never holds its own commit: the source commit
is recorded, then the file is committed, then the tag is made. A later use refuses when the tag is missing,
the file differs from the tagged one, the requested k is not validated, or any setting or code differs.
"""

import pytest

from src import summary_freeze as sf

SETTINGS = {"provider": "groq", "model": "openai/gpt-oss-120b",
            "generation": {"temperature": 1.0, "reasoning_effort": "medium", "max_completion_tokens": 4096},
            "prompt_version": "3", "configuration_version": "7", "correction": "targeted", "repair_tries": 2,
            "max_attempts": 2, "number_check": "2", "source_check": "1", "passage_version": "3",
            "pacer_version": "3"}
CODE = {"src/summarizer.py": "a" * 64, "src/summary_v2.py": "b" * 64, "requirements.txt": "c" * 64}


def _record(**changes):
    values = dict(design="direct", ks=[5], evidence={"data/eval/summary_v2_dev_k5_c5_p3.json": "d" * 64},
                  source_commit="1234abc", settings=SETTINGS, code=CODE, packages={"groq": "0.37.1"},
                  decision="Design: direct. Articles per summary (k): 5.")
    values.update(changes)
    return sf.freeze_record(**values)


def test_the_record_names_everything_the_summary_depends_on():
    record = _record()
    assert record["design"] == "direct" and record["ks"] == [5]
    assert record["settings"] == SETTINGS and record["code"] == CODE and record["packages"] == {"groq": "0.37.1"}
    assert record["evidence"] and record["source_commit"] == "1234abc" and record["tag"] == sf.FREEZE_TAG
    assert "created" in record


def test_the_current_settings_are_read_from_the_code():
    current = sf.current_settings()
    assert set(current) == set(SETTINGS)
    assert current["generation"]["max_completion_tokens"] == 4096
    assert (current["prompt_version"], current["configuration_version"], current["number_check"],
            current["passage_version"], current["correction"], current["repair_tries"], current["pacer_version"]) == \
        ("4", "7", "2", "3", "targeted", 2, "3")


def test_the_code_fingerprints_cover_the_summary_code_and_the_requirements():
    found = sf.code_fingerprints()
    for path in ("src/summarizer.py", "src/summary_v2.py", "src/llm.py", "src/relevance_check.py",
                 "src/ranking.py", "src/summary_passage.py", "src/boilerplate.py", "requirements.txt"):
        assert len(found[path]) == 64


@pytest.mark.parametrize("changes", [dict(design="three_step"), dict(ks=[]), dict(ks=["5"]), dict(evidence={}),
                                     dict(decision=""), dict(decision="   \n ")])
def test_an_incomplete_or_unknown_freeze_is_refused(changes):
    with pytest.raises(sf.FreezeError):
        _record(**changes)


def _problems(record, k=5, settings=SETTINGS, code=CODE, tag_text="same", file_text="same"):
    return sf.freeze_problems(record, k, settings, code, tag_text, file_text)


def test_a_matching_freeze_has_no_problems():
    assert _problems(_record()) == []


def test_a_missing_tag_or_a_changed_file_is_a_problem():
    assert any(sf.FREEZE_TAG in p for p in _problems(_record(), tag_text=None))
    assert any("differs from the tagged" in p for p in _problems(_record(), tag_text="old", file_text="new"))


def test_a_k_the_evidence_does_not_validate_is_refused():
    assert any("k = 10" in p for p in _problems(_record(), k=10))
    assert _problems(_record(ks=[5, 10]), k=10) == []


def test_every_changed_setting_and_code_file_is_named():
    changed = dict(SETTINGS, number_check="3", generation={**SETTINGS["generation"], "max_completion_tokens": 8192})
    found = _problems(_record(), settings=changed, code=dict(CODE, **{"src/summarizer.py": "e" * 64}))
    assert any("number_check" in p for p in found) and any("generation" in p for p in found)
    assert any("src/summarizer.py" in p for p in found)


def test_the_freeze_refuses_while_summary_code_has_uncommitted_changes():
    def fake_git(*args):
        return " M src/summarizer.py\n?? notes.txt\n"
    assert sf.uncommitted(fake_git) == ["src/summarizer.py"]
    assert sf.uncommitted(lambda *args: "") == []


# ---------- the pacer version and the written decision ----------

def test_the_pacer_version_is_a_frozen_setting():
    from src.llm import PACER_VERSION
    assert sf.current_settings()["pacer_version"] == PACER_VERSION
    found = _problems(_record(), settings=dict(SETTINGS, pacer_version="4"))
    assert found == ["the setting pacer_version differs from the freeze"]


def test_the_record_holds_the_written_decision():
    assert _record()["decision"] == "Design: direct. Articles per summary (k): 5."


def test_the_freeze_command_needs_the_decision_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sf, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(sf, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(sf, "uncommitted", lambda run_git, files=sf.CODE_FILES: [])
    with pytest.raises(sf.FreezeError, match="summary_freeze_decision.md"):
        sf._freeze_command("direct", "5")
    assert not (tmp_path / sf.FREEZE_FILE).exists()


def test_the_freeze_command_copies_and_fingerprints_the_decision(tmp_path, monkeypatch):
    import json
    from src import fresh_eval
    monkeypatch.setattr(sf, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(sf, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(sf, "uncommitted", lambda run_git, files=sf.CODE_FILES: [])
    monkeypatch.setattr(sf, "current_settings", lambda: SETTINGS)
    monkeypatch.setattr(sf, "code_fingerprints", lambda: CODE)
    monkeypatch.setattr(fresh_eval, "git", lambda *args: "abc123")
    monkeypatch.setattr(fresh_eval, "package_versions", lambda: {"groq": "x"})
    (tmp_path / sf.DECISION_FILE).write_text("# Decision\n\nDesign: direct.\n", encoding="utf-8")
    (tmp_path / "record.json").write_text("{}", encoding="utf-8")
    sf._freeze_command("direct", "5", "record.json")
    record = json.loads((tmp_path / sf.FREEZE_FILE).read_text(encoding="utf-8"))
    assert record["decision"] == "# Decision\n\nDesign: direct.\n"
    assert set(record["evidence"]) == {"record.json", f"data/eval/{sf.DECISION_FILE}"}


def test_the_decision_file_must_be_committed_before_the_freeze():
    def fake_git(*args):
        return "?? data/eval/summary_freeze_decision.md\n"
    assert sf.uncommitted(fake_git, sf.CODE_FILES + (f"data/eval/{sf.DECISION_FILE}",)) == \
        ["data/eval/summary_freeze_decision.md"]


def test_the_freeze_command_refuses_an_uncommitted_decision_file(tmp_path, monkeypatch):
    from src import fresh_eval
    monkeypatch.setattr(sf, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(sf, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(fresh_eval, "git",
                        lambda *args: "?? data/eval/summary_freeze_decision.md\n" if args[0] == "status" else "abc")
    (tmp_path / sf.DECISION_FILE).write_text("# Decision\n", encoding="utf-8")
    with pytest.raises(sf.FreezeError, match="summary_freeze_decision.md"):
        sf._freeze_command("direct", "5")
    assert not (tmp_path / sf.FREEZE_FILE).exists()
