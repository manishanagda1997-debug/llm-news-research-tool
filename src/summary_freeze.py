"""
The Stage 4 freeze: the record that fixes the summary configuration chosen on the development data.

The record (data/eval/summary_freeze.json) names the design, the k values the development evidence validates,
every setting the summary depends on, the fingerprints of the summary code and the requirements, the package
versions, the development evidence files and the source commit. It never holds its own commit. The order is:
the code is committed (its commit is the source commit), the record is written and committed, then the tag
FREEZE_TAG is made on that commit.

A later use (the Stage 7 transfer test, the app's tested setting) calls frozen_summary(k): it refuses when the
tag is missing, the file differs from the tagged one, k is not validated, or any setting or code file differs
from the record.

The written decision (data/eval/summary_freeze_decision.md) is required: its text is copied into the record and
the file is fingerprinted with the evidence.

Run (after the Stage 4 comparison is judged and scored, and the decision is written and committed):
    python -m src.summary_freeze freeze direct 5 <evidence files>
    then: git add data/eval/summary_freeze.json, commit, git tag freeze-summary
"""

import json
import sys
from datetime import datetime, timezone

from src.config import EVAL_DIR, ROOT_DIR

FREEZE_FILE = "summary_freeze.json"
DECISION_FILE = "summary_freeze_decision.md"
FREEZE_TAG = "freeze-summary"
CODE_FILES = ("src/summarizer.py", "src/summary_v2.py", "src/llm.py", "src/relevance_check.py", "src/ranking.py",
              "src/summary_passage.py", "src/boilerplate.py", "requirements.txt")


class FreezeError(Exception):
    pass


def current_settings() -> dict:
    """Every setting the summary depends on, read from the code as it is now."""
    from src import summarizer as sm
    from src import summary_v2 as sv
    from src.llm import PACER_VERSION
    from src.model_compare import CONFIGS
    from src.summary_passage import PASSAGE_VERSION
    config = CONFIGS[sv.MODEL_SETTING]
    return {"provider": config["provider"], "model": config["model"],
            "generation": {**config["generation"], "max_completion_tokens": sv.OUTPUT_LIMIT},
            "prompt_version": sm.PROMPT_VERSION, "configuration_version": sv.SUMMARY_VERSION,
            "correction": sv.CORRECTION, "repair_tries": sm.REPAIR_TRIES[sv.CORRECTION],
            "max_attempts": sm.MAX_ATTEMPTS, "number_check": sm.NUMBER_CHECK_VERSION,
            "source_check": sm.SOURCE_CHECK_VERSION, "passage_version": PASSAGE_VERSION,
            "pacer_version": PACER_VERSION}


def code_fingerprints(root=ROOT_DIR, files=CODE_FILES) -> dict[str, str]:
    from src.fresh_eval import file_fingerprint
    return {path: file_fingerprint(root / path) for path in files}


def freeze_record(design: str, ks: list[int], evidence: dict[str, str], source_commit: str, settings: dict,
                  code: dict[str, str], packages: dict[str, str], decision: str = "") -> dict:
    from src.summary_v2 import DESIGNS
    if design not in DESIGNS:
        raise FreezeError(f"Unknown design {design}; use one of {', '.join(DESIGNS)}.")
    if not ks or not all(isinstance(k, int) and k > 0 for k in ks):
        raise FreezeError("The validated k values must be positive whole numbers.")
    if not evidence:
        raise FreezeError("The freeze needs the development evidence files that support it.")
    if not decision.strip():
        raise FreezeError(f"The freeze needs the written decision ({DECISION_FILE}).")
    return {"design": design, "ks": sorted(ks), "settings": settings, "code": code, "packages": packages,
            "evidence": evidence, "decision": decision, "source_commit": source_commit, "tag": FREEZE_TAG,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def freeze_problems(record: dict, k: int, settings: dict, code: dict[str, str], tag_text: str | None,
                    file_text: str) -> list[str]:
    """Everything that stops a use of the frozen summary; empty when the use may go ahead."""
    if tag_text is None:
        return [f"the tag {FREEZE_TAG} does not exist or does not hold {FREEZE_FILE}"]
    found = []
    if tag_text.replace("\r\n", "\n") != file_text.replace("\r\n", "\n"):
        found.append(f"{FREEZE_FILE} differs from the tagged one")
    if k not in record["ks"]:
        found.append(f"the requested k = {k} is not validated (validated: {', '.join(map(str, record['ks']))})")
    found += [f"the setting {name} differs from the freeze" for name in sorted(set(record["settings"]) | set(settings))
              if record["settings"].get(name) != settings.get(name)]
    found += [f"{path} differs from the freeze" for path in sorted(set(record["code"]) | set(code))
              if record["code"].get(path) != code.get(path)]
    return found


def uncommitted(run_git, files=CODE_FILES) -> list[str]:
    """The summary code files with changes Git has not committed."""
    changed = [line.split()[-1] for line in run_git("status", "--porcelain").splitlines() if line.strip()]
    return [path for path in changed if path in files]


def _tag_text(run_git) -> str | None:
    try:
        return run_git("show", f"{FREEZE_TAG}:data/eval/{FREEZE_FILE}")
    except Exception:
        return None


def frozen_summary(k: int, run_git=None) -> dict:
    """The frozen record for a use with k articles; raises with every problem found."""
    from src.fresh_eval import git
    run_git = run_git or git
    path = EVAL_DIR / FREEZE_FILE
    if not path.exists():
        raise FreezeError(f"{FREEZE_FILE} does not exist: the Stage 4 summary is not frozen yet.")
    text = path.read_text(encoding="utf-8")
    record = json.loads(text)
    problems = freeze_problems(record, k, current_settings(), code_fingerprints(), _tag_text(run_git), text)
    if problems:
        raise FreezeError("The frozen summary cannot be used:\n" + "\n".join(problems))
    return record


def _freeze_command(design: str, ks_text: str, *evidence_files: str) -> None:
    from src.fresh_eval import file_fingerprint, git, package_versions
    path = EVAL_DIR / FREEZE_FILE
    if path.exists():
        raise FreezeError(f"{FREEZE_FILE} already exists; it is not overwritten.")
    dirty = uncommitted(git, CODE_FILES + (f"data/eval/{DECISION_FILE}",))
    if dirty:
        raise FreezeError("Commit these files first, so the source commit holds them: " + ", ".join(dirty))
    try:
        ks = [int(k) for k in ks_text.split(",")]
    except ValueError as error:
        raise FreezeError("k values: whole numbers separated by commas, for example 5 or 5,10") from error
    decision_path = EVAL_DIR / DECISION_FILE
    if not decision_path.exists():
        raise FreezeError(f"data/eval/{DECISION_FILE} does not exist: write and commit the decision first.")
    decision = decision_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    evidence = {}
    for name in evidence_files:
        if not (ROOT_DIR / name).exists():
            raise FreezeError(f"{name} does not exist.")
        evidence[name] = file_fingerprint(ROOT_DIR / name)
    evidence[f"data/eval/{DECISION_FILE}"] = file_fingerprint(decision_path)
    record = freeze_record(design, ks, evidence, git("rev-parse", "HEAD"), current_settings(), code_fingerprints(),
                           package_versions(), decision)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"Saved: data/eval/{FREEZE_FILE}\nNext: commit it, then: git tag {FREEZE_TAG}")


if __name__ == "__main__":
    try:
        if len(sys.argv) >= 4 and sys.argv[1] == "freeze":
            _freeze_command(*sys.argv[2:])
        else:
            print(__doc__)
    except FreezeError as error:
        print(f"[STOP] {error}")
        sys.exit(1)
