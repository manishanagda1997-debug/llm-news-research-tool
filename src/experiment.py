"""
The experiment record: everything that decides a model result.

    provider, model, generation settings   from the language model function (src/llm.py)
    relevance                              prompt version, batch size, workers, passage limit
    ranking                                threshold, list sizes, the main list size, the order of choice
    commit                                 recorded; the freeze check compares it, not this record

The development record saves it. The fresh score builds it again and refuses
to run when any part differs, so a model name changed in .env (outside the
freeze) cannot slip into the fresh test. The record never holds a key.
"""

import subprocess

from src import relevance_check as rc


def build_record(llm, ranking: dict, commit: str | None) -> dict:
    return {"provider": llm.provider, "model": llm.model_name, "generation": dict(llm.generation),
            "relevance": {"prompt_version": rc.PROMPT_VERSION, "batch_size": rc.BATCH_SIZE,
                          "workers": rc.WORKERS, "passage_limit": rc.MAX_PASSAGE_CHARS},
            "ranking": ranking, "commit": commit}


def record_problems(frozen: dict | None, current: dict) -> list[str]:
    if not frozen:
        return ["no experiment record in the development record"]
    return [f"{part}: {frozen.get(part)!r} in the development record, {current.get(part)!r} now"
            for part in ("provider", "model", "generation", "relevance", "ranking")
            if frozen.get(part) != current.get(part)]


def current_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None
