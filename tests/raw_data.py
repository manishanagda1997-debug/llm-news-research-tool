"""
The saved NewsAPI responses (data/raw/*.json) are evidence for the evaluation runs. A copy of the project without
them (for example a public repository, where NewsAPI's terms do not allow publishing the returned data) still runs
every test that does not read them; the few that do are skipped with this reason.
"""

import pytest

from src.config import RAW_DIR

RAW_NOTE = "needs the saved NewsAPI responses in data/raw, which are not included in this copy of the project"


def raw_data_present() -> bool:
    return any(RAW_DIR.glob("*.json"))


def require_raw_data() -> None:
    if not raw_data_present():
        pytest.skip(RAW_NOTE)
