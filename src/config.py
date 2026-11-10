"""
Settings for the News Research Tool.

All keys and settings are kept in one place. Keys are read from the .env
file, so they never appear inside the code.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project folders
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"              # news exactly as NewsAPI sends it
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned, numbered articles
EVAL_DIR = DATA_DIR / "eval"            # test queries and scores

# Read keys from the .env file in the project root
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    # API keys
    newsapi_key: str
    google_api_key: str
    groq_api_key: str

    # LLM settings
    llm_provider: str           # "gemini" or "groq"
    gemini_model: str
    groq_model: str
    groq_reasoning_effort: str  # low, medium or high (gpt-oss reasoning models)
    groq_rpm: int               # Groq plan limits per model: requests and tokens a minute
    groq_tpm: int
    gemini_rpm: int             # Gemini plan limits per model (free tier: 5 requests a minute, 20 a day)
    gemini_tpm: int
    temperature: float          # 0 = steady, factual answers

    # News settings
    language: str               # NewsAPI language code
    default_num_articles: int   # articles sent to the LLM per query
    max_num_articles: int       # upper limit shown in the app
    default_days_back: int      # free plan searches up to about 1 month back
    default_sort_by: str        # relevancy, popularity or publishedAt


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


settings = Settings(
    newsapi_key=os.getenv("NEWSAPI_KEY", "").strip(),
    google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
    groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
    llm_provider=(os.getenv("LLM_PROVIDER", "") or "gemini").strip().lower(),
    gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip(),
    groq_model=(os.getenv("GROQ_MODEL", "") or "openai/gpt-oss-120b").strip(),
    groq_reasoning_effort=(os.getenv("GROQ_REASONING_EFFORT", "") or "medium").strip().lower(),
    groq_rpm=_get_int("GROQ_RPM", 30),
    groq_tpm=_get_int("GROQ_TPM", 8000),
    gemini_rpm=_get_int("GEMINI_RPM", 5),
    gemini_tpm=_get_int("GEMINI_TPM", 250_000),
    temperature=_get_float("TEMPERATURE", 0.0),
    language="en",
    default_num_articles=_get_int("NUM_ARTICLES", 10),
    max_num_articles=20,
    default_days_back=7,
    default_sort_by="relevancy",
)


PROVIDER_KEYS = {"gemini": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY"}


def _empty(value: str) -> bool:
    return not value or value.startswith("your-")


def missing_keys(providers: tuple[str, ...] | None = None) -> list[str]:
    """Names of keys that are empty or still the placeholder: NewsAPI and each provider's key
    (the provider in LLM_PROVIDER when none are named)."""
    providers = providers or (settings.llm_provider,)
    values = {"NEWSAPI_KEY": settings.newsapi_key, "GOOGLE_API_KEY": settings.google_api_key,
              "GROQ_API_KEY": settings.groq_api_key}
    names = ["NEWSAPI_KEY"] + [PROVIDER_KEYS[p] for p in providers if p in PROVIDER_KEYS]
    return [name for name in names if _empty(values[name])]
