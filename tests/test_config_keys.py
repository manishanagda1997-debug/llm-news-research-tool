"""
Tests for the key check in src/config.py (no .env file is read: the settings are replaced).

missing_keys() names the keys that are empty or still the placeholder from
.env.example: always NEWSAPI_KEY, and the key of each provider asked for
(the provider in LLM_PROVIDER when none is named). Key values are never returned.
"""

import dataclasses

from src import config


def _settings(monkeypatch, **values):
    monkeypatch.setattr(config, "settings", dataclasses.replace(config.settings, **values))


def test_only_the_active_provider_key_is_needed(monkeypatch):
    _settings(monkeypatch, newsapi_key="n", google_api_key="", groq_api_key="g", llm_provider="groq")
    assert config.missing_keys() == []
    _settings(monkeypatch, newsapi_key="n", google_api_key="", groq_api_key="g", llm_provider="gemini")
    assert config.missing_keys() == ["GOOGLE_API_KEY"]


def test_both_providers_can_be_checked_together(monkeypatch):
    _settings(monkeypatch, newsapi_key="your-newsapi-key-here", google_api_key="k", groq_api_key="your-groq-key-here")
    assert config.missing_keys(("gemini", "groq")) == ["NEWSAPI_KEY", "GROQ_API_KEY"]


def test_key_values_are_never_returned(monkeypatch):
    _settings(monkeypatch, newsapi_key="SECRET-N", google_api_key="SECRET-G", groq_api_key="", llm_provider="groq")
    assert not any("SECRET" in name for name in config.missing_keys(("gemini", "groq")))



def test_the_app_always_needs_the_groq_key_whatever_llm_provider_says(monkeypatch):
    import check_setup
    _settings(monkeypatch, newsapi_key="n", google_api_key="k", groq_api_key="", llm_provider="gemini")
    assert config.missing_keys() == []                         # the development scripts' provider is fine
    assert check_setup.app_missing_keys() == ["GROQ_API_KEY"]  # but the app cannot summarize without Groq
    _settings(monkeypatch, newsapi_key="n", google_api_key="", groq_api_key="g", llm_provider="gemini")
    assert check_setup.app_missing_keys() == []                # Gemini is optional for the app


def test_the_app_provider_is_the_provider_of_the_summary_and_review_model():
    import check_setup
    from src import summary_v2 as sv
    from src.model_compare import CONFIGS
    assert CONFIGS[sv.MODEL_SETTING]["provider"] == check_setup.APP_PROVIDER


def test_the_example_file_says_the_app_needs_groq():
    from pathlib import Path
    example = (Path(config.__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    assert "Required: free key from https://console.groq.com/keys" in example
    assert "The app always uses Groq." in example
