"""
Tests that splitting src/llm.py keeps make_llm() exactly as it is (fake model class, no live calls).

build_chat_model(provider) returns the real LangChain chat model.
make_llm(provider) builds its model through build_chat_model() and keeps its
contract: prompt text in, answer text out; the provider and model name on the
function; every provider error becomes LLMUnavailable; Gemini at temperature 0
with the model name from the settings, and with no hidden retries
(max_retries=0), so rate-limit errors reach the project's own retry loop. Retries, validation and saved answers
live in the summary and relevance steps, which call this function and keep
their own tests.
"""

import dataclasses

import pytest

from src import llm as llm_module
from src.summarizer import LLMUnavailable


class Reply:
    def __init__(self, text):
        self.text = text


class FakeGemini:
    """Records how it is built; answers every prompt with a fixed text."""
    built = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prompts = []
        FakeGemini.built.append(kwargs)

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return Reply(f"answer to: {prompt}")


@pytest.fixture
def fake_gemini(monkeypatch):
    FakeGemini.built = []
    monkeypatch.setattr(llm_module, "_gemini_class", lambda: FakeGemini)
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(
        llm_module.settings, google_api_key="TEST-KEY-123", gemini_model="gemini-test-model"))
    return FakeGemini


# ---------- build_chat_model ----------

def test_build_chat_model_returns_the_langchain_model_with_the_project_settings(fake_gemini):
    model = llm_module.build_chat_model("gemini")
    assert isinstance(model, FakeGemini)
    assert model.kwargs == {"model": "gemini-test-model", "temperature": 0, "google_api_key": "TEST-KEY-123",
                            "max_retries": 0, "timeout": llm_module.REQUEST_TIMEOUT}


def test_build_chat_model_does_not_call_the_model(fake_gemini):
    model = llm_module.build_chat_model("gemini")
    assert model.prompts == []


def test_build_chat_model_refuses_an_unknown_provider(fake_gemini):
    with pytest.raises(ValueError):
        llm_module.build_chat_model("someone-else")


def test_build_chat_model_without_a_key_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(llm_module.settings, google_api_key=""))
    with pytest.raises(ValueError) as caught:
        llm_module.build_chat_model("gemini")
    assert "GOOGLE_API_KEY" in str(caught.value)


# ---------- make_llm keeps its contract ----------

def test_make_llm_builds_its_model_through_build_chat_model(fake_gemini):
    llm = llm_module.make_llm("gemini")
    assert fake_gemini.built == [{"model": "gemini-test-model", "temperature": 0, "google_api_key": "TEST-KEY-123",
                                "max_retries": 0, "timeout": llm_module.REQUEST_TIMEOUT}]
    assert llm.provider == "gemini" and llm.model_name == "gemini-test-model"


def test_make_llm_sends_the_prompt_unchanged_and_returns_text(fake_gemini):
    llm = llm_module.make_llm("gemini")
    answer = llm('Return JSON: {"a": 1}')
    assert answer == 'answer to: Return JSON: {"a": 1}' and isinstance(answer, str)


def test_make_llm_turns_a_provider_error_into_llm_unavailable(monkeypatch, fake_gemini):
    def broken(self, prompt):
        raise TimeoutError("read timed out")
    monkeypatch.setattr(FakeGemini, "invoke", broken)
    with pytest.raises(LLMUnavailable):
        llm_module.make_llm("gemini")("hello")


def test_an_injected_model_is_used_as_before_without_building_one(fake_gemini):
    injected = FakeGemini(model="injected")
    fake_gemini.built = []
    llm = llm_module.make_llm(chat_model=injected, model_name="injected")
    assert llm("x") == "answer to: x" and fake_gemini.built == []


def test_the_key_never_appears_in_output_or_errors(fake_gemini, capsys):
    llm = llm_module.make_llm("gemini")
    llm("hello")
    printed = capsys.readouterr()
    assert "TEST-KEY-123" not in printed.out + printed.err
    assert "TEST-KEY-123" not in repr(llm) + str(llm.provider) + str(llm.model_name)
