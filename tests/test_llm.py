"""
Tests for the model connection (no live calls: a fake chat model stands in).

make_llm() returns a plain function, prompt text in and answer text out, which
is what src.summarizer.summarize() expects. Any error from the provider
becomes LLMUnavailable, so the app shows the labelled fallback instead of
crashing, and the summary step does not ask again.
"""

import pytest

from src import llm as llm_module
from src.summarizer import LLMUnavailable


class Reply:
    def __init__(self, text):
        self.text = text


class FakeChat:
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.prompts = answer, error, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return Reply(self.answer)


def test_returns_the_answer_text():
    chat = FakeChat(answer='{"ok": true}')
    llm = llm_module.make_llm(chat_model=chat, model_name="fake-model")
    assert llm("hello") == '{"ok": true}'
    assert chat.prompts == ["hello"]


def test_records_the_provider_and_model_name():
    llm = llm_module.make_llm(chat_model=FakeChat(answer="x"), model_name="fake-model")
    assert llm.provider == "gemini" and llm.model_name == "fake-model"


@pytest.mark.parametrize("error", [
    Exception("429 RESOURCE_EXHAUSTED: quota exceeded"),
    TimeoutError("timed out"),
    ConnectionError("network is unreachable"),
    RuntimeError("500 internal error"),
])
def test_every_provider_error_becomes_llm_unavailable(error):
    llm = llm_module.make_llm(chat_model=FakeChat(error=error), model_name="fake-model")
    with pytest.raises(LLMUnavailable) as caught:
        llm("hello")
    assert str(error) in str(caught.value)


def test_an_empty_answer_is_returned_as_empty_text():
    llm = llm_module.make_llm(chat_model=FakeChat(answer=None), model_name="fake-model")
    assert llm("hello") == ""


def test_unknown_provider_is_refused():
    with pytest.raises(ValueError):
        llm_module.make_llm(provider="someone-else")


def test_openai_is_not_a_provider():
    # the providers are Gemini and Groq
    with pytest.raises(ValueError):
        llm_module.make_llm(provider="openai")


def test_gemini_without_a_key_gives_a_clear_error(monkeypatch):
    import dataclasses
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(llm_module.settings, google_api_key=""))
    with pytest.raises(ValueError) as caught:
        llm_module.make_llm(provider="gemini")
    assert "GOOGLE_API_KEY" in str(caught.value)


def test_gemini_is_created_with_temperature_zero(monkeypatch):
    import dataclasses
    created = {}

    class FakeGemini:
        def __init__(self, **options):
            created.update(options)

    monkeypatch.setattr(llm_module, "settings",
                        dataclasses.replace(llm_module.settings, google_api_key="test-key", gemini_model="gemini-test"))
    monkeypatch.setattr(llm_module, "_gemini_class", lambda: FakeGemini)
    llm = llm_module.make_llm(provider="gemini")
    assert created["model"] == "gemini-test" and created["temperature"] == 0
    assert created["google_api_key"] == "test-key"
    assert llm.model_name == "gemini-test"


def test_text_that_is_also_callable_is_used_directly():
    # newer LangChain: .text is a string that can also be called (calling it is deprecated)
    class CallableText(str):
        def __call__(self):
            raise AssertionError(".text() must not be called")

    class NewReply:
        text = CallableText("ready")

    class Chat:
        def invoke(self, prompt):
            return NewReply()

    llm = llm_module.make_llm(chat_model=Chat(), model_name="fake-model")
    assert llm("hello") == "ready"
