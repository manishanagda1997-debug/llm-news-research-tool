"""
Tests for the Groq provider, pacing and error handling in src/llm.py
(fake chat models, a fake clock and a fake sleep: no keys, no network, no waiting).

Providers: "gemini" and "groq", behind the same make_llm(). The provider comes
from the settings (LLM_PROVIDER, "gemini" when empty) unless the call names one.
Both chat models are built with max_retries=0, so every rate-limit error
reaches the project's own retry loop and is counted the same way.

    pacing      Groq's free plan allows 30 requests and 8,000 tokens a minute
                per model. A request waits until it fits both limits. A
                request that can never fit raises RequestTooLarge before it
                is sent, so the relevance step can split its batch.
    429         wait the time the provider gives (Retry-After header, or
                "try again in 7.5s" in the message), at most 3 retries; no
                retry without a given wait, and none when the wait is longer
                than the time left
    daily limit no retry: the limit lasts until the next day
    auth        no retry
    timeout     no retry
    5xx         two retries, after 2 and then 8 seconds (or the wait the
                provider gives)
    secrets     the key never appears in an error message
    usage       input, output and reasoning tokens are counted per call, and
                every wait (pacing, rate limit, server) is recorded
    settings    a call can name its model and generation settings; only the
                historical Gemini settings (temperature 0) keep the empty
                cache tag of the answers saved before Stage 3
"""

import dataclasses

import pytest

from src import llm as llm_module
from src.summarizer import LLMUnavailable

KEY = "gsk_TEST_SECRET_123"


class Reply:
    def __init__(self, text, usage=None, finish_reason=None):
        self.text = text
        self.usage_metadata = usage
        self.response_metadata = {"finish_reason": finish_reason} if finish_reason else {}


class Clock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(round(seconds, 3))
        self.now += seconds


class Response:
    def __init__(self, status, headers=None):
        self.status_code, self.headers = status, headers or {}


class ApiError(Exception):
    def __init__(self, message, status=None, headers=None):
        super().__init__(message)
        self.status_code = status
        self.response = Response(status, headers) if status else None


class Script:
    """A fake chat model that raises or answers in the given order."""

    def __init__(self, *steps):
        self.steps, self.prompts = list(steps), []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _llm(chat, clock, **options):
    return llm_module.make_llm(provider="groq", chat_model=chat, model_name="openai/gpt-oss-120b",
                               clock=clock, sleep=clock.sleep, **options)


# ---------- the provider ----------

class FakeGroq:
    built = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeGroq.built.append(kwargs)

    def invoke(self, prompt):
        return Reply("ok")


@pytest.fixture
def groq_settings(monkeypatch):
    FakeGroq.built = []
    monkeypatch.setattr(llm_module, "_groq_class", lambda: FakeGroq)
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(
        llm_module.settings, groq_api_key=KEY, groq_model="openai/gpt-oss-120b", groq_reasoning_effort="medium"))


def test_the_providers_are_gemini_and_groq():
    assert llm_module.PROVIDERS == ("gemini", "groq")
    with pytest.raises(ValueError):
        llm_module.make_llm(provider="openai")


def test_groq_is_built_with_the_project_settings_and_no_hidden_retries(groq_settings):
    model = llm_module.build_chat_model("groq")
    assert model.kwargs == {"model": "openai/gpt-oss-120b", "temperature": 0, "api_key": KEY, "max_retries": 0,
                            "timeout": llm_module.REQUEST_TIMEOUT, "reasoning_effort": "medium"}


def test_gemini_is_built_with_no_hidden_retries_too(monkeypatch):
    built = {}

    class FakeGemini:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(llm_module, "_gemini_class", lambda: FakeGemini)
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(
        llm_module.settings, google_api_key="G-KEY", gemini_model="gemini-test"))
    llm_module.build_chat_model("gemini")
    assert built["max_retries"] == 0 and built["timeout"] == llm_module.REQUEST_TIMEOUT
    assert built["model"] == "gemini-test" and built["temperature"] == 0


def test_groq_without_a_key_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(llm_module.settings, groq_api_key=""))
    with pytest.raises(ValueError) as caught:
        llm_module.build_chat_model("groq")
    assert "GROQ_API_KEY" in str(caught.value)


def test_the_provider_comes_from_the_settings_when_the_call_names_none(groq_settings, monkeypatch):
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(llm_module.settings, llm_provider="groq"))
    llm = llm_module.make_llm()
    assert llm.provider == "groq" and llm.model_name == "openai/gpt-oss-120b"


def test_the_llm_states_its_generation_settings_and_cache_tag(groq_settings):
    groq = llm_module.make_llm(provider="groq")
    assert groq.generation == {"temperature": 0, "reasoning_effort": "medium"}
    assert groq.cache_tag and "groq" in groq.cache_tag and "medium" in groq.cache_tag
    gemini = llm_module.make_llm(provider="gemini", chat_model=Script(Reply("x")), model_name="g")
    assert gemini.generation == {"temperature": 0} and gemini.cache_tag == ""    # saved Gemini answers stay valid


# ---------- pacing ----------

def test_requests_are_spaced_by_the_request_limit():
    clock = Clock()
    llm = _llm(Script(Reply("a"), Reply("b")), clock, pacing=llm_module.Pacing(rpm=30, tpm=100_000, clock=clock))
    llm("one")
    llm("two")
    assert clock.slept == [2.0]


def test_a_request_waits_until_the_token_window_has_room():
    clock = Clock()
    usage = {"input_tokens": 5000, "output_tokens": 1000, "total_tokens": 6000}
    llm = _llm(Script(Reply("a", usage), Reply("b")), clock, pacing=llm_module.Pacing(rpm=1000, tpm=8000, clock=clock))
    llm("x" * 300)
    llm("y" * 3000)                              # about 1,000 tokens plus the answer allowance: needs the window
    assert clock.slept and clock.now >= 60.0


def test_a_request_that_can_never_fit_raises_request_too_large_without_sending():
    clock = Clock()
    chat = Script(Reply("never"))
    llm = _llm(chat, clock, pacing=llm_module.Pacing(rpm=30, tpm=8000, clock=clock))
    with pytest.raises(llm_module.RequestTooLarge):
        llm("x" * 40_000)
    assert chat.prompts == []
    assert issubclass(llm_module.RequestTooLarge, LLMUnavailable)


def test_a_413_answer_is_request_too_large():
    clock = Clock()
    llm = _llm(Script(ApiError("Request too large for model", status=413)), clock)
    with pytest.raises(llm_module.RequestTooLarge):
        llm("x")


# ---------- 429 and other errors ----------

def test_a_429_waits_the_retry_after_header_and_tries_again():
    clock = Clock()
    llm = _llm(Script(ApiError("rate limit", 429, {"retry-after": "7"}), Reply("ok")), clock)
    assert llm("x") == "ok"
    assert clock.slept == [7.0] and llm.stats["rate_limited"] == 1 and llm.stats["retries"] == 1


@pytest.mark.parametrize("message, wait", [
    ("Rate limit reached on tokens per minute (TPM). Please try again in 7.5s.", 7.5),
    ("Rate limit reached on requests per minute (RPM). Please try again in 1m2.5s.", 62.5),
    ("Please try again in 450ms.", 0.45),
    ("429 RESOURCE_EXHAUSTED. Please retry in 12.0s.", 12.0),
])
def test_a_429_without_a_header_waits_the_time_in_the_message(message, wait):
    clock = Clock()
    llm = _llm(Script(ApiError(message, 429), Reply("ok")), clock)
    assert llm("x") == "ok" and clock.slept == [wait]


def test_a_429_without_any_given_wait_is_not_retried():
    clock = Clock()
    llm = _llm(Script(ApiError("429 too many requests", 429), Reply("never")), clock)
    with pytest.raises(LLMUnavailable) as caught:
        llm("x")
    assert caught.value.kind == "rate_limit" and clock.slept == []


def test_retries_stop_after_the_limit():
    clock = Clock()
    errors = [ApiError("rate limit", 429, {"retry-after": "1"}) for _ in range(llm_module.MAX_RETRIES + 1)]
    llm = _llm(Script(*errors, Reply("never")), clock)
    with pytest.raises(LLMUnavailable) as caught:
        llm("x")
    assert caught.value.kind == "rate_limit" and len(clock.slept) == llm_module.MAX_RETRIES


def test_no_wait_is_longer_than_the_time_left():
    clock = Clock()
    llm = _llm(Script(ApiError("rate limit", 429, {"retry-after": "90"}), Reply("never")), clock, time_budget=60)
    with pytest.raises(LLMUnavailable) as caught:
        llm("x")
    assert caught.value.kind == "rate_limit" and clock.slept == []
    assert "time left" in str(caught.value)


@pytest.mark.parametrize("message", [
    "Rate limit reached for model on tokens per day (TPD): Limit 200000. Please try again in 1m2s.",
    "Rate limit reached for model on requests per day (RPD): Limit 1000. Please try again in 9m.",
])
def test_a_daily_limit_is_not_retried(message):
    clock = Clock()
    llm = _llm(Script(ApiError(message, 429), Reply("never")), clock)
    with pytest.raises(LLMUnavailable) as caught:
        llm("x")
    assert caught.value.kind == "daily_limit" and clock.slept == []


@pytest.mark.parametrize("error, kind", [
    (ApiError("Invalid API Key", 401), "auth"),
    (ApiError("Forbidden", 403), "auth"),
    (TimeoutError("read timed out"), "timeout"),
    (type("APITimeoutError", (Exception,), {})("Request timed out."), "timeout"),
    (ValueError("something else"), "other"),
])
def test_errors_that_are_not_retried(error, kind):
    clock = Clock()
    chat = Script(error, Reply("never"))
    llm = _llm(chat, clock)
    with pytest.raises(LLMUnavailable) as caught:
        llm("x")
    assert caught.value.kind == kind and len(chat.prompts) == 1 and clock.slept == []


def test_the_key_never_appears_in_an_error(groq_settings):
    clock = Clock()
    llm = _llm(Script(ApiError(f"Invalid API Key: {KEY}", 401)), clock)
    with pytest.raises(LLMUnavailable) as caught:
        llm("x")
    assert KEY not in str(caught.value) and "[key hidden]" in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__   # a traceback cannot show it


def test_errors_are_counted_by_kind():
    clock = Clock()
    llm = _llm(Script(ApiError("rate limit", 429, {"retry-after": "1"}), Reply("ok"), TimeoutError("t")), clock)
    llm("x")
    with pytest.raises(LLMUnavailable):
        llm("y")
    assert llm.stats["calls"] == 3 and llm.stats["failures"] == {"timeout": 1}


# ---------- token use ----------

def test_token_use_is_counted_per_call_including_reasoning():
    clock = Clock()
    usage = {"input_tokens": 900, "output_tokens": 300, "total_tokens": 1200,
             "output_token_details": {"reasoning": 220}}
    llm = _llm(Script(Reply("a", usage), Reply("b", usage)), clock)
    llm("x")
    llm("y")
    assert llm.stats["input_tokens"] == 1800 and llm.stats["output_tokens"] == 600
    assert llm.stats["reasoning_tokens"] == 440
    assert [{k: c[k] for k in ("input", "output", "reasoning")} for c in llm.stats["per_call"]] == \
        [{"input": 900, "output": 300, "reasoning": 220}] * 2


def test_a_reply_without_usage_counts_nothing_and_still_works():
    clock = Clock()
    llm = _llm(Script(Reply("a")), clock)
    assert llm("x") == "a" and llm.stats["input_tokens"] == 0


# ---------- settings per configuration (Stage 3.3) ----------

def test_a_call_can_name_its_model_and_generation_settings(groq_settings):
    llm = llm_module.make_llm(provider="groq", model="qwen/qwen3.8-27b",
                              generation={"temperature": 1.0, "top_p": 0.95, "reasoning_effort": "medium",
                                          "reasoning_format": "hidden"})
    built = FakeGroq.built[-1]
    assert built["model"] == "qwen/qwen3.8-27b" and built["temperature"] == 1.0
    assert built["reasoning_effort"] == "medium" and built["reasoning_format"] == "hidden"
    assert built["model_kwargs"] == {"top_p": 0.95}
    assert llm.model_name == "qwen/qwen3.8-27b" and llm.generation["top_p"] == 0.95


def test_gemini_takes_temperature_and_thinking_level(monkeypatch):
    built = {}

    class FakeGemini:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(llm_module, "_gemini_class", lambda: FakeGemini)
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(llm_module.settings, google_api_key="G"))
    llm = llm_module.make_llm(provider="gemini", model="gemini-3.8-flash",
                              generation={"temperature": 1.0, "thinking_level": "medium"})
    assert built["temperature"] == 1.0 and built["thinking_level"] == "medium" and built["model"] == "gemini-3.8-flash"
    assert llm.cache_tag                       # new settings: never the saved temperature-0 answers


def test_only_the_historical_gemini_settings_keep_the_empty_cache_tag(monkeypatch):
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(llm_module.settings, google_api_key="G"))
    old = llm_module.make_llm(provider="gemini", chat_model=Script(Reply("x")), model_name="g")
    new = llm_module.make_llm(provider="gemini", chat_model=Script(Reply("x")), model_name="g",
                              generation={"temperature": 1.0, "thinking_level": "medium"})
    assert old.cache_tag == "" and new.cache_tag != "" and old.generation == {"temperature": 0}


def test_unknown_generation_settings_are_refused():
    with pytest.raises(ValueError):
        llm_module.make_llm(provider="groq", chat_model=Script(), model_name="m", generation={"temprature": 1})


# ---------- server errors: two retries, 2 s then 8 s ----------

def test_a_server_error_is_retried_twice_with_growing_waits():
    clock = Clock()
    llm = _llm(Script(ApiError("503 UNAVAILABLE high demand", 503), ApiError("503", 503), Reply("ok")), clock)
    assert llm("x") == "ok" and clock.slept == [2.0, 8.0] and llm.stats["retries"] == 2


def test_a_server_error_stops_after_two_retries():
    clock = Clock()
    errors = [ApiError("Bad gateway", 502) for _ in range(3)]
    llm = _llm(Script(*errors, Reply("never")), clock)
    with pytest.raises(LLMUnavailable) as caught:
        llm("x")
    assert caught.value.kind == "server" and clock.slept == [2.0, 8.0]


def test_a_server_error_with_a_given_wait_uses_that_wait():
    clock = Clock()
    llm = _llm(Script(ApiError("503", 503, {"retry-after": "5"}), Reply("ok")), clock)
    assert llm("x") == "ok" and clock.slept == [5.0]


def test_every_wait_is_recorded():
    clock = Clock()
    llm = _llm(Script(ApiError("rate limit", 429, {"retry-after": "3"}), ApiError("503", 503), Reply("ok")), clock,
               pacing=llm_module.Pacing(rpm=30, tpm=100_000, clock=clock))
    llm("x")
    # pacing needs no wait here: each retry starts at least 2 s after the request before it
    assert llm.stats["waits"] == [{"kind": "rate_limit", "seconds": 3.0}, {"kind": "server", "seconds": 2.0}]
    assert llm.stats["waited_seconds"] == pytest.approx(sum(w["seconds"] for w in llm.stats["waits"]))


def test_a_pacing_wait_is_recorded_as_pacing():
    clock = Clock()
    llm = _llm(Script(Reply("a"), Reply("b")), clock, pacing=llm_module.Pacing(rpm=30, tpm=100_000, clock=clock))
    llm("x")
    llm("y")
    assert llm.stats["waits"] == [{"kind": "pacing", "seconds": 2.0}] and llm.stats["waited_seconds"] == 2.0


# ---------- the learning pacer (Stage 3.3) ----------

class Headers:
    """A fake httpx response: only the headers matter."""

    def __init__(self, headers):
        self.headers = headers


def test_the_first_request_uses_the_careful_estimate():
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=Clock())
    assert pacing.estimate("x" * 3000) == 3000 // 3 + 1 + llm_module.ANSWER_ALLOWANCE


def test_later_estimates_learn_the_real_tokens_per_character_and_keep_the_answer_margin():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=clock)
    ticket = pacing.reserve(pacing.estimate("x" * 4000))
    pacing.settle(ticket, prompt_chars=4000, input_tokens=1000, output_tokens=600, cached_tokens=0)
    assert pacing.estimate("x" * 8000) == 2000 + 1 + llm_module.ANSWER_ALLOWANCE      # 0.25 tokens a character


def test_the_answer_margin_grows_with_the_largest_answer_and_never_shrinks_below_the_floor():
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=Clock())
    ticket = pacing.reserve(100)
    pacing.settle(ticket, prompt_chars=400, input_tokens=100, output_tokens=2000, cached_tokens=0)
    assert pacing.answer_margin() == round(2000 * llm_module.OUTPUT_MARGIN)
    small = llm_module.Pacing(rpm=30, tpm=8000, clock=Clock())
    small.settle(small.reserve(100), prompt_chars=400, input_tokens=100, output_tokens=50, cached_tokens=0)
    assert small.answer_margin() == llm_module.ANSWER_ALLOWANCE


def test_two_batches_of_4200_tokens_never_share_one_8000_token_minute():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=clock)
    pacing.reserve(4200)
    clock.now = 3.0
    assert pacing.wait_time(4200) == pytest.approx(57.0)
    assert pacing.wait_time(3700) == 0.0                     # a smaller request fits what is left


def test_a_request_in_progress_holds_its_reservation():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=1000, tpm=8000, clock=clock)
    pacing.reserve(6000)                                      # sent, no answer yet
    assert pacing.wait_time(3000) > 0


def test_the_reservation_becomes_the_real_use_without_cached_tokens():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=1000, tpm=8000, clock=clock)
    ticket = pacing.reserve(7000)
    pacing.settle(ticket, prompt_chars=4000, input_tokens=3000, output_tokens=500, cached_tokens=2000)
    assert pacing.used_tokens() == 1500                      # cached input does not count against the limit
    clock.now = 1.0                                          # past the requests-a-minute spacing
    assert pacing.wait_time(6000) == 0.0


def test_a_failed_request_keeps_its_reservation_and_invents_no_use():
    clock = Clock()
    chat = Script(TimeoutError("t"))
    pacing = llm_module.Pacing(rpm=1000, tpm=8000, clock=clock)
    llm = _llm(chat, clock, pacing=pacing)
    with pytest.raises(LLMUnavailable):
        llm("x" * 300)
    assert pacing.used_tokens() == pacing.estimate("x" * 300) and llm.stats["per_call"] == []


def test_groq_headers_tell_the_pacer_what_is_left():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=1000, tpm=8000, clock=clock)
    pacing.update_from_headers({"x-ratelimit-remaining-tokens": "1000", "x-ratelimit-reset-tokens": "7.5s"})
    assert pacing.wait_time(4000) == pytest.approx(7.5)      # the local count alone would say 0
    assert pacing.wait_time(900) == 0.0


@pytest.mark.parametrize("value, seconds", [("7.66s", 7.66), ("1m2.5s", 62.5), ("450ms", 0.45), ("2m59.56s", 179.56)])
def test_reset_times_in_groq_format_are_read(value, seconds):
    clock = Clock()
    pacing = llm_module.Pacing(rpm=1000, tpm=8000, clock=clock)
    pacing.update_from_headers({"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": value})
    assert pacing.wait_time(10) == pytest.approx(seconds)


def test_without_headers_the_local_count_still_works():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=1000, tpm=8000, clock=clock)
    pacing.update_from_headers({})
    pacing.reserve(7000)
    assert pacing.wait_time(2000) == pytest.approx(60.0)


def test_header_news_is_reduced_by_requests_sent_after_it():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=1000, tpm=8000, clock=clock)
    pacing.update_from_headers({"x-ratelimit-remaining-tokens": "5000", "x-ratelimit-reset-tokens": "20s"})
    pacing.reserve(3000)
    assert pacing.wait_time(2500) == pytest.approx(20.0)


def test_the_groq_model_reports_its_response_headers_to_the_pacer(groq_settings):
    llm = llm_module.make_llm(provider="groq")
    client = FakeGroq.built[-1]["http_client"]
    for hook in client.event_hooks["response"]:
        hook(Headers({"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "3s"}))
    assert llm.pacing.wait_time(10) == pytest.approx(3.0, abs=0.5)
    assert llm.stats["header_updates"] == 1


def test_a_ten_article_batch_that_fits_is_no_longer_split_after_the_first_answer():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=clock)
    first = "x" * 15000                                       # careful estimate: 5,001 + 1,500
    usage = {"input_tokens": 3600, "output_tokens": 640, "total_tokens": 4240}
    llm = _llm(Script(Reply("a", usage), Reply("b")), clock, pacing=pacing)
    llm(first)
    ten = "x" * 26000                                         # learned: 0.24 a character -> about 6,241 + 1,500
    assert pacing.estimate(ten) < 8000 and pacing.fits(pacing.estimate(ten))
    llm(ten)                                                  # sent, after the minute has room
    assert llm.stats["failures"] == {}


def test_every_request_records_its_estimate_and_its_real_use():
    clock = Clock()
    usage = {"input_tokens": 900, "output_tokens": 300, "total_tokens": 1200}
    llm = _llm(Script(Reply("a", usage)), clock, pacing=llm_module.Pacing(rpm=30, tpm=8000, clock=clock))
    llm("x" * 3000)
    call = llm.stats["per_call"][0]
    assert {k: call[k] for k in ("input", "output", "reasoning", "cached", "estimated", "prompt_chars",
                                 "finish_reason")} == {"input": 900, "output": 300, "reasoning": 0, "cached": 0,
                                                       "estimated": 3000 // 3 + 1 + llm_module.ANSWER_ALLOWANCE,
                                                       "prompt_chars": 3000, "finish_reason": None}


def test_one_odd_token_count_cannot_make_every_later_request_too_large():
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=Clock())
    pacing.settle(pacing.reserve(100), prompt_chars=300, input_tokens=5000, output_tokens=100, cached_tokens=0)
    assert pacing.tokens_per_char == llm_module.MAX_TOKENS_PER_CHAR
    assert pacing.fits(pacing.estimate("x" * 3000))


# ---------- fixes after the first full run ----------

def test_the_estimate_is_made_again_after_every_wait():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=1000, tpm=8000, clock=clock)
    pacing.done(pacing.reserve(7000))                         # an ended request fills most of the minute
    calls = []
    original = pacing.estimate

    def counting(prompt):
        calls.append(clock.now)
        return original(prompt)

    pacing.estimate = counting
    llm = _llm(Script(Reply("a")), clock, pacing=pacing)
    llm("x" * 3000)
    assert len(calls) >= 2 and calls[-1] >= 60.0              # estimated again once the minute has room


def test_the_first_answer_is_learned_before_another_request_estimates():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=1000, tpm=8000, clock=clock)
    in_flight = pacing.reserve(3000)                          # sent by another worker, no answer yet

    def sleep(seconds):                                       # while this request waits, the answer arrives
        clock.now += seconds
        pacing.settle(in_flight, prompt_chars=10000, input_tokens=2000, output_tokens=400, cached_tokens=0)
        pacing.done(in_flight)

    usage = {"input_tokens": 4000, "output_tokens": 300, "total_tokens": 4300}
    llm = llm_module.make_llm(provider="groq", chat_model=Script(Reply("b", usage)), model_name="m",
                              clock=clock, sleep=sleep, pacing=pacing)
    llm("x" * 20000)
    assert llm.stats["waits"][0]["kind"] == "learning"
    assert llm.stats["per_call"][0]["estimated"] == 4000 + 1 + llm_module.ANSWER_ALLOWANCE   # learned: 0.2 a char


def test_learning_never_waits_on_a_request_that_has_ended():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=1000, tpm=8000, clock=clock)
    llm = _llm(Script(TimeoutError("t"), Reply("b")), clock, pacing=pacing)
    with pytest.raises(LLMUnavailable):
        llm("x" * 300)                                        # failed: no answer, but no longer in flight
    assert llm("y" * 300) == "b"


def test_a_split_is_counted_as_a_split_not_as_a_failure():
    clock = Clock()
    llm = _llm(Script(Reply("never")), clock, pacing=llm_module.Pacing(rpm=30, tpm=8000, clock=clock))
    with pytest.raises(llm_module.RequestTooLarge):
        llm("x" * 40_000)
    assert llm.stats["splits"] == 1 and llm.stats["failures"] == {}


def test_the_wait_to_learn_never_lasts_longer_than_one_request_timeout():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=1000, tpm=100_000, clock=clock)
    pacing.reserve(100)                                       # in flight, and it never ends
    llm = _llm(Script(Reply("b")), clock, pacing=pacing)
    assert llm("x" * 300) == "b"
    learning = sum(w["seconds"] for w in llm.stats["waits"] if w["kind"] == "learning")
    assert llm_module.REQUEST_TIMEOUT <= learning <= llm_module.REQUEST_TIMEOUT + llm_module.LEARN_POLL


# ---------- Gemini pacing (free tier: 5 requests a minute, 20 a day) ----------

def test_gemini_gets_its_own_pacing_from_the_settings(monkeypatch):
    class FakeGemini:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(llm_module, "_gemini_class", lambda: FakeGemini)
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(
        llm_module.settings, google_api_key="G", gemini_rpm=5, gemini_tpm=250_000))
    llm = llm_module.make_llm(provider="gemini")
    assert llm.pacing is not None and llm.pacing.rpm == 5 and llm.pacing.tpm == 250_000


def test_five_requests_a_minute_are_spaced_twelve_seconds_apart():
    clock = Clock()
    llm = llm_module.make_llm(provider="gemini", chat_model=Script(Reply("a"), Reply("b")), model_name="g",
                              clock=clock, sleep=clock.sleep, pacing=llm_module.Pacing(rpm=5, tpm=250_000, clock=clock))
    llm("x")
    llm("y")
    assert clock.slept == [12.0]


def test_the_default_gemini_limits_are_the_free_tier_ones():
    from src import config
    assert config.settings.gemini_rpm == 5 and config.settings.gemini_tpm == 250_000


# ---------- the request log: where each planned request ends ----------
#
# Four different endings stay separate: the project refuses a request before sending it (preflight), the
# provider rejects it, the provider answers it, or the provider fails in another way. A content problem in
# an answer (broken JSON, an unconfirmed number) is judged later by the summary step, not here.

def test_a_request_refused_before_sending_is_marked_preflight_and_logged():
    clock = Clock()
    llm = _llm(Script(Reply("never")), clock, pacing=llm_module.Pacing(rpm=30, tpm=8000, clock=clock))
    with pytest.raises(llm_module.RequestTooLarge) as caught:
        llm("x" * 40_000)
    assert caught.value.stage == "preflight"
    assert llm.stats["requests"] == [{"estimated": caught.value.estimated, "prompt_chars": 40_000,
                                      "outcome": "preflight_too_large"}]
    assert caught.value.estimated > 8000


def test_a_413_from_the_provider_is_marked_provider_and_logged_separately():
    clock = Clock()
    llm = _llm(Script(ApiError("Request too large for model", status=413)), clock)
    with pytest.raises(llm_module.RequestTooLarge) as caught:
        llm("x")
    assert caught.value.stage == "provider"
    assert [r["outcome"] for r in llm.stats["requests"]] == ["provider_too_large"]


def test_an_answered_request_is_logged_once_even_after_a_retry():
    clock = Clock()
    llm = _llm(Script(ApiError("rate limit", 429, {"retry-after": "7"}), Reply("ok")), clock)
    llm("x" * 30)
    assert [r["outcome"] for r in llm.stats["requests"]] == ["answered"]
    assert llm.stats["requests"][0]["prompt_chars"] == 30


def test_another_provider_failure_is_logged_with_its_kind():
    clock = Clock()
    llm = _llm(Script(ApiError("auth failed", status=401)), clock)
    with pytest.raises(LLMUnavailable):
        llm("x")
    assert llm.stats["requests"][0]["outcome"].startswith("provider_")
    assert llm.stats["requests"][0]["outcome"] != "provider_too_large"


def test_the_finish_reason_of_each_answer_is_recorded():
    clock = Clock()
    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    llm = _llm(Script(Reply("a", usage, "stop"), Reply("b", usage, "length")), clock)
    llm("x")
    llm("y")
    assert [c["finish_reason"] for c in llm.stats["per_call"]] == ["stop", "length"]


# ---------- the per-request detail for diagnosis: timing and the provider's own count ----------

class SlowScript(Script):
    """Answers after the given number of seconds on the fake clock."""

    def __init__(self, clock, seconds, *steps):
        super().__init__(*steps)
        self.clock, self.seconds = clock, seconds

    def invoke(self, prompt):
        self.clock.now += self.seconds
        return super().invoke(prompt)


def test_each_call_records_when_it_was_sent_and_answered():
    clock = Clock()
    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    llm = _llm(SlowScript(clock, 4.0, Reply("a", usage)), clock)
    llm("x")
    call = llm.stats["per_call"][0]
    assert call["sent_at"] == 0.0 and call["answered_at"] == 4.0


def test_each_call_records_the_providers_remaining_tokens_before_it_was_sent():
    clock = Clock()
    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=clock)
    llm = _llm(Script(Reply("a", usage), Reply("b", usage)), clock, pacing=pacing)
    llm("x")
    assert llm.stats["per_call"][0]["provider_remaining"] is None          # no header seen yet
    pacing.update_from_headers({"x-ratelimit-remaining-tokens": "5200", "x-ratelimit-reset-tokens": "20s"})
    clock.now += 3.0
    llm("y")
    call = llm.stats["per_call"][1]
    assert call["provider_remaining"] == 5200 and call["header_age"] == 3.0
    assert call["window_used"] == 15                 # the project's own count before it: the first call's real use


def test_the_llm_states_the_output_limit_it_sends():
    clock = Clock()
    llm = _llm(Script(Reply("a")), clock)
    assert llm.output_limit is None                  # the project sends no output limit


# ---------- an explicit output limit (max_completion_tokens) ----------

CAPPED = {"temperature": 1.0, "reasoning_effort": "medium", "max_completion_tokens": 4096}


def test_the_output_limit_is_sent_as_max_completion_tokens(groq_settings):
    model = llm_module.build_chat_model("groq", generation=CAPPED)
    assert model.kwargs["model_kwargs"] == {"max_completion_tokens": 4096}
    assert "max_completion_tokens" not in model.kwargs and "max_tokens" not in model.kwargs


def test_the_real_outgoing_request_carries_the_output_limit(monkeypatch):
    """The request that leaves the computer, built by the installed LangChain Groq client (no network)."""
    langchain_groq = pytest.importorskip("langchain_groq")
    httpx = pytest.importorskip("httpx")
    import functools
    import json as json_module
    sent = {}

    def handler(request):
        sent["body"] = json_module.loads(request.content)
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                                         "choices": [{"index": 0, "finish_reason": "stop",
                                                      "message": {"role": "assistant", "content": "ok"}}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(llm_module, "_groq_class",
                        lambda: functools.partial(langchain_groq.ChatGroq, http_client=client))
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(
        llm_module.settings, groq_api_key=KEY, groq_model="openai/gpt-oss-120b", groq_reasoning_effort="medium"))
    llm_module.build_chat_model("groq", generation=CAPPED).invoke("hello")
    assert sent["body"]["max_completion_tokens"] == 4096
    assert "max_tokens" not in sent["body"]


# Pacer version 2. Groq counts a request as its input plus a few hundred tokens, not the whole output limit:
# its "Requested" values were 276 to 551 above the input (3,186 -> 3,462, 3,507, 3,737; about 1,170 -> 1,512).
# So the rate check uses the input estimate plus REQUEST_MARGIN; the output limit is still sent with every
# request and still checked against the model's context window. Real use is settled after each answer.

def test_the_pacer_counts_the_input_and_a_fixed_margin_not_the_whole_output_limit():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=clock)
    pacing.output_cap = 4096
    assert llm_module.REQUEST_MARGIN == 600
    assert pacing.estimate("x" * 3000) == 3000 // 3 + 1 + 600
    pacing.tokens_per_char, pacing.largest_answer = 0.25, 9000      # learned values do not change the margin
    assert pacing.estimate("x" * 4000) == 1000 + 1 + 600


def test_without_an_output_limit_the_pacer_keeps_its_learned_margin():
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=Clock())
    assert pacing.estimate("x" * 3000) == llm_module.estimate_tokens("x" * 3000)


def test_make_llm_passes_the_output_limit_to_the_pacer_and_states_it():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=clock)
    llm = llm_module.make_llm(provider="groq", chat_model=Script(Reply("a")), model_name="openai/gpt-oss-120b",
                              generation=CAPPED, pacing=pacing, clock=clock, sleep=clock.sleep)
    assert pacing.output_cap == 4096 and llm.output_limit == 4096
    assert "max_completion_tokens" in llm.cache_tag


def _capped(chat, clock, pacing):
    return llm_module.make_llm(provider="groq", chat_model=chat, model_name="openai/gpt-oss-120b",
                               generation=CAPPED, pacing=pacing, clock=clock, sleep=clock.sleep)


def test_once_the_rate_is_learned_a_request_that_fits_without_the_output_limit_is_sent():
    """The 2026-10-23 and 2026-10-24 direct requests of the first configuration 7 comparison: refused before
    sending by version 1 ("About 8,060" and "About 9,136" tokens with the 4,096 output limit counted)."""
    clock = Clock()
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=clock)
    pacing.tokens_per_char = 3186 / 15_537           # learned from a real answer
    chat = Script(Reply("ok", {"input_tokens": 4060, "output_tokens": 2500, "total_tokens": 6560}))
    assert _capped(chat, clock, pacing)("x" * 19_776) == "ok"       # 4,056 + 600 <= 8,000
    assert len(chat.prompts) == 1


def test_a_k10_direct_request_is_sent_and_the_provider_decides():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=clock)
    pacing.tokens_per_char = 3186 / 15_537
    chat = Script(Reply("ok", {"input_tokens": 6100, "output_tokens": 2800, "total_tokens": 8900}))
    assert _capped(chat, clock, pacing)("x" * 29_765) == "ok"       # about 6,104 + 600 <= 8,000


def test_once_the_rate_is_learned_a_request_whose_input_alone_cannot_fit_is_refused():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=clock)
    pacing.tokens_per_char = 0.34                    # 25,000 characters -> 8,500 + 600 > 8,000
    chat = Script(Reply("never"))
    with pytest.raises(llm_module.RequestTooLarge) as caught:
        _capped(chat, clock, pacing)("x" * 25_000)
    assert caught.value.stage == "preflight" and chat.prompts == []


def test_a_request_longer_than_the_context_window_is_refused_before_sending():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=clock)
    pacing.context_window = 6000                     # input (careful) + the 4,096 output limit must fit
    chat = Script(Reply("never"))
    with pytest.raises(llm_module.RequestTooLarge) as caught:
        _capped(chat, clock, pacing)("x" * 6_000)    # 2,001 + 4,096 > 6,000
    assert caught.value.stage == "preflight" and "context window" in str(caught.value) and chat.prompts == []


def test_make_llm_knows_the_context_window_of_the_groq_models():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=clock)
    _capped(Script(Reply("a")), clock, pacing)
    assert pacing.context_window == 131_072


def test_before_the_rate_is_learned_a_request_whose_input_fits_is_sent_alone_and_the_provider_decides():
    """The careful estimate (3 characters a token) is far above real news text (about 4.9), so with the whole
    output limit reserved it would refuse requests that fit. Sent alone, the provider refuses a request that is
    really too large (no tokens used), and its answer teaches the pacer the real rate."""
    clock = Clock()
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=clock)
    usage = {"input_tokens": 3186, "output_tokens": 2121, "total_tokens": 5307}
    chat = Script(Reply("ok", usage))
    llm = _capped(chat, clock, pacing)
    assert llm("x" * 15_537) == "ok"                 # careful: 5,180 + 4,096 > 8,000; real: 3,186 + output
    assert len(chat.prompts) == 1 and pacing.tokens_per_char is not None
    assert llm.stats["requests"][0]["outcome"] == "answered"


def test_before_the_rate_is_learned_even_a_large_request_is_sent_alone_and_the_provider_decides():
    """The careful estimate (3 characters a token) is far above real news text, so before any real count it
    never refuses on its own: the request goes alone, and a provider refusal costs no tokens."""
    clock = Clock()
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=clock)
    chat = Script(Reply("ok", {"input_tokens": 5100, "output_tokens": 2000, "total_tokens": 7100}))
    assert _capped(chat, clock, pacing)("x" * 25_000) == "ok"      # careful input alone: 8,334 > 8,000
    assert len(chat.prompts) == 1


def test_an_unlearned_large_request_waits_until_nothing_else_is_in_the_minute():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=clock)
    pacing.output_cap = 4096
    ticket = pacing.reserve(500)                      # something else is in this minute, still waiting for its answer
    assert pacing.unlearned_alone_allowed("x" * 15_537) is False
    pacing.done(ticket)                               # answered, but its tokens stay in the minute
    assert pacing.unlearned_alone_allowed("x" * 15_537) is False and pacing.wait_alone() == 60.0
    clock.now += 61
    assert pacing.unlearned_alone_allowed("x" * 15_537) is True



# Pacer version 3. Once the rate is learned, a request whose input estimate fits the minute but whose input
# plus REQUEST_MARGIN does not is no longer refused before sending: the margin is the project's own planning
# allowance, not a provider limit. Such a request goes alone (nothing in flight, nothing used in the last 60
# seconds) and Groq decides. The context-window check comes first; a request whose input alone cannot fit is
# still refused before sending. A provider 413 is final; 429 and 5xx keep their waits and identical resends.

TPM_413 = ("Error code: 413 - {'error': {'message': 'Request too large for model `openai/gpt-oss-120b` in "
           "organization `org_01kTESTorg9Z` service tier `on_demand` on tokens per minute (TPM): Limit 8000, "
           "Requested 8234, please reduce your message size and try again.', 'type': 'tokens', "
           "'code': 'rate_limit_exceeded'}}")


def _learned(clock, ratio=7683 / 33_162):
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=clock)
    pacing.tokens_per_char = ratio                     # learned on 14 October, k = 10
    return pacing


def test_the_pacer_version_is_3():
    assert llm_module.PACER_VERSION == "3"


def test_a_request_over_the_limit_only_because_of_the_margin_is_sent_alone_and_answered():
    """23 October, k = 10: about 7,683 input tokens + 600 = 8,284 was refused before sending by version 2."""
    clock = Clock()
    pacing = _learned(clock)
    chat = Script(Reply("ok", {"input_tokens": 7690, "output_tokens": 2500, "total_tokens": 10190}))
    llm = _capped(chat, clock, pacing)
    before = pacing.input_estimate("x" * 33_162)              # the answer teaches a new rate afterwards
    assert pacing.estimate("x" * 33_162) > 8000 >= before
    assert llm("x" * 33_162) == "ok" and len(chat.prompts) == 1
    entry = llm.stats["requests"][0]
    assert entry["admission"] == "alone_margin_only" and entry["outcome"] == "answered"
    assert entry["input_estimate"] == before and entry["estimated"] > 8000


def test_a_provider_413_on_a_margin_only_request_is_final_and_records_groqs_numbers():
    clock = Clock()
    chat = Script(ApiError(TPM_413, status=413), Reply("never"))
    llm = _capped(chat, clock, _learned(clock))
    with pytest.raises(llm_module.RequestTooLarge) as caught:
        llm("x" * 33_162)
    assert caught.value.stage == "provider" and len(chat.prompts) == 1 and clock.slept == []
    entry = llm.stats["requests"][0]
    assert entry["outcome"] == "provider_too_large" and entry["admission"] == "alone_margin_only"
    assert entry["provider_limit"] == 8000 and entry["provider_requested"] == 8234


def test_organisation_ids_and_keys_are_hidden_in_every_recorded_error_field(monkeypatch):
    import json
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(llm_module.settings, groq_api_key=KEY))
    clock = Clock()
    llm = _capped(Script(ApiError(TPM_413 + " key " + KEY, status=413)), clock, _learned(clock))
    with pytest.raises(llm_module.RequestTooLarge) as caught:
        llm("x" * 33_162)
    recorded = str(caught.value) + json.dumps(llm.stats)
    assert "org_01kTESTorg9Z" not in recorded and KEY not in recorded
    assert "org_[hidden]" in llm.stats["requests"][0]["error"] and "org_[hidden]" in str(caught.value)


def test_a_learned_request_whose_input_alone_is_over_the_limit_is_still_refused_before_sending():
    clock = Clock()
    chat = Script(Reply("never"))
    llm = _capped(chat, clock, _learned(clock, ratio=0.34))         # 25,000 characters -> 8,500 input tokens
    with pytest.raises(llm_module.RequestTooLarge) as caught:
        llm("x" * 25_000)
    assert caught.value.stage == "preflight" and chat.prompts == []
    assert llm.stats["requests"][0]["outcome"] == "preflight_too_large"


def test_the_context_window_check_comes_before_the_margin_only_admission():
    clock = Clock()
    pacing = _learned(clock)
    pacing.context_window = 14_000                   # careful input 11,055 + output limit 4,096 > 14,000
    chat = Script(Reply("never"))
    with pytest.raises(llm_module.RequestTooLarge) as caught:
        _capped(chat, clock, pacing)("x" * 33_162)
    assert "context window" in str(caught.value) and chat.prompts == []


def test_a_margin_only_request_waits_until_nothing_is_in_flight_and_the_minute_is_clear():
    clock = Clock()
    pacing = _learned(clock)
    pacing.output_cap = 4096
    ticket = pacing.reserve(500)
    assert pacing.alone_clear() is False             # in flight
    pacing.done(ticket)
    assert pacing.alone_clear() is False and pacing.wait_alone() == 60.0      # used within the last minute
    clock.now += 61
    assert pacing.alone_clear() is True


def test_a_margin_only_request_is_sent_only_after_the_earlier_use_leaves_the_minute():
    clock = Clock()
    pacing = _learned(clock)
    first = {"input_tokens": 900, "output_tokens": 500, "total_tokens": 1400}      # 0.225 a character: below the learned rate
    chat = Script(Reply("a", first), Reply("b", {"input_tokens": 7690, "output_tokens": 10, "total_tokens": 7700}))
    llm = _capped(chat, clock, pacing)
    llm("x" * 4000)
    sent_first = clock.now
    assert llm("x" * 33_162) == "b"
    assert clock.now - sent_first >= 60.0 and llm.stats["requests"][1]["admission"] == "alone_margin_only"
    assert llm.stats["requests"][0]["admission"] == "paced"


def test_a_429_on_a_margin_only_request_resends_the_identical_prompt_after_the_given_wait():
    clock = Clock()
    chat = Script(ApiError("rate limit", 429, {"retry-after": "7"}),
                  Reply("ok", {"input_tokens": 7690, "output_tokens": 10, "total_tokens": 7700}))
    llm = _capped(chat, clock, _learned(clock))
    prompt = "x" * 33_000 + "end"
    assert llm(prompt) == "ok"
    assert chat.prompts == [prompt, prompt] and 7.0 in clock.slept
    assert llm.stats["retries"] == 1 and llm.stats["rate_limited"] == 1


def test_a_retry_sends_an_identical_request_body(monkeypatch):
    """The real outgoing requests built by the installed LangChain Groq client (no network): model, messages,
    output limit and every other parameter are the same on the retry."""
    langchain_groq = pytest.importorskip("langchain_groq")
    httpx = pytest.importorskip("httpx")
    import functools
    import json as json_module
    bodies = []

    def handler(request):
        bodies.append(json_module.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(429, headers={"retry-after": "3"},
                                  json={"error": {"message": "Rate limit reached", "type": "tokens"}})
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                                         "choices": [{"index": 0, "finish_reason": "stop",
                                                      "message": {"role": "assistant", "content": "ok"}}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(llm_module, "_groq_class",
                        lambda: functools.partial(langchain_groq.ChatGroq, http_client=client))
    monkeypatch.setattr(llm_module, "settings", dataclasses.replace(
        llm_module.settings, groq_api_key=KEY, groq_model="openai/gpt-oss-120b", groq_reasoning_effort="medium"))
    clock = Clock()
    model = llm_module.build_chat_model("groq", generation=CAPPED)
    llm = llm_module.make_llm(provider="groq", chat_model=model, model_name="openai/gpt-oss-120b",
                              generation=CAPPED, clock=clock, sleep=clock.sleep)
    assert llm("hello") == "ok"
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert bodies[0]["max_completion_tokens"] == 4096 and bodies[0]["model"] == "openai/gpt-oss-120b"


def test_requests_that_fit_are_admitted_as_paced_and_unchanged():
    clock = Clock()
    pacing = _learned(clock)
    chat = Script(Reply("ok", {"input_tokens": 3000, "output_tokens": 10, "total_tokens": 3010}))
    llm = _capped(chat, clock, pacing)
    assert llm("x" * 12_000) == "ok"
    assert llm.stats["requests"][0]["admission"] == "paced" and clock.slept == []


def test_an_unlearned_large_request_is_admitted_alone_as_before():
    clock = Clock()
    pacing = llm_module.Pacing(rpm=30, tpm=8000, clock=clock)
    chat = Script(Reply("ok", {"input_tokens": 5100, "output_tokens": 2000, "total_tokens": 7100}))
    llm = _capped(chat, clock, pacing)
    assert llm("x" * 25_000) == "ok"
    assert llm.stats["requests"][0]["admission"] == "alone_unlearned"
