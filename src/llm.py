"""
The connection to the language model.

build_chat_model() returns the LangChain chat model itself (used by
langchain_config.py). make_llm() wraps it in a plain function: prompt text
in, answer text out. The summary and relevance steps only use that function,
so they do not depend on which company's model answers.

    provider "gemini"  Google Gemini through LangChain, temperature 0
    provider "groq"    an open model on Groq through LangChain, temperature 0
                       (default openai/gpt-oss-120b, a reasoning model)

The provider comes from LLM_PROVIDER in the settings ("gemini" when empty),
unless the call names one. Both chat models are built with max_retries=0, so
every rate-limit error reaches the retry loop below and is counted the same
way for both providers.

Pacing and errors:
    pacing       a request waits until it fits the requests-a-minute and
                 tokens-a-minute limits (Groq: 30 and 8,000 on the free plan);
                 one that can never fit raises RequestTooLarge before it is
                 sent, so the caller can split its batch. The estimate starts
                 careful and then learns from the real token counts, with an
                 answer margin; requests in progress hold their reservation;
                 Groq's rate-limit headers add what the provider reports
    rate limit   wait the time the provider gives (Retry-After header, or the
                 wait in the message), at most MAX_RETRIES times; no retry
                 without a given wait or when the wait is longer than the
                 time left
    daily limit, sign-in problem, timeout: no retry
    server error two retries, after SERVER_WAITS seconds (2, then 8), or
                 after the wait the provider gives
Every failure becomes LLMUnavailable with a `kind`. The app then shows the
labelled fallback instead of crashing.

The API keys are read from the settings (the .env file). They are never
printed, and an error message never shows them.

Live check (1 small request to the provider in the settings):
    python -m src.llm
"""

import json
import math
import re
import threading
import time
from collections import deque

from src.config import settings
from src.summarizer import LLMUnavailable

PROVIDERS = ("gemini", "groq")
REQUEST_TIMEOUT = 60            # seconds for one request
MAX_RETRIES = 3                 # rate-limit retries for one request
SERVER_WAITS = (2.0, 8.0)       # seconds before each retry after a server error
CHARS_PER_TOKEN = 3             # a careful estimate: real text has about 4 characters a token
ANSWER_ALLOWANCE = 1500         # tokens kept free for the answer and the model's reasoning (the floor)
MAX_TOKENS_PER_CHAR = 1.0       # text never has more tokens than characters
LEARN_POLL = 0.5                # seconds between checks while the first answer is awaited
OUTPUT_MARGIN = 1.25            # the answer margin: the largest answer seen so far, plus a quarter
PACER_VERSION = "3"             # 1 counted the whole output limit against the minute; 2 counts what Groq counts;
                                # 3 sends a learned request alone when only REQUEST_MARGIN pushes it over the minute
REQUEST_MARGIN = 600            # with an output limit: Groq counts the input plus a few hundred tokens (its
                                # "Requested" values were 276 to 551 above the input), not the whole limit
CONTEXT_WINDOW = {"openai/gpt-oss-120b": 131_072, "openai/gpt-oss-20b": 131_072}


class RequestTooLarge(LLMUnavailable):
    """The request can never fit the provider's token limit: split it, never cut an article."""

    kind = "too_large"


def _gemini_class():
    # imported here, so the tests and the rest of the code work without loading the provider libraries
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI


def _groq_class():
    from langchain_groq import ChatGroq
    return ChatGroq


def _text(reply) -> str:
    text = getattr(reply, "text", None)
    if callable(text) and not isinstance(text, str):   # older LangChain: text() is a method
        text = text()
    if text is None:
        text = getattr(reply, "content", "")
    return text if isinstance(text, str) else ""


def _provider(provider: str | None) -> str:
    provider = provider or settings.llm_provider or "gemini"
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}.")
    return provider


def _model_name(provider: str) -> str:
    """Check the provider and its key; return the model name from the settings."""
    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is missing. Add it to the .env file.")
        return settings.groq_model
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Add it to the .env file.")
    return settings.gemini_model


GENERATION_KEYS = {"gemini": {"temperature", "thinking_level"},
                   "groq": {"temperature", "top_p", "reasoning_effort", "reasoning_format", "max_completion_tokens"}}
HISTORICAL_GEMINI = {"temperature": 0}     # the settings of every Gemini answer saved before Stage 3


def _generation(provider: str, generation: dict | None = None) -> dict:
    """The generation settings: the ones the call names, or the project defaults."""
    if generation is not None:
        unknown = set(generation) - GENERATION_KEYS[provider]
        if unknown:
            raise ValueError(f"Unknown {provider} settings: {sorted(unknown)}")
        return dict(generation)
    if provider == "groq":
        return {"temperature": 0, "reasoning_effort": settings.groq_reasoning_effort}
    return dict(HISTORICAL_GEMINI)


def _cache_tag(provider: str, generation: dict) -> str:
    """Empty only for the historical Gemini settings, so the answers saved with them keep their keys.
    Any other settings save and read their own answers."""
    if provider == "gemini" and generation == HISTORICAL_GEMINI:
        return ""
    return provider + "|" + json.dumps(generation, sort_keys=True)


def build_chat_model(provider: str | None = None, model: str | None = None, generation: dict | None = None,
                     on_headers=None):
    """The LangChain chat model for the provider. It is built, not called.
    model and generation: a named configuration (Stage 3.3); the project defaults when None.
    on_headers: called with the headers of every Groq response (the rate-limit news for the pacer)."""
    provider = _provider(provider)
    model_name = model or _model_name(provider)
    _model_name(provider)                                   # the key check, also for a named model
    options = _generation(provider, generation)
    if provider == "groq":
        passed = {name: options.pop(name) for name in ("top_p", "max_completion_tokens") if name in options}
        extra = {"model_kwargs": passed} if passed else {}     # sent as they are, under Groq's own names
        if on_headers is not None:
            import httpx
            extra["http_client"] = httpx.Client(timeout=REQUEST_TIMEOUT,
                                                event_hooks={"response": [lambda r: on_headers(r.headers)]})
        return _groq_class()(model=model_name, api_key=settings.groq_api_key, max_retries=0,
                             timeout=REQUEST_TIMEOUT, **options, **extra)
    return _gemini_class()(model=model_name, google_api_key=settings.google_api_key,
                           max_retries=0, timeout=REQUEST_TIMEOUT, **options)


# ---------- pacing ----------

def estimate_tokens(prompt: str) -> int:
    """The careful estimate, used before any real token count is known."""
    return len(prompt) // CHARS_PER_TOKEN + 1 + ANSWER_ALLOWANCE


def _duration(text: str) -> float | None:
    """Groq's duration format: "7.66s", "1m2.5s", "450ms", "2m59.56s"."""
    match = re.fullmatch(r"\s*(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)(ms|s))?\s*", text or "")
    if not match or not any(match.groups()):
        return None
    hours, minutes, amount, unit = match.groups()
    seconds = float(amount or 0) / (1000 if unit == "ms" else 1)
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + seconds


class Pacing:
    """
    Keeps requests under the requests-a-minute and tokens-a-minute limits. Safe across threads.

    estimate     the careful estimate first; after a real answer, the measured tokens per
                 character (the highest seen) plus an answer margin that never falls below
                 ANSWER_ALLOWANCE and grows with the largest answer seen
    reserve      a request takes its estimate from the minute when it is sent, so requests
                 in progress hold their share; settle() swaps it for the real use (cached
                 input tokens do not count against the limit)
    headers      Groq's remaining tokens and reset time, when the responses carry them; a
                 request sent after the headers lowers what they report as left
    """

    def __init__(self, rpm: int, tpm: int, clock=time.monotonic):
        self.rpm, self.tpm, self.clock = rpm, tpm, clock
        self.interval = 60.0 / rpm
        self.last = None
        self.used: dict[int, list] = {}            # ticket -> [time, tokens]
        self.next_ticket = 0
        self.tokens_per_char = None
        self.largest_answer = 0
        self.header = None                          # (time, remaining tokens, reset seconds)
        self.output_cap = None                      # an explicit output limit: sent, and checked against the context
        self.context_window = None                  # the model's context window in tokens, when known
        self.in_flight: set[int] = set()            # sent, with no answer or error yet
        self.lock = threading.RLock()

    # estimates
    def answer_margin(self) -> int:
        """The tokens counted beyond the input: REQUEST_MARGIN when an output limit is set (what Groq counts),
        otherwise the largest answer seen so far plus a quarter, at least ANSWER_ALLOWANCE."""
        if self.output_cap:
            return REQUEST_MARGIN
        return max(ANSWER_ALLOWANCE, round(self.largest_answer * OUTPUT_MARGIN))

    def estimate(self, prompt: str) -> int:
        if self.tokens_per_char is None:
            return len(prompt) // CHARS_PER_TOKEN + 1 + self.answer_margin()
        return math.ceil(len(prompt) * self.tokens_per_char) + 1 + self.answer_margin()

    def fits(self, tokens: int) -> bool:
        return tokens <= self.tpm

    def careful_input(self, prompt: str) -> int:
        return len(prompt) // CHARS_PER_TOKEN + 1

    def input_estimate(self, prompt: str) -> int:
        """The input alone: the learned rate, or the careful estimate before anything is learned."""
        if self.tokens_per_char is None:
            return self.careful_input(prompt)
        return math.ceil(len(prompt) * self.tokens_per_char) + 1

    def margin_only_over(self, prompt: str) -> bool:
        """Pacer version 3: with an output limit set and the rate learned, the input fits the minute and only
        REQUEST_MARGIN (the project's planning allowance, not a provider limit) pushes the estimate over it.
        Such a request is sent alone and the provider decides."""
        return bool(self.output_cap) and self.tokens_per_char is not None and self.input_estimate(prompt) <= self.tpm

    def alone_clear(self) -> bool:
        """Nothing in flight and no tokens used in the last 60 seconds."""
        with self.lock:
            self._drop_old(self.clock())
            return not self.used and not self.in_flight

    def unlearned_and_input_fits(self, prompt: str) -> bool:
        """Before any real token count, the careful estimate (3 characters a token, about 1.6 times real news
        text) would refuse requests that fit. With an output limit set, such a request is sent alone instead and
        the provider decides; a provider refusal uses no tokens."""
        return bool(self.output_cap) and self.tokens_per_char is None

    def beyond_context(self, prompt: str) -> bool:
        """The input (careful estimate) and the whole output limit do not fit the model's context window."""
        return bool(self.context_window and self.output_cap) and \
            self.careful_input(prompt) + self.output_cap > self.context_window

    def unlearned_alone_allowed(self, prompt: str) -> bool:
        with self.lock:
            self._drop_old(self.clock())
            return self.unlearned_and_input_fits(prompt) and not self.used and not self.in_flight

    def wait_alone(self) -> float:
        """Seconds until nothing else is in the minute."""
        with self.lock:
            now = self.clock()
            self._drop_old(now)
            if self.in_flight:
                return LEARN_POLL
            return max((when + 60.0 - now for when, _ in self.used.values()), default=0.0)

    # the minute
    def _drop_old(self, now: float) -> None:
        for ticket in [t for t, (when, _) in self.used.items() if now - when >= 60.0]:
            del self.used[ticket]

    def used_tokens(self) -> int:
        with self.lock:
            self._drop_old(self.clock())
            return sum(tokens for _, tokens in self.used.values())

    def wait_time(self, tokens: int) -> float:
        with self.lock:
            now = self.clock()
            self._drop_old(now)
            wait = 0.0 if self.last is None else max(0.0, self.last + self.interval - now)
            entries = sorted(self.used.values())
            total = sum(t for _, t in entries)
            for when, spent in entries:             # wait until enough old use leaves the window
                if total + tokens <= self.tpm:
                    break
                total -= spent
                wait = max(wait, when + 60.0 - now)
            if self.header is not None:
                seen, remaining, reset = self.header
                if now - seen < reset:
                    sent_since = sum(t for when, t in entries if when >= seen)
                    if tokens > remaining - sent_since:
                        wait = max(wait, seen + reset - now)
            return wait

    def reserve(self, tokens: int) -> int:
        with self.lock:
            now = self.clock()
            self.last = now
            ticket = self.next_ticket
            self.next_ticket += 1
            self.used[ticket] = [now, tokens]
            self.in_flight.add(ticket)
            return ticket

    def done(self, ticket: int | None) -> None:
        """The request has ended, with an answer or an error."""
        with self.lock:
            self.in_flight.discard(ticket)

    def waiting_to_learn(self) -> bool:
        """Before the first real token count, a request waits for the one in progress instead of guessing."""
        with self.lock:
            return self.tokens_per_char is None and bool(self.in_flight)

    def settle(self, ticket: int, prompt_chars: int, input_tokens: int, output_tokens: int,
               cached_tokens: int = 0) -> None:
        """Swap the reservation for the real use and learn from it."""
        with self.lock:
            if ticket in self.used:
                self.used[ticket][1] = max(0, input_tokens - cached_tokens) + output_tokens
            if prompt_chars and input_tokens:
                ratio = min(input_tokens / prompt_chars, MAX_TOKENS_PER_CHAR)   # one odd reply cannot block the run
                self.tokens_per_char = max(ratio, self.tokens_per_char or 0.0)
            self.largest_answer = max(self.largest_answer, output_tokens)

    def release(self, ticket: int | None) -> None:
        with self.lock:
            self.used.pop(ticket, None)
            self.in_flight.discard(ticket)

    def update_from_headers(self, headers) -> None:
        get = headers.get if hasattr(headers, "get") else (lambda name: None)
        remaining, reset = get("x-ratelimit-remaining-tokens"), get("x-ratelimit-reset-tokens")
        if remaining is None or reset is None:
            return
        try:
            remaining = int(float(remaining))
        except ValueError:
            return
        seconds = _duration(reset)
        if seconds is None:
            return
        with self.lock:
            self.header = (self.clock(), remaining, seconds)


def _default_pacing(provider: str, clock=time.monotonic) -> Pacing | None:
    if provider == "groq":
        return Pacing(rpm=settings.groq_rpm, tpm=settings.groq_tpm, clock=clock)
    if provider == "gemini":
        return Pacing(rpm=settings.gemini_rpm, tpm=settings.gemini_tpm, clock=clock)
    return None


# ---------- errors ----------

WAIT_IN_MESSAGE = re.compile(r"(?:try again|retry) in (?:(\d+)m(?!s))?\s*([\d.]+)?\s*(ms|s)?", re.I)
DAILY = re.compile(r"per ?day|\(RPD\)|\(TPD\)", re.I)     # Groq: "per day (TPD)"; Google: "PerDay"


def _status(error) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _given_wait(error) -> float | None:
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    if value is not None:
        try:
            return float(value)
        except ValueError:
            pass
    match = WAIT_IN_MESSAGE.search(str(error))
    if not match or not (match.group(1) or match.group(2)):
        return None
    minutes = float(match.group(1) or 0)
    amount = float(match.group(2) or 0)
    if match.group(3) == "ms":
        amount /= 1000
    return round(minutes * 60 + amount, 3)


def error_kind(error) -> str:
    status, text, name = _status(error), str(error), type(error).__name__
    if status == 413 or "request too large" in text.lower():
        return "too_large"
    if status == 429 or "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower():
        return "daily_limit" if DAILY.search(text) else "rate_limit"
    if status in (401, 403) or name in ("AuthenticationError", "PermissionDeniedError"):
        return "auth"
    if isinstance(error, TimeoutError) or "Timeout" in name:
        return "timeout"
    if (status is not None and status >= 500) or name in ("InternalServerError", "ServiceUnavailable"):
        return "server"
    return "other"


ORGANISATION_ID = re.compile(r"org_[A-Za-z0-9]+")
PROVIDER_NUMBERS = re.compile(r"Limit\s+(\d+),\s*Requested\s+(\d+)")


def _hide_keys(text: str) -> str:
    """API keys and the provider's organisation IDs never reach a message or a record."""
    for key in (settings.groq_api_key, settings.google_api_key, settings.newsapi_key):
        if key:
            text = text.replace(key, "[key hidden]")
    return ORGANISATION_ID.sub("org_[hidden]", text)


def _provider_numbers(text: str) -> dict:
    """The limit and the requested tokens from a provider refusal ("Limit 8000, Requested 8234"), when given."""
    match = PROVIDER_NUMBERS.search(text)
    return {"provider_limit": int(match.group(1)), "provider_requested": int(match.group(2))} if match else {}


def _unavailable(kind: str, message: str) -> LLMUnavailable:
    error = (RequestTooLarge if kind == "too_large" else LLMUnavailable)(_hide_keys(message))
    error.kind = kind
    return error


def _finish_reason(reply) -> str | None:
    """Why the provider stopped writing: "stop" for a finished answer, "length" for a cut-off one."""
    metadata = getattr(reply, "response_metadata", None)
    return metadata.get("finish_reason") if isinstance(metadata, dict) else None


def _usage(reply) -> dict | None:
    usage = getattr(reply, "usage_metadata", None)
    if not isinstance(usage, dict):
        return None
    details = usage.get("output_token_details") or {}
    cached = (usage.get("input_token_details") or {}).get("cache_read")
    return {"input": int(usage.get("input_tokens") or 0), "output": int(usage.get("output_tokens") or 0),
            "reasoning": int(details.get("reasoning") or 0), "cached": int(cached or 0),
            "total": int(usage.get("total_tokens") or 0) or None}


# ---------- the function the project uses ----------

def make_llm(provider: str | None = None, chat_model=None, model_name: str | None = None,
             model: str | None = None, generation: dict | None = None,
             pacing: Pacing | None = None, clock=time.monotonic, sleep=time.sleep,
             time_budget: float | None = None):
    """
    A function that sends one prompt and returns the answer text.

    chat_model / model_name - a ready model (tests, or a model built elsewhere)
    model / generation      - a named configuration; the project defaults when None
    pacing                  - request and token limits; the provider's own by default
    time_budget             - seconds for the whole run; no wait goes past it
    """
    provider = _provider(provider)
    stats = {"calls": 0, "rate_limited": 0, "retries": 0, "failures": {}, "waits": [], "waited_seconds": 0.0,
             "header_updates": 0, "splits": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "per_call": [],
             "requests": []}
    stats_lock = threading.Lock()

    def on_headers(headers) -> None:
        pacing.update_from_headers(headers)
        with stats_lock:
            stats["header_updates"] += 1               # 0 after a run: the headers never reach the pacer

    if chat_model is None:
        model_name = model or _model_name(provider)
        pacing = pacing or _default_pacing(provider, clock)
        chat_model = build_chat_model(provider, model, generation, on_headers=on_headers if pacing else None)
    deadline = None if time_budget is None else clock() + time_budget
    generation = _generation(provider, generation)
    if pacing is not None and generation.get("max_completion_tokens"):
        pacing.output_cap = generation["max_completion_tokens"]
        pacing.context_window = pacing.context_window or CONTEXT_WINDOW.get(model or model_name or "")

    def time_left() -> float | None:
        return None if deadline is None else deadline - clock()

    def pause(kind: str, seconds: float) -> None:
        with stats_lock:
            stats["waits"].append({"kind": kind, "seconds": round(seconds, 3)})
            stats["waited_seconds"] = round(stats["waited_seconds"] + seconds, 3)
        sleep(seconds)

    def fail(kind: str, message: str) -> LLMUnavailable:
        with stats_lock:
            stats["failures"][kind] = stats["failures"].get(kind, 0) + 1
        return _unavailable(kind, message)

    def split(message: str, stage: str, tokens: int | None = None) -> RequestTooLarge:
        """stage: "preflight" (the project refuses it before sending) or "provider" (the provider rejects it)."""
        with stats_lock:
            stats["splits"] += 1                       # the caller splits the batch: not a failure
        error = _unavailable("too_large", message)
        error.stage, error.estimated = stage, tokens
        return error

    def wait_turn(prompt: str) -> tuple[int | None, int, str | None]:
        """Wait until the request may go; return its reservation ticket, its estimate and how it is admitted
        ("paced", "alone_unlearned" or "alone_margin_only"). The estimate is made again after every wait, so it
        uses everything learned meanwhile. The context-window check comes first."""
        if pacing is None:
            return None, estimate_tokens(prompt), None
        learning_since = clock()
        while True:
            with pacing.lock:
                if pacing.waiting_to_learn() and clock() - learning_since < REQUEST_TIMEOUT:
                    kind, wait = "learning", LEARN_POLL
                else:
                    tokens = pacing.estimate(prompt)
                    if pacing.beyond_context(prompt):
                        raise split(f"About {pacing.careful_input(prompt)} input tokens and an output limit of "
                                    f"{pacing.output_cap} do not fit the model's context window of "
                                    f"{pacing.context_window} tokens.", "preflight", tokens)
                    if not pacing.fits(tokens):
                        if pacing.unlearned_and_input_fits(prompt):
                            admission = "alone_unlearned"
                        elif pacing.margin_only_over(prompt):
                            admission = "alone_margin_only"
                        else:
                            raise split(f"About {tokens} tokens; the limit is {pacing.tpm} tokens a minute.",
                                        "preflight", tokens)
                        if pacing.alone_clear():
                            return pacing.reserve(tokens), tokens, admission   # holds the minute; settled on the answer
                        kind, wait = "alone", pacing.wait_alone()
                    else:
                        kind, wait = "pacing", pacing.wait_time(tokens)
                        if wait <= 0:
                            return pacing.reserve(tokens), tokens, "paced"
            left = time_left()
            if left is not None and wait > left:
                raise fail("rate_limit", f"The wait for the rate limit ({wait:.1f} s) is longer than the time left.")
            pause(kind, wait)

    started = clock()

    def _before_sending(tokens: int) -> dict:
        """For diagnosis: when the request leaves, the project's own count of the minute's tokens before it, and
        the provider's last reported remaining tokens with the age of that report (numbers only, no text)."""
        detail = {"sent_at": round(clock() - started, 3), "window_used": None, "provider_remaining": None,
                  "header_age": None}
        if pacing is not None:
            detail["window_used"] = pacing.used_tokens() - tokens
            header = pacing.header
            if header is not None:
                seen, remaining, _ = header
                detail.update(provider_remaining=remaining, header_age=round(clock() - seen, 3))
        return detail

    def ask(prompt: str) -> str:
        """One planned request; the request log keeps its last estimate and how it ends."""
        entry = {"estimated": None, "prompt_chars": len(prompt), "outcome": None}
        try:
            text = send(prompt, entry)
            entry["outcome"] = "answered"
            return text
        except RequestTooLarge as error:
            stage = getattr(error, "stage", "provider")
            entry["outcome"] = f"{stage}_too_large"
            if stage == "preflight":
                entry["estimated"] = error.estimated
            else:
                entry.update(error=str(error), **_provider_numbers(str(error)))
            raise
        except LLMUnavailable as error:
            entry["outcome"] = f"provider_{getattr(error, 'kind', 'unknown')}"
            entry["error"] = str(error)
            raise
        finally:
            with stats_lock:
                stats["requests"].append(entry)

    def send(prompt: str, entry: dict) -> str:
        retries = server_retries = 0
        while True:
            ticket, tokens, admission = wait_turn(prompt)
            entry["estimated"] = tokens
            if admission is not None:
                entry.update(admission=admission, input_estimate=pacing.input_estimate(prompt))
            before = _before_sending(tokens)
            with stats_lock:
                stats["calls"] += 1
            try:
                reply = chat_model.invoke(prompt)
            except Exception as error:      # every provider problem: wait and retry, or no answer this time
                if pacing is not None:
                    pacing.done(ticket)
                kind = error_kind(error)
                if kind == "too_large":
                    raise split(f"{type(error).__name__}: {error}", "provider") from None
                message = f"{type(error).__name__}: {error}"
                if kind == "rate_limit":
                    if pacing is not None:
                        pacing.release(ticket)          # a refused request uses no tokens
                    with stats_lock:
                        stats["rate_limited"] += 1
                    wait, left = _given_wait(error), time_left()
                    if wait is None or retries >= MAX_RETRIES:
                        raise fail(kind, message) from None
                    if left is not None and wait > left:
                        raise fail(kind, f"{message} (the wait of {wait} s is longer than the time left)") from None
                    retries += 1
                    with stats_lock:
                        stats["retries"] += 1
                    pause(kind, wait)
                    continue
                if kind == "server" and server_retries < len(SERVER_WAITS):
                    wait, left = _given_wait(error) or SERVER_WAITS[server_retries], time_left()
                    if left is not None and wait > left:
                        raise fail(kind, f"{message} (the wait of {wait} s is longer than the time left)") from None
                    server_retries += 1
                    with stats_lock:
                        stats["retries"] += 1
                    pause(kind, wait)
                    continue
                raise fail(kind, message) from None
            if pacing is not None:
                pacing.done(ticket)
            usage = _usage(reply)
            if usage:
                with stats_lock:
                    stats["input_tokens"] += usage["input"]
                    stats["output_tokens"] += usage["output"]
                    stats["reasoning_tokens"] += usage["reasoning"]
                    stats["per_call"].append({**{k: usage[k] for k in ("input", "output", "reasoning", "cached")},
                                              "estimated": tokens, "prompt_chars": len(prompt),
                                              "finish_reason": _finish_reason(reply), **before,
                                              "answered_at": round(clock() - started, 3)})
                if pacing is not None:
                    pacing.settle(ticket, prompt_chars=len(prompt), input_tokens=usage["input"],
                                  output_tokens=usage["output"], cached_tokens=usage["cached"])
            return _text(reply)

    ask.provider = provider
    ask.model_name = model_name
    ask.generation = generation
    ask.cache_tag = _cache_tag(provider, generation)
    ask.stats = stats
    ask.pacing = pacing
    ask.output_limit = generation.get("max_completion_tokens") or getattr(chat_model, "max_tokens", None)
    return ask


if __name__ == "__main__":
    llm = make_llm()
    try:
        answer = llm("Reply with exactly one word: ready")
        print(f"[OK]   {llm.provider} answered (model: {llm.model_name}): {answer.strip()}")
        print(f"       tokens: {llm.stats['per_call']}")
    except LLMUnavailable as error:
        print(f"[FAIL] {llm.provider} ({llm.model_name}) is unavailable ({error.kind}): {error}")
