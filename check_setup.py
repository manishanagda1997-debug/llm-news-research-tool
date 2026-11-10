"""
Quick setup check. Run once after filling in the .env file:

    python check_setup.py

It checks that the keys are present and working: 1 NewsAPI request, and 1
small request to each model provider that has a key. The app needs NewsAPI
and Groq: its summaries and its optional review always use Groq. Gemini is
optional; it is used only in the Stage 2 model comparison. The keys
themselves are never printed.
"""

import sys

from src.config import PROVIDER_KEYS, missing_keys, settings

APP_PROVIDER = "groq"              # the app's summary and review always use Groq (src/model_compare.py, CONFIGS)


def app_missing_keys() -> list[str]:
    """The keys the app needs: NewsAPI and Groq, whatever LLM_PROVIDER says (it only chooses the provider of the
    development scripts)."""
    return missing_keys((APP_PROVIDER,))


def check_keys_present() -> bool:
    missing = app_missing_keys()
    if missing:
        print(f"[FAIL] Missing in .env: {', '.join(missing)}")
        print("       Copy .env.example to .env and paste your keys.")
        return False
    print(f"[OK]   NewsAPI key and the {APP_PROVIDER} key found in .env")
    return True


def check_newsapi() -> bool:
    from newsapi import NewsApiClient

    try:
        client = NewsApiClient(api_key=settings.newsapi_key)
        result = client.get_everything(q="Infosys", language="en", page_size=3)
        total = result.get("totalResults", 0)
        print(f"[OK]   NewsAPI works ({total} articles found for 'Infosys')")
        return True
    except Exception as error:
        print(f"[FAIL] NewsAPI: {error}")
        return False


def check_provider(provider: str) -> bool:
    from src.llm import make_llm
    from src.summarizer import LLMUnavailable

    llm = make_llm(provider)
    try:
        reply = llm("Reply with exactly one word: ready")
    except LLMUnavailable as error:
        print(f"[FAIL] {provider} ({llm.model_name}): {getattr(error, 'kind', 'other')}: {error}")
        return False
    print(f"[OK]   {provider} works (model: {llm.model_name}, reply: {reply.strip()}, "
          f"tokens: {llm.stats['per_call']})")
    return True


if __name__ == "__main__":
    if not check_keys_present():
        sys.exit(1)
    news_ok = check_newsapi()
    results = {}
    for provider in PROVIDER_KEYS:
        if provider == APP_PROVIDER or not missing_keys((provider,)):
            results[provider] = check_provider(provider)
        else:
            print(f"[--]   {provider}: no key in .env (optional)")
    print()
    ok = news_ok and results.get(APP_PROVIDER, False)
    print("Setup complete." if ok else "Fix the [FAIL] lines above and run again.")
