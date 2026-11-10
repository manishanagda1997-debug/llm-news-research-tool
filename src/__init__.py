"""
News Research Tool - core package.

This folder holds all the logic of the tool:
    config.py           settings and API keys
    news_fetcher.py     gets news articles from NewsAPI       (next step)
    preprocess.py       cleans and numbers the articles       (next step)
    prompts.py          instruction text for the LLM          (next step)
    langchain_config.py LLM setup and summary chain           (next step)
    utils.py            caching, export and small helpers     (next step)

app.py (the Streamlit screen) imports from here, for example:
    from src.config import settings
"""

__version__ = "0.1.0"
