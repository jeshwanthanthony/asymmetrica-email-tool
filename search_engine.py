from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import re, time, random
import os

# ------------------------------------------------------------------------------
# main search function
# ------------------------------------------------------------------------------

# search_engine.py
from duckduckgo_search import DDGS

def duckduckgo_search(query: str, max_results: int = 20) -> list[str]:
    """
    DuckDuckGo search using the lightweight DDGS API (no Selenium needed).
    Returns a list of clean URLs.
    """
    urls = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            href = r.get("href")
            if href and href.startswith("http"):
                urls.append(href)
    return urls
