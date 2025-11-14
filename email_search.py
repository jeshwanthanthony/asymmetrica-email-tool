# ──────────────────────────────────────────────────────────────────────────────
# email_search.py
# Collect and/or guess valid e‑mail addresses for a given company domain
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import re
import time
import random
from typing import Iterable, List

import requests
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------- 
# config
# -----------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (EmailHunterBot/1.0; +https://example.com/bot)"}
DEFAULT_PATHS = ["", "contact", "team", "about", "our-team", "leadership", "who-we-are"]
BINARY_EXT_RE = re.compile(r"\.(pdf|png|jpe?g|gif|zip|docx?|pptx?)$", re.I)

# domains we don’t want to treat as “real” e‑mails
BLACKLISTED_EMAIL_DOMAINS = {
    "linkedin.com",
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "youtube.com",
    "example.com",  # avoid placeholders
}

# Try to import utils helpers (gracefully fallback if not present)
try:
    from utils import generate_email_formats, verify_email
except :
    def generate_email_formats(first: str, last: str, domain: str) -> List[str]:
        # very small fallback set
        return [f"{first}@{domain}", f"{first}.{last}@{domain}", f"{first}{last}@{domain}"]

    def verify_email(email: str, domain: str) -> bool:  # naïve “always true” fallback
        return True


# ----------------------------------------------------------------------------- 
# public API
# -----------------------------------------------------------------------------
def discover_emails(
    domain_or_url: str,
    *,
    candidate_names: Iterable[str] | None = None,
    max_pages: int = 6,
    verify: bool = True,
) -> List[str]:
    """
    High‑level helper that combines:

    • **Scraping**    – extracts any e‑mails already visible on public pages.  
    • **Guessing**    – for each *candidate name* (e.g. "Alice Jones") generates
                        common patterns (<first>.<last>@) and optionally verifies.

    Parameters
    ----------
    domain_or_url : "example.com" or full URL.
    candidate_names : Iterable of full‑name strings (optional).
    max_pages     : Stop traversal after this many unique pages.
    verify        : If True, run SMTP “RCPT TO” check via `utils.verify_email`.

    Returns
    -------
    List[str]     : Deduplicated, lower‑case addresses.
    """
    base = _normalise_base(domain_or_url)
    if not base:
        return []

    found: set[str] = set()

    # ---------------------------------------------------- 1) page scraping
    pages_to_visit = [f"{base}/{p}".rstrip("/") for p in DEFAULT_PATHS][:max_pages]
    visited = set()

    for url in pages_to_visit:
        if url in visited or BINARY_EXT_RE.search(url):
            continue
        visited.add(url)

        html_text = _safe_get(url)
        if not html_text:
            continue

        found |= _extract_emails_from_text(html_text)

        # opportunistic crawl: look for more “contact/team/about” links
        if len(visited) < max_pages:
            soup = BeautifulSoup(html_text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if any(key in href.lower() for key in ["contact", "team", "about"]):
                    next_url = _absolute(base, href)
                    if next_url and next_url not in visited and not BINARY_EXT_RE.search(next_url):
                        pages_to_visit.append(next_url)

        time.sleep(random.uniform(0.3, 1.0))  # polite delay

    # ---------------------------------------------------- 2) pattern guessing
    if candidate_names:
        domain = base.split("//")[-1]                # strip scheme
        first_last_pairs = _tokenise_names(candidate_names)

        for first, last in first_last_pairs:
            for guess in generate_email_formats(first, last, domain):
                guess = guess.lower()
                if guess in found:
                    continue
                if verify and not verify_email(guess, domain):
                    continue
                found.add(guess)

    return sorted(found)


# ----------------------------------------------------------------------------- 
# helpers
# -----------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
HTTP_RE  = re.compile(r"^https?://", re.I)

def _normalise_base(domain_or_url: str) -> str | None:
    """
    Turn “example.com” → “https://example.com”.  Strip trailing slashes.
    """
    if not domain_or_url:
        return None
    url = domain_or_url.strip()
    if not HTTP_RE.match(url):
        url = "https://" + url
    return url.rstrip("/")

def _absolute(base: str, link: str) -> str | None:
    """
    Make a link absolute relative to *base*.
    """
    if HTTP_RE.match(link):
        return link
    if link.startswith("/"):
        return base + link
    return None  # ignore relative non‑root links for simplicity here

def _safe_get(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        if resp.status_code == 200 and "text/html" in resp.headers.get("Content-Type", ""):
            return resp.text
    except Exception as e:
        print(f"[email_search] GET {url} failed: {e}")
    return None

def _extract_emails_from_text(text: str) -> set[str]:
    raw = {m.lower() for m in EMAIL_RE.findall(text)}
    return {e for e in raw if _email_ok(e)}

def _email_ok(email: str) -> bool:
    if any(bad in email for bad in BLACKLISTED_EMAIL_DOMAINS):
        return False
    return True

def _tokenise_names(names: Iterable[str]) -> List[tuple[str, str]]:
    """
    Very light name splitter ⇒ [(first, last), …].
    Skips obviously invalid tokens.
    """
    pairs: list[tuple[str, str]] = []
    for full in names:
        parts = full.strip().split()
        if len(parts) < 2:
            continue
        first, last = parts[0].lower(), parts[-1].lower()
        if not (first.isalpha() and last.isalpha()):
            continue
        pairs.append((first, last))
    return pairs


# ----------------------------------------------------------------------------- 
# CLI demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    dom = sys.argv[1] if len(sys.argv) > 1 else "asymmetrica.com"
    names = ["Maria Gomez", "John Smith", "Jane Doe"]
    emails = discover_emails(dom, candidate_names=names, max_pages=5)
    print(f"{len(emails)} addresses for {dom}:\n" + "\n".join(emails))