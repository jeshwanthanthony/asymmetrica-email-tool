# ──────────────────────────────────────────────────────────────────────────────
# utils.py  —  shared low‑level helpers for the CrewAI lead‑gen project
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import re
import time
import random
import socket
import smtplib
from pathlib import Path
from typing import Iterable, List, Tuple

import requests
import dns.resolver
from dotenv import load_dotenv

import requests
import mimetypes


# -----------------------------------------------------------------------------
#   1)  Environment + OpenAI key handling
# -----------------------------------------------------------------------------
_ENV_LOADED = False

def load_env(dotenv_path: str | None = None) -> None:
    """
    Read a .env file once per process.  
    Exposes OPENAI_API_KEY and any other secret variables.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    load_dotenv(dotenv_path or Path(".env"))
    _ENV_LOADED = True

def get_openai_key() -> str | None:
    load_env()
    return os.getenv("OPENAI_API_KEY")


# -----------------------------------------------------------------------------
#   2)  Normalisation helpers
# -----------------------------------------------------------------------------
def normalize_email(email: str) -> str:
    """lower‑case & strip whitespace."""
    return email.strip().lower()

def normalize_phone(phone: str) -> str:
    """Remove anything except digits and leading ‘+’."""
    return re.sub(r"[^\d+]", "", phone.strip())

def list_to_str(items: Iterable, sep: str = " | ") -> str:
    """Pretty join while gracefully handling non‑lists."""
    if isinstance(items, str):
        return items
    return sep.join(map(str, items))


# -----------------------------------------------------------------------------
#   3)  E‑mail pattern generator + SMTP verify
# -----------------------------------------------------------------------------
_DEFAULT_PATTERNS = [
    "{first}@{d}",
    "{first}.{last}@{d}",
    "{first}{last}@{d}",
    "{first}_{last}@{d}",
    "{f}{last}@{d}",
    "{first}{l}@{d}",
    "{first[0]}{last}@{d}",
    "{last}@{d}",
]

def generate_email_formats(first: str, last: str, domain: str,
                           patterns: Iterable[str] | None = None) -> List[str]:
    """
    Produce common address permutations.  
    `first`, `last` should already be **lower‑case**.
    """
    first, last = first.strip(), last.strip()
    d = domain.lower()
    result = []
    for tpl in (patterns or _DEFAULT_PATTERNS):
        # allow shorthand tokens {f} and {l} for first/last initials
        addr = tpl.format(first=first,
                          last=last,
                          f=first[0],
                          l=last[0],
                          d=d)
        result.append(addr.lower())
    return result


def verify_email(address: str, domain: str, timeout: int = 7) -> bool:
    """
    *Very* lightweight SMTP “RCPT TO” probe.  
    Returns **True** if MX responds with 250 (accepted) or 251 (will forward).

    NB: many modern servers grey‑list or tarp‑it — treat failures as *maybe*.
    """
    try:
        records = dns.resolver.resolve(domain, "MX", lifetime=timeout)
        mx_host = str(sorted(records, key=lambda r: r.preference)[0].exchange)

        smtp = smtplib.SMTP(timeout=timeout)
        smtp.connect(mx_host)
        smtp.helo(socket.gethostname() or "localhost")
        smtp.mail("probe@yourdomain.org")
        code, _ = smtp.rcpt(address)
        smtp.quit()
        return code in (250, 251)
    except Exception as e:
        print(f"[verify_email] {address} → {e}")
        return False


# -----------------------------------------------------------------------------
#   4)  HTTP/HTML helper (used by email_search + tasks)
# -----------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (LeadGenBot/2.0 +https://example.com)"}

def safe_get(url: str, timeout: int = 10) -> str | None:
    """
    GET a URL and return the **HTML text** (or None on failure / non‑HTML).  
    Random delay is injected to be polite.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code == 200 and "text/html" in resp.headers.get("Content-Type", ""):
            # Minor random sleep to reduce hammering
            time.sleep(random.uniform(0.2, 0.8))
            return resp.text
    except Exception as e:
        print(f"[safe_get] {url} failed: {e}")
    return None


# -----------------------------------------------------------------------------
#   5)  Simple skip‑list logic  (optional but handy for crawlers)
# -----------------------------------------------------------------------------
def load_skip_domains(file_path: str = "skip_domains.txt") -> set[str]:
    """Load a newline‑separated file of domains to ignore."""
    if not Path(file_path).is_file():
        return set()
    with open(file_path, encoding="utf‑8") as fh:
        return {ln.strip().lower() for ln in fh if ln.strip()}


def should_skip(url: str, skip_set: set[str]) -> bool:
    """Return True if the base domain of *url* is in *skip_set*."""
    domain = url.split("//")[-1].split("/")[0].lower()
    return domain in skip_set

def safe_scrape(url: str) -> str | None:
    """
    Attempts to fetch the page content safely.
    Skips if the page is forbidden (403), is a PDF, or other issues.

    Returns:
      - HTML text if success
      - None if skipped
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    
        content_type = response.headers.get("Content-Type", "").lower()
    
        # If it’s a PDF or binary file, skip
        if "application/pdf" in content_type or "application/octet-stream" in content_type:
            return None
    
        # Extra: if URL ends with .pdf but headers are weird
        if url.lower().endswith(".pdf"):
            return None
    
        return response.text
    
    except requests.exceptions.HTTPError as e:
        if response.status_code == 403:
            return None  # Forbidden
        return None  # Other errors
    except Exception:
        return None  # General failure

def discover_emails(domain: str, candidate_names: list[str], max_pages: int = 6, verify: bool = False) -> list[str]:
    """
    Try to generate and verify possible email addresses based on domain and candidate full names.

    Args:
        domain (str): Domain to construct emails (e.g., 'example.com').
        candidate_names (list[str]): List of names like ["John Doe", "Jane Smith"].
        max_pages (int): Not used currently (placeholder for future web scrape search).
        verify (bool): Whether to verify emails via SMTP MX check.

    Returns:
        list[str]: List of verified or plausible email addresses.
    """

    if not domain:
        return []

    def generate_formats(first, last, domain):
        return [
            f"{first[0]}{last}@{domain}", f"{first[0]}.{last}@{domain}",
            f"{first[0]}_{last}@{domain}", f"{first[0]}-{last}@{domain}", f"{first}@{domain}",
            f"{first}{last}@{domain}", f"{first}.{last}@{domain}", f"{first}_{last}@{domain}",
            f"{first}-{last}@{domain}", f"{first[0]}{last[0]}@{domain}", f"{first[0]}.{last[0]}@{domain}",
            f"{first[0]}-{last[0]}@{domain}", f"{first}{last[0]}@{domain}", f"{first}.{last[0]}@{domain}",
            f"{first}_{last[0]}@{domain}", f"{first}-{last[0]}@{domain}", f"{last}@{domain}", f"{last[0]}@{domain}"
        ]

    def verify_email(email, domain):
        try:
            records = dns.resolver.resolve(domain, "MX")
            mx_record = str(records[0].exchange)
            local_host = socket.gethostname()
            smtp_server = smtplib.SMTP(timeout=10)
            smtp_server.connect(mx_record)
            smtp_server.helo(local_host)
            smtp_server.mail('test@example.com')  # dummy sender
            code, _ = smtp_server.rcpt(email)
            smtp_server.quit()
            return code == 250
        except Exception:
            return False

    def is_catch_all(domain):
        """Check if the domain accepts random emails."""
        random_email = ''.join(random.choices(string.ascii_lowercase, k=10)) + "@" + domain
        return verify_email(random_email, domain)

    found_emails = []

    if verify and is_catch_all(domain):
        return["Unable to verify emails (catch-all domain)"]

    for full_name in candidate_names:
        name_parts = full_name.split()
        if len(name_parts) < 2:
            continue
        first, last = name_parts[0].lower(), name_parts[-1].lower()
        candidates = generate_formats(first, last, domain)

        # try each possible format
        for email in candidates:
            if verify:
                if verify_email(email, domain):
                    found_emails.append(email)
                    break  # first valid email found → stop
            else:
                # if not verifying, just take the first plausible format
                found_emails.append(candidates[0])
                break

    return found_emails