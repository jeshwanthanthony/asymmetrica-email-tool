# main.py
# ──────────────────────────────────────────────────────────
# Streamlit UI → Inline Email Generator using original Preqin columns
# ──────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
load_dotenv()

import io
import json
import streamlit as st
import pandas as pd
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# crewai imports remain (not used here, but kept if you need the pipeline later)
from crewai import Crew
from agents import AgentFactory
from tasks import TaskFactory

import utils as U
import search_engine as SE
import email_search as ES
from email_tool import email_tool  # our working email sender

st.set_page_config(page_title="Investment Lead Generator", page_icon="🌱")
st.title("🌱 Investment Lead Generator ＋ Inline Email Generator")

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
def first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name from candidates that exists in df (exact match)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def add_minimal_normalized_columns(firm_df: pd.DataFrame) -> pd.DataFrame:
    """
    IMPORTANT:
    - Do NOT rename or drop original Preqin columns.
    - ONLY add minimal normalized columns required by email_tool:
      'Company Name', 'Email', 'Investment Interests'
    """
    df = firm_df.copy()

    # Company Name
    name_col = first_present(df, ["FIRM NAME", "LOCAL LANGUAGE FIRM NAME", "Company Name"])
    if name_col and "Company Name" not in df.columns:
        df["Company Name"] = df[name_col]
    elif "Company Name" not in df.columns:
        df["Company Name"] = ""

    # Email (prefer Preqin priority-email fields before fallback to EMAIL)
    email_candidates = [
        "PE: PRIORITY CONTACT EMAIL",
        "RE: PREFERRED INITIAL CONTACT EMAIL",
        "NR: PREFERRED INITIAL CONTACT EMAIL",
        "INF: PREFERRED INITIAL CONTACT EMAIL",
        "PRIORITY CONTACT EMAIL",
        "EMAIL",
    ]
    email_col = first_present(df, email_candidates)
    if email_col and "Email" not in df.columns:
        df["Email"] = df[email_col]
    elif "Email" not in df.columns:
        df["Email"] = ""

    # Investment Interests: use BACKGROUND if present; otherwise Strategy Preferences
    if "Investment Interests" not in df.columns:
        if "BACKGROUND" in df.columns:
            df["Investment Interests"] = df["BACKGROUND"]
        elif "PE: STRATEGY PREFERENCES" in df.columns:
            df["Investment Interests"] = df["PE: STRATEGY PREFERENCES"]
        else:
            df["Investment Interests"] = ""

    return df

# ──────────────────────────────────────────────────────────
# Upload & Inline Email Generator (front and center)
# ──────────────────────────────────────────────────────────
uploaded_xlsx = st.file_uploader("Upload your CRM (.xlsx) with Preqin sheets", type=["xlsx"])

if uploaded_xlsx is None:
    st.info("Upload a Preqin-exported Excel file (e.g., sheet 'Preqin_Export' + 'Contacts_Export') to begin.")
else:
    try:
        with st.spinner("⏳ Reading your Excel..."):
            excel_file = pd.ExcelFile(uploaded_xlsx, engine="openpyxl")

            # Firm-level sheet (keep original columns)
            if "Preqin_Export" in excel_file.sheet_names:
                firm_df = excel_file.parse("Preqin_Export")
            else:
                firm_df = excel_file.parse(0)

            # Contact-level sheet (optional)
            contacts_df = None
            if "Contacts_Export" in excel_file.sheet_names:
                contacts_df = excel_file.parse("Contacts_Export")

        # Add ONLY the three minimal normalized columns; do not touch original names
        working_df = add_minimal_normalized_columns(firm_df)

        # Quick sanity check: required by email_tool
        required = {"Company Name", "Email", "Investment Interests"}
        missing = [c for c in required if c not in working_df.columns]
        if missing:
            st.error(f"Missing required columns for email tool: {missing}")
        else:
            st.subheader("✉️ Inline Email Generator")
            # Pass the full row with ALL original Preqin columns + the 3 normalized ones
            email_tool(working_df, contacts_df)

    except Exception as e:
        st.error(f"Failed to read Excel: {e}")

# ──────────────────────────────────────────────────────────
# (Optional) Keep logging for debug
# ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.DEBUG)
