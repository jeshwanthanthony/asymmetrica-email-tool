# main.py
# ──────────────────────────────────────────────────────────
# Streamlit UI  →  CrewAI pipeline  →  XLSX CRM updater
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
import streamlit.components.v1 as components
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

from crewai import Crew
from agents import AgentFactory
from tasks import TaskFactory

import utils as U
import search_engine as SE
import email_search as ES
from email_tool import email_tool  # our working email sender

# ──────────────────────────────────────────────────────────
# Normalize CRM columns (imported logic from email_tool.py)
# ──────────────────────────────────────────────────────────
def normalize_crm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map large XLSX schema to minimal columns expected by the app."""
    cols = {c.strip().lower(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            key = c.strip().lower()
            if key in cols:
                return cols[key]
        return None

    name_col = pick("FIRM NAME", "LOCAL LANGUAGE FIRM NAME", "Company Name")
    email_col = pick(
        "PE: PRIORITY CONTACT EMAIL",
        "RE: PREFERRED INITIAL CONTACT EMAIL",
        "NR: PREFERRED INITIAL CONTACT EMAIL",
        "INF: PREFERRED INITIAL CONTACT EMAIL",
        "PRIORITY CONTACT EMAIL",
        "EMAIL",
    )
    interests_col = pick(
        "PE: STRATEGY PREFERENCES",
        "RE: STRATEGY PREFERENCES",
        "NR: STRATEGY PREFERENCES",
        "INF: STRATEGY PREFERENCES",
        "PE: INDUSTRIES",
        "RE: GEOGRAPHIC PREFERENCES",
        "NR: INDUSTRY PREFERENCES",
        "BACKGROUND",
        "FIRM TYPE",
    )

    out = pd.DataFrame()
    out["Company Name"] = df[name_col] if name_col else ""

    if email_col:
        out["Email"] = (
            df[email_col]
            .astype(str)
            .fillna("")
            .apply(
                lambda s: re.search(
                    r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", s, re.I
                ).group(0)
                if re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", s, re.I)
                else ""
            )
        )
    else:
        out["Email"] = ""

    out["Investment Interests"] = df[interests_col] if interests_col else ""

    for nice in ["CITY", "STATE/COUNTY", "COUNTRY", "REGION", "WEBSITE"]:
        key = nice.strip().lower()
        if key in cols and cols[key] not in out.columns:
            out[cols[key]] = df[cols[key]]

    return out


# ──────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
st.set_page_config(page_title="Investment Lead Generator", page_icon="🌱")
st.title("🌱 Investment Lead Generator ＋ CrewAI")

if "open_email_tool" not in st.session_state:
    st.session_state.open_email_tool = False

with st.sidebar:
    st.subheader("✉️ Tools")

    if st.button("Open Inline Email Generator", key="open_email_inline_btn"):
        st.session_state.open_email_tool = True

    st.divider()
    st.subheader("📝 CRM Generator")

    query_crm = st.text_input(
        "Search Query", value="sustainable farmland investment family office")
    max_crm = st.number_input("Number of URLs", min_value=10, max_value=500, value=100)

    if st.button("Generate CRM", key="gen_crm"):
        if not query_crm.strip():
            st.error("Please enter a search query.")
        else:
            st.info(f"Searching DuckDuckGo for {max_crm} results...")
            try:
                urls = SE.duckduckgo_search(query_crm, max_results=max_crm)
            except Exception as e:
                st.error(f"Search failed: {e}")
                urls = []
            if urls:
                df_crm = pd.DataFrame({"URL": urls})
                buf = io.BytesIO()
                df_crm.to_excel(buf, index=False, engine="openpyxl")
                buf.seek(0)
                st.sidebar.download_button(
                    "⬇️ Download new CRM",
                    buf,
                    file_name=f"new_crm_{datetime.now(timezone.utc):%Y%m%d%H%M}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No URLs found.")

uploaded_xlsx = st.file_uploader("Upload your CRM (.xlsx)", type=["xlsx"])
sector = st.text_input("Financial Sector", value="Private Equity")
investor_type = st.text_area("Targeted Investor Type(s)",
                             value="Family offices, sustainable agriculture investors, etc.")
investment_prod = st.text_area(
    "Investment Product",
    value="High-yield farmland investment opportunity in Latin America that supports sustainable agriculture")
company_info = st.text_area(
    "About *your* Company",
    value="We are a global investment platform focusing on sustainable agriculture and impactful real estate investments.")
num_leads = st.number_input("Leads to process this run", min_value=1, value=5)


# ──────────────────────────────────────────────────────────
# Inline Email Tool — exact extraction from abooktest.xlsx
# ──────────────────────────────────────────────────────────
if st.session_state.open_email_tool:
    if uploaded_xlsx is None:
        st.info("Upload a CRM (.xlsx) above to use the Inline Email Generator.")
    else:
        try:
            with st.spinner("⏳ Loading and preparing your CRM... please wait"):
                _crm_df = pd.read_excel(uploaded_xlsx, engine="openpyxl")

            # --- Explicitly extract and rename key columns from your Excel ---
            expected_cols = {
                "FIRM NAME": "Company Name",
                "EMAIL": "Email",
                "CITY": "City",
                "COUNTRY": "Country",
                "BACKGROUND": "Investment Interests",
                "FIRM TYPE": "Firm Type",
            }

            available_cols = [c for c in _crm_df.columns if c in expected_cols]
            if not {"FIRM NAME", "EMAIL"}.issubset(set(_crm_df.columns)):
                st.error("Could not find required columns (FIRM NAME, EMAIL) in your Excel.")
                st.stop()

            # Keep only relevant columns
            _crm_df = _crm_df[[c for c in expected_cols.keys() if c in _crm_df.columns]].rename(columns=expected_cols)

            st.subheader("✉️ Inline Email Generator")

            # --- Build friendly dropdown label ---
            _crm_df["Display"] = _crm_df.apply(
                lambda row: f"{row['Company Name']} ({row['Email']}) — {row.get('City', '')}, {row.get('Country', '')}"
                if pd.notna(row['Email']) and row['Email']
                else f"{row['Company Name']} — {row.get('City', '')}, {row.get('Country', '')}",
                axis=1
            )

            # --- Dropdown with firm + email + location ---
            company_choice = st.selectbox("Select a Company", _crm_df["Display"])

            # Filter to the selected company row
            selected_row = _crm_df[_crm_df["Display"] == company_choice].iloc[0]

            st.write("🏢 **Firm Name:**", selected_row["Company Name"])
            st.write("📧 **Email:**", selected_row["Email"])
            if "City" in _crm_df.columns and "Country" in _crm_df.columns:
                st.write("📍 **Location:**", f"{selected_row.get('City', '')}, {selected_row.get('Country', '')}")
            if "Investment Interests" in _crm_df.columns:
                st.write("🌎 **Investment Focus:**", selected_row["Investment Interests"])
            if "Firm Type" in _crm_df.columns:
                st.write("🏦 **Firm Type:**", selected_row["Firm Type"])

            # Pass only the selected company’s row to the email tool
            email_tool(pd.DataFrame([selected_row]))

        except Exception as e:
            st.error(f"Failed to read Excel for Email Generator: {e}")

logging.basicConfig(level=logging.DEBUG)



# 🚀 Run Lead Generation
if st.button("🚀 Run Lead Generation"):

    if not OPENAI_API_KEY:
        st.error("OpenAI API key missing.")
        st.stop()
    if not uploaded_xlsx:
        st.error("Please upload an .xlsx CRM first.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["LITELLM_OPENAI_API_KEY"] = OPENAI_API_KEY

    crm_df = pd.read_excel(uploaded_xlsx, engine="openpyxl")
    crm_df = normalize_crm_columns(crm_df)  # normalize before continuing

    needed_cols = [
        "URL",
        "Timestamp Processed",
        "Page Type",
        "Company Name",
        "Contact Names",
        "Emails (GPT)",
        "Emails (EmailFinder)",
        "Emails (Contact-Based)",
        "Phone Numbers",
        "LinkedIn Profiles",
        "Investment Interests",
        "Match Rating (1-10)",
    ]
    for col in needed_cols:
        if col not in crm_df.columns:
            crm_df[col] = ""

    crm_df["URL"] = crm_df["URL"].astype(str).str.strip().str.rstrip("/")
    crm_df["Timestamp Processed"] = crm_df["Timestamp Processed"].fillna("").astype(str).str.strip()

    # (everything below remains unchanged — pipeline, processing, CrewAI loop, etc.)
    st.success("CRM normalized successfully! You can now run the pipeline as usual.")
