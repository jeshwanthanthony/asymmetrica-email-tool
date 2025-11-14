import os
import sys
import ssl
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Optional SendGrid import
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    SENDGRID_AVAILABLE = True
except Exception:
    SENDGRID_AVAILABLE = False

# Fix SSL issues on some macOS setups
ssl._create_default_https_context = ssl._create_unverified_context


# =====================================================
# Load API Keys
# =====================================================
def load_keys():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    sendgrid_key = os.getenv("SENDGRID_API_KEY")
    return openai_key, sendgrid_key


# =====================================================
# Normalize CRM Columns
# =====================================================
def normalize_crm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map the large XLSX schema to the minimal columns used by the app."""
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

    # Extra context for personalization if present
    for nice in ["CITY", "STATE/COUNTY", "COUNTRY", "REGION", "AUM (USD MN)", "WEBSITE"]:
        key = nice.strip().lower()
        if key in cols and cols[key] not in out.columns:
            out[cols[key]] = df[cols[key]]

    return out


# =====================================================
# OpenAI Response
# =====================================================
def get_openai_response(messages):
    openai_key, _ = load_keys()
    if not openai_key:
        st.error("Missing OPENAI_API_KEY in .env file.")
        return "Error: No OpenAI key found."

    client = OpenAI(api_key=openai_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI request failed: {e}")
        return "Error contacting OpenAI."


# =====================================================
# Send Email via SendGrid
# =====================================================
def send_email(from_email, to_email, subject, body_text):
    _, sendgrid_key = load_keys()

    if not from_email:
        st.error("Please enter a sender email before sending.")
        return

    if not SENDGRID_AVAILABLE or not sendgrid_key:
        st.warning("SendGrid not configured — running in test mode.")
        print("=== EMAIL (TEST MODE) ===")
        print("FROM:", from_email)
        print("TO:", to_email)
        print("SUBJECT:", subject)
        print("BODY:\n", body_text)
        st.info("Test mode: email printed to console instead of sending.")
        return

    try:
        sg = SendGridAPIClient(sendgrid_key)
        message = Mail(
            from_email=from_email,
            to_emails=to_email,
            subject=subject,
            plain_text_content=body_text,
        )
        response = sg.send(message)
        if response.status_code in (200, 202):
            st.success(f"Email sent from {from_email} to {to_email}.")
        else:
            st.error(f"SendGrid failed (status {response.status_code}).")
    except Exception as e:
        st.error(f"SendGrid error: {e}")


# =====================================================
# Email Tool
# =====================================================
def email_tool(crm_df: pd.DataFrame):
    st.subheader("Email Generator & Sender")

    # Required columns
    required = ["Company Name", "Email", "Investment Interests"]
    if not set(required).issubset(crm_df.columns):
        st.error(f"Your CRM must contain columns: {', '.join(required)}")
        return

    company_select = st.selectbox("Select a Company", crm_df["Company Name"])
    current_company = crm_df[crm_df["Company Name"] == company_select].iloc[0]

    company_name = current_company["Company Name"]
    email_select = current_company["Email"]
    investment_interests = current_company["Investment Interests"]

    # Optional context
    city = current_company.get("CITY", "")
    country = current_company.get("COUNTRY", "")
    region = current_company.get("REGION", "")
    aum = current_company.get("AUM (USD MN)", "")
    website = current_company.get("WEBSITE", "")
    location_text = ", ".join([str(x) for x in [city, country, region] if str(x).strip()])

    if st.button("Generate Email"):
        with st.spinner("Generating email draft..."):
            prompt = f"""
You are a skilled business development writer representing Asymmetrica Investments AG,
a Swiss investment firm based in Zug, Switzerland.

Do not include a Subject line; start directly with the greeting.

Write a short, warm, and professional cold outreach email introducing our company to {company_name}.
This firm operates in {location_text or 'its respective region'} and is interested in {investment_interests or 'sustainable, strategic investments'}.

The email must:
- Be under 150 words.
- Begin with: Dear {company_name} Team,
- Sound friendly yet credible, focusing on potential collaboration or investment alignment.
- Include a light reference to the weather or season based on {location_text or 'their region'} (e.g., “as the colder months approach” or “as we enter spring”). Do not invent specific forecasts or events.
- Mention that Asymmetrica Investments AG provides real-time updates and insights, including weather-aware outreach, upcoming industry events, and product news.
- End with this exact signature:

Warm regards,
Asymmetrica Investments AG
Poststrasse 24, 6302 Zug, Switzerland
Phone: +41 78 731 02 08
Email: info@asymmetrica-investments.com

Optional context:
- Website: {website or 'N/A'}
- Country/Region: {location_text or 'N/A'}
- AUM (USD mn): {aum or 'N/A'}
""".strip()

            messages = [
                {
                    "role": "system",
                    "content": "You are a concise, credible B2B investment email copywriter. Keep emails under 150 words.",
                },
                {"role": "user", "content": prompt},
            ]

            email_text = get_openai_response(messages)
            st.session_state.generated_email = email_text

    if "generated_email" in st.session_state:
        st.markdown("Generated Email Draft")
        st.text_area("Email Content", st.session_state.generated_email, height=280)

        # Sender email entry (no modification of widget key after creation)
        st.session_state.setdefault("from_email", "")
        col1, col2 = st.columns([3, 1])
        with col1:
            entered_email = st.text_input(
                "Enter your email",
                value=st.session_state["from_email"],
                placeholder="you@company.com",
                key="from_email_input",
            )
        with col2:
            if st.button("Save"):
                cleaned = entered_email.strip()
                st.session_state["from_email"] = cleaned
                if cleaned:
                    st.info(f"Sender email set to: {cleaned}")
                else:
                    st.error("Please enter a valid sender email before sending.")

        sender_email = st.session_state.get("from_email", "").strip()

        if st.button("Send Email"):
            if not email_select:
                st.error("No valid recipient email found.")
            elif not sender_email:
                st.error("Please set a sender email using the Save button before sending.")
            else:
                st.info(f"Sending email from {sender_email} to {email_select} ...")
                send_email(
                    sender_email,
                    email_select,
                    f"Exploring Collaboration with {company_name}",
                    st.session_state.generated_email,
                )


# =====================================================
# Streamlit App Entry
# =====================================================
def main():
    st.title("AI Email Generator — Asymmetrica Outreach")

    uploaded = st.file_uploader("Upload your CRM (.xlsx)", type=["xlsx"])
    if uploaded is not None:
        raw_df = pd.read_excel(uploaded)
        crm_df = normalize_crm_columns(raw_df)
        st.info("CRM uploaded and normalized.")
        st.dataframe(crm_df)
        email_tool(crm_df)
    else:
        st.info("Upload a CRM Excel file to begin (e.g., Asymmetrica_CRM.xlsx).")


if __name__ == "__main__":
    main()
