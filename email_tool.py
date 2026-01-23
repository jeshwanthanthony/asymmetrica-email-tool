import os
import ssl
import re
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
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

# -----------------------------------------------------
# Asymmetrica philosophy (summarized for the model)
# -----------------------------------------------------
ASYMMETRICA_SUMMARY = """
Asymmetrica Investments AG is a Swiss impact investment manager that structures
real-asset investments with asymmetric return profiles. We first create a
downside “floor” via stable, inflation-resilient cash flows and then seek
superior upside with attractive risk-reward.

Our principles include: investing in projects with high return on invested
capital (ROIC); robustness to changes in interest rates; ownership of scarce,
productive real assets such as avocado orchards in water-secure regions;
strong, visible cash flows; buying below intrinsic value in inefficient and
underfollowed markets (e.g., agriculture in emerging markets); focusing on
growing but idiosyncratic, uncorrelated niches; and targeting high margins
supported by structural barriers to entry.
""".strip()


# =====================================================
# Load API Keys
# =====================================================
def load_keys():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    sendgrid_key = os.getenv("SENDGRID_API_KEY")
    return openai_key, sendgrid_key


# =====================================================
# Normalize CRM Columns (firm-level)
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
# Small helper to find a column by candidate names
# =====================================================
def find_column(df: pd.DataFrame, candidates):
    if df is None:
        return None
    cols = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in cols:
            return cols[key]
    return None


# =====================================================
# Fetch basic website text for context (no heavy scraping)
# =====================================================
def fetch_website_context(website: str, max_chars: int = 5000) -> str:
    """
    Try scraping multiple institutional subpages.
    Return combined clean text.
    If none work, fall back to main page.
    """
    if not website:
        return ""

    base = website.strip()
    if not base.startswith(("http://", "https://")):
        base = "https://" + base

    # Pages to test (relative)
    candidate_paths = [
        "",  # homepage first
        "about",
        "about-us",
        "aboutus",
        "who-we-are",
        "team",
        "strategy",
        "investment",
        "investment/strategy",
        "our-approach",
        "philosophy",
        "mission",
    ]

    collected_texts = []
    tested = 0

    for path in candidate_paths:
        if tested >= 3:    # only collect up to 3 valid pages
            break

        # Build URL
        if path:
            url = base.rstrip("/") + "/" + path
        else:
            url = base

        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code >= 400:
                continue
        except Exception:
            continue

        tested += 1

        # Parse and clean text
        try:
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove useless tags
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ").split())
            if text:
                collected_texts.append(text)
        except Exception:
            continue

    # If nothing collected → fallback to original homepage scrape
    if not collected_texts:
        try:
            resp = requests.get(base, timeout=5)
            if resp.status_code < 400:
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                fallback_text = " ".join(soup.get_text(separator=" ").split())
                return fallback_text[:max_chars]
        except Exception:
            return ""

    # Merge all valid texts and trim
    combined = "\n\n".join(collected_texts)
    return combined[:max_chars]


# =====================================================
# OpenAI Response (kept for other callers, not used for file-attach flow)
# =====================================================
def get_openai_response(messages):
    openai_key, _ = load_keys()
    if not openai_key:
        st.error("Missing OPENAI_API_KEY in .env file.")
        return "Error: No OpenAI key found."

    client = OpenAI(api_key=openai_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI request failed: {e}")
        return "Error contacting OpenAI."


# =====================================================
# Helpers for email body formatting / SendGrid template
# =====================================================
def prepare_email_body_for_template(body_text: str) -> str:
    """Clean and convert plain text into HTML-friendly lines for the template."""
    # Strip any bold markers just in case
    clean = body_text.replace("**", "")

    # Normalize newlines
    clean = clean.replace("\r\n", "\n")

    # Turn line breaks into <br> so HTML respects them
    html_body = clean.replace("\n\n", "<br><br>").replace("\n", "<br>")

    return html_body


# =====================================================
# Send Email via SendGrid (with dynamic template + optional CC)
# =====================================================
# =====================================================
# Send Email via SendGrid (plain text, optional CC)
# =====================================================
# =====================================================
# Send Email via SendGrid (plain text, optional CC)
# =====================================================
def send_email(from_email, to_email, subject, body_text, cc_list=None):
    _, sendgrid_key = load_keys()

    if not from_email:
        st.error("Please enter a sender email before sending.")
        return

    if not SENDGRID_AVAILABLE or not sendgrid_key:
        st.warning("SendGrid not configured — running in test mode.")
        print("=== EMAIL (TEST MODE) ===")
        print("FROM:", from_email)
        print("TO:", to_email)
        print("CC:", cc_list)
        print("SUBJECT:", subject)
        print("BODY:\n", body_text)
        return

    try:
        sg = SendGridAPIClient(sendgrid_key)

        message = Mail(
            from_email=from_email,
            to_emails=to_email,
        )

        # ✅ REQUIRED: Template ID
        message.template_id = "d-6d042f7969624cdd87217c826865819a"

        # ✅ REQUIRED: Dynamic data
        message.dynamic_template_data = {
            "subject": subject,
            "email_body": body_text,   # ← plain text, untouched
        }

        # Optional CC
        if cc_list:
            message.cc = cc_list

        response = sg.send(message)

        if response.status_code in (200, 202):
            st.success(f"Email sent to {to_email}")
        else:
            st.error(f"SendGrid failed: {response.status_code}")

    except Exception as e:
        st.error(f"SendGrid error: {e}")



# =====================================================
# Mandate helpers (outside the function as requested)
# =====================================================
def safe_get(df_row, col_name):
    """Get column text safely, clean, fallback to empty."""
    if col_name in df_row and pd.notna(df_row[col_name]):
        text = str(df_row[col_name]).strip()
        return text if text and text.lower() != "nan" else ""
    return ""


# global list (outside)
mandate_text_parts = []


def compute_mandate_text() -> str:
    """Join the global mandate parts into final text (outside)."""
    return "\n\n".join(mandate_text_parts)


# =====================================================
# Email Tool  (uses firm sheet + contacts sheet)
# =====================================================
def email_tool(crm_df: pd.DataFrame, contacts_df: pd.DataFrame = None):
    st.subheader("Email Generator & Sender")

    required = ["Company Name", "Email", "Investment Interests"]
    if not set(required).issubset(crm_df.columns):
        st.error(f"Your CRM must contain columns: {', '.join(required)}")
        return

    # ---- 1. Select firm ----
    company_select = st.selectbox("Select a Company", crm_df["Company Name"])
    current_company = crm_df[crm_df["Company Name"] == company_select].iloc[0]

    company_name = current_company["Company Name"]
    firm_email = str(current_company.get("Email", "")).strip()

    # -------------------------------------------------------
    # BUILD MANDATE TEXT (append to global list; combine outside)
    # -------------------------------------------------------
    global mandate_text_parts
    mandate_text_parts.clear()  # reset each selection

    background = safe_get(current_company, "BACKGROUND")
    strategy_pref = safe_get(current_company, "PE: STRATEGY PREFERENCES")
    industry_pref = safe_get(current_company, "PE: INDUSTRIES")
    geo_pref = safe_get(current_company, "PE: GEOGRAPHIC PREFERENCES")
    firm_type = safe_get(current_company, "FIRM TYPE")
    city_val = safe_get(current_company, "CITY") or str(current_company.get("City", "")).strip()
    country_val = safe_get(current_company, "COUNTRY") or str(current_company.get("Country", "")).strip()
    region_val = safe_get(current_company, "REGION") or str(current_company.get("Region", "")).strip()
    aum_val = safe_get(current_company, "AUM (USD MN)")

    if background:
        mandate_text_parts.append(f"BACKGROUND:\n{background}")
    if strategy_pref:
        mandate_text_parts.append(f"STRATEGY PREFERENCES:\n{strategy_pref}")
    if industry_pref:
        mandate_text_parts.append(f"INDUSTRY FOCUS:\n{industry_pref}")
    if geo_pref:
        mandate_text_parts.append(f"GEOGRAPHIC PREFERENCES:\n{geo_pref}")
    if firm_type:
        mandate_text_parts.append(f"FIRM TYPE:\n{firm_type}")

    location_txt = ", ".join([v for v in [city_val, country_val] if v])
    if location_txt:
        mandate_text_parts.append(f"LOCATION:\n{location_txt}")
    if aum_val:
        mandate_text_parts.append(f"AUM (USD MN):\n{aum_val}")

    # combine outside (via helper), then use inside
    mandate_text = compute_mandate_text()

    # -------------------------------------------------------
    # SHOW MANDATE TEXT IN THE UI
    # -------------------------------------------------------
    st.markdown("### 🧭 Investor Mandate Summary")
    if mandate_text:
        st.text_area(
            "Investor Mandate (auto-assembled from Preqin columns)",
            mandate_text,
            height=250,
        )
    else:
        st.info("No mandate information found for this investor.")

    # -------------------------------------------------------
    # Website handling (show extracted text)
    # -------------------------------------------------------
    website = (
        str(current_company.get("WEBSITE") or current_company.get("Website") or "")
        .strip()
    )

    website_text = ""
    if website:
        st.write("🌐 **Fetching website content…**")
        website_text = fetch_website_context(website)
        if website_text:
            st.markdown("### 🔎 Extracted Website Text")
            st.text_area(
                "Website Text (auto-extracted)",
                website_text,
                height=200,
            )
        else:
            st.info("No website text could be extracted or the website did not load.")
    else:
        st.info("No website provided in this row.")

    # build a location string for the prompt
    location_text = ", ".join(
        [str(x) for x in [city_val, country_val, region_val] if str(x).strip()]
    )

    # pull these for the prompt (support both normalized and raw Preqin names)
    investment_interests = (
        str(current_company.get("Investment Interests", "")).strip()
        or background
    )
    strategy_prefs = (
        str(current_company.get("Strategy Preferences", "")).strip()
        or strategy_pref
    )

    # ---- 2. Look up person-level contacts in Contacts_Export ----
    recipient_email = firm_email
    greeting_recipient = f"{company_name} Team"
    contact_entries = []

    if contacts_df is not None and not contacts_df.empty:
        investor_col = find_column(contacts_df, ["INVESTOR"])
        if investor_col:
            normalized_company = str(company_name).strip().lower()
            investor_series = (
                contacts_df[investor_col]
                .astype(str)
                .fillna("")
                .str.strip()
                .str.lower()
            )
            company_contacts = contacts_df[investor_series == normalized_company]

            if not company_contacts.empty:
                name_col = find_column(contacts_df, ["NAME", "CONTACT NAME"])
                title_col = find_column(contacts_df, ["TITLE"])
                role_col = find_column(contacts_df, ["ROLE"])
                email_col = find_column(contacts_df, ["EMAIL", "EMAIL ADDRESS"])

                def clean(row, col):
                    if not col or col not in row.index:
                        return ""
                    val = row[col]
                    return "" if pd.isna(val) else str(val).strip()

                for _, row in company_contacts.iterrows():
                    name = clean(row, name_col)
                    title = clean(row, title_col)
                    role = clean(row, role_col)
                    contact_email = clean(row, email_col)

                    full_name = " ".join(x for x in [title, name] if x).strip()
                    display_name = full_name or name or "Contact"
                    label_email = contact_email or "No email available"
                    label_role = role or "No role listed"

                    contact_entries.append(
                        {
                            "label": f"{display_name} — {label_email} — {label_role}",
                            "name": name,
                            "title": title,
                            "role": role,
                            "email": contact_email,
                            "is_investment_role": (
                                ("investment" in role.lower())
                                or ("portfolio" in role.lower())
                            )
                            if role
                            else False,
                        }
                    )
            else:
                st.caption(
                    "No individual contacts found for this firm — using firm-level email."
                )
        else:
            st.warning(
                "Contacts sheet does not contain an 'INVESTOR' column — using firm-level email."
            )

    # ---- 3. Second dropdown: pick contact ----
    if contact_entries:
        contact_labels = [c["label"] for c in contact_entries]
        preferred_idx = next(
            (i for i, c in enumerate(contact_entries) if c["is_investment_role"]),
            0,
        )

        selected_label = st.selectbox(
            "Select a Contact at this Firm", contact_labels, index=preferred_idx
        )
        selected_contact = contact_entries[contact_labels.index(selected_label)]

        if selected_contact.get("email"):
            recipient_email = selected_contact["email"]

        full_name = " ".join(
            x for x in [selected_contact.get("title"), selected_contact.get("name")] if x
        ).strip()
        if full_name:
            greeting_recipient = full_name

    greeting_line = f"Dear {greeting_recipient},"

    # -------------------------------------------------------
    # PRODUCT / EXECUTIVE SUMMARY UPLOAD (NEW FEATURE)
    # -------------------------------------------------------
    st.markdown("### 📄 Upload Investment Product Material")

    int uploaded = 0

    uploaded_product_file = st.file_uploader(
        "Upload Executive Summary / Deck / Product Document (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        key="product_upload",
    )

    if uploaded_product_file is not None:
        st.success(f"Uploaded: {uploaded_product_file.name}")
       
    else:
        st.info("You can optionally upload a product document. The email will still work without it.")

    # -------------------------------------------------------
    # Subject line logic (depends on whether product/AVO file is uploaded)
    # -------------------------------------------------------
    if uploaded_product_file is not None:
        uploaded = 1
        email_subject = "Opportunistic Farmland Fund – Executive Summary & Investment Highlights"
    else:
        email_subject = f"Exploring Collaboration with {company_name}"
    
    has_product_file = uploaded_product_file is not None

    # ---- 4. Generate email with correct greeting + firm philosophy + website + PRODUCT FILE ----
    if st.button("Generate Email"):
        with st.spinner("Generating email draft..."):
            # Build website context string (re-use website_text already fetched above)
            website_context = (
                f"Here is public text extracted from the firm's website ({website}):\n"
                f"{website_text}\n\n"
                "Use this only to infer their focus and how our strategy could complement it. "
                "Do not invent specific facts that are not clearly implied."
                if website_text
                else "No website content was available; keep the description of the firm generic and conservative."
            )

            openai_key, _ = load_keys()
            if not openai_key:
                st.error("Missing OPENAI_API_KEY in .env file.")
                return

            client = OpenAI(api_key=openai_key)

            # ------------------ upload product file to OpenAI (if provided) ------------------
            file_content_part = [
                {"type": "input_text", "text": ""}  # will be replaced below
            ]
            # we will rebuild it properly right after constructing prompt_text

            file_attachment_part = []
            if uploaded_product_file is not None:
                try:
                    uploaded = client.files.create(
                        file=(
                            uploaded_product_file.name,
                            uploaded_product_file.getvalue(),
                            "application/octet-stream",
                        ),
                        purpose="user_data",
                    )
                    file_attachment_part = [
                        {
                            "type": "input_file",
                            "file_id": uploaded.id,
                        }
                    ]
                except Exception as e:
                    st.error(f"Failed to upload product file to OpenAI: {e}")
                    file_attachment_part = []

            # Main user prompt text (no fake product text, we rely on file attachment)
            prompt_text = f"""
You are writing on behalf of Asymmetrica Investments AG. Your task is to generate a
human, concise, naturally-written outreach email to a potential investment partner.
The tone must feel professional, thoughtful, and written by a real person — NEVER AI-sounding.

Always follow these global rules:
• Never mention any individual sender’s name unless the template explicitly requires it.
• Never add extra sections beyond what is described.
• Never change the greeting line. It must begin exactly as:
  {greeting_line}
• When referencing the firm, rely only on the provided information — do not invent facts.
• When a product document is attached, you MUST follow the structure rules below EXACTLY.
• When NO product document is attached, follow the non-attachment instructions at the end.

────────────────────────────────────────────────────────
ASSET INFORMATION AVAILABLE TO YOU
────────────────────────────────────────────────────────

Asymmetrica overview (use to describe who we are, briefly and naturally):
{ASYMMETRICA_SUMMARY}

Target firm profile (use for personalization):
• Company Name: {company_name}
• Location: {location_text or 'N/A'}
• Strategy Preferences: {strategy_prefs or 'N/A'}
• Background / Investment Interests: {investment_interests or 'N/A'}

Investor mandate details (MUST be reviewed and used for personalization):
{mandate_text}

Public website context (MUST be reviewed; use only what is clearly implied):
{website_context}

PRODUCT DOCUMENT ATTACHED: {has_product_file}


────────────────────────────────────────────────────────
STRUCTURE & CONTENT RULES (CRITICAL)
────────────────────────────────────────────────────────

• The email MUST use bullet points for all investment highlights.
• Bullet points must be plain text bullets (• or -), NOT markdown formatting.
• The section titled **“Fund Structure & Commitments” MUST remain STATIC**:
  - Do NOT change numbers, ranges, structure, or wording in that section.
  - Reproduce it exactly as provided below.
• Other sections MAY be lightly tailored based on the investor’s mandate,
  strategy preferences, or profile — but must remain conservative and factual.
• Do NOT exaggerate, speculate, or invent performance claims.

If PRODUCT DOCUMENT ATTACHED = true:
→ Follow ONLY the section titled:
  "EMAIL FLOW (WHEN PRODUCT DOCUMENT IS ATTACHED)"

If PRODUCT DOCUMENT ATTACHED = false:
→ Follow ONLY the section titled:
  "WHEN NO PRODUCT DOCUMENT IS ATTACHED"


Do NOT combine or mix instructions from both sections.
Follow exactly ONE path.

────────────────────────────────────────────────────────
EMAIL FLOW (WHEN PRODUCT DOCUMENT IS ATTACHED)
────────────────────────────────────────────────────────

The email MUST begin with:
1. A warm, human introduction about Asymmetrica Investments AG.
2. One or two personalized lines explaining WHY you are reaching out to this investor
   (based on mandate, strategy preferences, location, or website context).
3. A soft, natural transition into the attached executive summary.

Next say "Attached is the executive summary for the AVO Capital Fund, a structured opportunity designed for long-term capital partners seeking exposure to high-margin agriculture and inflation-resilient assets. Key highlights include:"

Now base the following information depending on the investor's profile
Performance & Profitability
• Target Net Return (IRR): 20–25% annually in USD.
• Underlying Asset: Export-focused superfood farms with EBITDA margins > 50%.
• Strategic Positioning: Deployment across top-producing regions creates arbitrage opportunities supported by generational turnover.
• Distribution Advantage: Collaboration with Avoworks, the 4th largest U.S. avocado importer, securing premium market access.

Portfolio Diversification & Inflation Hedge
• Farmland provides structural protection against inflation and developed-market volatility.
• Cash flows remain uncorrelated with real estate, VC, and traditional private equity.
• Asset class remains undercapitalized despite rising global demand.

BUT NOW THIS: 
Fund Structure & Commitments  (THIS SECTION IS STATIC — DO NOT MODIFY)
• Fund Size: USD 50–70M target, scalable to USD 150M capacity.
• First Closing: USD 20M.
• Investment Allocation:
  - 70% orchard acquisitions (brownfield/greenfield)
  - 30% CAPEX efficiency upgrades and processing improvements
• Minimum Ticket Size: USD 1M.
• Typical investment size: USD $4M per asset (combined initial + follow-up).
• Distribution Profile: 8–14% target yield, semi-annual.
• First compartment of an umbrella structure fund.

Closing lines (adapt tone slightly, but keep meaning):
If there is interest, we can provide access to our data room, including the Investment Memorandum, PPM, and full due diligence materials.
Happy to schedule a discussion once you’ve reviewed the attached material.
Looking forward to your feedback.

────────────────────────────────────────────────────────
WHEN NO PRODUCT DOCUMENT IS ATTACHED
────────────────────────────────────────────────────────

If NO document is attached:
• Write a brief, warm, professional outreach email.
• Begin with the exact greeting line: {greeting_line}
• Use Asymmetrica’s summary, the investor mandate, and website context to explain strategic fit.
• Do NOT include deal terms, performance figures, or product specifics.
• Do NOT include a signature.
• End with this exact final sentence and nothing after it:
  "Would you be available for a brief call to discuss further?"

────────────────────────────────────────────────────────
IMPORTANT OUTPUT REQUIREMENTS
────────────────────────────────────────────────────────
• Output ONLY the email body.
• No headers, no explanations, no meta-comments.
• Maintain a human, confident, professional tone.
""".strip()


            # Build the content for Responses API
            content_parts = [
                {"type": "input_text", "text": prompt_text}
            ] + file_attachment_part

            try:
                response = client.responses.create(
                    model="gpt-4o",
                    input=[
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    ],
                )

                # Responses API: extract the generated text safely
                email_text = ""
                try:
                    email_text = response.output[0].content[0].text
                except Exception:
                    # fallback: stringifying whole response if shape changes
                    email_text = str(response)
            except Exception as e:
                st.error(f"OpenAI request failed: {e}")
                return

            # Remove any markdown bold markers like **this**
            email_text = email_text.replace("**", "")

            st.session_state.generated_email = email_text
            st.session_state["draft_email_text"] = email_text

    # ---- 5. Edit + send ----
    if "generated_email" in st.session_state:
        st.markdown("Generated Email Draft")
        st.session_state.setdefault(
            "draft_email_text", st.session_state.get("generated_email", "")
        )

        draft_col, save_col = st.columns([3, 1])
        with draft_col:
            st.text_area(
                "Email Content",
                key="draft_email_text",
                height=280,
            )
        with save_col:
            if st.button("Save Text"):
                st.session_state["generated_email"] = st.session_state["draft_email_text"]
                st.success("Draft saved.")

        # sender email
        st.session_state.setdefault("from_email", "")
        col1, col2 = st.columns([3, 1])
        with col1:
            entered_email = st.text_input(
                "Enter your email, MUST SAVE after entered",
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
                    st.error("Please enter a valid sender email before sending. Verify via Sendgrind")

        # CC EMAILS (optional)
        st.markdown("### CC (Optional)")

        if "cc_list" not in st.session_state:
            st.session_state.cc_list = []

        cc_input = st.text_input(
            "Add CC Email",
            placeholder="cc@example.com",
            key="cc_input",
        )

        if st.button("Add CC"):
            email = cc_input.strip()
            if email:
                st.session_state.cc_list.append(email)
                st.success(f"Added CC: {email}")
            else:
                st.error("Please enter a valid email.")

        if st.session_state.cc_list:
            st.write("**Current CC Recipients:**")
            for e in st.session_state.cc_list:
                st.write("- " + e)
        else:
            st.caption("No CC emails added yet.")

        sender_email = st.session_state.get("from_email", "").strip()
        final_recipient = recipient_email or firm_email

        if st.button("Send Email"):
            if not final_recipient:
                st.error("No valid recipient email found.")
            elif not sender_email:
                st.error("Please set a valid sender email before sending.")
            else:
                st.info(f"Sending email from {sender_email} to {final_recipient} ...")
                send_email(
                    sender_email,
                    final_recipient,
                    email_subject,
                    st.session_state.get(
                        "draft_email_text",
                        st.session_state.get("generated_email", "")
                    ),
                    cc_list=st.session_state.cc_list if st.session_state.cc_list else None,
                )
