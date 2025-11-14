# test_duolingo_crm.py
import pandas as pd
from datetime import datetime
from search_engine import duckduckgo_search

# Step 1: Define the test query
query = "Duolingo AI language learning startup investments"

# Step 2: Fetch URLs (using the new DDGS method)
urls = duckduckgo_search(query, max_results=10)

# Step 3: Build a CRM-like DataFrame
crm_df = pd.DataFrame({
    "URL": urls,
    "Timestamp Processed": "",
    "Page Type": "",
    "Company Name": "",
    "Contact Names": "",
    "Emails (GPT)": "",
    "Emails (EmailFinder)": "",
    "Emails (Contact-Based)": "",
    "Phone Numbers": "",
    "LinkedIn Profiles": "",
    "Investment Interests": "",
    "Match Rating (1-10)": "",
})

# Step 4: Save to Excel file
filename = f"test_crm_duolingo_{datetime.now():%Y%m%d%H%M}.xlsx"
crm_df.to_excel(filename, index=False, engine="openpyxl")

print(f"✅ Created test CRM file: {filename}")
print(crm_df.head(5))
