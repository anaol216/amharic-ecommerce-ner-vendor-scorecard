import pandas as pd
import re
import os

# Define file paths
input_file = 'C:\\Users\\Desta\\Pictures\\10 academy\\week4\\amharic-ecommerce-ner-vendor-scorecard\\telegram_data.csv'
output_file = 'C:\\Users\\Desta\\Pictures\\10 academy\\week4\\amharic-ecommerce-ner-vendor-scorecard\\cleaned_vendor_data.csv'

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Load raw data
df = pd.read_csv(input_file)

# Helper functions for text extraction
def extract_price(text):
    match = re.search(r"ዋጋ[:\- ]*(\d{1,3}(?:,\d{3})*|\d+)\s*ብር", text)
    if match:
        return int(match.group(1).replace(",", ""))
    return None

def extract_product_name(text):
    lines = text.strip().split("\n")
    return lines[0].strip() if lines else ""

def extract_stock(text):
    return "Limited" if "ውስን ፍሬ" in text or "limited" in text.lower() else "Available"

def extract_contacts(text):
    return ", ".join(re.findall(r"0\d{9}", text))

def extract_address(text):
    sections = re.split(r"(?:አድራሻ|ቁ\.1|ቁ\.2)", text)
    if len(sections) > 1:
        return sections[1].split("ፎቅ")[0].strip()
    return ""

# Apply cleaning functions
df["Message"] = df["Message"].fillna("")

df["Product Name"] = df["Message"].apply(extract_product_name)
df["Price (ETB)"] = df["Message"].apply(extract_price)
df["Stock Status"] = df["Message"].apply(extract_stock)
df["Contacts"] = df["Message"].apply(extract_contacts)
df["Address"] = df["Message"].apply(extract_address)
df["Channel"] = df["Channel Username"]
df["Date"] = df["Date"]

# Select and clean structured columns
structured_df = df[["Product Name", "Price (ETB)", "Stock Status", "Contacts", "Address", "Channel", "Date"]]
structured_df = structured_df.dropna(subset=["Product Name", "Price (ETB)"])

# Save cleaned data
structured_df.to_csv(output_file, index=False)
print(f"Cleaned data saved to: {output_file}")
