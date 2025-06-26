import pandas as pd

# Load the cleaned CSV file
input_file = "C:\\Users\\Desta\\Pictures\\10 academy\\week4\\amharic-ecommerce-ner-vendor-scorecard\\cleaned_vendor_data.csv"  # Replace with your actual file path
df = pd.read_csv(input_file)

# Define simple keyword-to-category mapping
category_keywords = {
    "dish": "Kitchen Equipment",
    "rack": "Home Storage",
    "hanger": "Home Storage",
    "glass dispenser": "Kitchen Equipment",
    "juicer": "Kitchen Appliance",
    "outlet": "Electrical Appliance",
    "cabinet": "Furniture",
    "toilet": "Bathroom",
    "water dispenser": "Kitchen Appliance",
    "pest repeller": "Home Electronics",
    "cloth": "Home Storage",
    "plastic": "Home Storage",
}

def categorize_product(name):
    name_lower = name.lower()
    for keyword, category in category_keywords.items():
        if keyword in name_lower:
            return category
    return "Other"  # fallback category

# Apply the category function to each row
df["Category"] = df["Product Name"].apply(categorize_product)

# Save the new categorized file
output_file = "categorized_products.csv"
df.to_csv(output_file, index=False)

print(f"Categorized data saved to {output_file}")
