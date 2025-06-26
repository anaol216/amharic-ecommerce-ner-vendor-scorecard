import pandas as pd
import numpy as np

# === Load Data ===
df = pd.read_csv("data/telegram_data_with_views.csv")

# === Preprocess ===
df["Date"] = pd.to_datetime(df["Date"])
df["Week"] = df["Date"].dt.isocalendar().week

# === Group by Channel ===
vendor_groups = df.groupby("Channel")

# === Output Container ===
scorecard = []

# === Scoring Weights ===
w_views = 0.5
w_posts = 0.3
w_price = 0.2

for vendor, group in vendor_groups:
    total_posts = len(group)
    weeks_active = group["Week"].nunique()
    posts_per_week = total_posts / weeks_active if weeks_active else 0

    avg_views = group["Views"].mean()
    avg_price = group["Price (ETB)"].mean()

    # Top product info
    top_post = group.loc[group["Views"].idxmax()]
    top_product = top_post["Product Name"]
    top_price = top_post["Price (ETB)"]
    
    # Lending Score (customizable)
    lending_score = (avg_views * w_views) + (posts_per_week * w_posts * 10) + (avg_price * w_price / 100)

    scorecard.append({
        "Vendor": vendor,
        "Avg Views/Post": round(avg_views, 2),
        "Posts/Week": round(posts_per_week, 2),
        "Avg Price (ETB)": round(avg_price, 2),
        "Top Product": top_product,
        "Top Price": top_price,
        "Lending Score": round(lending_score, 2)
    })

# === Save to CSV ===
score_df = pd.DataFrame(scorecard)
score_df.to_csv("vendor_scorecard.csv", index=False)
print("Vendor scorecard saved to 'vendor_scorecard.csv'")
