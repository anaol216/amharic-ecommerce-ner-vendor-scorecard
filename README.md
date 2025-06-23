# 🛍️ Amharic E-Commerce NER & Vendor Scorecard

This project builds a Named Entity Recognition (NER) system to extract product, price, and location data from Amharic Telegram-based e-commerce posts. It also includes a vendor scoring system to support micro-lending decisions for small businesses.

---

## 📌 Key Features

- 🔎 Extracts entities like `Product`, `Price`, `Location`, `Delivery Fee`, and `Contact Info`
- 🤖 Fine-tunes multilingual models (XLM-Roberta, AfroXLMR, etc.) for Amharic NER
- 📊 Scores vendors based on posting activity, views, and product prices
- 🧠 Uses SHAP and LIME for model explainability

---

## 📁 Folder Structure

amharic-ecommerce-ner/
├── data/ # Raw, cleaned, and labeled Telegram data
├── scripts/ # Python scripts for scraping, training, scoring
├── notebooks/ # Jupyter/Colab notebooks
├── models/ # Fine-tuned NER models
├── tests/ # Pytest unit tests
├── reports/ # Project reports (PDF)
├── .vscode/ # VS Code settings
├── .github/workflows/ # GitHub Actions CI


---

## 🚀 How to Run

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
