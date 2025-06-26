# 🧠 Task 5: Model Interpretability Report

## 🔍 Tools Used
- **SHAP (SHapley Additive Explanations)** for token-wise attribution
- **LIME (Local Interpretable Model-agnostic Explanations)** for local prediction explanation

## 🧪 Sample Text Analyzed
- `ይህ የውሃ መቀመጫ ዋጋ 300 ብር ነው።`

## ✅ SHAP Observations
- Top contributing tokens: `ዋጋ`, `300`, `ብር` were highly attributed to `B-PRICE/I-PRICE`.
- Weak attribution observed on location names like `አዲስ`.

## ✅ LIME Observations
- Model performed well in segmenting price entities.
- Confused between `B-Product` and `B-LOC` in some texts.
- Needs better handling of entity overlap in multi-entity sentences.

## ⚠️ Challenges Identified
- Overlapping entities caused inconsistency.
- Rare words in Amharic had poor embedding representations.

## 💡 Recommendations
- Add more labeled examples of rare Amharic locations.
- Introduce character-level embeddings or lexicon features.

