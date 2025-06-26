import shap
import numpy as np
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from lime.lime_text import LimeTextExplainer

# Load model and tokenizer
model_name = "your_fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Example text
text = "ይህ የውሃ መቀመጫ ዋጋ 300 ብር ነው።"

# === SHAP === #
print("Computing SHAP values...")
explainer = shap.Explainer(ner_pipeline)
shap_values = explainer([text])

# Plot SHAP
shap.plots.text(shap_values[0])

# === LIME === #
print("Running LIME explanation...")
class_names = ["O", "B-Product", "I-Product", "B-PRICE", "I-PRICE", "B-LOC", "I-LOC"]
lime_explainer = LimeTextExplainer(class_names=class_names)

def predict_proba(texts):
    preds = []
    for t in texts:
        output = ner_pipeline(t)
        label_probs = np.zeros(len(class_names))
        for entity in output:
            idx = class_names.index(entity['entity_group'])
            label_probs[idx] += 1
        label_probs /= label_probs.sum()
        preds.append(label_probs)
    return np.array(preds)

exp = lime_explainer.explain_instance(text, predict_proba, num_features=6)
exp.show_in_notebook()
