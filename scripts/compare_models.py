import subprocess

models = [
    "xlm-roberta-base",
    "distilbert-base-multilingual-cased",
    "bert-base-multilingual-cased"
]

labels = "O,B-Product,I-Product,B-PRICE,I-PRICE,B-LOC,I-LOC"
data_path = "../telegram_ner_data.conll.txt"

for model_name in models:
    output_dir = f"../models/{model_name.replace('/', '_')}"
    print(f"\n--- Fine-tuning {model_name} ---\n")
    subprocess.run([
        "python", "train_model.py",
        "--model_name", model_name,
        "--data_path", data_path,
        "--output_dir", output_dir,
        "--labels", labels
    ])
