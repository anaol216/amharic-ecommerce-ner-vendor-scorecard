import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_metric
import numpy as np
import os
from data_utils import prepare_hf_dataset, tokenize_and_align_labels

def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [label_list[l] for l in label if l != -100]
        for label in labels
    ]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    metric = load_metric("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main(args):
    label_list = args.labels.split(",")
    dataset = prepare_hf_dataset(args.data_path, label_list)

    # Split dataset into train/validation
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(label_list)
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, label_list),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"Evaluation results: {metrics}")

    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CoNLL data file")
    parser.add_argument("--output_dir", type=str, default="./model_output", help="Where to save the model")
    parser.add_argument("--labels", type=str, required=True, help="Comma separated list of labels")
    args = parser.parse_args()
    main(args)
