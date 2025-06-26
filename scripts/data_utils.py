import os
from datasets import load_dataset, Dataset
import pandas as pd
from transformers import AutoTokenizer

def read_conll_data(conll_file_path):
    # Reads CoNLL format text file into list of sentences with tokens and labels
    sentences = []
    tokens = []
    labels = []
    
    with open(conll_file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": labels})
                    tokens = []
                    labels = []
            else:
                splits = line.split()
                tokens.append(splits[0])
                labels.append(splits[1])
        if tokens:
            sentences.append({"tokens": tokens, "ner_tags": labels})

    return sentences

def prepare_hf_dataset(conll_file_path, label_list):
    data = read_conll_data(conll_file_path)
    
    # Map labels to ids
    label_to_id = {label: i for i, label in enumerate(label_list)}
    
    # Convert labels to ids
    for item in data:
        item["ner_tags"] = [label_to_id[label] for label in item["ner_tags"]]
    
    return Dataset.from_list(data)

def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
