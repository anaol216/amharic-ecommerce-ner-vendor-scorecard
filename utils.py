from datasets import DatasetDict
import pandas as pd

label_list = ["O", "B-Product", "I-Product", "B-PRICE", "I-PRICE", "B-LOC", "I-LOC"]
label_map = {label: i for i, label in enumerate(label_list)}

def conll_to_dataset(filepath, tokenizer):
    sentences, labels = [], []
    with open(filepath, encoding="utf8") as f:
        tokens, tags = [], []
        for line in f:
            if line.strip() == "":
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                token, tag = line.strip().split()
                tokens.append(token)
                tags.append(tag)

    encodings = tokenizer(sentences, is_split_into_words=True, padding=True, truncation=True)
    encoded_labels = []

    for i, label in enumerate(labels):
        word_ids = encodings.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else label_map[label[word_id]] for word_id in word_ids]
        encoded_labels.append(label_ids)

    encodings["labels"] = encoded_labels

    dataset = DatasetDict({
        "train": Dataset.from_dict(encodings),
        "validation": Dataset.from_dict(encodings)  # use same for validation here
    })

    return dataset
