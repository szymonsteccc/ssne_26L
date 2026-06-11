import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

SEED = 42
MAX_LEN = 128

MODELS = [
    "allegro/herbert-base-cased",
    "dkleczek/bert-base-polish-cased-v1",
    "sdadas/polish-roberta-base-v2"
]


# =====================================================
# DATA
# =====================================================

df = pd.read_csv("hate_train.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["label"]
)


# =====================================================
# DATASET
# =====================================================

class HateDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        )

        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item


def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1
    }


# =====================================================
# MODEL COMPARISON
# =====================================================

results = []

for model_name in MODELS:

    print("=" * 50)
    print(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = HateDataset(
        train_df["sentence"],
        train_df["label"].values,
        tokenizer
    )

    val_dataset = HateDataset(
        val_df["sentence"],
        val_df["label"].values,
        tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    args = TrainingArguments(
        output_dir=f"./tmp_{model_name.split('/')[-1]}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=20,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    metrics = trainer.evaluate()

    results.append({
        "model": model_name,
        "f1": metrics["eval_f1"],
        "accuracy": metrics["eval_accuracy"]
    })

results_df = pd.DataFrame(results)

print("\nRESULTS")
print(results_df.sort_values("f1", ascending=False))

best_model_name = results_df.sort_values(
    "f1",
    ascending=False
).iloc[0]["model"]

print(f"\nBEST MODEL: {best_model_name}")


# =====================================================
# RETRAIN ON FULL TRAIN SET
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(best_model_name)

full_dataset = HateDataset(
    df["sentence"],
    df["label"].values,
    tokenizer
)

best_model = AutoModelForSequenceClassification.from_pretrained(
    best_model_name,
    num_labels=2
)

args = TrainingArguments(
    output_dir="./final_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=best_model,
    args=args,
    train_dataset=full_dataset
)

trainer.train()


# =====================================================
# TEST PREDICTIONS
# =====================================================

with open("hate_test_data.txt", "r", encoding="utf8") as f:
    test_texts = [line.strip() for line in f]

enc = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LEN,
    return_tensors="pt"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

best_model.to(device)

enc = {
    k: v.to(device)
    for k, v in enc.items()
}

with torch.no_grad():

    outputs = best_model(**enc)

preds = outputs.logits.argmax(dim=1).cpu().numpy()

pd.DataFrame(preds).to_csv(
    "pred.csv",
    header=False,
    index=False
)

print("pred.csv zapisany")