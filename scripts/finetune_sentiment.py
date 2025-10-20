#!/usr/bin/env python3
"""Fine-tune the sentiment model using the reviewed ground-truth dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
FINETUNE_DIR = Path("models") / "sentiment_finetuned"
GROUND_TRUTH_PATH = Path("data") / "ground_truth" / "entity_sentiment_finetune.csv"
MIN_ROWS = 2001
LABEL_MAP = {"negative": 0, "neg": 0, "neutral": 1, "neu": 1, "positive": 2, "pos": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


class SentimentDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_dataset() -> pd.DataFrame:
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            "entity_sentiment_finetune.csv not found. Run update_entity_sentiment_labels.py first."
        )
    df = pd.read_csv(GROUND_TRUTH_PATH, sep=";", encoding="utf-8-sig")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(LABEL_MAP)].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    return df


def tokenize_dataset(tokenizer, texts: List[str]):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )


def main(args: argparse.Namespace) -> None:
    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        print(f"ⓘ {exc}")
        return

    if len(df) < MIN_ROWS:
        print(
            f"ⓘ Only {len(df)} examples available (minimum required: {MIN_ROWS}). Skipping training."
        )
        return

    df["label_id"] = df["label"].map(LABEL_MAP)
    train_df, eval_df = train_test_split(
        df[["text", "label_id"]],
        test_size=args.eval_ratio,
        random_state=args.seed,
        stratify=df["label_id"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_enc = tokenize_dataset(tokenizer, train_df["text"].tolist())
    eval_enc = tokenize_dataset(tokenizer, eval_df["text"].tolist())

    train_dataset = SentimentDataset(train_enc, train_df["label_id"].tolist())
    eval_dataset = SentimentDataset(eval_enc, eval_df["label_id"].tolist())

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
    )

    training_args = TrainingArguments(
        output_dir=str(FINETUNE_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINETUNE_DIR))
    tokenizer.save_pretrained(str(FINETUNE_DIR))
    print(f"✔ Modelo de sentimiento guardado en {FINETUNE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning del modelo de sentimiento con ground truth.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
