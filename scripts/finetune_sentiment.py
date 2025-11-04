#!/usr/bin/env python3
"""Fine-tune the sentiment model using the reviewed ground-truth dataset."""

from __future__ import annotations

import argparse
import os
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
GROUND_TRUTH_PATH = Path("data") / "ground_truth" / "entity_sentiment_labels.csv"
MIN_ROWS = 200
LABEL_MAP = {"negative": 0, "neg": 0, "neutral": 1, "neu": 1, "positive": 2, "pos": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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


def load_dataset(ground_truth_path: Path) -> pd.DataFrame:
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"{ground_truth_path} not found. Run update_entity_sentiment_labels.py first.")

    df = pd.read_csv(ground_truth_path, sep=";", encoding="utf-8-sig")
    if "sentiment_manual" not in df.columns:
        raise ValueError("Expected column 'sentiment_manual' in ground-truth file.")

    df["sentiment_manual"] = df["sentiment_manual"].astype(str).str.strip().str.lower()
    df = df[df["sentiment_manual"].isin(LABEL_MAP)].copy()

    df["entity"] = df.get("entity", "").astype(str).str.strip()
    df["snippet"] = df.get("snippet", "").astype(str).str.strip()
    df["alias_hit"] = df.get("alias_hit", "").astype(str)
    df = df[(df["entity"] != "") & (df["snippet"] != "")]

    def _compose_text(row: pd.Series) -> str:
        entity = row["entity"]
        alias = str(row.get("alias_hit", "")).strip()
        alias_hint = f" ({alias})" if alias and alias.lower() not in entity.lower() else ""
        snippet = row["snippet"]
        return f"[ENTITY] {entity}{alias_hint} ||| {snippet}"

    df["text"] = df.apply(_compose_text, axis=1)
    df["label"] = df["sentiment_manual"]
    df["group_id"] = df.get("item_id", "").astype(str).replace("", pd.NA)
    if df["group_id"].isna().all():
        df["group_id"] = df.index.astype(str)
    return df[["text", "label", "group_id"]]


def tokenize_dataset(tokenizer, texts: List[str], *, max_length: int):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )


def main(args: argparse.Namespace) -> None:
    try:
        df = load_dataset(Path(args.ground_truth_path))
    except FileNotFoundError as exc:
        print(f"ⓘ {exc}")
        return
    except ValueError as exc:
        print(f"⚠️ {exc}")
        return

    if len(df) < args.min_rows:
        print(f"ⓘ Only {len(df)} examples available (minimum required: {args.min_rows}). Skipping training.")
        return

    df["label_id"] = df["label"].map(LABEL_MAP)
    if args.group_split:
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=args.eval_ratio,
            random_state=args.seed,
        )
        split_indices = next(splitter.split(df, groups=df["group_id"]))
        train_idx, eval_idx = split_indices
        train_df = df.iloc[train_idx][["text", "label_id"]].reset_index(drop=True)
        eval_df = df.iloc[eval_idx][["text", "label_id"]].reset_index(drop=True)
    else:
        train_df, eval_df = train_test_split(
            df[["text", "label_id"]],
            test_size=args.eval_ratio,
            random_state=args.seed,
            stratify=df["label_id"],
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_enc = tokenize_dataset(tokenizer, train_df["text"].tolist(), max_length=args.max_length)
    eval_enc = tokenize_dataset(tokenizer, eval_df["text"].tolist(), max_length=args.max_length)

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
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        fp16=False,
        use_cpu=not args.use_mps,
        use_mps_device=args.use_mps,
    )

    class_weights = None
    if args.use_class_weights:
        counts = np.bincount(train_df["label_id"], minlength=len(ID2LABEL))
        inv_freq = np.sum(counts) / (counts + 1e-9)
        class_weights = inv_freq / inv_freq.sum()
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "macro_f1": macro_f1}

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            labels = labels.to(logits.device).long()
            weight = class_weights.to(logits.device) if class_weights is not None else None
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer_cls = WeightedTrainer if args.use_class_weights else Trainer

    trainer = trainer_cls(
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
    parser.add_argument("--ground-truth-path", type=str, default=str(GROUND_TRUTH_PATH))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS)
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Aplica ponderación inversa por frecuencia en la función de pérdida.",
    )
    parser.add_argument(
        "--group-split",
        action="store_true",
        help="Garantiza que el mismo item_id no aparece en train y eval.",
    )
    parser.add_argument(
        "--use-mps",
        action="store_true",
        help="Enable Apple Metal (MPS) acceleration; CPU is used by default.",
    )
    args = parser.parse_args()

    main(args)
