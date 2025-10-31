#!/usr/bin/env python3
"""Train lightweight classifiers for topic and subtopic labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

FACTS_POSTS_PATH = Path("data") / "processed" / "facts_posts.csv"
MODEL_PATH = Path("models") / "topic_classifier" / "topic_classifier.joblib"
MIN_SAMPLES = 200
EMPTY_TOKENS = {"", "nan", "none", "null", "n/a", "na"}


def _canonical_topic_id(value) -> str:
    if value is None:
        return ""
    try:
        return str(int(float(value)))
    except Exception:
        token = str(value).strip()
        if token.endswith(".0"):
            try:
                return str(int(float(token)))
            except Exception:
                pass
        return token


def _clean_label(value) -> str:
    if value is None:
        return ""
    token = str(value).strip()
    if token.lower() in EMPTY_TOKENS:
        return ""
    return token


def _filter_min_samples(df: pd.DataFrame, column: str, min_samples: int) -> pd.DataFrame:
    counts = df[column].value_counts()
    valid = counts[counts >= min_samples].index
    return df[df[column].isin(valid)].copy()


def _majority_label(series: pd.Series) -> str:
    values = [_clean_label(val) for val in series if _clean_label(val)]
    if not values:
        return ""
    counter = Counter(values)
    most_common = counter.most_common()
    top_count = most_common[0][1]
    candidates = sorted([label for label, count in most_common if count == top_count])
    return candidates[0] if candidates else ""


def load_training_data(min_samples: int) -> pd.DataFrame:
    if not FACTS_POSTS_PATH.exists():
        raise FileNotFoundError(
            "facts_posts.csv not found. Ejecuta process_all.py para generarlo."
        )
    df = pd.read_csv(FACTS_POSTS_PATH, sep=";", encoding="utf-8-sig")
    required_cols = {"text_clean", "topic_id", "manual_label_topic", "manual_label_subtopic"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Faltan columnas en facts_posts.csv: {sorted(missing)}")

    df["text_clean"] = df["text_clean"].astype(str).str.strip()
    df["manual_label_topic"] = df["manual_label_topic"].apply(_clean_label)
    df["manual_label_subtopic"] = df["manual_label_subtopic"].apply(_clean_label)
    df["topic_id_key"] = df["topic_id"].apply(_canonical_topic_id)

    df = df[
        (df["text_clean"] != "")
        & (df["manual_label_topic"] != "")
        & (df["manual_label_subtopic"] != "")
        & (df["topic_id_key"] != "")
        & (df["topic_id_key"] != "-1")
    ].copy()

    df = _filter_min_samples(df, "topic_id_key", min_samples)
    df = _filter_min_samples(df, "manual_label_topic", max(2, min_samples // 2))
    df = _filter_min_samples(df, "manual_label_subtopic", max(2, min_samples // 2))

    return df


def _train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int,
    inverse_reg: float,
    seed: int,
) -> LogisticRegression:
    topic_clf = LogisticRegression(
        max_iter=max_iter,
        C=inverse_reg,
        multi_class="auto",
        random_state=seed,
        class_weight="balanced",
    )
    topic_clf.fit(X, y)
    return topic_clf


def _encode_topics(
    texts: pd.Series,
    model_name: str,
    *,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(
        texts.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    return np.asarray(embeddings)


def build_models(
    embeddings: np.ndarray,
    manual_topic_labels: pd.Series,
    manual_subtopic_labels: pd.Series,
    *,
    max_iter: int,
    inverse_reg: float,
    seed: int,
) -> Dict[str, object]:
    manual_topic_encoder = LabelEncoder()
    y_manual_topic = manual_topic_encoder.fit_transform(manual_topic_labels)

    manual_subtopic_encoder = LabelEncoder()
    y_manual_subtopic = manual_subtopic_encoder.fit_transform(manual_subtopic_labels)

    manual_topic_clf = _train_classifier(
        embeddings,
        y_manual_topic,
        max_iter=max_iter,
        inverse_reg=inverse_reg,
        seed=seed,
    )
    manual_subtopic_clf = _train_classifier(
        embeddings,
        y_manual_subtopic,
        max_iter=max_iter,
        inverse_reg=inverse_reg,
        seed=seed,
    )

    return {
        "manual_topic_clf": manual_topic_clf,
        "manual_subtopic_clf": manual_subtopic_clf,
        "manual_topic_label_encoder": manual_topic_encoder,
        "manual_subtopic_label_encoder": manual_subtopic_encoder,
    }


def run_cross_validation(
    embeddings: np.ndarray,
    manual_topic_labels: pd.Series,
    manual_subtopic_labels: pd.Series,
    *,
    folds: int,
    seed: int,
    max_iter: int,
    inverse_reg: float,
) -> None:
    manual_topic_encoder = LabelEncoder()
    y_manual_topic = manual_topic_encoder.fit_transform(manual_topic_labels)

    manual_subtopic_encoder = LabelEncoder()
    y_manual_subtopic = manual_subtopic_encoder.fit_transform(manual_subtopic_labels)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    topic_acc: List[float] = []
    topic_f1: List[float] = []
    sub_acc: List[float] = []
    sub_f1: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(embeddings, y_manual_topic), start=1):
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_topic_train, y_topic_val = y_manual_topic[train_idx], y_manual_topic[val_idx]
        y_sub_train, y_sub_val = y_manual_subtopic[train_idx], y_manual_subtopic[val_idx]

        topic_clf = _train_classifier(X_train, y_topic_train, max_iter=max_iter, inverse_reg=inverse_reg, seed=seed)
        sub_clf = _train_classifier(X_train, y_sub_train, max_iter=max_iter, inverse_reg=inverse_reg, seed=seed)

        topic_pred = topic_clf.predict(X_val)
        sub_pred = sub_clf.predict(X_val)

        topic_acc.append(accuracy_score(y_topic_val, topic_pred))
        topic_f1.append(f1_score(y_topic_val, topic_pred, average="macro", zero_division=0))
        sub_acc.append(accuracy_score(y_sub_val, sub_pred))
        sub_f1.append(f1_score(y_sub_val, sub_pred, average="macro", zero_division=0))

        print(
            f"Fold {fold_idx}/{folds} — topic acc: {topic_acc[-1]:.3f}, topic F1_macro: {topic_f1[-1]:.3f}, "
            f"subtopic acc: {sub_acc[-1]:.3f}, subtopic F1_macro: {sub_f1[-1]:.3f}"
        )

    def _summary(name: str, values: List[float]) -> str:
        mean = float(np.mean(values))
        std = float(np.std(values))
        return f"{name}: {mean:.3f} ± {std:.3f}"

    print("\nCross-validation summary (macro metrics):")
    print("  " + _summary("Topic accuracy", topic_acc))
    print("  " + _summary("Topic F1", topic_f1))
    print("  " + _summary("Subtopic accuracy", sub_acc))
    print("  " + _summary("Subtopic F1", sub_f1))


def main(args: argparse.Namespace) -> None:
    try:
        df = load_training_data(args.min_samples_per_class)
    except (FileNotFoundError, KeyError) as exc:
        print(f"ⓘ {exc}")
        return

    if len(df) < MIN_SAMPLES:
        print(
            f"ⓘ Only {len(df)} samples available (minimum {MIN_SAMPLES}). Skipping training."
        )
        return

    texts = df["text_clean"].astype(str)
    embeddings = _encode_topics(
        texts,
        args.embedding_model,
        batch_size=args.batch_size,
        normalize=not args.no_normalize_embeddings,
    )

    if args.cv_folds > 1:
        print(f"Running {args.cv_folds}-fold cross-validation...")
        run_cross_validation(
            embeddings,
            df["manual_label_topic"],
            df["manual_label_subtopic"],
            folds=args.cv_folds,
            seed=args.seed,
            max_iter=args.max_iter,
            inverse_reg=args.inverse_reg,
        )
    else:
        print("ⓘ Cross-validation disabled (use --cv-folds > 1 to enable).")

    model_bundle = build_models(
        embeddings,
        df["manual_label_topic"],
        df["manual_label_subtopic"],
        seed=args.seed,
        max_iter=args.max_iter,
        inverse_reg=args.inverse_reg,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "kind": "sentence-transformer-multitask",
            "embedding_model_name": args.embedding_model,
            "normalize_embeddings": not args.no_normalize_embeddings,
            "inference_batch_size": args.inference_batch_size,
            "manual_topic_clf": model_bundle["manual_topic_clf"],
            "manual_subtopic_clf": model_bundle["manual_subtopic_clf"],
            "manual_topic_label_encoder": model_bundle["manual_topic_label_encoder"],
            "manual_subtopic_label_encoder": model_bundle["manual_subtopic_label_encoder"],
            "vectorizer": None,
        },
        MODEL_PATH,
    )
    print(f"✔ Topic classifier saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train topic and subtopic classifiers from the manual ground truth."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--inverse-reg", type=float, default=1.0)
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model to embed topic terms.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-normalize-embeddings", action="store_true")
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--min-samples-per-class", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds (set to 0/1 to disable).")
    args = parser.parse_args()

    main(args)
