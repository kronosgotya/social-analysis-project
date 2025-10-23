#!/usr/bin/env python3
"""Train lightweight classifiers for topic and subtopic labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

GROUND_TRUTH_PATH = Path("data") / "ground_truth" / "topics_manual_labels.csv"
MODEL_PATH = Path("models") / "topic_classifier" / "topic_classifier.joblib"
MIN_TOPICS = 201


def _terms_to_text(value: str) -> str:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return " ".join(str(term) for term in parsed if term)
    except json.JSONDecodeError:
        pass
    return value.replace("[", " ").replace("]", " ").replace(",", " ")


def load_dataset() -> pd.DataFrame:
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            "topics_manual_labels.csv not found. Run update_topics_manual_labels.py first."
        )
    df = pd.read_csv(GROUND_TRUTH_PATH, sep=";", encoding="utf-8-sig")
    for col in ["manual_label_topic", "manual_label_subtopic"]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' is missing from topics_manual_labels.csv")
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["manual_label_topic"] != "") & (df["manual_label_subtopic"] != "")].copy()
    df["topic_terms"] = df["topic_terms"].astype(str).str.strip().apply(_terms_to_text)
    df = df[df["topic_terms"] != ""]
    return df


def build_models(
    texts: pd.Series,
    topic_labels: pd.Series,
    subtopic_labels: pd.Series,
    *,
    max_features: int,
    max_iter: int,
    inverse_reg: float,
    seed: int,
) -> Dict[str, object]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features)
    X = vectorizer.fit_transform(texts)

    topic_clf = LogisticRegression(
        max_iter=max_iter,
        C=inverse_reg,
        multi_class="auto",
        random_state=seed,
    )
    topic_clf.fit(X, topic_labels)

    subtopic_clf = LogisticRegression(
        max_iter=max_iter,
        C=inverse_reg,
        multi_class="auto",
        random_state=seed,
    )
    subtopic_clf.fit(X, subtopic_labels)

    return {
        "vectorizer": vectorizer,
        "topic_clf": topic_clf,
        "subtopic_clf": subtopic_clf,
    }


def evaluate_models(models: Dict[str, object], texts: pd.Series, topic_labels: pd.Series, subtopic_labels: pd.Series) -> None:
    vectorizer = models["vectorizer"]
    topic_clf = models["topic_clf"]
    subtopic_clf = models["subtopic_clf"]

    X = vectorizer.transform(texts)
    topic_preds = topic_clf.predict(X)
    subtopic_preds = subtopic_clf.predict(X)

    print("\n=== Topic classification report ===")
    print(classification_report(topic_labels, topic_preds))
    print("\n=== Subtopic classification report ===")
    print(classification_report(subtopic_labels, subtopic_preds))


def main(args: argparse.Namespace) -> None:
    try:
        df = load_dataset()
    except (FileNotFoundError, KeyError) as exc:
        print(f"ⓘ {exc}")
        return

    if len(df) < MIN_TOPICS:
        print(
            f"ⓘ Only {len(df)} topics with manual labels are available (minimum {MIN_TOPICS}). Skipping training."
        )
        return

    X_train, X_eval, y_topic_train, y_topic_eval, y_subtopic_train, y_subtopic_eval = train_test_split(
        df["topic_terms"],
        df["manual_label_topic"],
        df["manual_label_subtopic"],
        test_size=args.eval_ratio,
        random_state=args.seed,
        stratify=df["manual_label_topic"],
    )

    models = build_models(
        X_train,
        y_topic_train,
        y_subtopic_train,
        max_features=args.max_features,
        max_iter=args.max_iter,
        inverse_reg=args.inverse_reg,
        seed=args.seed,
    )

    evaluate_models(models, X_eval, y_topic_eval, y_subtopic_eval)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODEL_PATH)
    print(f"✔ Topic classifier saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train topic and subtopic classifiers from the manual ground truth."
    )
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--inverse-reg", type=float, default=1.0)
    args = parser.parse_args()

    main(args)
