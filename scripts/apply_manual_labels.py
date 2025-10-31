#!/usr/bin/env python3
"""
apply_manual_labels.py

Propagates the manual topic labels curated in
`data/ground_truth/topics_manual_labels.csv` into every CSV under
`data/processed/` that exposes the trio:
    - topic_id
    - manual_label_topic
    - manual_label_subtopic

It also emits `results/topics/topics_missing_manual_labels.csv` with the list of
topic IDs present in the processed outputs but still lacking a manual label.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd

ENCODING = "utf-8-sig"
SEP = ";"
EMPTY_TOKENS = {"", "nan", "none", "null", "n/a", "na"}

DATA_PROCESSED_DIR = Path("data") / "processed"
MANUAL_LABELS_PATH = Path("data") / "ground_truth" / "topics_manual_labels.csv"
MISSING_OUTPUT_PATH = Path("results") / "topics" / "topics_missing_manual_labels.csv"


def _normalize_topic_id(value) -> str:
    if value is None:
        return ""
    token = str(value).strip()
    if token.lower() in EMPTY_TOKENS:
        return ""
    try:
        return str(int(float(token)))
    except Exception:
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


def load_manual_label_map() -> Tuple[Dict[str, str], Dict[str, str]]:
    if not MANUAL_LABELS_PATH.exists():
        raise FileNotFoundError(
            "topics_manual_labels.csv not found. Run scripts/topics_map_labels.py first."
        )
    df = pd.read_csv(MANUAL_LABELS_PATH, sep=SEP, encoding=ENCODING, low_memory=False)
    required = {"topic_id", "manual_label_topic", "manual_label_subtopic"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{MANUAL_LABELS_PATH} is missing columns: {sorted(missing)}")

    df["topic_id_key"] = df["topic_id"].apply(_normalize_topic_id)
    df["manual_label_topic"] = df["manual_label_topic"].apply(_clean_label)
    df["manual_label_subtopic"] = df["manual_label_subtopic"].apply(_clean_label)

    topic_map: Dict[str, str] = {}
    subtopic_map: Dict[str, str] = {}
    for _, row in df.iterrows():
        key = row["topic_id_key"]
        if not key:
            continue
        topic_label = row["manual_label_topic"]
        subtopic_label = row["manual_label_subtopic"]
        if topic_label:
            topic_map[key] = topic_label
        if subtopic_label:
            subtopic_map[key] = subtopic_label
    return topic_map, subtopic_map


def apply_labels_to_csv(path: Path, topic_map: Dict[str, str], subtopic_map: Dict[str, str]) -> Tuple[int, int]:
    df = pd.read_csv(path, sep=SEP, encoding=ENCODING, low_memory=False)
    if "topic_id" not in df.columns:
        return 0, 0

    if "manual_label_topic" not in df.columns:
        df["manual_label_topic"] = ""
    if "manual_label_subtopic" not in df.columns:
        df["manual_label_subtopic"] = ""

    df["topic_id_key"] = df["topic_id"].apply(_normalize_topic_id)
    df["manual_label_topic"] = df["manual_label_topic"].apply(_clean_label)
    df["manual_label_subtopic"] = df["manual_label_subtopic"].apply(_clean_label)

    before_topic = (df["manual_label_topic"] != "").sum()
    before_subtopic = (df["manual_label_subtopic"] != "").sum()

    df["manual_label_topic"] = df.apply(
        lambda row: topic_map.get(row["topic_id_key"], row["manual_label_topic"]),
        axis=1,
    )
    df["manual_label_subtopic"] = df.apply(
        lambda row: subtopic_map.get(row["topic_id_key"], row["manual_label_subtopic"]),
        axis=1,
    )

    after_topic = (df["manual_label_topic"] != "").sum()
    after_subtopic = (df["manual_label_subtopic"] != "").sum()

    df.drop(columns=["topic_id_key"], inplace=True)
    df.to_csv(path, sep=SEP, index=False, encoding=ENCODING)

    return after_topic - before_topic, after_subtopic - before_subtopic


def list_missing_topics(processed_files: List[Path], topic_map: Dict[str, str]) -> None:
    topic_ids_in_outputs = set()
    for path in processed_files:
        try:
            df = pd.read_csv(path, sep=SEP, encoding=ENCODING, usecols=["topic_id"], low_memory=False)
        except Exception:
            continue
        df["topic_id_key"] = df["topic_id"].apply(_normalize_topic_id)
        topic_ids_in_outputs.update(df["topic_id_key"].dropna().tolist())

    missing = sorted(
        tid for tid in topic_ids_in_outputs
        if tid and tid not in topic_map
    )

    if not missing:
        if MISSING_OUTPUT_PATH.exists():
            MISSING_OUTPUT_PATH.unlink()
        print("ⓘ No missing manual labels detected across processed outputs.")
        return

    MISSING_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"topic_id": missing}).to_csv(
        MISSING_OUTPUT_PATH, sep=SEP, index=False, encoding=ENCODING
    )
    print(f"ⓘ Missing labels exported to {MISSING_OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propagate manual labels from data/ground_truth/topics_manual_labels.csv into data/processed/*.csv"
    )
    parser.parse_args()

    topic_map, subtopic_map = load_manual_label_map()
    if not topic_map and not subtopic_map:
        print("ⓘ No manual labels present in topics_manual_labels.csv.")
        return

    processed_files = sorted(DATA_PROCESSED_DIR.glob("*.csv"))
    if not processed_files:
        print("ⓘ No CSV files found in data/processed/.")
        return

    updated_any = False
    for path in processed_files:
        topic_delta, subtopic_delta = apply_labels_to_csv(path, topic_map, subtopic_map)
        if topic_delta or subtopic_delta:
            updated_any = True
            print(f"✔ {path.name}: updated {topic_delta} topic labels / {subtopic_delta} subtopic labels")

    if not updated_any:
        print("ⓘ Processed files already contained the manual labels; no changes applied.")

    list_missing_topics(processed_files, topic_map)


if __name__ == "__main__":
    main()
