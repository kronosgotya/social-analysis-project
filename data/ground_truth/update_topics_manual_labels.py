#!/usr/bin/env python3
"""Synchronise topics_manual_labels.csv with detected topics and manage fine-tune datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ENCODING = "utf-8-sig"
SEP = ";"
MIN_TOPICS_FINETUNE = 201


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_topics() -> pd.DataFrame:
    processed_dir = _project_root() / "data" / "processed"
    assignments_path = processed_dir / "topics_assignments.csv"
    if not assignments_path.exists():
        raise FileNotFoundError("topics_assignments.csv not found in data/processed")

    assignments = pd.read_csv(assignments_path, sep=SEP, encoding=ENCODING)
    if assignments.empty:
        return assignments

    required = ["topic_id", "topic_label", "topic_terms"]
    missing = [col for col in required if col not in assignments.columns]
    if missing:
        raise KeyError(f"Missing columns in topics_assignments.csv: {', '.join(missing)}")

    topics = (
        assignments[required]
        .drop_duplicates(subset="topic_id", keep="first")
        .rename(columns={"topic_label": "auto_label"})
    )
    topics["topic_id"] = topics["topic_id"].astype(str)
    topics["topic_terms"] = topics["topic_terms"].astype(str)
    topics["auto_label"] = topics["auto_label"].astype(str)
    return topics.reset_index(drop=True)


def _maybe_export_topic_finetune(df: pd.DataFrame, gt_dir: Path) -> None:
    if df.empty:
        return
    if "manual_label_topic" not in df.columns or "manual_label_subtopic" not in df.columns:
        return

    reviewed = df[
        (df["manual_label_topic"].astype(str).str.strip() != "")
        & (df["manual_label_subtopic"].astype(str).str.strip() != "")
    ].copy()

    if len(reviewed) < MIN_TOPICS_FINETUNE:
        print(
            f"ⓘ Only {len(reviewed)} topics with topic/subtopic labels (minimum {MIN_TOPICS_FINETUNE}). Skipping fine-tune export."
        )
        return

    export_cols = [
        "topic_id",
        "topic_terms",
        "auto_label",
        "manual_label_topic",
        "manual_label_subtopic",
    ]

    finetune_df = reviewed[export_cols]
    finetune_path = gt_dir / "topics_manual_finetune.csv"
    finetune_df.to_csv(finetune_path, sep=SEP, index=False, encoding=ENCODING)
    print(
        f"✔ topics_manual_finetune.csv generated with {len(finetune_df)} topics."
    )


def main(force_sort: bool) -> None:
    gt_dir = _project_root() / "data" / "ground_truth"
    gt_path = gt_dir / "topics_manual_labels.csv"

    topics = _load_topics()
    if topics.empty:
        print("ⓘ topics_assignments.csv has no rows; nothing to sync.")
        return

    if gt_path.exists():
        manual = pd.read_csv(gt_path, sep=SEP, encoding=ENCODING)
        manual["topic_id"] = manual["topic_id"].astype(str)
    else:
        manual = pd.DataFrame()

    base_columns = ["topic_id", "topic_terms", "auto_label"]
    default_extra = [
        "manual_label_topic",
        "manual_label_subtopic",
        "include_dashboard",
        "notes",
    ]

    extra_columns = [col for col in manual.columns if col not in base_columns]
    if not extra_columns:
        extra_columns = default_extra
    else:
        for col in default_extra:
            if col not in extra_columns:
                extra_columns.append(col)

    if manual.empty:
        manual = pd.DataFrame(columns=base_columns + extra_columns)

    manual = manual.copy()
    for col in extra_columns:
        if col not in manual.columns:
            manual[col] = ""

    manual = manual.set_index("topic_id", drop=False)

    for _, row in topics.iterrows():
        topic_id = row["topic_id"]
        topic_terms = row["topic_terms"]
        auto_label = row["auto_label"]

        if topic_id in manual.index:
            manual.at[topic_id, "topic_terms"] = topic_terms
            manual.at[topic_id, "auto_label"] = auto_label
        else:
            manual.loc[topic_id] = {
                "topic_id": topic_id,
                "topic_terms": topic_terms,
                "auto_label": auto_label,
                **{col: "" for col in extra_columns},
            }

    if force_sort:
        manual = manual.sort_index()

    output = manual[base_columns + extra_columns].reset_index(drop=True)
    output.to_csv(gt_path, sep=SEP, index=False, encoding=ENCODING)
    print(f"✔ topics_manual_labels.csv updated ({len(output)} topics).")

    _maybe_export_topic_finetune(output, gt_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update topics_manual_labels.csv with all detected topics"
    )
    parser.add_argument(
        "--sort", action="store_true", help="Sort the output by topic_id"
    )
    args = parser.parse_args()

    main(force_sort=args.sort)
