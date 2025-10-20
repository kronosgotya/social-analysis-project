#!/usr/bin/env python3
"""Populate entity_sentiment_labels.csv with new samples and export fine-tune dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable, Tuple, Dict

import pandas as pd

DEFAULT_SAMPLE_SIZE = 25
ENCODING = "utf-8-sig"
SEP = ";"
SCAN_BASE = 5000  # filas máximas a escanear en all_platforms por ejecución

REQUIRED_POST_COLS = [
    "item_id",
    "timestamp",
    "source",
    "text_clean",
    "sentiment_label",
    "entity_mentions",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    return out[columns]


def _existing_keys(df: pd.DataFrame) -> set[Tuple[str, str, str]]:
    if df.empty:
        return set()
    required = ["item_id", "entity", "alias_hit"]
    if any(col not in df.columns for col in required):
        return set()
    return {
        (
            str(row.get("item_id", "")).strip(),
            str(row.get("entity", "")).strip(),
            str(row.get("alias_hit", "")).strip(),
        )
        for _, row in df.iterrows()
    }


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "item_id",
            "timestamp",
            "source",
            "entity",
            "alias_hit",
            "snippet",
            "sentiment_model",
            "stance_model",
        ]
    )


def _prepare_candidates(
    sample_size: int,
    seed: int | None,
    existing_keys: set[Tuple[str, str, str]],
) -> pd.DataFrame:
    if sample_size <= 0:
        return _empty_frame()

    posts_path = _project_root() / "data" / "processed" / "all_platforms.csv"
    if not posts_path.exists():
        print("ⓘ all_platforms.csv not found; no new samples were generated.")
        return _empty_frame()

    rng = random.Random(seed) if seed is not None else random.Random()
    reservoir: list[Dict[str, str]] = []
    seen = 0

    scan_limit = max(SCAN_BASE, sample_size * 200)
    skip_window = scan_limit
    skip_target = rng.randint(0, skip_window)
    skipped = 0
    processed = 0

    with posts_path.open(encoding=ENCODING, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=SEP)
        fieldnames = reader.fieldnames or []
        missing = [col for col in REQUIRED_POST_COLS if col not in fieldnames]
        if missing:
            raise KeyError(
                "Faltan columnas en all_platforms.csv: " + ", ".join(missing)
            )
        stance_available = "stance" in fieldnames

        for row in reader:
            if skipped < skip_target:
                skipped += 1
                continue
            processed += 1
            if processed > scan_limit:
                break

            item_id = (row.get("item_id") or "").strip()
            if not item_id:
                continue

            try:
                mentions = json.loads(row.get("entity_mentions", "[]") or "[]")
            except json.JSONDecodeError:
                continue
            if not isinstance(mentions, list):
                continue

            timestamp = (row.get("timestamp") or "").strip()
            source = (row.get("source") or "").strip()
            snippet = (row.get("text_clean") or "").strip()
            sentiment_model = (row.get("sentiment_label") or "").strip()
            stance_model = (row.get("stance") or "").strip() if stance_available else ""

            for mention in mentions:
                if not isinstance(mention, dict):
                    continue
                entity = (mention.get("entity") or "").strip()
                if not entity:
                    continue
                alias = (mention.get("alias") or entity).strip()
                key = (item_id, entity, alias)
                if key in existing_keys:
                    continue

                seen += 1
                record = {
                    "item_id": item_id,
                    "timestamp": timestamp,
                    "source": source,
                    "entity": entity,
                    "alias_hit": alias,
                    "snippet": snippet,
                    "sentiment_model": sentiment_model,
                    "stance_model": stance_model,
                }

                if len(reservoir) < sample_size:
                    reservoir.append(record)
                else:
                    idx = rng.randint(0, seen - 1)
                    if idx < sample_size:
                        reservoir[idx] = record

    if not reservoir:
        return _empty_frame()

    return pd.DataFrame(reservoir)


def _maybe_export_finetune_dataset(df: pd.DataFrame, gt_dir: Path) -> None:
    if df.empty or "sentiment_manual" not in df.columns:
        return

    reviewed_mask = df["sentiment_manual"].astype(str).str.strip() != ""
    reviewed_count = int(reviewed_mask.sum())
    output_path = gt_dir / "entity_sentiment_finetune.csv"

    if reviewed_count <= 2000:
        print(
            f"ⓘ Only {reviewed_count} reviewed rows (minimum required: 2001). Skipping sentiment fine-tuning dataset export."
        )
        return

    columns = [
        "item_id",
        "timestamp",
        "source",
        "entity",
        "alias_hit",
        "snippet",
        "sentiment_manual",
        "sentiment_model",
    ]
    if "stance_manual" in df.columns:
        columns.append("stance_manual")

    finetune_df = df.loc[reviewed_mask, columns].copy()
    finetune_df = finetune_df.rename(columns={"snippet": "text", "sentiment_manual": "label"})
    finetune_df = finetune_df[finetune_df["text"].astype(str).str.strip() != ""]

    if finetune_df.empty:
        print(
            "ⓘ No valid texts available for the sentiment fine-tuning dataset."
        )
        return

    finetune_df.to_csv(output_path, sep=SEP, index=False, encoding=ENCODING)
    print(
        f"✔ Dataset de fine-tuning de sentimiento generado en {output_path.name} ({len(finetune_df)} filas)."
    )


def main(sample_size: int, seed: int | None, dry_run: bool) -> None:
    project_root = _project_root()
    gt_dir = project_root / "data" / "ground_truth"
    gt_path = gt_dir / "entity_sentiment_labels.csv"

    if gt_path.exists():
        existing = pd.read_csv(gt_path, sep=SEP, encoding=ENCODING)
    else:
        existing = pd.DataFrame(
            columns=[
                "item_id",
                "timestamp",
                "source",
                "entity",
                "alias_hit",
                "snippet",
                "sentiment_model",
                "stance_model",
                "sentiment_manual",
                "stance_manual",
                "notes",
            ]
        )

    keys = _existing_keys(existing)
    candidates = _prepare_candidates(sample_size=sample_size, seed=seed, existing_keys=keys)

    if candidates.empty:
        print("ⓘ No eligible mentions found for sampling.")
        _maybe_export_finetune_dataset(existing, gt_dir)
        return

    required_columns = list(existing.columns)
    for col in ["sentiment_manual", "stance_manual", "notes"]:
        if col not in required_columns:
            required_columns.append(col)
            existing[col] = existing.get(col, "")

    for col in required_columns:
        if col not in candidates.columns:
            candidates[col] = ""

    new_rows = _ensure_columns(candidates, required_columns)

    updated = pd.concat([existing, new_rows], ignore_index=True)
    updated = updated.drop_duplicates(
        subset=["item_id", "entity", "alias_hit"], keep="first"
    )

    if dry_run:
        print(new_rows)
        return

    updated.to_csv(gt_path, index=False, sep=SEP, encoding=ENCODING)
    print(
        f"✔ entity_sentiment_labels.csv updated with {len(new_rows)} new rows (total {len(updated)})."
    )

    _maybe_export_finetune_dataset(updated, gt_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Actualizar entity_sentiment_labels.csv con nuevas muestras aleatorias"
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument(
        "--seed", type=int, default=None, help="Semilla opcional para reproducibilidad"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Imprime las filas seleccionadas sin escribir el archivo"
    )
    args = parser.parse_args()

    main(sample_size=args.sample_size, seed=args.seed, dry_run=args.dry_run)
