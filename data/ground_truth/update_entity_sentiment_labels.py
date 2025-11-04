#!/usr/bin/env python3
"""Populate entity_sentiment_labels.csv with new OTAN/Rusia samples and export fine-tune dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_SAMPLE_SIZE = 25
ENCODING = "utf-8-sig"
SEP = ";"
SCAN_BASE = 5000  # maximum rows to scan in all_platforms per execution

REQUIRED_POST_COLS = [
    "item_id",
    "timestamp",
    "source",
    "text_clean",
    "sentiment_label",
    "entity_mentions",
]

CANONICAL_OTAN = {
    "nato",
    "otan",
    "нато",
}

CANONICAL_RUSIA = {
    "rusia",
    "russia",
    "россия",
    "россия ",
    "росія",
    "рф",
    "россии",
    "россии ",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_entity(raw_entity: str) -> Optional[str]:
    value = (raw_entity or "").strip().lower()
    if not value:
        return None
    normalized = value.replace("ё", "е").replace("́", "")
    if normalized in CANONICAL_OTAN:
        return "OTAN"
    if normalized in CANONICAL_RUSIA:
        return "Rusia"
    compact = "".join(ch for ch in normalized if ch.isalnum())
    if compact in {"otan", "nato"}:
        return "OTAN"
    if compact in {"rusia", "russia", "россия", "росія", "рф"}:
        return "Rusia"
    return None


def _read_existing(gt_path: Path) -> Tuple[List[str], List[List[str]]]:
    if not gt_path.exists():
        header = [
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
        return header, []

    with gt_path.open(encoding=ENCODING, newline="") as handle:
        reader = csv.reader(handle, delimiter=SEP)
        rows = list(reader)

    if not rows:
        header = [
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
        return header, []

    header = rows[0]
    data_rows = [row + [""] * (len(header) - len(row)) for row in rows[1:]]
    return header, data_rows


def _ensure_columns(header: List[str], rows: List[List[str]], required: Iterable[str]) -> Tuple[List[str], List[List[str]]]:
    header_mut = header[:]
    rows_mut = [row[:] for row in rows]
    for col in required:
        if col not in header_mut:
            header_mut.append(col)
            for row in rows_mut:
                row.append("")
    return header_mut, rows_mut


def _existing_keys(header: Sequence[str], rows: Sequence[Sequence[str]]) -> set:
    try:
        idx_item = header.index("item_id")
        idx_entity = header.index("entity")
        idx_alias = header.index("alias_hit")
    except ValueError:
        return set()

    keys = set()
    for row in rows:
        if idx_item >= len(row) or idx_entity >= len(row) or idx_alias >= len(row):
            continue
        key = (row[idx_item].strip(), row[idx_entity].strip(), row[idx_alias].strip())
        if all(key):
            keys.add(key)
    return keys


def _compute_target_counts(sample_size: int, available: Dict[str, int]) -> Dict[str, int]:
    total = sum(available.values())
    if total <= 0 or sample_size <= 0:
        return {}

    base_counts: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []

    for source, count in available.items():
        exact = sample_size * (count / total)
        floor_val = int(exact)
        base_counts[source] = floor_val
        remainders.append((exact - floor_val, source))

    assigned = sum(base_counts.values())
    remaining = sample_size - assigned

    if remaining > 0:
        remainders.sort(reverse=True)
        for _, source in remainders:
            if remaining <= 0:
                break
            base_counts[source] = base_counts.get(source, 0) + 1
            remaining -= 1

    return {source: count for source, count in base_counts.items() if count > 0}


def _prepare_candidates(
    sample_size: int,
    seed: Optional[int],
    existing_keys: set,
) -> List[Dict[str, str]]:
    if sample_size <= 0:
        return []

    posts_path = _project_root() / "data" / "processed" / "all_platforms.csv"
    if not posts_path.exists():
        print("ⓘ all_platforms.csv not found; no new samples were generated.")
        return []

    rng = random.Random(seed)
    capacity_limit = max(sample_size * 5, 100)

    candidates_by_source: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    seen_by_source: Dict[str, int] = defaultdict(int)
    available_counts: Dict[str, int] = defaultdict(int)
    collected_keys: set = set()

    with posts_path.open(encoding=ENCODING, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=SEP)
        fieldnames = reader.fieldnames or []
        missing = [col for col in REQUIRED_POST_COLS if col not in fieldnames]
        if missing:
            raise KeyError("Faltan columnas en all_platforms.csv: " + ", ".join(missing))
        stance_available = "stance" in fieldnames

        for row in reader:
            item_id = (row.get("item_id") or "").strip()
            if not item_id:
                continue

            mentions_raw = row.get("entity_mentions", "") or "[]"
            try:
                mentions = json.loads(mentions_raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(mentions, list):
                continue

            timestamp = (row.get("timestamp") or "").strip()
            source = (row.get("source") or "").strip()
            if not source:
                continue
            snippet = (row.get("text_clean") or "").strip()
            sentiment_model = (row.get("sentiment_label") or "").strip()
            stance_model = (row.get("stance") or "").strip() if stance_available else ""

            for mention in mentions:
                if not isinstance(mention, dict):
                    continue
                entity = _canonical_entity(mention.get("entity"))
                if entity is None:
                    continue
                alias = (mention.get("alias") or mention.get("entity") or entity).strip()
                key = (item_id, entity, alias)
                if key in existing_keys or key in collected_keys:
                    continue

                record = {
                    "item_id": item_id,
                    "timestamp": timestamp,
                    "source": source,
                    "entity": entity,
                    "alias_hit": alias,
                    "snippet": snippet,
                    "sentiment_model": sentiment_model,
                    "stance_model": stance_model,
                    "sentiment_manual": "",
                    "stance_manual": "",
                    "notes": "",
                }

                available_counts[source] += 1
                seen_by_source[source] += 1

                bucket = candidates_by_source[source]
                if len(bucket) < capacity_limit:
                    bucket.append(record)
                    collected_keys.add(key)
                else:
                    idx = rng.randint(0, seen_by_source[source] - 1)
                    if idx < capacity_limit:
                        replacement = bucket[idx]
                        replacement_key = (
                            replacement["item_id"],
                            replacement["entity"],
                            replacement["alias_hit"],
                        )
                        collected_keys.discard(replacement_key)
                        bucket[idx] = record
                        collected_keys.add(key)

    target_counts = _compute_target_counts(sample_size, available_counts)
    if not target_counts:
        return []

    selected: List[Dict[str, str]] = []
    used_keys: set = set()
    per_source_selected: Dict[str, int] = defaultdict(int)

    for source, target in target_counts.items():
        bucket = candidates_by_source.get(source, [])
        if not bucket:
            continue
        candidates = bucket[:]
        rng.shuffle(candidates)
        for record in candidates:
            if per_source_selected[source] >= target:
                break
            key = (record["item_id"], record["entity"], record["alias_hit"])
            if key in used_keys:
                continue
            selected.append(record)
            used_keys.add(key)
            per_source_selected[source] += 1

    if len(selected) < sample_size:
        leftovers: List[Dict[str, str]] = []
        for bucket in candidates_by_source.values():
            for record in bucket:
                key = (record["item_id"], record["entity"], record["alias_hit"])
                if key in used_keys:
                    continue
                leftovers.append(record)
        rng.shuffle(leftovers)
        for record in leftovers:
            if len(selected) >= sample_size:
                break
            key = (record["item_id"], record["entity"], record["alias_hit"])
            if key in used_keys:
                continue
            selected.append(record)
            used_keys.add(key)

    return selected


def _deduplicate_rows(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> List[List[str]]:
    try:
        idx_item = header.index("item_id")
        idx_entity = header.index("entity")
        idx_alias = header.index("alias_hit")
    except ValueError:
        return [list(row) for row in rows]

    seen = set()
    deduped: List[List[str]] = []
    for row in rows:
        if idx_item >= len(row) or idx_entity >= len(row) or idx_alias >= len(row):
            deduped.append(list(row))
            continue
        key = (row[idx_item].strip(), row[idx_entity].strip(), row[idx_alias].strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(list(row))
    return deduped


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", encoding=ENCODING, newline="") as handle:
        writer = csv.writer(handle, delimiter=SEP)
        writer.writerow(header)
        for row in rows:
            padded = list(row) + [""] * (len(header) - len(row))
            writer.writerow(padded[: len(header)])


def _maybe_export_finetune_dataset(header: Sequence[str], rows: Sequence[Sequence[str]], gt_dir: Path) -> None:
    try:
        idx_sentiment = header.index("sentiment_manual")
    except ValueError:
        return

    reviewed = [row for row in rows if idx_sentiment < len(row) and row[idx_sentiment].strip()]
    reviewed_count = len(reviewed)
    output_path = gt_dir / "entity_sentiment_finetune.csv"

    if reviewed_count <= 2000:
        print(
            f"ⓘ Only {reviewed_count} reviewed rows (minimum required: 2001). Skipping sentiment fine-tuning dataset export."
        )
        return

    indices = {col: header.index(col) if col in header else None for col in [
        "item_id",
        "timestamp",
        "source",
        "entity",
        "alias_hit",
        "snippet",
        "sentiment_manual",
        "sentiment_model",
        "stance_manual",
    ]}

    finetune_rows: List[List[str]] = []
    for row in reviewed:
        text_idx = indices["snippet"]
        if text_idx is None or text_idx >= len(row):
            continue
        if not (row[text_idx] or "").strip():
            continue
        finetune_row = [
            row[indices["item_id"]] if indices["item_id"] is not None else "",
            row[indices["timestamp"]] if indices["timestamp"] is not None else "",
            row[indices["source"]] if indices["source"] is not None else "",
            row[indices["entity"]] if indices["entity"] is not None else "",
            row[indices["alias_hit"]] if indices["alias_hit"] is not None else "",
            row[text_idx],
            row[indices["sentiment_manual"]] if indices["sentiment_manual"] is not None else "",
            row[indices["sentiment_model"]] if indices["sentiment_model"] is not None else "",
        ]
        if indices["stance_manual"] is not None:
            finetune_row.append(row[indices["stance_manual"]])
        finetune_rows.append(finetune_row)

    if not finetune_rows:
        print("ⓘ No valid texts available for the sentiment fine-tuning dataset.")
        return

    output_header = [
        "item_id",
        "timestamp",
        "source",
        "entity",
        "alias_hit",
        "text",
        "label",
        "sentiment_model",
    ]
    if indices["stance_manual"] is not None:
        output_header.append("stance_manual")

    with output_path.open("w", encoding=ENCODING, newline="") as handle:
        writer = csv.writer(handle, delimiter=SEP)
        writer.writerow(output_header)
        writer.writerows(finetune_rows)

    print(
        f"✔ Dataset de fine-tuning de sentimiento generado en {output_path.name} ({len(finetune_rows)} filas)."
    )


def _dicts_to_rows(header: Sequence[str], records: Sequence[Dict[str, str]]) -> List[List[str]]:
    index_map = {name: idx for idx, name in enumerate(header)}
    rows: List[List[str]] = []
    for record in records:
        row = [""] * len(header)
        for key, value in record.items():
            if key not in index_map:
                continue
            row[index_map[key]] = value
        rows.append(row)
    return rows


def main(sample_size: int, seed: Optional[int], dry_run: bool) -> None:
    project_root = _project_root()
    gt_dir = project_root / "data" / "ground_truth"
    gt_path = gt_dir / "entity_sentiment_labels.csv"

    header, existing_rows = _read_existing(gt_path)
    header, existing_rows = _ensure_columns(
        header,
        existing_rows,
        ["sentiment_manual", "stance_manual", "notes"],
    )

    existing_keys = _existing_keys(header, existing_rows)
    candidates = _prepare_candidates(sample_size, seed, existing_keys)

    if not candidates:
        print("ⓘ No eligible mentions found for sampling.")
        combined_rows = _deduplicate_rows(header, existing_rows)
        _maybe_export_finetune_dataset(header, combined_rows, gt_dir)
        return

    new_rows = _dicts_to_rows(header, candidates)
    combined_rows = _deduplicate_rows(header, existing_rows + new_rows)

    if dry_run:
        writer = csv.writer(sys.stdout, delimiter=SEP)
        writer.writerow(header)
        writer.writerows(new_rows)
        return

    _write_csv(gt_path, header, combined_rows)
    print(
        f"✔ entity_sentiment_labels.csv updated with {len(new_rows)} new rows (total {len(combined_rows)})."
    )

    _maybe_export_finetune_dataset(header, combined_rows, gt_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Actualizar entity_sentiment_labels.csv con nuevas muestras aleatorias (OTAN/Rusia)"
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument(
        "--seed", type=int, default=None, help="Semilla opcional para reproducibilidad"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Imprime las filas seleccionadas sin escribir el archivo"
    )
    args = parser.parse_args()

    try:
        main(sample_size=args.sample_size, seed=args.seed, dry_run=args.dry_run)
    except KeyboardInterrupt:  # pragma: no cover - allow graceful interruption
        print("\nInterrumpido por el usuario.")
