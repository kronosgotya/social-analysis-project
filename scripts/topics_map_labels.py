#!/usr/bin/env python3
"""
topics_map_labels.py

Reaplica etiquetas manuales a un topic_info.csv (mezclado por idioma) usando
los backups de results/topics/topic_backup/*.csv. No depende de pandas para
evitar errores de parseo con CSV heterogéneos.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_OUTPUT = Path("data") / "ground_truth" / "topics_manual_labels.csv"
TERMS_THRESHOLD = 0.2  # similitud mínima para aceptar un mapeo automático
CSV_KWARGS = {"quotechar": '"', "escapechar": "\\"}

csv.field_size_limit(10_000_000)
_EMPTY_TOKENS = {"", "nan", "none", "null", "n/a", "na"}


def _auto_sep(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")
    first_line = sample.splitlines()[0] if sample else ""
    if ";" in first_line:
        return ";"
    if sample.count(";") > sample.count(","):
        return ";"
    return ","


def _parse_terms(raw: object) -> List[str]:
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []

    def _normalize_token(token: str) -> str:
        return token.strip().strip("'\"").strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            tokens: List[str] = []
            for item in parsed:
                if isinstance(item, str) and "dtype:" in item:
                    extra = _extract_terms_from_series_text(item, 0, None)
                    if extra:
                        tokens.extend(extra)
                elif isinstance(item, (list, tuple)):
                    tokens.extend(str(t).strip() for t in item if str(t).strip())
                else:
                    token = str(item).strip()
                    if token:
                        tokens.append(token)
            if tokens:
                return tokens
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(t).strip() for t in parsed if str(t).strip()]
    except Exception:
        pass

    # Maneja cadenas generadas desde Series de pandas (e.g. "0 [term1, term2]\n1 [term3, ...]")
    bracketed: List[str] = []
    current = ""
    depth = 0
    for char in text:
        if char == "[":
            if depth == 0:
                current = ""
            else:
                current += char
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
            if depth == 0:
                bracketed.append(current)
                current = ""
            else:
                current += char
        else:
            if depth > 0:
                current += char
    if bracketed:
        tokens: List[str] = []
        for chunk in bracketed:
            parts = [p for p in chunk.split(",") if p.strip()]
            tokens.extend(_normalize_token(p) for p in parts if _normalize_token(p))
        if tokens:
            return tokens

    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p]


def _set_from_terms(raw: Iterable[str]) -> set:
    return {str(t).strip().lower() for t in raw if str(t).strip()}


def _clean_label(value: object) -> str:
    if value is None:
        return ""
    token = str(value).strip()
    if token.lower() in _EMPTY_TOKENS:
        return ""
    return token


@dataclass
class ManualTopic:
    topic_id: str
    terms: List[str]
    auto_label: str
    manual_topic: str
    manual_subtopic: str
    include_dashboard: str
    notes: str

    def similarity(self, terms: Sequence[str]) -> float:
        left = _set_from_terms(self.terms)
        right = _set_from_terms(terms)
        if not left or not right:
            return 0.0
        intersection = left & right
        if not intersection:
            return 0.0
        union = left | right
        return float(len(intersection) / len(union))


def _read_csv_rows(path: Path, sep: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=sep, **CSV_KWARGS)
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def load_backup_manuals(backup_dir: Path) -> List[ManualTopic]:
    if not backup_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de backups: {backup_dir}")
    candidates = sorted(backup_dir.glob("topics_manual_labels_*.csv"))
    latest_path: Optional[Path] = None
    if candidates:
        latest_path = max(candidates, key=lambda p: p.stem)
    direct_latest = backup_dir / "topics_manual_labels_latest.csv"
    if direct_latest.exists():
        latest_path = direct_latest
    if latest_path is None:
        raise FileNotFoundError(
            f"No se encontraron archivos topics_manual_labels_*.csv en {backup_dir}"
        )
    sep = _auto_sep(latest_path)
    rows = _read_csv_rows(latest_path, sep)
    records: List[ManualTopic] = []
    for row in rows:
        records.append(
            ManualTopic(
                topic_id=_clean_label(row.get("topic_id")),
                terms=_parse_terms(row.get("topic_terms")),
                auto_label=_clean_label(row.get("auto_label")),
                manual_topic=_clean_label(row.get("manual_label_topic")),
                manual_subtopic=_clean_label(row.get("manual_label_subtopic")),
                include_dashboard=_clean_label(row.get("include_dashboard")),
                notes=_clean_label(row.get("notes")),
            )
        )
    return records


def _extract_terms_from_series_text(text: str, position: int, topic_value: Optional[str]) -> Optional[List[str]]:
    if not isinstance(text, str) or "[" not in text:
        return None
    text = text.replace("\\n", "\n")
    lines = text.splitlines()
    candidates: Dict[int, List[str]] = {}
    fallback: List[List[str]] = []
    term_pattern = re.compile(r"\[(.+?)\]")
    for line in lines:
        if "Name:" in line or "dtype:" in line:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        match = term_pattern.search(stripped)
        if not match:
            continue
        chunk = match.group(1)
        tokens = [tok.strip().strip("'\"") for tok in chunk.split(",") if tok.strip()]
        if not tokens:
            continue
        index_token = stripped.split("[", 1)[0].strip().strip("'\"")
        try:
            idx = int(index_token)
            candidates[idx] = tokens
        except ValueError:
            fallback.append(tokens)
    if candidates:
        if position in candidates:
            return candidates[position]
        try:
            topic_idx = int(topic_value) if topic_value is not None else None
        except Exception:
            topic_idx = None
        if topic_idx is not None and topic_idx in candidates:
            return candidates[topic_idx]
        first_key = min(candidates.keys())
        return candidates[first_key]
    if fallback:
        return fallback[0]
    return None


def load_topic_info(path: Path) -> List[Dict[str, str]]:
    sep = _auto_sep(path)
    rows = _read_csv_rows(path, sep)
    if not rows:
        return []
    if "Topic" not in rows[0]:
        raise ValueError(f"{path} no tiene columna 'Topic'.")
    for idx, row in enumerate(rows):
        raw_terms = row.get("top_terms") or row.get("Representation")
        tokens = _extract_terms_from_series_text(raw_terms, idx, row.get("Topic"))
        if not tokens and row.get("Representation"):
            rep_tokens = _parse_terms(row["Representation"])
            if rep_tokens:
                tokens = rep_tokens
        if tokens:
            row["top_terms"] = json.dumps(tokens, ensure_ascii=False)
    return rows


def _build_subtopic_parent_map_from_manuals(manuals: List[ManualTopic]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    conflicts: set[str] = set()
    for manual in manuals:
        topic = manual.manual_topic
        subtopic = manual.manual_subtopic
        if not topic or not subtopic:
            continue
        if subtopic not in mapping:
            mapping[subtopic] = topic
        elif mapping[subtopic] != topic:
            conflicts.add(subtopic)
    for sub in conflicts:
        mapping.pop(sub, None)
    return mapping


def _build_subtopic_parent_map_from_rows(rows: List[Dict[str, str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    conflicts: set[str] = set()
    for row in rows:
        topic = _clean_label(row.get("manual_label_topic"))
        subtopic = _clean_label(row.get("manual_label_subtopic"))
        if not topic or not subtopic:
            continue
        if subtopic not in mapping:
            mapping[subtopic] = topic
        elif mapping[subtopic] != topic:
            conflicts.add(subtopic)
    for sub in conflicts:
        mapping.pop(sub, None)
    return mapping


def _enforce_subtopic_hierarchy_rows(rows: List[Dict[str, str]], mapping: Dict[str, str]) -> None:
    if not mapping:
        return
    for row in rows:
        subtopic = _clean_label(row.get("manual_label_subtopic"))
        if not subtopic:
            continue
        parent = mapping.get(subtopic)
        if parent:
            row["manual_label_topic"] = parent


def build_mapping(
    topic_rows: List[Dict[str, str]],
    manuals: List[ManualTopic],
    *,
    threshold: float = TERMS_THRESHOLD,
) -> List[Dict[str, str]]:
    manual_mapping = _build_subtopic_parent_map_from_manuals(manuals)
    rows: List[Dict[str, str]] = []
    for row in topic_rows:
        topic_id_raw = row.get("Topic", "-1")
        try:
            topic_id = int(topic_id_raw)
        except Exception:
            topic_id = -1
        top_terms = _parse_terms(row.get("top_terms") or row.get("TopTerms") or row.get("Representation"))
        auto_label = _clean_label(row.get("Name") or row.get("Representation"))
        count_val = _clean_label(row.get("Count"))
        lang_group = _clean_label(row.get("lang_group"))
        original_topic = _clean_label(row.get("Topic_original"))

        best: Optional[Tuple[ManualTopic, float]] = None
        for manual in manuals:
            sim = manual.similarity(top_terms)
            if best is None or sim > best[1]:
                best = (manual, sim)

        manual_topic = ""
        manual_subtopic = ""
        include_dashboard = ""
        notes = ""
        matched_id = ""
        similarity = 0.0

        if best is not None and best[1] >= threshold:
            matched, similarity = best
            manual_topic = matched.manual_topic
            manual_subtopic = matched.manual_subtopic
            include_dashboard = matched.include_dashboard
            notes = matched.notes
            matched_id = matched.topic_id
            if not auto_label:
                auto_label = matched.auto_label

        rows.append(
            {
                "topic_id": str(topic_id),
                "topic_terms": json.dumps(top_terms, ensure_ascii=False),
                "auto_label": auto_label,
                "manual_label_topic": manual_topic,
                "manual_label_subtopic": manual_subtopic,
                "include_dashboard": include_dashboard,
                "notes": notes,
                "matched_previous_topic": matched_id,
                "similarity": f"{similarity:.3f}",
                "count": count_val,
                "lang_group": lang_group,
                "topic_original": original_topic,
                "name": _clean_label(row.get("Name")),
            }
        )

    # Jerarquía: primero la que proviene de los manuales históricos, luego la que surge del propio mapeo.
    if manual_mapping:
        _enforce_subtopic_hierarchy_rows(rows, manual_mapping)
    dynamic_mapping = _build_subtopic_parent_map_from_rows(rows)
    if dynamic_mapping:
        _enforce_subtopic_hierarchy_rows(rows, dynamic_mapping)
    return rows


def write_output(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "topic_id",
        "topic_terms",
        "auto_label",
        "manual_label_topic",
        "manual_label_subtopic",
        "include_dashboard",
        "notes",
        "matched_previous_topic",
        "similarity",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";", **CSV_KWARGS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"✔︎ Etiquetas mapeadas guardadas en {output_path}")


def write_missing_labels(rows: List[Dict[str, str]], output_path: Path) -> None:
    missing = [row for row in rows if not _clean_label(row.get("manual_label_topic"))]
    if not missing:
        if output_path.exists():
            output_path.unlink()
        print("ⓘ No se encontraron tópicos sin etiqueta manual.")
        return
    fieldnames = [
        "topic_id",
        "count",
        "lang_group",
        "auto_label",
        "topic_terms",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";", **CSV_KWARGS)
        writer.writeheader()
        for row in missing:
            writer.writerow({
                "topic_id": row.get("topic_id", ""),
                "count": row.get("count", ""),
                "lang_group": row.get("lang_group", ""),
                "auto_label": row.get("auto_label", ""),
                "topic_terms": row.get("topic_terms", ""),
            })
    print(f"ⓘ Resumen de tópicos sin etiqueta guardado en {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reaplicar etiquetas manuales a nuevos tópicos.")
    parser.add_argument(
        "--topic-info",
        type=Path,
        default=Path("results") / "topics" / "topic_info.csv",
        help="CSV generado por summarize_topics con los tópicos actuales.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("results") / "topics" / "topic_backup",
        help="Directorio donde residen los respaldos topics_manual_labels_*.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Archivo de salida (default: {DEFAULT_OUTPUT.as_posix()}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=TERMS_THRESHOLD,
        help="Similitud mínima de términos para aceptar un mapeo automático (0-1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topic_rows = load_topic_info(args.topic_info)
    manuals = load_backup_manuals(args.backup_dir)
    mapped_rows = build_mapping(topic_rows, manuals, threshold=args.threshold)
    write_output(mapped_rows, args.output)
    missing_path = Path("results") / "topics" / "topics_missing_manual_labels.csv"
    write_missing_labels(mapped_rows, missing_path)


if __name__ == "__main__":
    main()
