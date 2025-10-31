from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, date
import sys
import re
import ast
from typing import Dict, Optional, List, Set, Any, Iterable
from collections import Counter

# --- Fix ruta para importar src/*
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import joblib  # noqa: E402
from transformers.utils.logging import set_verbosity_warning  # noqa: E402
set_verbosity_warning()  # menos ruido en consola

from src.utils import (  # noqa: E402
    ensure_dirs, normalize_source, add_engagement, add_dominant_emotion,
    export_tableau_csv, emotions_to_long
)
from src.preprocessing import load_telegram, load_x, unify_frames  # noqa: E402
# SentenceTransformer para clasificador embeddings
from sentence_transformers import SentenceTransformer  # noqa: E402
from src.network import (  # noqa: E402
    build_x_graph, graph_metrics, export_gexf,
    edges_from_x, nodes_metrics_df
)
from src.topics_bertopic import fit_topics, summarize_topics  # noqa: E402
from src.entities_runtime import load_entities  # noqa: E402
from src.entity_analysis import (  # noqa: E402
    extract_entity_mentions,
    score_entity_mentions,
    summarize_entity_mentions,
    serialize_mentions_for_export,
    aggregate_mentions_per_item,
)


def _ensure_json_array(value):
    if isinstance(value, (list, tuple)):
        try:
            return json.dumps(list(value), ensure_ascii=False)
        except TypeError:
            return json.dumps(list(value), ensure_ascii=False, default=str)
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "[]"
        if stripped.startswith("[") or stripped.startswith("{"):
            return stripped
        return json.dumps(stripped, ensure_ascii=False)
    if pd.isna(value):
        return "[]"
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


ENCODING = "utf-8-sig"
SEP = ";"
TOPIC_CLASSIFIER_PATH = Path("models") / "topic_classifier" / "topic_classifier.joblib"
TOPIC_MANUAL_PATH = Path("data") / "ground_truth" / "topics_manual_labels.csv"
DEFAULT_CLASSIFIER_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_EMBEDDER_CACHE: Dict[str, SentenceTransformer] = {}


def _get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDER_CACHE[model_name]


def _load_topic_classifier() -> Optional[Dict[str, Any]]:
    if not TOPIC_CLASSIFIER_PATH.exists():
        return None
    try:
        model_bundle = joblib.load(TOPIC_CLASSIFIER_PATH)
    except Exception as exc:
        print(f"ⓘ Could not load topic classifier ({exc}).")
        return None
    keys = set(model_bundle.keys())
    kind = model_bundle.get("kind")
    if kind == "sentence-transformer":
        required = {"topic_clf", "subtopic_clf", "topic_label_encoder", "subtopic_label_encoder", "embedding_model_name"}
        if not required.issubset(keys):
            print("ⓘ Topic classifier (embeddings) missing expected components; ignoring.")
            return None
    elif kind == "sentence-transformer-multitask":
        required = {
            "manual_topic_clf",
            "manual_subtopic_clf",
            "manual_topic_label_encoder",
            "manual_subtopic_label_encoder",
            "embedding_model_name",
        }
        if not required.issubset(keys):
            # compatibilidad con bundles antiguos
            legacy_required = {
                "topic_id_clf",
                "manual_topic_clf",
                "manual_subtopic_clf",
                "topic_id_label_encoder",
                "manual_topic_label_encoder",
                "manual_subtopic_label_encoder",
                "embedding_model_name",
            }
            if legacy_required.issubset(keys):
                # dejamos pasar; las rutas que usan topic_id_clf lo ignorarán
                pass
            else:
                print("ⓘ Topic classifier (multitask) missing expected components; ignoring.")
                return None
    else:
        required_keys = {"vectorizer", "topic_clf", "subtopic_clf"}
        if not required_keys.issubset(keys):
            print("ⓘ Topic classifier file is missing expected components; ignoring.")
            return None
        model_bundle.setdefault("kind", "tfidf")
    return model_bundle


def _load_manual_topic_labels() -> pd.DataFrame:
    if not TOPIC_MANUAL_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(TOPIC_MANUAL_PATH, sep=SEP, encoding=ENCODING)
    df["topic_id"] = df["topic_id"].astype(str)
    for col in ["manual_label_topic", "manual_label_subtopic"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)
    return df[["topic_id", "manual_label_topic", "manual_label_subtopic"]]


def _topic_terms_to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(term) for term in value if term)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return " ".join(str(term) for term in parsed if term)
            except json.JSONDecodeError:
                pass
        return text.replace(",", " ").replace("[", " ").replace("]", " ")
    return str(value)


def _normalize_label(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _majority_label(series: pd.Series) -> str:
    values = [_normalize_label(val) for val in series if _normalize_label(val)]
    if not values:
        return ""
    counter = Counter(values)
    most_common = counter.most_common()
    top_count = most_common[0][1]
    candidates = sorted([label for label, count in most_common if count == top_count])
    return candidates[0] if candidates else ""


def _first_non_empty(seq: Iterable[object]) -> object:
    for val in seq:
        if isinstance(val, list) and val:
            return val
        if isinstance(val, str) and val.strip():
            return val
    return []


_SERIES_LINE_PATTERN = re.compile(r"^\s*\d+\s+\[(.*)\]\s*$")


def _parse_series_style_terms(text: str) -> List[str]:
    cleaned = text.replace("\\n", "\n")
    tokens: List[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or "Name:" in line or "dtype" in line:
            continue
        match = _SERIES_LINE_PATTERN.match(line)
        if not match:
            continue
        inside = match.group(1)
        tokens.extend([tok.strip().strip("'\"") for tok in inside.split(",") if tok.strip()])
    return tokens


def _serialize_terms(value: object) -> str:
    """
    Serializa términos a JSON garantizando una lista plana de strings legibles.
    Maneja valores anidados (list[list[str]], tuplas...) y strings con
    representaciones de listas.
    """

    if isinstance(value, pd.Series):
        return _serialize_terms(value.tolist())

    def _maybe_parse_container(text: str) -> object:
        stripped = text.strip()
        if not stripped:
            return []
        if not (stripped.startswith("[") and stripped.endswith("]")):
            series_tokens = _parse_series_style_terms(stripped)
            if series_tokens:
                return series_tokens
            return text
        try:
            parsed = json.loads(stripped)
            return parsed
        except Exception:
            try:
                parsed = ast.literal_eval(stripped)
                return parsed
            except Exception:
                series_tokens = _parse_series_style_terms(stripped)
                if series_tokens:
                    return series_tokens
                return text

    def _flatten_terms(obj: object) -> Iterable[str]:
        if obj is None:
            return
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                yield from _flatten_terms(item)
            return
        if isinstance(obj, str):
            parsed = _maybe_parse_container(obj)
            if parsed is obj:
                token = obj.strip()
                if token:
                    yield token
            else:
                yield from _flatten_terms(parsed)
            return
        token = str(obj).strip()
        if token:
            yield token

    seen: Set[str] = set()
    ordered: List[str] = []
    for term in _flatten_terms(value):
        if term not in seen:
            seen.add(term)
            ordered.append(term)
    formatted = ", ".join(f"'{tok}'" for tok in ordered)
    return f"[{formatted}]"


def _build_subtopic_parent_map(df: pd.DataFrame, topic_col: str, subtopic_col: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    conflicts: Set[str] = set()
    if df is None or df.empty:
        return mapping
    for _, row in df.iterrows():
        topic = _normalize_label(row.get(topic_col))
        subtopic = _normalize_label(row.get(subtopic_col))
        if not subtopic or not topic:
            continue
        if subtopic not in mapping:
            mapping[subtopic] = topic
        elif mapping[subtopic] != topic:
            conflicts.add(subtopic)
    for sub in conflicts:
        mapping.pop(sub, None)
    return mapping


def _enforce_subtopic_hierarchy(df: pd.DataFrame, topic_col: str, subtopic_col: str, mapping: Dict[str, str]) -> None:
    if df is None or df.empty or not mapping:
        return
    if topic_col not in df.columns or subtopic_col not in df.columns:
        return
    subtopics = df[subtopic_col].astype(str).str.strip()
    replacement = subtopics.map(mapping)
    mask = subtopics != ""
    mask &= replacement.notna()
    if mask.any():
        df.loc[mask, topic_col] = replacement[mask]


def _apply_classifier_predictions(
    classifier: Optional[Dict[str, Any]],
    texts: List[str],
    assignments: List[Dict[str, Any]],
) -> None:
    if classifier is None or not assignments or not texts:
        return
    kind = classifier.get("kind")
    if kind not in {"sentence-transformer", "sentence-transformer-multitask"}:
        return

    embed_name = classifier.get("embedding_model_name", DEFAULT_CLASSIFIER_EMBED_MODEL)
    normalize_embeddings = classifier.get("normalize_embeddings", True)
    batch_size = int(classifier.get("inference_batch_size", 128))
    embedder = _get_embedder(embed_name)
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize_embeddings,
    )

    manual_topic_preds: List[str]
    manual_subtopic_preds: List[str]

    if kind == "sentence-transformer-multitask":
        manual_topic_encoder = classifier.get("manual_topic_label_encoder")
        manual_subtopic_encoder = classifier.get("manual_subtopic_label_encoder")
        manual_topic_clf = classifier.get("manual_topic_clf")
        manual_subtopic_clf = classifier.get("manual_subtopic_clf")
        if (
            manual_topic_encoder is None
            or manual_subtopic_encoder is None
            or manual_topic_clf is None
            or manual_subtopic_clf is None
        ):
            return

        manual_topic_preds = manual_topic_encoder.inverse_transform(manual_topic_clf.predict(embeddings))
        manual_subtopic_preds = manual_subtopic_encoder.inverse_transform(manual_subtopic_clf.predict(embeddings))

        for idx, assign in enumerate(assignments):
            pred_topic = manual_topic_preds[idx] if idx < len(manual_topic_preds) else ""
            if pred_topic and not _normalize_label(assign.get("manual_label_topic")):
                assign["manual_label_topic"] = pred_topic

            pred_subtopic = manual_subtopic_preds[idx] if idx < len(manual_subtopic_preds) else ""
            if pred_subtopic and not _normalize_label(assign.get("manual_label_subtopic")):
                assign["manual_label_subtopic"] = pred_subtopic

    elif kind == "sentence-transformer":
        manual_topic_encoder = classifier.get("topic_label_encoder")
        manual_subtopic_encoder = classifier.get("subtopic_label_encoder")
        manual_topic_clf = classifier.get("topic_clf")
        manual_subtopic_clf = classifier.get("subtopic_clf")
        if manual_topic_encoder is None or manual_subtopic_encoder is None:
            return
        manual_topic_preds = manual_topic_encoder.inverse_transform(manual_topic_clf.predict(embeddings))
        manual_subtopic_preds = manual_subtopic_encoder.inverse_transform(manual_subtopic_clf.predict(embeddings))

        for idx, assign in enumerate(assignments):
            pred_topic = manual_topic_preds[idx] if idx < len(manual_topic_preds) else ""
            if pred_topic and not _normalize_label(assign.get("manual_label_topic")):
                assign["manual_label_topic"] = pred_topic

            pred_subtopic = manual_subtopic_preds[idx] if idx < len(manual_subtopic_preds) else ""
            if pred_subtopic and not _normalize_label(assign.get("manual_label_subtopic")):
                assign["manual_label_subtopic"] = pred_subtopic

def _derive_facts_posts_tableau(
    input_path: Path,
    *,
    output_filename: str = "facts_posts_tableau.csv",
    trunc_len: int = 200,
    sep: str = ";",
    encoding: str = "utf-8-sig",
) -> None:
    if not input_path.exists():
        print(f"ⓘ Could not find {input_path.name}; skipping Tableau derivative.")
        return

    df = pd.read_csv(input_path, sep=sep, encoding=encoding, dtype=str)

    total_rows = len(df)
    if total_rows == 0:
        output_path = input_path.parent / output_filename
        export_tableau_csv(df, str(output_path))
        print(f"ⓘ {output_filename} generated (empty dataset).")
        return

    timestamp_series = df.get("timestamp", pd.Series(["" for _ in range(total_rows)]))
    text_series = df.get("text_clean", pd.Series(["" for _ in range(total_rows)]))
    invalid_timestamp = 0
    text_null_count = int(text_series.isna().sum())

    def _extract_date(value) -> str:
        nonlocal invalid_timestamp
        if pd.isna(value):
            invalid_timestamp += 1
            return ""
        if isinstance(value, (datetime, date)):
            return value.strftime("%Y-%m-%d")
        text = str(value).strip()
        if not text or text.lower() == "nat":
            invalid_timestamp += 1
            return ""
        match = re.search(r"\d{4}-\d{2}-\d{2}", text)
        if match:
            return match.group(0)
        try:
            parsed = pd.to_datetime(text, errors="raise")
            return parsed.date().isoformat()
        except Exception:
            invalid_timestamp += 1
            return ""

    def _truncate_text(value) -> str:
        if pd.isna(value):
            return ""
        text = str(value).replace("\r", " ").replace("\n", " ").replace("\t", " ")
        return text.strip()[:trunc_len]

    def _entity_polarities(value: object):
        if value in (None, "", "[]"):
            return []
        try:
            mentions = json.loads(value) if isinstance(value, str) else value
        except Exception:
            return []
        if not isinstance(mentions, list):
            return []
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        labels: Dict[str, Dict[str, int]] = {}
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            entity = str(mention.get("entity") or "").strip()
            if not entity:
                continue
            try:
                score_val = float(mention.get("sentiment_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score_val = 0.0
            label = str(mention.get("sentiment_label") or "").strip().lower()
            if label in {"positive", "pos"}:
                polarity = score_val
            elif label in {"negative", "neg"}:
                polarity = -score_val
            else:
                polarity = 0.0
            sums[entity] = sums.get(entity, 0.0) + polarity
            counts[entity] = counts.get(entity, 0) + 1
            labels.setdefault(entity, {})[label] = labels.setdefault(entity, {}).get(label, 0) + 1

        results = []
        for entity, total in sums.items():
            count = counts.get(entity, 1)
            avg = total / count if count else 0.0
            entity_labels = labels.get(entity, {})
            dominant_label = "neutral"
            if entity_labels:
                dominant_label = max(entity_labels.items(), key=lambda kv: kv[1])[0] or "neutral"
            results.append({
                "entity": entity,
                "avg_polarity": round(avg, 6),
                "mentions": count,
                "dominant_label": dominant_label,
            })
        return results

    df["date"] = timestamp_series.apply(_extract_date)
    df["text_trunc"] = text_series.apply(_truncate_text)
    if "entity_mentions" in df.columns:
        df["entity_sentiment_polarity"] = df["entity_mentions"].apply(_entity_polarities).apply(
            lambda rows: json.dumps(rows, ensure_ascii=False)
        )
    else:
        df["entity_sentiment_polarity"] = [json.dumps([], ensure_ascii=False)] * total_rows

    if "sentiment_polarity" in df.columns:
        df = df.drop(columns=["sentiment_polarity"])

    output_path = input_path.parent / output_filename
    export_tableau_csv(df, str(output_path))

    print(
        "✔ {file} generated → {rows} rows | invalid timestamp: {bad_ts} | empty text_clean: {null_text}".format(
            file=output_filename,
            rows=total_rows,
            bad_ts=invalid_timestamp,
            null_text=text_null_count,
        )
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telegram", type=str, default="")
    ap.add_argument("--x", type=str, default="")
    ap.add_argument("--max_rows", type=int, default=0, help="0 = todos")
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="GPU id (0), CPU (-1), o MPS (-2 / mps) en Apple Silicon",
    )
    ap.add_argument("--emotion_model", type=str, default="joeddav/xlm-roberta-large-xnli",
                    help="Modelo zero-shot para emociones")
    ap.add_argument("--entities", type=str, default="OTAN,Rusia",
                    help="Lista separada por comas de entidades para análisis condicionado")
    ap.add_argument("--entities_file", type=str, default="",
                    help="Archivo YAML/JSON/TXT con entidades (opcional)")
    ap.add_argument("--entity_window", type=int, default=160,
                    help="Ventana en caracteres alrededor de la mención (contexto)")
    args = ap.parse_args()

    raw_device = args.device
    if raw_device is None:
        coerced_device = None
    else:
        if isinstance(raw_device, str):
            token = raw_device.strip().lower()
            if token in {"", "none"}:
                coerced_device = None
            elif token in {"-1", "cpu"}:
                coerced_device = -1
            elif token in {"-2", "mps"}:
                coerced_device = -2
            else:
                try:
                    coerced_device = int(token)
                except Exception as exc:
                    raise ValueError(f"Valor de --device no reconocido: {raw_device}") from exc
        else:
            coerced_device = raw_device
    args.device = coerced_device

    ensure_dirs(
        "data/processed",
        "results/graphs",
        "results/charts",
        "results/topics",
        "models/bertopic/global",
    )

    # --- Telegram
    df_tg = None
    if args.telegram and Path(args.telegram).exists():
        df_tg = load_telegram(args.telegram)
        if args.max_rows > 0:
            df_tg = df_tg.head(args.max_rows)
        df_tg = add_engagement(df_tg)
        df_tg_export = df_tg.copy()
        if "geo_country_distribution" in df_tg_export.columns:
            df_tg_export["geo_country_distribution"] = df_tg_export["geo_country_distribution"].apply(
                _ensure_json_array
            )
        export_tableau_csv(df_tg_export, "data/processed/telegram_preprocessed.csv")
        print("✔ TG procesado → data/processed/telegram_preprocessed.csv")

    # --- X
    df_x = None
    if args.x and Path(args.x).exists():
        df_x = load_x(args.x)
        if args.max_rows > 0:
            df_x = df_x.head(args.max_rows)
        df_x = add_engagement(df_x)
        export_tableau_csv(df_x, "data/processed/x_preprocessed.csv")
        print("✔ X procesado → data/processed/x_preprocessed.csv")

        # Network (X)
        G = build_x_graph(df_x)
        metrics = nodes_metrics_df(G)
        export_tableau_csv(metrics, "results/graphs/x_nodes_metrics.csv")
        export_gexf(G, "results/graphs/x_interactions.gexf")
        # Edges para Tableau/Gephi
        edges = edges_from_x(df_x)
        if not edges.empty:
            export_tableau_csv(edges, "data/processed/x_edges.csv")
        print("✔ Red X → .gexf, x_nodes_metrics.csv, x_edges.csv")

    if df_tg is None and df_x is None:
        print("No input CSV files found. Use --telegram and/or --x to provide sources.")
        return

    entities = load_entities(args.entities, args.entities_file)

    # --- Unificado
    df_all = unify_frames(df_tg, df_x)
    if not df_all.empty:
        df_all = normalize_source(df_all)
        df_all = add_engagement(df_all)

        # Topic modeling (BERTopic) sobre todo el corpus disponible
        topic_text_col = "text_topic" if "text_topic" in df_all.columns else "text_clean"
        if topic_text_col == "text_topic":
            non_empty_ratio = (
                df_all["text_topic"].astype(str).str.strip() != ""
            ).mean()
            if non_empty_ratio < 0.7:
                print(
                    f"ⓘ text_topic sólo está disponible en {non_empty_ratio:.1%} de los posts;"
                    " usando text_clean para BERTopic."
                )
                topic_text_col = "text_clean"

        topic_mask = df_all[topic_text_col].astype(str).str.strip() != ""
        if topic_mask.any():
            topic_df = df_all[topic_mask].copy()
            lang_series = topic_df.get("lang", pd.Series([""] * len(topic_df), index=topic_df.index)).astype(str).str.strip().str.lower()

            topic_classifier_bundle = _load_topic_classifier()

            cyrillic_langs = {"ru", "uk", "bg", "be", "sr", "mk"}
            western_core_langs = {"en", "pl", "de", "fr", "it", "es", "pt", "nl"}
            western_scandi_langs = {"sv", "fi", "da", "no"}
            western_baltic_langs = {"lv", "lt", "et"}
            western_central_langs = {"cs", "sk", "hu", "ro", "tr"}

            group_configs = [
                {
                    "name": "cyrillic",
                    "mask": lang_series.isin(cyrillic_langs),
                    "fit_kwargs": {
                        "min_topic_size": 4,
                        "umap_kwargs": {"n_neighbors": 45},
                        "hdbscan_kwargs": {"min_cluster_size": 4, "min_samples": 1},
                    },
                },
                {
                    "name": "western_core",
                    "mask": lang_series.isin(western_core_langs),
                    "fit_kwargs": {
                        "min_topic_size": 7,
                        "umap_kwargs": {"n_neighbors": 32},
                        "hdbscan_kwargs": {"min_cluster_size": 7, "min_samples": 1},
                    },
                },
                {
                    "name": "western_scandi",
                    "mask": lang_series.isin(western_scandi_langs),
                    "fit_kwargs": {
                        "min_topic_size": 5,
                        "umap_kwargs": {"n_neighbors": 38},
                        "hdbscan_kwargs": {"min_cluster_size": 5, "min_samples": 1},
                    },
                },
                {
                    "name": "western_baltic",
                    "mask": lang_series.isin(western_baltic_langs),
                    "fit_kwargs": {
                        "min_topic_size": 3,
                        "umap_kwargs": {"n_neighbors": 36},
                        "hdbscan_kwargs": {"min_cluster_size": 3, "min_samples": 1},
                    },
                },
                {
                    "name": "western_central",
                    "mask": lang_series.isin(western_central_langs),
                    "fit_kwargs": {
                        "min_topic_size": 6,
                        "umap_kwargs": {"n_neighbors": 35},
                        "hdbscan_kwargs": {"min_cluster_size": 6, "min_samples": 1},
                    },
                },
                {
                    "name": "other",
                    "mask": pd.Series([True] * len(topic_df), index=topic_df.index),
                    "fit_kwargs": {
                        "min_topic_size": 5,
                        "umap_kwargs": {"n_neighbors": 35},
                        "hdbscan_kwargs": {"min_cluster_size": 5, "min_samples": 1},
                    },
                },
            ]

            topic_assignments: List[Dict[str, object]] = []
            summary_frames: List[pd.DataFrame] = []
            topics_tables: List[pd.DataFrame] = []
            global_topic_counter = 0
            used_mask = pd.Series([False] * len(topic_df), index=topic_df.index)

            group_summaries: List[str] = []
            group_metrics: List[Dict[str, object]] = []

            for group_cfg in group_configs:
                group_name = group_cfg["name"]
                group_mask = group_cfg["mask"] & ~used_mask
                if not group_mask.any():
                    continue
                used_mask |= group_mask
                group_df = topic_df[group_mask].copy()

                docs = group_df[topic_text_col].astype(str).tolist()
                ids = group_df["item_id"].astype(str).tolist()
                dts = group_df["timestamp"].astype(str).tolist()
                lang_codes = group_df["lang"].astype(str).tolist() if "lang" in group_df.columns else None
                docs_clean = group_df["text_clean"].astype(str).tolist() if "text_clean" in group_df.columns else docs

                cache_dir = (Path("models") / "bertopic" / f"group_{group_name}").as_posix()

                print(f"▶ BERTopic grupo {group_name} ({len(docs)} documentos)")

                topic_out = fit_topics(
                    docs,
                    ids=ids,
                    cache_dir=cache_dir,
                    min_topic_size=group_cfg["fit_kwargs"]["min_topic_size"],
                    n_gram_range=(1, 2),
                    seed=42,
                    lang_codes=lang_codes,
                    apply_reduce_outliers=group_cfg.get("apply_reduce_outliers", False),
                    reduce_outliers_kwargs=group_cfg.get("reduce_outliers_kwargs"),
                    umap_kwargs=group_cfg["fit_kwargs"].get("umap_kwargs"),
                    hdbscan_kwargs=group_cfg["fit_kwargs"].get("hdbscan_kwargs"),
                )

                assignments = topic_out["assignments"]
                _apply_classifier_predictions(topic_classifier_bundle, docs_clean, assignments)

                summary = summarize_topics(
                    topic_out["model"],
                    docs,
                    ids,
                    dts,
                    assignments=assignments,
                )

                mapping: Dict[int, int] = {}
                for original_tid in sorted({a["topic_id"] for a in assignments if int(a["topic_id"]) >= 0}):
                    mapping[int(original_tid)] = global_topic_counter
                    global_topic_counter += 1

                for assign in assignments:
                    original_tid = int(assign["topic_id"])
                    assign["topic_id_original"] = original_tid
                    assign["lang_group"] = group_name
                    if original_tid >= 0:
                        assign["topic_id"] = mapping.get(original_tid, original_tid)

                if "topic_id" in summary.columns:
                    summary["topic_id_original"] = summary["topic_id"].astype(int)
                    summary["topic_id"] = summary["topic_id"].astype(int).map(lambda tid: mapping.get(tid, tid))
                else:
                    summary["topic_id_original"] = -1
                summary["lang_group"] = group_name

                topic_assignments.extend(assignments)
                summary_frames.append(summary)

                topics_table_group = topic_out.get("model_info", {}).get("topics_table")
                if topics_table_group is not None:
                    topics_table_group = topics_table_group.copy()
                    if "Topic" in topics_table_group.columns:
                        topics_table_group["Topic_original"] = topics_table_group["Topic"].astype(int)
                        topics_table_group["Topic"] = topics_table_group["Topic"].astype(int).map(
                            lambda tid: mapping.get(tid, tid)
                        )
                    topics_table_group["lang_group"] = group_name
                    topics_tables.append(topics_table_group)

                model_info = topic_out.get("model_info", {})
                n_outliers = model_info.get("n_outliers", "n/a")
                n_docs = model_info.get("n_documents", len(docs))
                group_summaries.append(
                    f"{group_name}: docs={n_docs}, outliers={n_outliers}"
                )
                group_metrics.append({
                    "lang_group": group_name,
                    "n_documents": n_docs,
                    "n_outliers": n_outliers,
                    "outlier_ratio": model_info.get("outlier_ratio"),
                    "n_topics": model_info.get("n_topics"),
                })

            if group_summaries:
                print("ⓘ Resumen por grupo -> " + "; ".join(group_summaries))
            if group_metrics:
                metrics_df = pd.DataFrame(group_metrics)
                export_tableau_csv(metrics_df, "results/topics/topic_group_metrics.csv")

            # Reordenar tópicos globalmente por volumen para evitar bloques por idioma
            new_topic_id_map: Dict[int, int] = {}
            if topics_tables:
                combined_table_preview = pd.concat(topics_tables, ignore_index=True)
                positive_table = combined_table_preview[combined_table_preview["Topic"] >= 0].copy()
                if not positive_table.empty:
                    positive_table = (
                        positive_table[["Topic", "Count"]]
                        .groupby("Topic", as_index=False)["Count"]
                        .max()
                        .sort_values(["Count", "Topic"], ascending=[False, True])
                    )
                    new_topic_id_map = {
                        int(row.Topic): idx for idx, row in enumerate(positive_table.itertuples(index=False))
                    }

            if new_topic_id_map:
                for assign in topic_assignments:
                    tid = int(assign.get("topic_id", -1))
                    if tid >= 0 and tid in new_topic_id_map:
                        assign["topic_id"] = new_topic_id_map[tid]
                for summary in summary_frames:
                    if "topic_id" in summary.columns:
                        topic_vals = pd.to_numeric(summary["topic_id"], errors="coerce")
                        mask = topic_vals.notna() & (topic_vals.astype(int) >= 0)
                        if mask.any():
                            current_ids = topic_vals.loc[mask].astype(int)
                            mapped_ids = current_ids.map(new_topic_id_map)
                            mapped_ids = mapped_ids.fillna(current_ids)
                            summary.loc[mask, "topic_id"] = mapped_ids.astype(int)
                for table in topics_tables:
                    topic_vals = pd.to_numeric(table["Topic"], errors="coerce")
                    mask = topic_vals.notna() & (topic_vals.astype(int) >= 0)
                    if mask.any():
                        current_ids = topic_vals.loc[mask].astype(int)
                        mapped_ids = current_ids.map(new_topic_id_map)
                        mapped_ids = mapped_ids.fillna(current_ids)
                        table.loc[mask, "Topic"] = mapped_ids.astype(int)

            uid_to_topic = {a["uid"]: a for a in topic_assignments}

            df_all["topic_id"] = df_all["item_id"].astype(str).map(
                lambda uid: uid_to_topic.get(uid, {}).get("topic_id")
            )
            df_all["topic_label"] = df_all["item_id"].astype(str).map(
                lambda uid: uid_to_topic.get(uid, {}).get("label")
            )
            df_all["topic_score"] = df_all["item_id"].astype(str).map(
                lambda uid: uid_to_topic.get(uid, {}).get("score")
            )
            df_all["topic_terms"] = df_all["item_id"].astype(str).map(
                lambda uid: uid_to_topic.get(uid, {}).get("terms")
            )

            assignments_df = pd.DataFrame(topic_assignments)
            assignments_df = assignments_df.rename(
                columns={"uid": "item_id", "label": "topic_label", "score": "topic_score", "terms": "topic_terms"}
            )

            summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
            topics_table = pd.concat(topics_tables, ignore_index=True) if topics_tables else pd.DataFrame()
            if not topics_table.empty and "Count" in topics_table.columns:
                topics_table = topics_table.sort_values(["Count", "Topic"], ascending=[False, True]).reset_index(drop=True)

            manual_topics = _load_manual_topic_labels()
            manual_topic_map = {}
            manual_subtopic_map = {}
            manual_hierarchy_map: Dict[str, str] = {}
            if not manual_topics.empty:
                manual_topic_map = manual_topics.set_index("topic_id")["manual_label_topic"].to_dict()
                manual_subtopic_map = manual_topics.set_index("topic_id")["manual_label_subtopic"].to_dict()
                manual_hierarchy_map = _build_subtopic_parent_map(
                    manual_topics,
                    "manual_label_topic",
                    "manual_label_subtopic",
                )

            summary_df["topic_id_str"] = summary_df.get("topic_id", pd.NA).astype(str)
            summary_df["manual_label_topic"] = summary_df["topic_id_str"].map(manual_topic_map).fillna("")
            summary_df["manual_label_subtopic"] = summary_df["topic_id_str"].map(manual_subtopic_map).fillna("")
            _enforce_subtopic_hierarchy(summary_df, "manual_label_topic", "manual_label_subtopic", manual_hierarchy_map)

            if topic_classifier_bundle is None:
                topic_classifier_bundle = _load_topic_classifier()
            if (
                topic_classifier_bundle is not None
                and topic_classifier_bundle.get("kind") == "sentence-transformer"
                and not summary_df.empty
            ):
                if "top_terms" in summary_df.columns:
                    term_source = summary_df["top_terms"].apply(_topic_terms_to_text)
                elif "topic_terms" in summary_df.columns:
                    term_source = summary_df["topic_terms"].apply(_topic_terms_to_text)
                else:
                    term_source = pd.Series([""] * len(summary_df), index=summary_df.index)
                term_source = term_source.fillna("").astype(str)

                topic_clf = topic_classifier_bundle["topic_clf"]
                subtopic_clf = topic_classifier_bundle["subtopic_clf"]

                if topic_classifier_bundle.get("kind") == "sentence-transformer":
                    embed_name = topic_classifier_bundle.get("embedding_model_name", DEFAULT_CLASSIFIER_EMBED_MODEL)
                    normalize_embeddings = topic_classifier_bundle.get("normalize_embeddings", True)
                    batch_size = int(topic_classifier_bundle.get("inference_batch_size", 128))
                    embedder = _get_embedder(embed_name)
                    embeddings = embedder.encode(
                        term_source.tolist(),
                        batch_size=batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=normalize_embeddings,
                    )
                    topic_encoder = topic_classifier_bundle["topic_label_encoder"]
                    subtopic_encoder = topic_classifier_bundle["subtopic_label_encoder"]
                    topic_preds = pd.Series(
                        topic_encoder.inverse_transform(topic_clf.predict(embeddings)),
                        index=summary_df.index,
                    )
                    subtopic_preds = pd.Series(
                        subtopic_encoder.inverse_transform(subtopic_clf.predict(embeddings)),
                        index=summary_df.index,
                    )
                else:
                    vectorizer = topic_classifier_bundle.get("vectorizer")
                    if vectorizer is None:
                        topic_preds = pd.Series(["" for _ in range(len(summary_df))], index=summary_df.index)
                        subtopic_preds = pd.Series(["" for _ in range(len(summary_df))], index=summary_df.index)
                    else:
                        X_features = vectorizer.transform(term_source.tolist())
                        topic_preds = pd.Series(topic_clf.predict(X_features), index=summary_df.index)
                        subtopic_preds = pd.Series(subtopic_clf.predict(X_features), index=summary_df.index)

                mask_topic_missing = summary_df["manual_label_topic"].astype(str).str.strip() == ""
                summary_df.loc[mask_topic_missing, "manual_label_topic"] = topic_preds[mask_topic_missing]

                mask_subtopic_missing = summary_df["manual_label_subtopic"].astype(str).str.strip() == ""
                summary_df.loc[mask_subtopic_missing, "manual_label_subtopic"] = subtopic_preds[mask_subtopic_missing]

            hierarchy_map = _build_subtopic_parent_map(summary_df, "manual_label_topic", "manual_label_subtopic")
            _enforce_subtopic_hierarchy(summary_df, "manual_label_topic", "manual_label_subtopic", hierarchy_map)

            final_topic_map = summary_df.set_index("topic_id_str")["manual_label_topic"].to_dict()
            final_subtopic_map = summary_df.set_index("topic_id_str")["manual_label_subtopic"].to_dict()

            if not assignments_df.empty:
                assignments_df["manual_label_topic"] = (
                    assignments_df["topic_id"].astype(str).map(final_topic_map).fillna("")
                )
                assignments_df["manual_label_subtopic"] = (
                    assignments_df["topic_id"].astype(str).map(final_subtopic_map).fillna("")
                )
                _enforce_subtopic_hierarchy(assignments_df, "manual_label_topic", "manual_label_subtopic", hierarchy_map)

                required_topic_cols = {"Count", "manual_label_topic", "manual_label_subtopic", "top_terms"}
                for col in required_topic_cols:
                    if col not in topics_table.columns:
                        topics_table[col] = pd.NA

                topics_counts = assignments_df.groupby("topic_id", as_index=False).agg(
                    Count=("item_id", "count"),
                    manual_label_topic=("manual_label_topic", _majority_label),
                    manual_label_subtopic=("manual_label_subtopic", _majority_label),
                    topic_terms=("topic_terms", _first_non_empty),
                )

                if not topics_table.empty and "Topic" in topics_table.columns:
                    topics_table["Topic"] = pd.to_numeric(topics_table["Topic"], errors="coerce")
                    topics_table = topics_table.dropna(subset=["Topic"])
                    topics_table["Topic"] = topics_table["Topic"].astype(int)
                    topics_table = topics_table.drop_duplicates("Topic", keep="first").set_index("Topic")
                else:
                    topics_table = pd.DataFrame().set_index(pd.Index([], name="Topic"))

                for _, row in topics_counts.iterrows():
                    tid = int(row["topic_id"])
                    if tid not in topics_table.index:
                        topics_table.loc[tid, "Count"] = 0
                    topics_table.at[tid, "Count"] = row["Count"]
                    topics_table.at[tid, "manual_label_topic"] = row["manual_label_topic"]
                    topics_table.at[tid, "manual_label_subtopic"] = row["manual_label_subtopic"]
                    topic_terms_value = _serialize_terms(row["topic_terms"])
                    topics_table.at[tid, "top_terms"] = topic_terms_value

                topics_table = topics_table.reset_index()
                topics_table = topics_table.sort_values(["Count", "Topic"], ascending=[False, True]).reset_index(drop=True)

            if "topic_terms" in assignments_df.columns:
                assignments_df["topic_terms"] = assignments_df["topic_terms"].apply(_serialize_terms)
            export_tableau_csv(assignments_df, "data/processed/topics_assignments.csv")

            if "top_terms" in summary_df.columns:
                summary_df["top_terms"] = summary_df["top_terms"].apply(_serialize_terms)
            summary_df = summary_df.drop(columns=["topic_id_str"])
            export_tableau_csv(summary_df, "data/processed/topics_summary_daily.csv")

            if "TopTerms" in topics_table.columns:
                topics_table = topics_table.rename(columns={"TopTerms": "top_terms"})
            if "topic_terms" in topics_table.columns:
                topics_table = topics_table.rename(columns={"topic_terms": "top_terms"})
            if "Representation" in topics_table.columns:
                topics_table["top_terms"] = topics_table["Representation"]
            elif "top_terms" in topics_table.columns:
                topics_table["top_terms"] = topics_table["top_terms"].apply(_serialize_terms)

            topic_id_col = None
            for candidate in ["Topic", "topic_id"]:
                if candidate in topics_table.columns:
                    topic_id_col = candidate
                    break
            if topic_id_col:
                topics_table["manual_label_topic"] = (
                    topics_table[topic_id_col].astype(str).map(final_topic_map).fillna("")
                )
                topics_table["manual_label_subtopic"] = (
                    topics_table[topic_id_col].astype(str).map(final_subtopic_map).fillna("")
                )
                _enforce_subtopic_hierarchy(topics_table, "manual_label_topic", "manual_label_subtopic", hierarchy_map)
            if "Count" in topics_table.columns:
                topics_table["Count"] = pd.to_numeric(topics_table["Count"], errors="coerce").fillna(0).astype(int)
            topics_table = topics_table.loc[:, ~topics_table.columns.duplicated()]
            preferred_cols = [
                "Topic",
                "topic_id",
                "Count",
                "Name",
                "Representation",
                "top_terms",
                "manual_label_topic",
                "manual_label_subtopic",
            ]
            ordered = [col for col in preferred_cols if col in topics_table.columns]
            topics_table = topics_table[ordered + [col for col in topics_table.columns if col not in ordered]]
            export_tableau_csv(topics_table, "results/topics/topic_info.csv")

            df_all["topic_id_str"] = df_all["topic_id"].astype(str)
            df_all["manual_label_topic"] = df_all["topic_id_str"].map(final_topic_map).fillna("")
            df_all["manual_label_subtopic"] = df_all["topic_id_str"].map(final_subtopic_map).fillna("")
            mask_topic_override = df_all["manual_label_topic"].astype(str).str.strip() != ""
            df_all.loc[mask_topic_override, "topic_label"] = df_all.loc[mask_topic_override, "manual_label_topic"]
            _enforce_subtopic_hierarchy(df_all, "manual_label_topic", "manual_label_subtopic", hierarchy_map)
            df_all = df_all.drop(columns=["topic_id_str"])
        else:
            df_all["topic_id"] = pd.NA
            df_all["topic_label"] = pd.NA
            df_all["topic_score"] = pd.NA

        # Entity-conditioned sentiment + emociones
        if entities:
            caption_col = "text_caption_clean" if "text_caption_clean" in df_all.columns else None
            summary_col = "text_summary_clean" if "text_summary_clean" in df_all.columns else None
            mentions = extract_entity_mentions(
                df_all,
                entities,
                text_col="text_clean",
                id_col="item_id",
                topic_id_col="topic_id",
                topic_label_col="topic_label",
                context_window=args.entity_window,
                caption_col=caption_col,
                summary_col=summary_col,
                caption_weight=0.8,
                summary_weight=0.2,
            )
            mentions_df = score_entity_mentions(
                mentions,
                sentiment_device=args.device,
                emotion_device=args.device,
                emotion_model=args.emotion_model,
            )
            if not mentions_df.empty:
                mentions_export = serialize_mentions_for_export(mentions_df)
                export_tableau_csv(mentions_export, "data/processed/entity_mentions.csv")
                summary_mentions = summarize_entity_mentions(mentions_df)
                export_tableau_csv(summary_mentions, "data/processed/entity_topic_summary.csv")
                item_summary = aggregate_mentions_per_item(mentions_df)
                if not item_summary.empty:
                    df_all = df_all.merge(item_summary, on="item_id", how="left")

                item_to_mentions = {}
                for rec in mentions_df.to_dict(orient="records"):
                    payload = {
                        "entity": rec.get("entity"),
                        "alias": rec.get("alias"),
                        "stance": rec.get("stance"),
                        "stance_value": rec.get("stance_value"),
                        "sentiment_label": rec.get("sentiment_label"),
                        "sentiment_score": rec.get("sentiment_score"),
                        "emotion_label": rec.get("emotion_label"),
                        "topic_id": rec.get("topic_id"),
                        "topic_label": rec.get("topic_label"),
                        "topic_score": rec.get("topic_score"),
                        "impact_score": rec.get("impact_score"),
                        "engagement": rec.get("engagement"),
                        "reach": rec.get("reach"),
                        "snippet": rec.get("snippet"),
                        "text_source": rec.get("text_source"),
                        "sentiment_dist": rec.get("sentiment_dist"),
                        "emotion_scores": rec.get("emotion_scores"),
                    }
                    item_id_key = str(rec.get("item_id"))
                    item_to_mentions.setdefault(item_id_key, []).append(payload)

                df_all["entity_mentions"] = df_all["item_id"].astype(str).apply(
                    lambda uid: json.dumps(item_to_mentions.get(uid, []), ensure_ascii=False)
                )
            else:
                print("ⓘ No mentions found for the configured entities.")
                df_all["entity_mentions"] = df_all["item_id"].astype(str).apply(lambda _: "[]")
        else:
            print("ⓘ Análisis de entidades omitido (sin entidades configuradas).")
            df_all["entity_mentions"] = df_all["item_id"].astype(str).apply(lambda _: "[]")

        numeric_defaults = {
            "impact_score": 0.0,
            "impact_score_mean": 0.0,
            "n_entity_mentions": 0,
            "stance_value": 0.0,
        }
        for col, default in numeric_defaults.items():
            if col in df_all.columns:
                df_all[col] = df_all[col].fillna(default)
            else:
                df_all[col] = default

        if "stance" in df_all.columns:
            df_all["stance"] = df_all["stance"].fillna("neu")
        else:
            df_all["stance"] = "neu"

        if "entities_detected" in df_all.columns:
            df_all["entities_detected"] = df_all["entities_detected"].apply(
                lambda v: list(v) if isinstance(v, (list, tuple, set)) else []
            )
        else:
            df_all["entities_detected"] = [[] for _ in range(len(df_all))]

        if "sentiment_dist" in df_all.columns:
            df_all["sentiment_dist"] = df_all["sentiment_dist"].apply(
                lambda v: v if isinstance(v, dict) else {}
            )
        else:
            df_all["sentiment_dist"] = [{} for _ in range(len(df_all))]

        if "emotion_scores" in df_all.columns:
            df_all["emotion_scores"] = df_all["emotion_scores"].apply(
                lambda v: v if isinstance(v, dict) else {}
            )
        else:
            df_all["emotion_scores"] = [{} for _ in range(len(df_all))]

        if "related_entities" in df_all.columns:
            df_all["related_entities"] = df_all["related_entities"].apply(
                lambda v: v if isinstance(v, str) else json.dumps(v or [], ensure_ascii=False)
            )
        else:
            df_all["related_entities"] = [json.dumps([], ensure_ascii=False) for _ in range(len(df_all))]

        if "entity_sentiment_polarity" not in df_all.columns:
            df_all["entity_sentiment_polarity"] = [json.dumps([], ensure_ascii=False) for _ in range(len(df_all))]

        if "topic_terms" in df_all.columns:
            df_all["topic_terms"] = df_all["topic_terms"].apply(
                lambda v: list(v) if isinstance(v, (list, tuple))
                else ([] if v in (None, "") else [str(v)])
            )
        else:
            df_all["topic_terms"] = [[] for _ in range(len(df_all))]

        df_all = add_dominant_emotion(df_all)

        # Base de hechos para Tableau
        facts_cols = [
            "source","timestamp","author","author_location","item_id","lang","geolocation","geo_country_distribution",
            "sentiment_label","sentiment_score","stance","stance_value",
            "impact_score","impact_score_mean","n_entity_mentions","entities_detected",
            "emotion_label","emotion_scores","emoji_count","text_clean","text_topic","topic_terms","link","likes","retweets","replies","quotes","views",
            "topic_id","topic_label","topic_score","manual_label_topic","manual_label_subtopic","related_entities","entity_sentiment_polarity","entity_mentions"
        ]
        facts_cols = [c for c in facts_cols if c in df_all.columns]
        facts = df_all[facts_cols].copy()

        def _to_dict_safe(val):
            if isinstance(val, dict):
                return val
            if isinstance(val, str) and val.strip().startswith("{"):
                try:
                    return json.loads(val)
                except Exception:
                    return {}
            return {}

        if "emotion_scores" in df_all.columns:
            emotion_matrix = pd.json_normalize(df_all["emotion_scores"].apply(_to_dict_safe)).fillna(0.0)
            if not emotion_matrix.empty:
                emotion_matrix = emotion_matrix.reindex(facts.index).fillna(0.0)
                emotion_matrix.columns = [f"emotion_prob_{c}" for c in emotion_matrix.columns]
                facts = pd.concat([facts, emotion_matrix], axis=1)
        if "impact_score" in facts.columns:
            facts["impact_score"] = facts["impact_score"].fillna(0.0)
        if "impact_score_mean" in facts.columns:
            facts["impact_score_mean"] = facts["impact_score_mean"].fillna(0.0)
        if "stance" in facts.columns:
            facts["stance"] = facts["stance"].fillna("neu")
        if "entities_detected" in facts.columns:
            facts["entities_detected"] = facts["entities_detected"].apply(
                lambda v: json.dumps(list(v), ensure_ascii=False)
                if isinstance(v, (list, tuple, set))
                else ("[]" if pd.isna(v) or v == "" else str(v))
            )
        if "entity_mentions" in facts.columns:
            facts["entity_mentions"] = facts["entity_mentions"].fillna("[]")
        if "topic_terms" in facts.columns:
            facts["topic_terms"] = facts["topic_terms"].apply(
                lambda v: json.dumps(list(v), ensure_ascii=False)
                if isinstance(v, (list, tuple))
                else ("[]" if pd.isna(v) or v == "" else str(v))
            )
        if "geo_country_distribution" in facts.columns:
            facts["geo_country_distribution"] = facts["geo_country_distribution"].apply(
                _ensure_json_array
            )
        if "emotion_scores" in facts.columns:
            facts["emotion_scores"] = facts["emotion_scores"].apply(
                lambda v: json.dumps(v, ensure_ascii=False)
                if isinstance(v, dict)
                else ("{}" if pd.isna(v) or v == "" else str(v))
            )
        # engagement coherente (0 si falta)
        if set(["likes","retweets","replies","quotes"]).issubset(facts.columns):
            facts["engagement"] = facts[["likes","retweets","replies","quotes"]].fillna(0).astype(float).sum(axis=1)
        else:
            facts["engagement"] = 0

        export_tableau_csv(facts, "data/processed/facts_posts.csv")
        _derive_facts_posts_tableau(Path("data/processed/facts_posts.csv"))

        # Emotions long
        emo_long = emotions_to_long(df_all)
        if not emo_long.empty:
            export_tableau_csv(emo_long, "data/processed/emotions_long.csv")

        if "entities_detected" in df_all.columns:
            df_all["entities_detected"] = df_all["entities_detected"].apply(
                lambda v: json.dumps(list(v), ensure_ascii=False)
                if isinstance(v, (list, tuple, set))
                else ("[]" if pd.isna(v) or v == "" else str(v))
            )
        if "sentiment_dist" in df_all.columns:
            df_all["sentiment_dist"] = df_all["sentiment_dist"].apply(
                lambda v: json.dumps(v, ensure_ascii=False)
                if isinstance(v, dict)
                else ("{}" if pd.isna(v) or v == "" else str(v))
            )
        if "emotion_scores" in df_all.columns:
            df_all["emotion_scores"] = df_all["emotion_scores"].apply(
                lambda v: json.dumps(v, ensure_ascii=False)
                if isinstance(v, dict)
                else ("{}" if pd.isna(v) or v == "" else str(v))
            )
        if "topic_terms" in df_all.columns:
            df_all["topic_terms"] = df_all["topic_terms"].apply(
                lambda v: json.dumps(list(v), ensure_ascii=False)
                if isinstance(v, (list, tuple))
                else ("[]" if pd.isna(v) or v == "" else str(v))
            )
        if "geo_country_distribution" in df_all.columns:
            df_all["geo_country_distribution"] = df_all["geo_country_distribution"].apply(
                _ensure_json_array
            )

        # Unificado completo
        export_tableau_csv(df_all, "data/processed/all_platforms.csv")
        print("✔ facts_posts.csv, emotions_long.csv, all_platforms.csv generados")

if __name__ == "__main__":
    main()


# === Finalize manual and classifier labels on processed CSVs ===

import atexit

def _normalize_topic_id_value(v):
    try:
        import pandas as _pd
    except Exception:
        # Minimal normalization w/o pandas
        if v is None:
            return ""
        s = str(v).strip()
        try:
            f = float(s)
            i = int(f)
            return str(i) if f == float(i) else s
        except Exception:
            return s
    if _pd.isna(v):
        return ""
    s = str(v).strip()
    try:
        f = float(s)
        i = int(f)
        return str(i) if f == float(i) else s
    except Exception:
        return s

def _load_manual_label_table():
    import pandas as pd
    p = Path("data") / "ground_truth" / "topics_manual_labels.csv"
    if not p.exists():
        print("ⓘ Ground truth topics_manual_labels.csv not found; skipping manual merge.")
        return pd.DataFrame()
    df = pd.read_csv(p, sep=";", encoding="utf-8-sig", low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    for col in ("topic_id","manual_label_topic","manual_label_subtopic"):
        if col not in df.columns:
            print("ⓘ Ground truth missing columns; skipping manual merge.")
            return pd.DataFrame()
    df["topic_id_norm"] = df["topic_id"].apply(_normalize_topic_id_value)
    return df[["topic_id_norm","manual_label_topic","manual_label_subtopic"]].drop_duplicates()

def _load_classifier_bundle():
    p = Path("models") / "topic_classifier" / "topic_classifier.joblib"
    if not p.exists():
        return None
    try:
        import joblib  # type: ignore
        bundle = joblib.load(p)
        return bundle
    except Exception as e:
        print(f"ⓘ Could not load topic classifier: {e}")
        return None

def _predict_manual_labels(bundle, texts_series):
    kind = bundle.get("kind")
    model_name = bundle.get("embedding_model_name")
    if not model_name:
        raise RuntimeError("Classifier bundle missing 'embedding_model_name'.")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(f"sentence-transformers unavailable: {e}")
    embedder = SentenceTransformer(model_name)
    texts = texts_series.fillna("").astype(str).tolist()
    embs = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    import numpy as np
    X = np.array(embs)

    if kind == "sentence-transformer":
        topic_clf = bundle["topic_clf"]
        sub_clf = bundle["subtopic_clf"]
        le_topic = bundle["topic_label_encoder"]
        le_sub = bundle["subtopic_label_encoder"]
        t_idx = topic_clf.predict(X)
        s_idx = sub_clf.predict(X)
        import pandas as pd
        t = pd.Series(le_topic.inverse_transform(t_idx), index=texts_series.index)
        s = pd.Series(le_sub.inverse_transform(s_idx), index=texts_series.index)
        return t, s
    elif kind == "sentence-transformer-multitask":
        manual_topic_clf = bundle["manual_topic_clf"]
        manual_subtopic_clf = bundle["manual_subtopic_clf"]
        manual_topic_encoder = bundle["manual_topic_label_encoder"]
        manual_subtopic_encoder = bundle["manual_subtopic_label_encoder"]
        t_idx = manual_topic_clf.predict(X)
        s_idx = manual_subtopic_clf.predict(X)
        import pandas as pd
        t = pd.Series(manual_topic_encoder.inverse_transform(t_idx), index=texts_series.index)
        s = pd.Series(manual_subtopic_encoder.inverse_transform(s_idx), index=texts_series.index)
        return t, s
    else:
        raise RuntimeError(f"Unknown classifier kind: {kind}")

def _export_with_bom(df, path_str):
    # Use project export helper if available to keep ; and utf-8-sig
    try:
        export_tableau_csv(df, path_str)  # from src.utils
    except Exception:
        import pandas as pd
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path_str, sep=";", encoding="utf-8-sig", index=False)

def _apply_labels_to_dataframe(df, name, manuals, bundle):
    import pandas as pd
    if df is None or df.empty:
        return df

    cols = {c.lower(): c for c in df.columns}
    topic_col = cols.get("topic_id")
    if not topic_col:
        return df

    if topic_col != "topic_id":
        df = df.rename(columns={topic_col: "topic_id"})

    manual_topic_col = cols.get("manual_label_topic")
    if manual_topic_col and manual_topic_col != "manual_label_topic":
        df = df.rename(columns={manual_topic_col: "manual_label_topic"})
    manual_subtopic_col = cols.get("manual_label_subtopic")
    if manual_subtopic_col and manual_subtopic_col != "manual_label_subtopic":
        df = df.rename(columns={manual_subtopic_col: "manual_label_subtopic"})

    if "manual_label_topic" not in df.columns:
        df["manual_label_topic"] = pd.NA
    if "manual_label_subtopic" not in df.columns:
        df["manual_label_subtopic"] = pd.NA

    df["topic_id_norm"] = df["topic_id"].apply(_normalize_topic_id_value)

    before_t = df["manual_label_topic"].notna().sum()
    before_s = df["manual_label_subtopic"].notna().sum()

    if manuals is not None and not manuals.empty:
        df = df.merge(manuals, on="topic_id_norm", how="left", suffixes=("", "_manual_ref"))
        for col in ("manual_label_topic", "manual_label_subtopic"):
            ref = f"{col}_manual_ref"
            if ref in df.columns:
                df[col] = df[col].where(df[col].notna() & (df[col].astype(str).str.strip() != ""), df[ref])
                df.drop(columns=[ref], inplace=True, errors="ignore")

    text_col = cols.get("text") or cols.get("text_clean")
    if bundle is not None and (text_col or "text" in df.columns):
        working_text_col = "text"
        if text_col and text_col != "text" and text_col in df.columns:
            df = df.rename(columns={text_col: working_text_col})
        if working_text_col in df.columns:
            try:
                mask = df["manual_label_topic"].isna() | (df["manual_label_topic"].astype(str).str.strip() == "")
                if mask.any():
                    t_pred, s_pred = _predict_manual_labels(bundle, df.loc[mask, working_text_col])
                    df.loc[mask, "manual_label_topic"] = df.loc[mask, "manual_label_topic"].fillna(t_pred)
                    df.loc[mask, "manual_label_subtopic"] = df.loc[mask, "manual_label_subtopic"].fillna(s_pred)
            except Exception as e:
                print(f"ⓘ Classifier skipped for {name}: {e}")
        if text_col and text_col != "text":
            df = df.rename(columns={working_text_col: text_col})

    after_t = df["manual_label_topic"].notna().sum()
    after_s = df["manual_label_subtopic"].notna().sum()
    df.drop(columns=["topic_id_norm"], inplace=True, errors="ignore")

    if topic_col and topic_col != "topic_id":
        df = df.rename(columns={"topic_id": topic_col})
    if manual_topic_col and manual_topic_col != "manual_label_topic":
        df = df.rename(columns={"manual_label_topic": manual_topic_col})
    if manual_subtopic_col and manual_subtopic_col != "manual_label_subtopic":
        df = df.rename(columns={"manual_label_subtopic": manual_subtopic_col})

    print(f"✔ {name}: topics {before_t}→{after_t} / subtopics {before_s}→{after_s}")
    return df

def _finalize_processed_topic_labels():
    import pandas as pd
    processed = Path("data") / "processed"
    if not processed.exists():
        return

    manuals = _load_manual_label_table()
    bundle = _load_classifier_bundle()

    # Fixed set we know about + autodiscovery
    candidates = [
        "facts_posts.csv",
        "facts_posts_tableau.csv",
        "all_platforms.csv",
        "entity_mentions.csv",
        "entity_topic_summary.csv",
    ]
    # Add any other csvs with topic_id
    for p in processed.glob("*.csv"):
        if p.name not in candidates:
            try:
                df_probe = pd.read_csv(p, sep=";", encoding="utf-8-sig", nrows=5, low_memory=False)
            except Exception:
                try:
                    df_probe = pd.read_csv(p, sep=",", encoding="utf-8", nrows=5, low_memory=False)
                except Exception:
                    continue
            if "topic_id" in [c.lower() for c in df_probe.columns]:
                candidates.append(p.name)

    for name in sorted(set(candidates)):
        p = processed / name
        if not p.exists():
            continue
        try:
            try:
                df = pd.read_csv(p, sep=";", encoding="utf-8-sig", low_memory=False)
            except Exception:
                df = pd.read_csv(p, sep=",", encoding="utf-8", low_memory=False)
            df2 = _apply_labels_to_dataframe(df, name, manuals, bundle)
            _export_with_bom(df2, str(p))
        except Exception as e:
            print(f"ⓘ Skipped {name}: {e}")

# Register finalize step to run after the script completes its normal pipeline
atexit.register(_finalize_processed_topic_labels)
