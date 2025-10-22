from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime, date
import sys
import re
from typing import Dict, Optional, List, Set

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


def _load_topic_classifier() -> Optional[Dict[str, object]]:
    if not TOPIC_CLASSIFIER_PATH.exists():
        return None
    try:
        model_bundle = joblib.load(TOPIC_CLASSIFIER_PATH)
    except Exception as exc:
        print(f"ⓘ Could not load topic classifier ({exc}).")
        return None
    required_keys = {"vectorizer", "topic_clf", "subtopic_clf"}
    if not required_keys.issubset(set(model_bundle.keys())):
        print("ⓘ Topic classifier file is missing expected components; ignoring.")
        return None
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
    ap.add_argument("--device", type=int, default=None, help="GPU id (0) o CPU (-1)")
    ap.add_argument("--emotion_model", type=str, default="joeddav/xlm-roberta-large-xnli",
                    help="Modelo zero-shot para emociones")
    ap.add_argument("--entities", type=str, default="OTAN,Rusia",
                    help="Lista separada por comas de entidades para análisis condicionado")
    ap.add_argument("--entities_file", type=str, default="",
                    help="Archivo YAML/JSON/TXT con entidades (opcional)")
    ap.add_argument("--entity_window", type=int, default=160,
                    help="Ventana en caracteres alrededor de la mención (contexto)")
    args = ap.parse_args()

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
            docs = topic_df[topic_text_col].astype(str).tolist()
            ids = topic_df["item_id"].astype(str).tolist()
            dts = topic_df["timestamp"].astype(str).tolist()
            topic_out = fit_topics(
                docs,
                ids=ids,
                cache_dir="models/bertopic/global",
                min_topic_size=20,
                n_gram_range=(1, 2),
                seed=42,
            )
            topic_assignments = topic_out["assignments"]
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

            summary_df = summarize_topics(
                topic_out["model"],
                docs,
                ids,
                dts,
                assignments=topic_assignments,
            )

            manual_topics = _load_manual_topic_labels()
            manual_topic_map = {}
            manual_subtopic_map = {}
            if not manual_topics.empty:
                manual_topic_map = manual_topics.set_index("topic_id")["manual_label_topic"].to_dict()
                manual_subtopic_map = manual_topics.set_index("topic_id")["manual_label_subtopic"].to_dict()

            summary_df["topic_id_str"] = summary_df.get("topic_id", pd.NA).astype(str)
            summary_df["manual_label_topic"] = summary_df["topic_id_str"].map(manual_topic_map).fillna("")
            summary_df["manual_label_subtopic"] = summary_df["topic_id_str"].map(manual_subtopic_map).fillna("")

            topic_classifier = _load_topic_classifier()
            if topic_classifier is not None and not summary_df.empty:
                term_source = None
                if "top_terms" in summary_df.columns:
                    term_source = summary_df["top_terms"].apply(_topic_terms_to_text)
                elif "topic_terms" in summary_df.columns:
                    term_source = summary_df["topic_terms"].apply(_topic_terms_to_text)
                else:
                    term_source = pd.Series([""] * len(summary_df), index=summary_df.index)

                vectorizer = topic_classifier["vectorizer"]
                topic_clf = topic_classifier["topic_clf"]
                subtopic_clf = topic_classifier["subtopic_clf"]
                X_features = vectorizer.transform(term_source.tolist())
                topic_preds = pd.Series(topic_clf.predict(X_features), index=summary_df.index)
                subtopic_preds = pd.Series(subtopic_clf.predict(X_features), index=summary_df.index)

                mask_topic_missing = summary_df["manual_label_topic"].astype(str).str.strip() == ""
                summary_df.loc[mask_topic_missing, "manual_label_topic"] = topic_preds[mask_topic_missing]

                mask_subtopic_missing = summary_df["manual_label_subtopic"].astype(str).str.strip() == ""
                summary_df.loc[mask_subtopic_missing, "manual_label_subtopic"] = subtopic_preds[mask_subtopic_missing]

            final_topic_map = summary_df.set_index("topic_id_str")["manual_label_topic"].to_dict()
            final_subtopic_map = summary_df.set_index("topic_id_str")["manual_label_subtopic"].to_dict()

            if not assignments_df.empty:
                assignments_df["manual_label_topic"] = (
                    assignments_df["topic_id"].astype(str).map(final_topic_map).fillna("")
                )
                assignments_df["manual_label_subtopic"] = (
                    assignments_df["topic_id"].astype(str).map(final_subtopic_map).fillna("")
                )

            if "topic_terms" in assignments_df.columns:
                assignments_df["topic_terms"] = assignments_df["topic_terms"].apply(
                    lambda terms: json.dumps(terms, ensure_ascii=False) if isinstance(terms, list)
                    else ("[]" if terms in (None, "") else str(terms))
                )
            export_tableau_csv(assignments_df, "data/processed/topics_assignments.csv")

            if "top_terms" in summary_df.columns:
                summary_df["top_terms"] = summary_df["top_terms"].apply(
                    lambda terms: json.dumps(terms, ensure_ascii=False) if isinstance(terms, list)
                    else ("[]" if terms in (None, "") else str(terms))
                )
            summary_df = summary_df.drop(columns=["topic_id_str"])
            export_tableau_csv(summary_df, "data/processed/topics_summary_daily.csv")

            topics_table = topic_out["model_info"]["topics_table"]
            if "TopTerms" in topics_table.columns:
                topics_table = topics_table.rename(columns={"TopTerms": "top_terms"})
            if "topic_terms" in topics_table.columns:
                topics_table = topics_table.rename(columns={"topic_terms": "top_terms"})
            if "TopTerms" in topics_table.columns:
                topics_table["TopTerms"] = topics_table["TopTerms"].apply(
                    lambda terms: json.dumps(terms, ensure_ascii=False) if isinstance(terms, list)
                    else ("[]" if terms in (None, "") else str(terms))
                )
                topics_table = topics_table.rename(columns={"TopTerms": "top_terms"})
            if "top_terms" in topics_table.columns:
                topics_table["top_terms"] = topics_table["top_terms"].apply(
                    lambda terms: json.dumps(terms, ensure_ascii=False) if isinstance(terms, list)
                    else ("[]" if terms in (None, "") else str(terms))
                )
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
            export_tableau_csv(topics_table, "results/topics/topic_info.csv")

            df_all["topic_id_str"] = df_all["topic_id"].astype(str)
            df_all["manual_label_topic"] = df_all["topic_id_str"].map(final_topic_map).fillna("")
            df_all["manual_label_subtopic"] = df_all["topic_id_str"].map(final_subtopic_map).fillna("")
            mask_topic_override = df_all["manual_label_topic"].astype(str).str.strip() != ""
            df_all.loc[mask_topic_override, "topic_label"] = df_all.loc[mask_topic_override, "manual_label_topic"]
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
