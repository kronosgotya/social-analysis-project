from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

# --- Fix ruta para importar src/*
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from transformers.utils.logging import set_verbosity_warning  # noqa: E402
set_verbosity_warning()  # menos ruido en consola

from src.utils import (  # noqa: E402
    ensure_dirs, normalize_source, add_engagement, add_dominant_emotion,
    export_tableau_csv, emotions_to_long
)
from src.preprocessing import load_telegram, load_x, unify_frames  # noqa: E402
from src.sentiment import add_sentiment  # noqa: E402
from src.emotions import add_emotions  # noqa: E402
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
        df_tg = add_sentiment(df_tg, text_col="text_clean", device=args.device)
        df_tg = add_emotions(df_tg, text_col="text_clean", device=args.device, model_name=args.emotion_model)
        df_tg = add_dominant_emotion(df_tg)
        export_tableau_csv(df_tg, "data/processed/telegram_sentiment.csv")
        print("✔ TG procesado → data/processed/telegram_sentiment.csv")

    # --- X
    df_x = None
    if args.x and Path(args.x).exists():
        df_x = load_x(args.x)
        if args.max_rows > 0:
            df_x = df_x.head(args.max_rows)
        df_x = add_sentiment(df_x, text_col="text_clean", device=args.device)
        df_x = add_emotions(df_x, text_col="text_clean", device=args.device, model_name=args.emotion_model)
        df_x = add_engagement(df_x)
        df_x = add_dominant_emotion(df_x)
        export_tableau_csv(df_x, "data/processed/x_sentiment.csv")
        print("✔ X procesado → data/processed/x_sentiment.csv")

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
        print("No se encontraron CSV de entrada. Usa --telegram y/o --x")
        return

    entities = load_entities(args.entities, args.entities_file)

    # --- Unificado
    df_all = unify_frames(df_tg, df_x)
    if not df_all.empty:
        df_all = normalize_source(df_all)

        # Topic modeling (BERTopic) sobre todo el corpus disponible
        topic_mask = df_all["text_clean"].astype(str).str.strip() != ""
        if topic_mask.any():
            topic_df = df_all[topic_mask].copy()
            docs = topic_df["text_clean"].astype(str).tolist()
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

            assignments_df = pd.DataFrame(topic_assignments)
            assignments_df = assignments_df.rename(
                columns={"uid": "item_id", "label": "topic_label", "score": "topic_score"}
            )
            export_tableau_csv(assignments_df, "data/processed/topics_assignments.csv")

            summary_df = summarize_topics(topic_out["model"], docs, ids, dts)
            export_tableau_csv(summary_df, "data/processed/topics_summary_daily.csv")

            topics_table = topic_out["model_info"]["topics_table"]
            export_tableau_csv(topics_table, "results/topics/topic_info.csv")
        else:
            df_all["topic_id"] = pd.NA
            df_all["topic_label"] = pd.NA
            df_all["topic_score"] = pd.NA

        # Entity-conditioned sentiment + emotions
        if entities:
            mentions = extract_entity_mentions(
                df_all,
                entities,
                text_col="text_clean",
                id_col="item_id",
                topic_id_col="topic_id",
                topic_label_col="topic_label",
                context_window=args.entity_window,
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

                item_to_mentions = {}
                for rec in mentions_df.to_dict(orient="records"):
                    payload = {
                        "entity": rec.get("entity"),
                        "alias": rec.get("alias"),
                        "stance": rec.get("stance"),
                        "sentiment_label": rec.get("sentiment_label"),
                        "sentiment_score": rec.get("sentiment_score"),
                        "emotion_label": rec.get("emotion_label"),
                        "topic_id": rec.get("topic_id"),
                        "topic_label": rec.get("topic_label"),
                        "topic_score": rec.get("topic_score"),
                        "snippet": rec.get("snippet"),
                        "sentiment_dist": rec.get("sentiment_dist"),
                        "emotion_scores": rec.get("emotion_scores"),
                    }
                    item_id_key = str(rec.get("item_id"))
                    item_to_mentions.setdefault(item_id_key, []).append(payload)

                df_all["entity_mentions"] = df_all["item_id"].astype(str).apply(
                    lambda uid: json.dumps(item_to_mentions.get(uid, []), ensure_ascii=False)
                )
            else:
                print("ⓘ No se encontraron menciones de las entidades configuradas.")
                df_all["entity_mentions"] = "[]"
        else:
            print("ⓘ Análisis de entidades omitido (sin entidades configuradas).")
            df_all["entity_mentions"] = "[]"

        # Base de hechos para Tableau
        facts_cols = [
            "source","timestamp","author","item_id","lang",
            "sentiment_label","sentiment_score","emotion_label",
            "emoji_count","text_clean","link","likes","retweets","replies","quotes","views",
            "topic_id","topic_label","topic_score","entity_mentions"
        ]
        facts_cols = [c for c in facts_cols if c in df_all.columns]
        facts = df_all[facts_cols].copy()
        # engagement coherente (0 si falta)
        if set(["likes","retweets","replies","quotes"]).issubset(facts.columns):
            facts["engagement"] = facts[["likes","retweets","replies","quotes"]].fillna(0).astype(float).sum(axis=1)
        else:
            facts["engagement"] = 0

        export_tableau_csv(facts, "data/processed/facts_posts.csv")

        # Emotions long
        emo_long = emotions_to_long(df_all)
        if not emo_long.empty:
            export_tableau_csv(emo_long, "data/processed/emotions_long.csv")

        # Unificado completo
        export_tableau_csv(df_all, "data/processed/all_platforms.csv")
        print("✔ facts_posts.csv, emotions_long.csv, all_platforms.csv generados")

if __name__ == "__main__":
    main()
