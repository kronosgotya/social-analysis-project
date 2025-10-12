from __future__ import annotations
import argparse
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telegram", type=str, default="")
    ap.add_argument("--x", type=str, default="")
    ap.add_argument("--max_rows", type=int, default=0, help="0 = todos")
    ap.add_argument("--device", type=int, default=None, help="GPU id (0) o CPU (-1)")
    ap.add_argument("--emotion_model", type=str, default="joeddav/xlm-roberta-large-xnli",
                    help="Modelo zero-shot para emociones")
    args = ap.parse_args()

    ensure_dirs("data/processed", "results/graphs", "results/charts")

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

    # --- Unificado
    df_all = unify_frames(df_tg, df_x)
    if not df_all.empty:
        df_all = normalize_source(df_all)

        # Base de hechos para Tableau
        facts_cols = [
            "source","timestamp","author","item_id","lang",
            "sentiment_label","sentiment_score","emotion_label",
            "emoji_count","text_clean","link","likes","retweets","replies","quotes"
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
