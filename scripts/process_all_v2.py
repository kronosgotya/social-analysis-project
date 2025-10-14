# -*- coding: utf-8 -*-
"""
process_all_v2.py
Orquestador alternativo que NO hardcodea entidades.
Uso:
  python scripts/process_all_v2.py --telegram data/raw/telegram.csv --x data/raw/x.csv \
      --entities "OTAN,Rusia" --date-from 2025-09-01 --date-to 2025-09-30 \
      --export-long-entity true
"""
from __future__ import annotations
import argparse, json
import pandas as pd
from pathlib import Path

from src.entities_runtime import load_entities
from src.geoloc import infer_country
from src.topics_bertopic import fit_topics
from src.stance_entity import stance_and_sentiment
from src.emotions_entity import emotions_for_entity
from src.impact import impact_score

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--telegram", type=str, required=False)
    p.add_argument("--x", type=str, required=False)
    p.add_argument("--date-from", type=str, required=False)
    p.add_argument("--date-to", type=str, required=False)
    p.add_argument("--entities", type=str, required=False, help="CSV: 'EntidadA,EntidadB'")
    p.add_argument("--entities-file", type=str, required=False, help="YAML/JSON/TXT con entidades")
    p.add_argument("--export-long-entity", type=str, default="false")
    p.add_argument("--outdir", type=str, default="data/processed")
    return p.parse_args()

def load_df(path: str, platform: str) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8")
    df["platform"] = platform
    return df

def main():
    args = parse_args()
    ents = load_entities(args.entities, args.entities_file)  # lista de EntitySpec

    df_tg = load_df(args.telegram, "telegram")
    df_x  = load_df(args.x, "x")
    df = pd.concat([df_tg, df_x], ignore_index=True)

    # Normaliza campos mínimos esperados del contrato
    for col in ["uid","dt","tz_message","tz_author","lang","author","channel","permalink","text","media_kind","engagement"]:
        if col not in df.columns: df[col] = None

    # Filtro por fechas (si se pide)
    if args.date_from:
        df = df[df["dt"] >= args.date_from]
    if args.date_to:
        df = df[df["dt"] <= args.date_to]

    # Geoloc heurística
    geo_country, geo_conf, geo_basis = [], [], []
    for tz_m, tz_a, lang, text in zip(df["tz_message"], df["tz_author"], df["lang"], df["text"]):
        c, conf, basis = infer_country(tz_m, tz_a, lang, text)
        geo_country.append(c); geo_conf.append(conf); geo_basis.append(basis)
    df["geo_country"] = geo_country
    df["geo_conf"] = geo_conf
    df["geo_basis"] = geo_basis

    # Topics (placeholder)
    from src.topics_bertopic import fit_topics

    docs = df["text"].fillna("").tolist()
    ids = df["uid"].astype(str).tolist()
    topic_out = fit_topics(
        docs,
        ids=ids,
        cache_dir="models/bertopic",      # persistir/recargar modelo
        min_topic_size=20,
        n_gram_range=(1, 2),
        seed=42,
    )
    uid2topic = {a["uid"]: a for a in topic_out["assignments"]}
    df["topics"] = df["uid"].astype(str).map(
        lambda u: json.dumps([uid2topic[u]], ensure_ascii=False)
    )

    # Entities por fila (a partir de selección usuario)
    # Para este MVP: si texto contiene alias -> marcamos presencia
    def detect_presence(text: str):
        found = []
        t = (text or "").lower()
        for e in ents:
            for al in (e.aliases or [e.name]):
                if al.lower() in t:
                    found.append({"name": e.name, "type": e.type, "alias": al})
                    break
        return json.dumps(found, ensure_ascii=False)

    df["entities"] = df["text"].map(detect_presence)

    # Métricas por entidad (wide) para dos entidades más frecuentes (si hay)
    # NOTA: mantenemos columnas genéricas para hasta 2 entidades; en producción, export-long-entity=True
    top_entities = [e.name for e in ents[:2]]

    for ent in top_entities:
        stance_list, sent_list, emo_list, impact_list = [], [], [], []
        for text, engagement in zip(df["text"], df["engagement"]):
            stance, sent = stance_and_sentiment(text or "", ent)
            emo = emotions_for_entity(text or "", ent)
            imp = impact_score(ent, sent, stance, emo.get("label"), engagement, None, None)
            stance_list.append(stance); sent_list.append(sent); emo_list.append(emo.get("label")); impact_list.append(imp)
        prefix = f"{ent}".replace(" ", "_")
        df[f"stance_entity_{prefix}"] = stance_list
        df[f"sentiment_entity_{prefix}"] = sent_list
        df[f"emotions_entity_{prefix}"] = emo_list
        df[f"impact_entity_{prefix}"] = impact_list

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_main = outdir / "facts_enriched.csv"
    df.to_csv(out_main, index=False, encoding="utf-8-sig")

    # Forma long por entidad (si se pide)
    if (args.export_long_entity or "").lower() in ("1","true","yes"):
        rows = []
        for _, r in df.iterrows():
            ents_found = json.loads(r["entities"]) if r["entities"] else []
            for e in ents_found:
                name = e["name"]
                prefix = name.replace(" ", "_")
                rows.append({
                    "uid": r["uid"],
                    "platform": r["platform"],
                    "dt": r["dt"],
                    "entity": name,
                    "topic_id": json.loads(r["topics"])[0]["topic_id"] if r["topics"] else None,
                    "topic_label": json.loads(r["topics"])[0]["label"] if r["topics"] else None,
                    "stance": r.get(f"stance_entity_{prefix}"),
                    "sentiment": r.get(f"sentiment_entity_{prefix}"),
                    "emotion": r.get(f"emotions_entity_{prefix}"),
                    "impact": r.get(f"impact_entity_{prefix}"),
                    "geo_country": r.get("geo_country"),
                    "geo_conf": r.get("geo_conf"),
                })
        pd.DataFrame(rows).to_csv(outdir / "facts_enriched_entity.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()