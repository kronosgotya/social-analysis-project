"""
Herramientas para detección y análisis de entidades con sentimiento/emociones condicionadas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import json
import re

import pandas as pd

from .entities_runtime import EntitySpec
from .sentiment import SentimentScorer
from .emotions import EmotionClassifier, DEFAULT_EMOTIONS

_DEFAULT_ENTITY_ALIASES = {
    "OTAN": ["OTAN", "NATO", "N.A.T.O", "North Atlantic Treaty Organization"],
    "Rusia": [
        "Rusia", "Russia", "Federación Rusa", "Russian Federation",
        "России", "Россия", "русские", "россия"
    ],
}


@dataclass
class MentionCandidate:
    item_id: str
    entity: str
    alias: str
    source: Optional[str]
    timestamp: Optional[str]
    topic_id: Optional[int]
    topic_label: Optional[str]
    topic_score: Optional[float]
    lang: Optional[str]
    full_text: str
    snippet: str


def _alias_list(spec: EntitySpec) -> List[str]:
    aliases = list(dict.fromkeys((spec.aliases or []) + [spec.name]))
    extra = _DEFAULT_ENTITY_ALIASES.get(spec.name)
    if extra:
        for alias in extra:
            if alias not in aliases:
                aliases.append(alias)
    return [a for a in aliases if isinstance(a, str) and a.strip()]


def _snippet(text: str, alias: str, window: int = 160) -> str:
    """
    Extrae ventana de texto alrededor de la mención; fallback al texto completo si no se encuentra.
    """
    clean = text.strip()
    if not clean:
        return clean
    text_lower = clean.lower()
    alias_lower = alias.lower()
    idx = text_lower.find(alias_lower)
    if idx == -1:
        return clean
    start = max(0, idx - window)
    end = min(len(clean), idx + len(alias) + window)
    snip = clean[start:end].strip()
    if start > 0:
        snip = "... " + snip
    if end < len(clean):
        snip = snip + " ..."
    return snip


def extract_entity_mentions(
    df: pd.DataFrame,
    entities: Sequence[EntitySpec],
    *,
    text_col: str = "text_clean",
    id_col: str = "item_id",
    topic_id_col: str = "topic_id",
    topic_label_col: str = "topic_label",
    context_window: int = 160,
) -> List[MentionCandidate]:
    mentions: List[MentionCandidate] = []
    if df.empty or not entities:
        return mentions

    for _, row in df.iterrows():
        text_raw = row.get(text_col) or row.get("text") or ""
        text = str(text_raw)
        if not text.strip():
            continue
        text_casefold = text.casefold()
        for spec in entities:
            aliases = _alias_list(spec)
            matched_alias: Optional[str] = None
            for alias in aliases:
                alias_cf = alias.casefold()
                if not alias_cf:
                    continue
                # Usa boundaries aproximados para evitar falsos positivos (palabras completas)
                pattern = re.compile(rf"(?<!\w){re.escape(alias_cf)}(?!\w)")
                if pattern.search(text_casefold):
                    matched_alias = alias
                    break
            if matched_alias:
                topic_id_raw = row.get(topic_id_col)
                topic_id_val: Optional[int] = None
                if topic_id_raw is not None and str(topic_id_raw).strip() != "":
                    try:
                        topic_id_val = int(float(topic_id_raw))
                    except (TypeError, ValueError):
                        topic_id_val = None

                topic_score_raw = row.get("topic_score")
                topic_score_val: Optional[float] = None
                if topic_score_raw not in (None, "", "nan"):
                    try:
                        topic_score_val = float(topic_score_raw)
                    except (TypeError, ValueError):
                        topic_score_val = None

                mentions.append(
                    MentionCandidate(
                        item_id=str(row.get(id_col, "")),
                        entity=spec.name,
                        alias=matched_alias,
                        source=str(row.get("source")) if row.get("source") is not None else None,
                        timestamp=str(row.get("timestamp")) if row.get("timestamp") is not None else None,
                        topic_id=topic_id_val,
                        topic_label=str(row.get(topic_label_col)) if row.get(topic_label_col) is not None else None,
                        topic_score=topic_score_val,
                        lang=str(row.get("lang")) if row.get("lang") is not None else None,
                        full_text=text,
                        snippet=_snippet(text, matched_alias, window=context_window),
                    )
                )
    return mentions


def score_entity_mentions(
    mentions: Sequence[MentionCandidate],
    *,
    sentiment_device: Optional[int] = None,
    sentiment_batch_size: int = 64,
    emotion_device: Optional[int] = None,
    emotion_batch_size: int = 16,
    emotion_model: str = "joeddav/xlm-roberta-large-xnli",
    emotion_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    if not mentions:
        return pd.DataFrame(
            columns=[
                "item_id", "entity", "alias", "source", "timestamp", "topic_id", "topic_label",
                "lang", "full_text", "snippet", "sentiment_label", "sentiment_score",
                "sentiment_dist", "stance", "stance_value",
                "emotion_label", "emotion_scores",
            ]
        )

    mentions_df = pd.DataFrame([m.__dict__ for m in mentions])
    snippets = mentions_df["snippet"].astype(str).tolist()

    # Sentiment (targeted)
    sentiment_scorer = SentimentScorer(device=sentiment_device)
    sentiment_labels: List[str] = []
    sentiment_scores: List[float] = []
    sentiment_dists: List[Dict[str, float]] = []

    for start in range(0, len(snippets), sentiment_batch_size):
        batch = snippets[start:start + sentiment_batch_size]
        results = sentiment_scorer.score_batch(batch)
        for lab, conf, dist in results:
            sentiment_labels.append(lab)
            sentiment_scores.append(conf)
            sentiment_dists.append(dist)

    mentions_df["sentiment_label"] = sentiment_labels
    mentions_df["sentiment_score"] = sentiment_scores
    mentions_df["sentiment_dist"] = sentiment_dists
    mentions_df["stance"] = mentions_df["sentiment_label"].map(
        {"positive": "pos", "negative": "neg"}
    ).fillna("neu")
    mentions_df["stance_value"] = mentions_df["stance"].map({"pos": 1.0, "neg": -1.0, "neu": 0.0})

    # Emotions
    emotion_label_list = emotion_labels or DEFAULT_EMOTIONS
    emotion_clf = EmotionClassifier(
        labels=emotion_labels or DEFAULT_EMOTIONS,
        device=emotion_device,
        model_name=emotion_model,
    )
    emotion_labels_out: List[Optional[str]] = []
    emotion_scores_out: List[Dict[str, float]] = []

    for start in range(0, len(snippets), emotion_batch_size):
        batch = snippets[start:start + emotion_batch_size]
        results = emotion_clf.score_batch(batch)
        for top, dist in results:
            # garantiza dict con todas las etiquetas para agregaciones posteriores
            norm_dist = {lbl: float(dist.get(lbl, 0.0)) for lbl in emotion_label_list}
            total = sum(norm_dist.values())
            if total > 0:
                norm_dist = {k: v / total for k, v in norm_dist.items()}
            emotion_labels_out.append(top)
            emotion_scores_out.append(norm_dist)

    mentions_df["emotion_label"] = emotion_labels_out
    mentions_df["emotion_scores"] = emotion_scores_out
    return mentions_df


def summarize_entity_mentions(mentions_df: pd.DataFrame) -> pd.DataFrame:
    if mentions_df.empty:
        return pd.DataFrame(
            columns=[
                "entity", "topic_id", "topic_label", "n_posts", "n_mentions",
                "mean_sentiment_score", "stance_index", "stance_label",
                "sentiment_prob_positive", "sentiment_prob_negative", "sentiment_prob_neutral",
                "top_emotion",
            ]
        )

    # Probabilidades explícitas para agregación
    sent_probs = pd.json_normalize(mentions_df["sentiment_dist"]).fillna(0.0)
    if sent_probs.empty:
        sent_probs = pd.DataFrame(columns=["positive", "negative", "neutral"])
    for key in ["positive", "negative", "neutral"]:
        if key not in sent_probs.columns:
            sent_probs[key] = 0.0
    sent_probs = sent_probs[["positive", "negative", "neutral"]]
    sent_probs.columns = [f"sentiment_prob_{c}" for c in ["positive", "negative", "neutral"]]
    emo_probs = pd.json_normalize(mentions_df["emotion_scores"]).fillna(0.0)
    emo_probs.columns = [f"emotion_prob_{c.lower()}" for c in emo_probs.columns]
    desired_emotion_cols = sorted(
        set(emo_probs.columns) | {f"emotion_prob_{lbl}" for lbl in DEFAULT_EMOTIONS}
    )
    for col in desired_emotion_cols:
        if col not in emo_probs.columns:
            emo_probs[col] = 0.0
    emo_probs = emo_probs[desired_emotion_cols]
    enriched = pd.concat([mentions_df, sent_probs, emo_probs], axis=1)

    group_cols = ["entity", "topic_id", "topic_label"]
    summary = enriched.groupby(group_cols, dropna=False).agg(
        n_posts=("item_id", "nunique"),
        n_mentions=("item_id", "count"),
        mean_sentiment_score=("sentiment_score", "mean"),
        mean_topic_score=("topic_score", "mean"),
        stance_index=("stance_value", "mean"),
        sentiment_prob_positive=("sentiment_prob_positive", "mean"),
        sentiment_prob_negative=("sentiment_prob_negative", "mean"),
        sentiment_prob_neutral=("sentiment_prob_neutral", "mean"),
    )

    emotion_cols = [c for c in enriched.columns if c.startswith("emotion_prob_")]
    for col in emotion_cols:
        summary[col] = enriched.groupby(group_cols, dropna=False)[col].mean()

    def _mode(series: pd.Series) -> Optional[str]:
        mode = series.mode(dropna=True)
        return str(mode.iloc[0]) if not mode.empty else None

    summary["sentiment_label_mode"] = enriched.groupby(group_cols, dropna=False)["sentiment_label"].apply(_mode)
    summary["emotion_label_mode"] = enriched.groupby(group_cols, dropna=False)["emotion_label"].apply(_mode)

    def _stance_label(val: float) -> str:
        if pd.isna(val):
            return "neu"
        if val > 0.15:
            return "pos"
        if val < -0.15:
            return "neg"
        return "neu"

    summary["stance_label"] = summary["stance_index"].apply(_stance_label)

    if emotion_cols:
        summary["top_emotion"] = summary[emotion_cols].idxmax(axis=1).apply(
            lambda s: s.replace("emotion_prob_", "") if isinstance(s, str) else None
        )

    summary = summary.reset_index()
    return summary


def serialize_mentions_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte columnas dict a JSON string para export_tableau_csv.
    """
    if df.empty:
        return df

    df_out = df.copy()
    for col in ["sentiment_dist", "emotion_scores"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else ""
            )
    return df_out
