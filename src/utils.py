# -*- coding: utf-8 -*-
"""
Utilidades comunes:
- ensure_dirs
- normalización de 'source'
- engagement total
- inferencia de emoción dominante desde distribuciones
- emotions_to_long
- exportación CSV 'amigable' para Excel/Tableau (UTF-8 BOM, ';')
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Union
import csv
import json

import pandas as pd
import os


def ensure_dirs(*paths: Union[str, Path]) -> None:
    """Crea directorios si no existen (parents=True)."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def normalize_source(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza 'source' a {'X','Telegram'}."""
    df = df.copy()
    if "source" in df.columns:
        df["source"] = (
            df["source"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"x": "X", "twitter": "X", "tw": "X", "telegram": "Telegram", "tg": "Telegram"})
            .fillna("X")
        )
    return df


def add_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """Añade 'engagement' = likes + retweets + replies + quotes (ausentes -> 0)."""
    df = df.copy()
    for c in ["likes", "retweets", "replies", "quotes"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["engagement"] = df[["likes", "retweets", "replies", "quotes"]].sum(axis=1)
    return df


def add_dominant_emotion(df: pd.DataFrame, scores_col: str = "emotion_scores") -> pd.DataFrame:
    """
    Si no hay 'emotion_label', la infiere a partir de 'emotion_scores' (dict o JSON).
    """
    df = df.copy()

    def _to_dict(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str) and x.strip().startswith("{"):
            try:
                return json.loads(x)
            except Exception:
                return {}
        return {}

    if scores_col in df.columns:
        dists = df[scores_col].apply(_to_dict)
        if "emotion_label" not in df.columns:
            df["emotion_label"] = None
        mask = df["emotion_label"].isna() | (df["emotion_label"].astype(str).str.strip() == "")
        df.loc[mask, "emotion_label"] = dists.apply(lambda d: max(d, key=d.get) if isinstance(d, dict) and d else None)
    return df


def emotions_to_long(df: pd.DataFrame, scores_col: str = "emotion_scores") -> pd.DataFrame:
    """
    Convierte distribuciones de emociones a formato largo:
    - item_id, source, emotion, prob (+ metadatos útiles)
    """
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        item = r.get("item_id") or r.get("messageId") or r.get("tweet_id")
        src = r.get("source")
        dist = r.get(scores_col)
        if isinstance(dist, str):
            try:
                dist = json.loads(dist)
            except Exception:
                dist = {}
        if isinstance(dist, dict):
            for emo, prob in dist.items():
                try:
                    probf = float(prob)
                except Exception:
                    continue
                rows.append(
                    {
                        "item_id": item,
                        "source": src,
                        "emotion": emo,
                        "prob": probf,
                        "sentiment_label": r.get("sentiment_label"),
                        "lang": r.get("lang"),
                        "author": r.get("author"),
                        "timestamp": r.get("timestamp"),
                    }
                )
    return pd.DataFrame(rows)


def export_tableau_csv(df: pd.DataFrame, path: str) -> None:
    """
    Exporta con:
    - UTF-8 con BOM (utf-8-sig) para Excel
    - Separador ';' (habitual en ES/EU)
    - Sin índice
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=";", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")


def coerce_datetime_series(s: pd.Series) -> pd.Series:
    """Convierte a datetime sin romper por formatos mezclados (NaT si no parsea)."""
    return pd.to_datetime(s, errors="coerce")


_MPS_WARNING_EMITTED = False


def resolve_pipeline_device(device: Union[int, str, None]) -> Union[int, "torch.device"]:
    """
    Normaliza la selección de dispositivo para pipelines de Transformers:
    - None → CPU (-1)
    - -1 → CPU
    - -2 / "mps" → torch.device("mps") si está disponible, si no CPU
    - >=0 → GPU id
    """
    global _MPS_WARNING_EMITTED
    if device is None:
        return -1
    if isinstance(device, str):
        token = device.strip().lower()
        if token == "mps":
            device = -2
        else:
            try:
                device = int(token)
            except Exception:
                return -1
    if isinstance(device, bool):
        device = int(device)
    if isinstance(device, int):
        if device == -1:
            return -1
        if device == -2:
            try:
                import torch  # type: ignore
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
                    return torch.device("mps")
            except Exception:
                pass
            if not _MPS_WARNING_EMITTED:
                print("ⓘ MPS no disponible; se usará CPU.")
                _MPS_WARNING_EMITTED = True
            return -1
        return device
    return -1
