# -*- coding: utf-8 -*-
"""
topics_bertopic.py
Implementación productiva de tópicos con BERTopic (multilingüe) + caché opcional.

Funciones clave:
- fit_topics(docs, ids, cache_dir, **opts) -> dict
- transform_docs(model, docs, ids) -> List[dict]
- summarize_topics(model, docs, ids, dts) -> pd.DataFrame

Uso típico:
  from src.topics_bertopic import fit_topics, transform_docs, summarize_topics
  out = fit_topics(docs, ids, cache_dir="models/bertopic", min_topic_size=20)
  assignments = out["assignments"]  # [{uid, topic_id, label, score}, ...]
  model = out["model"]
  df_sum = summarize_topics(model, docs, ids, dts)
"""
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Evita warnings de numba/umap si no hay aceleración
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -------- Embeddings --------
_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Alternativas (más lentas pero algo mejores):
# "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

def _build_embeddings(
    docs: List[str],
    model_name: str = _DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    normalize_embeddings: bool = True,
) -> Tuple[np.ndarray, SentenceTransformer]:
    st = SentenceTransformer(model_name)
    emb = st.encode(
        docs,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings,
    )
    return emb, st

# -------- Fit / Load --------
def fit_topics(
    docs: List[str],
    ids: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    *,
    min_topic_size: int = 20,
    n_gram_range: Tuple[int, int] = (1, 2),
    seed: int = 42,
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
) -> Dict[str, Any]:
    """
    Entrena BERTopic y devuelve asignaciones + modelo para reuso.
    Si cache_dir existe con un modelo guardado, lo carga y usa transform.
    """
    assert len(docs) > 0, "No hay documentos para topic modeling."
    if ids is None:
        ids = [str(i) for i in range(len(docs))]

    model_path = os.path.join(cache_dir, "bertopic_model") if cache_dir else None
    loaded = False
    model: Optional[BERTopic] = None

    if model_path and os.path.isdir(model_path):
        try:
            model = BERTopic.load(model_path)
            loaded = True
        except Exception:
            loaded = False

    if loaded and model is not None:
        topics, probs = model.transform(docs)
    else:
        # Embeddings (multilingüe y rápido)
        emb, _ = _build_embeddings(docs, model_name=embedding_model)

        # Configuración BERTopic (sin reducir demasiado en CPU)
        model = BERTopic(
            min_topic_size=min_topic_size,
            n_gram_range=n_gram_range,
            calculate_probabilities=True,
            language="multilingual",
            verbose=True,
            seed=seed,
        )
        topics, probs = model.fit_transform(docs, embeddings=emb)

        if model_path:
            os.makedirs(model_path, exist_ok=True)
            model.save(model_path)

    # Etiquetas legibles
    topic_labels = model.get_topic_info()
    # topic_labels: columns -> Topic, Count, Name
    topic_name = {int(r["Topic"]): r["Name"] for _, r in topic_labels.iterrows()}

    # Ensamblar assignments
    assignments: List[Dict[str, Any]] = []
    for i, (uid, t) in enumerate(zip(ids, topics)):
        label = topic_name.get(int(t), "unassigned")
        score = float(np.max(probs[i])) if probs is not None and len(probs[i]) else 0.0
        assignments.append({"uid": uid, "topic_id": int(t), "label": label, "score": score})

    return {
        "assignments": assignments,
        "model": model,
        "model_info": {
            "n_topics": int((topic_labels["Topic"] >= 0).sum()),
            "topics_table": topic_labels,
        },
    }

def transform_docs(model: BERTopic, docs: List[str], ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Asigna tópicos usando un modelo ya entrenado (p. ej. producción).
    """
    if ids is None:
        ids = [str(i) for i in range(len(docs))]
    topics, probs = model.transform(docs)
    topic_labels = model.get_topic_info()
    topic_name = {int(r["Topic"]): r["Name"] for _, r in topic_labels.iterrows()}

    out = []
    for i, (uid, t) in enumerate(zip(ids, topics)):
        label = topic_name.get(int(t), "unassigned")
        score = float(np.max(probs[i])) if probs is not None and len(probs[i]) else 0.0
        out.append({"uid": uid, "topic_id": int(t), "label": label, "score": score})
    return out

def summarize_topics(
    model: BERTopic,
    docs: List[str],
    ids: List[str],
    dts: List[str],
) -> pd.DataFrame:
    """
    Devuelve resumen temporal por topic_id/label/fecha (YYYY-MM-DD):
    - volume, share, top_words
    """
    assignments = transform_docs(model, docs, ids)
    df = pd.DataFrame(assignments)
    df["dt"] = pd.Series(dts).astype(str)
    # volumen por día y topic
    grp = df.groupby(["dt", "topic_id", "label"]).size().reset_index(name="volume")
    # share por día
    daily = grp.groupby("dt")["volume"].transform("sum")
    grp["share"] = grp["volume"] / daily

    # palabras clave por topic
    topic_words = {}
    for t in grp["topic_id"].unique():
        words = model.get_topic(t)  # [(word, weight), ...]
        if words:
            topic_words[t] = ", ".join([w for w, _ in words[:8]])
        else:
            topic_words[t] = ""
    grp["top_words"] = grp["topic_id"].map(topic_words)
    return grp.sort_values(["dt", "volume"], ascending=[True, False])