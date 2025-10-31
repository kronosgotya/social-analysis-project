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
import random
from typing import List, Dict, Any, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from stopwordsiso import stopwords as stopwords_iso

# Evita warnings de numba/umap si no hay aceleración y fuerza un solo hilo OpenMP
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

# -------- Embeddings --------
_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_TOPIC_MODEL_VERSION = "2024-10-stopwords-lemma"
# Alternativas (más lentas pero algo mejores):
# "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

_VECTOR_STOPWORD_LANGS = {
    "en", "es", "ru", "uk", "pl", "cs", "sk", "ro", "fr", "de",
    "it", "pt", "tr", "sv", "fi", "lt", "lv", "et", "da", "no",
    "nl", "bg", "sr", "mk", "hu"
}

def _collect_stopwords() -> List[str]:
    words = set()
    for lang in _VECTOR_STOPWORD_LANGS:
        try:
            words.update(stopwords_iso(lang))
        except Exception:
            continue
    words.update({"https", "http", "amp", "rt"})
    return sorted(words)

_GLOBAL_VECTOR_STOPWORDS = _collect_stopwords()

def _build_vectorizer(n_gram_range: Tuple[int, int]) -> CountVectorizer:
    return CountVectorizer(
        stop_words=_GLOBAL_VECTOR_STOPWORDS,
        token_pattern=r"(?u)\b\w{3,}\b",
        lowercase=True,
        ngram_range=n_gram_range,
        min_df=1,
    )

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


def _dedupe_terms(terms: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for term in terms:
        if not isinstance(term, str):
            continue
        norm = term.strip().lower()
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(term.strip())
    return out


def _get_topic_terms(model: BERTopic, top_n: int = 8) -> Dict[int, List[str]]:
    topics: Dict[int, List[str]] = {}
    for topic_id, words in (model.get_topics() or {}).items():
        try:
            tid = int(topic_id)
        except Exception:
            continue
        if not words:
            topics[tid] = []
            continue
        selected = [w for w, _ in words[:top_n] if isinstance(w, str)]
        topics[tid] = _dedupe_terms(selected)
    return topics


def _topics_degenerate(topic_terms_map: Dict[int, List[str]]) -> bool:
    valid = [terms for tid, terms in topic_terms_map.items() if int(tid) >= 0 and terms]
    if len(valid) <= 2:
        return True
    noisy = 0
    for terms in valid:
        leading = [t.strip().lower() for t in terms[:3] if isinstance(t, str)]
        if not leading:
            noisy += 1
            continue
        if all(val in {"", "nan", "nan nan"} for val in leading):
            noisy += 1
    return noisy >= max(1, len(valid) // 2)

def _maybe_reduce_outliers(
    model: BERTopic,
    docs: List[str],
    topics: Iterable[int],
    probs: Optional[np.ndarray],
    *,
    enabled: bool,
    reduce_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Aplica BERTopic.reduce_outliers (>=0.17.3) si está disponible y devuelve
    los tópicos/probabilidades actualizados.
    """
    topics_array = np.asarray(list(topics)) if not isinstance(topics, np.ndarray) else topics
    if not enabled or probs is None:
        return topics_array, probs
    if len(docs) == 0:
        return topics_array, probs
    attempt_params: List[Dict[str, Any]] = []
    base_kwargs = dict(reduce_kwargs or {})
    attempt_params.append(base_kwargs)
    if "strategy" not in base_kwargs:
        attempt_params.append({"strategy": "probabilities"})
    result = None
    last_exc: Optional[Exception] = None
    for kwargs in attempt_params:
        try:
            result = model.reduce_outliers(
                docs,
                topics=topics_array,
                probabilities=probs,
                **kwargs,
            )
            last_exc = None
            break
        except TypeError:
            # Compatibilidad con firmas antiguas: reduce_outliers(docs, topics, probs, **kwargs)
            try:
                result = model.reduce_outliers(docs, topics_array, probs, **kwargs)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                continue
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        print(f"⚠️ reduce_outliers falló ({last_exc}); se omite.")
        return topics_array, probs

    new_topics = topics_array
    new_probs = probs
    if isinstance(result, tuple):
        if len(result) >= 2:
            new_topics, new_probs = result[0], result[1]
        elif len(result) == 1:
            new_topics = result[0]
    elif result is not None:
        new_topics = result

    new_topics_arr = np.asarray(new_topics)
    if new_probs is not None and not isinstance(new_probs, np.ndarray):
        new_probs = np.asarray(new_probs)

    try:
        model.update_topics(docs, topics=new_topics_arr)
    except Exception:
        # update_topics no siempre está disponible/soportado (p. ej. sin vectorizer)
        pass
    return new_topics_arr, new_probs
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
    apply_reduce_outliers: bool = True,
    reduce_outliers_kwargs: Optional[Dict[str, Any]] = None,
    umap_kwargs: Optional[Dict[str, Any]] = None,
    hdbscan_kwargs: Optional[Dict[str, Any]] = None,
    lang_codes: Optional[Iterable[Optional[str]]] = None,
) -> Dict[str, Any]:
    """
    Entrena BERTopic y devuelve asignaciones + modelo para reuso.
    Si cache_dir existe con un modelo guardado, lo carga y usa transform.

    Args:
        docs: Lista de textos preprocesados/listos para embeddings.
        ids: Identificadores únicos alineados con docs.
        cache_dir: Ruta opcional para persistir/cargar el modelo.
        min_topic_size: Tamaño mínimo esperado por tópico (input a BERTopic).
        n_gram_range: Rango de n-gramas para CountVectorizer.
        seed: Semilla reproducible para UMAP/HDBSCAN.
        embedding_model: Nombre del modelo SentenceTransformer multilingüe.
        apply_reduce_outliers: Si True, aplica reduce_outliers tras fit/transform.
        reduce_outliers_kwargs: Parámetros extra para reduce_outliers (p. ej. {"strategy": "distribute"}).
        umap_kwargs: Overrides para instanciar UMAP.
        hdbscan_kwargs: Overrides para instanciar HDBSCAN.
        lang_codes: Secuencia opcional de códigos de idioma alineada con docs.
    """
    assert len(docs) > 0, "No hay documentos para topic modeling."
    if ids is None:
        ids = [str(i) for i in range(len(docs))]
    if lang_codes is not None:
        lang_list_input = list(lang_codes)
        if len(lang_list_input) != len(docs):
            raise ValueError("lang_codes debe tener igual longitud que docs.")
    else:
        lang_list_input = None

    model_path = os.path.join(cache_dir, "bertopic_model.pkl") if cache_dir else None
    loaded = False
    model: Optional[BERTopic] = None

    if model_path and os.path.isfile(model_path):
        try:
            model = BERTopic.load(model_path)
            loaded = getattr(model, "_insight_topic_version", None) == _TOPIC_MODEL_VERSION
            if not loaded:
                model = None
        except Exception:
            loaded = False

    emb, embedder = _build_embeddings(docs, model_name=embedding_model)

    # Configuración BERTopic (sin reducir demasiado en CPU)
    umap_defaults = dict(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
        low_memory=True,
        verbose=False,
    )
    if umap_kwargs:
        umap_defaults.update(umap_kwargs)
    umap_model = UMAP(**umap_defaults)

    hdbscan_defaults = dict(
        min_cluster_size=max(5, min_topic_size // 2),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        core_dist_n_jobs=1,
    )
    if hdbscan_kwargs:
        hdbscan_defaults.update(hdbscan_kwargs)
    hdbscan_model = HDBSCAN(**hdbscan_defaults)

    topic_terms_map: Optional[Dict[int, List[str]]] = None
    reduce_kwargs = {"strategy": "distribute"}
    if reduce_outliers_kwargs:
        reduce_kwargs.update(reduce_outliers_kwargs)

    def _train_new() -> tuple[BERTopic, np.ndarray, Optional[np.ndarray], Dict[int, List[str]]]:
        random.seed(seed)
        np.random.seed(seed)
        model_kwargs = dict(
            min_topic_size=min_topic_size,
            calculate_probabilities=True,
            language="multilingual",
            verbose=True,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            embedding_model=embedder,
            vectorizer_model=_build_vectorizer(n_gram_range),
        )
        new_model = BERTopic(**model_kwargs)
        new_topics, new_probs = new_model.fit_transform(docs, embeddings=emb)
        new_topics, new_probs = _maybe_reduce_outliers(
            new_model,
            docs,
            new_topics,
            new_probs,
            enabled=apply_reduce_outliers,
            reduce_kwargs=reduce_kwargs,
        )
        setattr(new_model, "_bertopic_embedding_model_name", embedding_model)
        setattr(new_model, "_insight_topic_version", _TOPIC_MODEL_VERSION)
        topic_terms_local = _get_topic_terms(new_model)
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            new_model.save(model_path)
        return new_model, new_topics, new_probs, topic_terms_local

    if loaded and model is not None:
        try:
            model.set_embedding_model(embedder)
            setattr(model, "_bertopic_embedding_model_name", embedding_model)
            setattr(model, "_insight_topic_version", _TOPIC_MODEL_VERSION)
        except Exception:
            pass
        topics, probs = model.transform(docs, embeddings=emb)
        topic_terms_map = _get_topic_terms(model)
        if _topics_degenerate(topic_terms_map):
            print("⚠️ Modelo BERTopic cacheado devolvió tópicos degenerados; reentrenando desde cero.")
            model, topics, probs, topic_terms_map = _train_new()
            loaded = False
        else:
            topics, probs = _maybe_reduce_outliers(
                model,
                docs,
                topics,
                probs,
                enabled=apply_reduce_outliers,
                reduce_kwargs=reduce_kwargs,
            )
    else:
        model, topics, probs, topic_terms_map = _train_new()

    if topic_terms_map is None:
        topic_terms_map = _get_topic_terms(model)

    # Etiquetas legibles
    topic_labels = model.get_topic_info().copy()
    topic_labels["Topic"] = topic_labels["Topic"].astype(int)
    topic_labels["TopTerms"] = topic_labels["Topic"].map(lambda tid: topic_terms_map.get(tid, []))
    topic_labels["Name"] = topic_labels.apply(
        lambda row: " / ".join(row["TopTerms"][:3]) if row["TopTerms"] else row.get("Name"),
        axis=1,
    )
    topic_name = {int(r["Topic"]): r["Name"] for _, r in topic_labels.iterrows()}
    topics_arr = np.asarray(topics, dtype=int)
    probs_arr = np.asarray(probs) if probs is not None else None
    outlier_mask = topics_arr == -1

    if lang_list_input is not None:
        lang_list = []
        for lang in lang_list_input:
            if lang is None:
                lang_list.append("")
                continue
            if isinstance(lang, str):
                token = lang.strip().lower()
                lang_list.append(token)
                continue
            lang_list.append(str(lang).strip().lower())
    else:
        lang_list = [""] * len(ids)

    # Ensamblar assignments
    assignments: List[Dict[str, Any]] = []
    for i, uid in enumerate(ids):
        tid = int(topics_arr[i])
        label = topic_name.get(tid, "unassigned")
        score = 0.0
        if probs_arr is not None and i < len(probs_arr) and len(probs_arr[i]):
            score = float(np.max(probs_arr[i]))
        assignments.append({
            "uid": uid,
            "topic_id": tid,
            "label": label,
            "score": score,
            "terms": topic_terms_map.get(tid, []),
            "lang": lang_list[i],
            "is_outlier": tid == -1,
        })

    n_docs = len(docs)
    n_outliers = int(outlier_mask.sum()) if n_docs else 0

    return {
        "assignments": assignments,
        "model": model,
        "model_info": {
            "n_documents": n_docs,
            "n_outliers": n_outliers,
            "outlier_ratio": float(n_outliers / n_docs) if n_docs else 0.0,
            "n_topics": int((topic_labels["Topic"] >= 0).sum()),
            "topics_table": topic_labels,
            "umap_params": umap_defaults,
            "hdbscan_params": hdbscan_defaults,
            "apply_reduce_outliers": apply_reduce_outliers,
        },
    }

def transform_docs(model: BERTopic, docs: List[str], ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Asigna tópicos usando un modelo ya entrenado (p. ej. producción).
    """
    if ids is None:
        ids = [str(i) for i in range(len(docs))]
    topics = None
    probs = None
    topic_terms_map = _get_topic_terms(model)
    if getattr(model, "embedding_model", None) is None:
        embed_name = getattr(model, "_bertopic_embedding_model_name", _DEFAULT_EMBEDDING_MODEL)
        try:
            embedder = SentenceTransformer(embed_name)
            model.set_embedding_model(embedder)
            topics, probs = model.transform(docs)
        except Exception:
            emb, _ = _build_embeddings(docs, model_name=embed_name)
            topics, probs = model.transform(docs, embeddings=emb)
    if topics is None or probs is None:
        topics, probs = model.transform(docs)
    topic_labels = model.get_topic_info().copy()
    topic_labels["Topic"] = topic_labels["Topic"].astype(int)
    topic_name = {}
    for _, row in topic_labels.iterrows():
        tid = int(row["Topic"])
        terms = topic_terms_map.get(tid, [])
        topic_name[tid] = " / ".join(terms[:3]) if terms else row.get("Name", str(tid))

    out = []
    for i, (uid, t) in enumerate(zip(ids, topics)):
        tid = int(t)
        label = topic_name.get(tid, "unassigned")
        score = float(np.max(probs[i])) if probs is not None and len(probs[i]) else 0.0
        out.append({
            "uid": uid,
            "topic_id": tid,
            "label": label,
            "score": score,
            "terms": topic_terms_map.get(tid, []),
        })
    return out

def summarize_topics(
    model: BERTopic,
    docs: List[str],
    ids: List[str],
    dts: List[str],
    assignments: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Devuelve resumen temporal por topic_id/label/fecha (YYYY-MM-DD):
    - volume, share, top_words
    """
    if assignments is None:
        assignments = transform_docs(model, docs, ids)
    df = pd.DataFrame(assignments)
    df["dt"] = pd.Series(dts).astype(str)
    # volumen por día y topic
    grp = df.groupby(["dt", "topic_id", "label"]).size().reset_index(name="volume")
    # share por día
    daily = grp.groupby("dt")["volume"].transform("sum")
    grp["share"] = grp["volume"] / daily

    terms_map = _get_topic_terms(model)
    grp["top_terms"] = grp["topic_id"].map(lambda t: terms_map.get(int(t), []))
    grp["top_words"] = grp["top_terms"].apply(lambda terms: ", ".join(terms))
    return grp.sort_values(["dt", "volume"], ascending=[True, False])
