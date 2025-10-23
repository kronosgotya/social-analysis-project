from __future__ import annotations
import os
import pandas as pd
from transformers import pipeline, AutoTokenizer
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

DEFAULT_EMOTIONS = [
    "joy", "anger", "sadness", "fear", "disgust",
    "surprise", "love", "optimism", "shame", "guilt"
]

_OFFLINE_FLAG = {"1", "true", "yes", "on"}

def _is_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _OFFLINE_FLAG

class EmotionClassifier:
    """
    Zero-shot multilingüe (XNLI). Por defecto modelo grande.
    Opcional ligero: 'MoritzLaurer/mMiniLMv2-L6-mnli-xnli'
    """
    def __init__(
        self,
        labels: Optional[List[str]] = None,
        device: Optional[int] = None,
        model_name: str = "joeddav/xlm-roberta-large-xnli"
    ):
        self.labels = labels or DEFAULT_EMOTIONS
        offline = _is_offline()
        if offline:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            local_files_only=offline,
        )
        self.clf = pipeline(
            task="zero-shot-classification",
            model=model_name,
            tokenizer=tokenizer,
            device=device if device is not None else -1,
        )

    def score_batch(self, texts: List[str]) -> List[Tuple[Optional[str], Dict[str, float]]]:
        if not texts:
            return []
        outputs = self.clf(texts, candidate_labels=self.labels, multi_label=False)
        if isinstance(outputs, dict):
            outputs = [outputs]

        results: List[Tuple[Optional[str], Dict[str, float]]] = []
        for out in outputs:
            labels = out.get("labels", []) or []
            scores = out.get("scores", []) or []
            dist = {lbl: float(score) for lbl, score in zip(labels, scores)}
            top = labels[0] if labels else None
            results.append((top, dist))
        return results


def _score_emotion_series(
    series: pd.Series,
    classifier: EmotionClassifier,
    labels: List[str],
    batch_size: int,
) -> Dict[int, Tuple[Optional[str], Dict[str, float]]]:
    results: Dict[int, Tuple[Optional[str], Dict[str, float]]] = {}
    buf: List[str] = []
    idxs: List[int] = []
    for idx, text in series.fillna("").items():
        value = str(text)
        if not value.strip():
            continue
        buf.append(value)
        idxs.append(idx)
        if len(buf) >= batch_size:
            batch_scores = classifier.score_batch(buf)
            for res, idx_out in zip(batch_scores, idxs):
                top, dist = res
                norm_dist = {lbl: float(dist.get(lbl, 0.0)) for lbl in labels}
                total = sum(norm_dist.values())
                if total > 0:
                    norm_dist = {k: v / total for k, v in norm_dist.items()}
                results[idx_out] = (top, norm_dist)
            buf, idxs = [], []
    if buf:
        batch_scores = classifier.score_batch(buf)
        for res, idx_out in zip(batch_scores, idxs):
            top, dist = res
            norm_dist = {lbl: float(dist.get(lbl, 0.0)) for lbl in labels}
            total = sum(norm_dist.values())
            if total > 0:
                norm_dist = {k: v / total for k, v in norm_dist.items()}
            results[idx_out] = (top, norm_dist)
    return results

def add_emotions(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    batch_size: int = 16,
    device: Optional[int] = None,
    labels: Optional[List[str]] = None,
    model_name: str = "joeddav/xlm-roberta-large-xnli",
    out_label_col: str = "emotion_label",
    out_scores_col: str = "emotion_scores",
    *,
    caption_col: Optional[str] = None,
    summary_col: Optional[str] = None,
    caption_weight: float = 0.8,
    summary_weight: float = 0.2,
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"No existe la columna de texto '{text_col}' en el DataFrame.")
    if caption_col and caption_col not in df.columns:
        raise ValueError(f"No existe la columna de caption '{caption_col}' en el DataFrame.")
    if summary_col and summary_col not in df.columns:
        raise ValueError(f"No existe la columna de summary '{summary_col}' en el DataFrame.")

    # Inicializa salidas
    if out_label_col not in df.columns:
        df[out_label_col] = pd.NA
    if out_scores_col not in df.columns:
        df[out_scores_col] = pd.NA

    # Índices con texto NO vacío
    label_list = labels or DEFAULT_EMOTIONS
    clf = EmotionClassifier(labels=label_list, device=device, model_name=model_name)

    if caption_col is None and summary_col is None:
        scores = _score_emotion_series(df[text_col], clf, label_list, batch_size)
        for idx, (top, dist) in scores.items():
            df.at[idx, out_label_col] = top
            df.at[idx, out_scores_col] = dist
        return df

    caption_series = (
        df[caption_col].fillna("") if caption_col else pd.Series([""] * len(df), index=df.index)
    )
    summary_series = (
        df[summary_col].fillna("") if summary_col else pd.Series([""] * len(df), index=df.index)
    )

    caption_scores = _score_emotion_series(caption_series, clf, label_list, batch_size) if caption_col else {}
    summary_scores = _score_emotion_series(summary_series, clf, label_list, batch_size) if summary_col else {}

    for idx in df.index:
        caption_text = str(caption_series.at[idx]).strip()
        summary_text = str(summary_series.at[idx]).strip()

        has_caption = bool(caption_text)
        has_summary = bool(summary_text)
        if not has_caption and not has_summary:
            continue

        cap_weight = caption_weight if has_caption else 0.0
        sum_weight = summary_weight if has_summary else 0.0
        if cap_weight == 0.0 and sum_weight == 0.0:
            continue
        if cap_weight > 0.0 and idx not in caption_scores:
            cap_weight = 0.0
        if sum_weight > 0.0 and idx not in summary_scores:
            sum_weight = 0.0
        if cap_weight == 0.0 and sum_weight == 0.0:
            continue

        if cap_weight > 0.0 and sum_weight > 0.0:
            total = cap_weight + sum_weight
            cap_weight /= total
            sum_weight /= total
        elif cap_weight > 0.0:
            cap_weight = 1.0
        elif sum_weight > 0.0:
            sum_weight = 1.0

        cap_dist = caption_scores[idx][1] if cap_weight > 0.0 else {}
        sum_dist = summary_scores[idx][1] if sum_weight > 0.0 else {}

        all_labels = set(cap_dist.keys()) | set(sum_dist.keys()) | set(label_list)
        combined = {}
        for label in all_labels:
            combined[label] = (
                cap_weight * float(cap_dist.get(label, 0.0))
                + sum_weight * float(sum_dist.get(label, 0.0))
            )
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}
        if not combined:
            continue
        best_label = max(combined, key=combined.get)
        df.at[idx, out_label_col] = best_label
        df.at[idx, out_scores_col] = combined

    return df
