from __future__ import annotations
import os
import pandas as pd
from transformers import pipeline, AutoTokenizer
from typing import List, Tuple, Dict, Optional
from pathlib import Path

_OFFLINE_FLAG = {"1", "true", "yes", "on"}
_DEFAULT_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
_FINETUNE_DIR = Path(__file__).resolve().parents[1] / "models" / "sentiment_finetuned"

def _is_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _OFFLINE_FLAG

class SentimentScorer:
    """
    Multilenguaje (redes sociales) con XLM-R.
    Usamos top_k=None (en lugar de return_all_scores) para evitar warnings.
    """
    def __init__(self, device: Optional[int] = None, max_length: int = 256, model_name: Optional[str] = None):
        offline = _is_offline()
        if offline:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        chosen_model = model_name or _DEFAULT_MODEL_NAME
        if model_name is None and _FINETUNE_DIR.exists():
            chosen_model = str(_FINETUNE_DIR)

        tokenizer = AutoTokenizer.from_pretrained(
            chosen_model,
            use_fast=False,
            local_files_only=offline,
        )
        self.clf = pipeline(
            task="text-classification",
            model=chosen_model,
            tokenizer=tokenizer,
            device=device if device is not None else -1,
            truncation=True,
            max_length=max_length,
            top_k=None,  # devuelve lista de dicts con todas las etiquetas
        )

    def score_batch(self, texts: List[str]) -> List[Tuple[str, float, Dict[str, float]]]:
        if not texts:
            return []
        outputs = self.clf(texts)
        # Normaliza: list[list[{'label':..., 'score':...}, ...]]
        if isinstance(outputs, dict):
            outputs = [outputs]
        if outputs and isinstance(outputs[0], dict):
            outputs = [[d for d in outputs]]

        results: List[Tuple[str, float, Dict[str, float]]] = []
        for scores in outputs:
            dist = {s["label"].lower(): float(s["score"]) for s in scores}
            label = max(dist, key=dist.get) if dist else "neutral"
            conf = dist.get(label, 0.0)
            results.append((label, conf, dist))
        return results


def _score_text_series(
    series: pd.Series,
    scorer: SentimentScorer,
    batch_size: int,
) -> Dict[int, Tuple[str, float, Dict[str, float]]]:
    results: Dict[int, Tuple[str, float, Dict[str, float]]] = {}
    buf: List[str] = []
    idxs: List[int] = []
    for idx, text in series.fillna("").items():
        value = str(text)
        if not value.strip():
            continue
        buf.append(value)
        idxs.append(idx)
        if len(buf) >= batch_size:
            batch_scores = scorer.score_batch(buf)
            for res, idx_out in zip(batch_scores, idxs):
                results[idx_out] = res
            buf, idxs = [], []
    if buf:
        batch_scores = scorer.score_batch(buf)
        for res, idx_out in zip(batch_scores, idxs):
            results[idx_out] = res
    return results

def add_sentiment(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    batch_size: int = 64,
    device: Optional[int] = None,
    max_length: int = 256,
    out_label_col: str = "sentiment_label",
    out_score_col: str = "sentiment_score",
    out_dist_col: str = "sentiment_dist",
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

    scorer = SentimentScorer(device=device, max_length=max_length)

    if caption_col is None and summary_col is None:
        scores = _score_text_series(df[text_col], scorer, batch_size)
        if scores:
            indices = list(scores.keys())
            df.loc[indices, out_label_col] = [scores[i][0] for i in indices]
            df.loc[indices, out_score_col] = [scores[i][1] for i in indices]
            df.loc[indices, out_dist_col] = [scores[i][2] for i in indices]
        return df

    caption_series = (
        df[caption_col].fillna("") if caption_col else pd.Series([""] * len(df), index=df.index)
    )
    summary_series = (
        df[summary_col].fillna("") if summary_col else pd.Series([""] * len(df), index=df.index)
    )

    caption_scores = _score_text_series(caption_series, scorer, batch_size) if caption_col else {}
    summary_scores = _score_text_series(summary_series, scorer, batch_size) if summary_col else {}

    combined_results: Dict[int, Tuple[str, float, Dict[str, float]]] = {}

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

        combined_dist: Dict[str, float] = {}
        if cap_weight > 0.0:
            _, _, cap_dist = caption_scores[idx]
        else:
            cap_dist = {}
        if sum_weight > 0.0:
            _, _, sum_dist = summary_scores[idx]
        else:
            sum_dist = {}

        all_labels = set(cap_dist.keys()) | set(sum_dist.keys())
        if not all_labels:
            continue

        for label in all_labels:
            combined_dist[label] = (
                cap_weight * float(cap_dist.get(label, 0.0))
                + sum_weight * float(sum_dist.get(label, 0.0))
            )
        if not combined_dist:
            continue
        best_label = max(combined_dist, key=combined_dist.get)
        best_score = combined_dist[best_label]
        combined_results[idx] = (best_label, best_score, combined_dist)

    if combined_results:
        indices = list(combined_results.keys())
        df.loc[indices, out_label_col] = [combined_results[i][0] for i in indices]
        df.loc[indices, out_score_col] = [combined_results[i][1] for i in indices]
        df.loc[indices, out_dist_col] = [combined_results[i][2] for i in indices]

    return df
