from __future__ import annotations
import pandas as pd
from transformers import pipeline
from typing import List, Tuple, Dict, Optional

class SentimentScorer:
    """
    Multilenguaje (redes sociales) con XLM-R.
    Usamos top_k=None (en lugar de return_all_scores) para evitar warnings.
    """
    def __init__(self, device: Optional[int] = None, max_length: int = 256):
        self.clf = pipeline(
            task="text-classification",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
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

def add_sentiment(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    batch_size: int = 64,
    device: Optional[int] = None,
    max_length: int = 256,
    out_label_col: str = "sentiment_label",
    out_score_col: str = "sentiment_score",
    out_dist_col: str = "sentiment_dist",
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"No existe la columna de texto '{text_col}' en el DataFrame.")
    scorer = SentimentScorer(device=device, max_length=max_length)

    labels_idx, confs_idx, dists_idx = [], [], []
    buf, idxs = [], []

    for i, t in list(df[text_col].fillna("").items()):
        buf.append(str(t))
        idxs.append(i)
        if len(buf) >= batch_size:
            res = scorer.score_batch(buf)
            for (lab, conf, dist), i0 in zip(res, idxs):
                labels_idx.append((i0, lab))
                confs_idx.append((i0, conf))
                dists_idx.append((i0, dist))
            buf, idxs = [], []

    if buf:
        res = scorer.score_batch(buf)
        for (lab, conf, dist), i0 in zip(res, idxs):
            labels_idx.append((i0, lab))
            confs_idx.append((i0, conf))
            dists_idx.append((i0, dist))

    if labels_idx:
        df.loc[[i for i, _ in labels_idx], out_label_col] = [lab for _, lab in labels_idx]
    if confs_idx:
        df.loc[[i for i, _ in confs_idx], out_score_col] = [c for _, c in confs_idx]
    if dists_idx:
        df.loc[[i for i, _ in dists_idx], out_dist_col] = [d for _, d in dists_idx]
    return df
