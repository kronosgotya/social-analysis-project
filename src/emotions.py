from __future__ import annotations
import pandas as pd
from transformers import pipeline
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

DEFAULT_EMOTIONS = [
    "joy", "anger", "sadness", "fear", "disgust",
    "surprise", "love", "optimism", "shame", "guilt"
]

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
        self.clf = pipeline(
            task="zero-shot-classification",
            model=model_name,
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

def add_emotions(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    batch_size: int = 16,
    device: Optional[int] = None,
    labels: Optional[List[str]] = None,
    model_name: str = "joeddav/xlm-roberta-large-xnli",
    out_label_col: str = "emotion_label",
    out_scores_col: str = "emotion_scores",
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"No existe la columna de texto '{text_col}' en el DataFrame.")

    # Inicializa salidas
    if out_label_col not in df.columns:
        df[out_label_col] = pd.NA
    if out_scores_col not in df.columns:
        df[out_scores_col] = pd.NA

    # Índices con texto NO vacío
    non_empty_idx, non_empty_texts = [], []
    for i, t in df[text_col].fillna("").items():
        s = str(t).strip()
        if s:
            non_empty_idx.append(i)
            non_empty_texts.append(s)

    if not non_empty_texts:
        return df

    clf = EmotionClassifier(labels=labels, device=device, model_name=model_name)

    # Procesar en lotes
    for start in tqdm(range(0, len(non_empty_texts), batch_size), desc="Emotions"):
        batch_texts = non_empty_texts[start:start + batch_size]
        batch_indices = non_empty_idx[start:start + batch_size]
        results = clf.score_batch(batch_texts)
        for (top, dist), idx in zip(results, batch_indices):
            df.at[idx, out_label_col] = top
            df.at[idx, out_scores_col] = dist

    return df
