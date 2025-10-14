# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional

def impact_score(entity: str,
                 sentiment: Optional[float],
                 stance: Optional[str],
                 emotion_label: Optional[str],
                 engagement: Optional[int],
                 recency_days: Optional[float],
                 topic_importance: Optional[float]) -> int:
    """
    Escala 0..100 (simple). Sustituible por LLM en n8n.
    """
    base = 50
    if sentiment is not None:
        base += int(20 * sentiment)  # -1..1 -> -20..+20
    if stance == "pos": base += 10
    elif stance == "neg": base -= 10
    if engagement: base += min(20, int((engagement ** 0.5)))
    if topic_importance: base += int(10 * topic_importance)
    return max(0, min(100, base))