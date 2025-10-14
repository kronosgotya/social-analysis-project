# -*- coding: utf-8 -*-
"""
stance_entity.py
Devuelve (stance, sentiment) para una ENTIDAD proporcionada por el usuario.
"""
from __future__ import annotations
from typing import Tuple

def stance_and_sentiment(text: str, entity: str) -> Tuple[str, float]:
    """
    Placeholder: lógica mínima. Sustituir por zero-shot o clasificador ligero.
    """
    if entity.lower() in text.lower():
        # heurística muy simple
        return "neu", 0.0
    return "neu", 0.0