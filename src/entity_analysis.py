"""
Herramientas para detección y análisis de entidades con sentimiento/emociones condicionadas.
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
from tqdm.auto import tqdm

from .entities_runtime import EntitySpec, load_entities
from .sentiment import SentimentScorer
from .emotions import EmotionClassifier, DEFAULT_EMOTIONS

PRIMARY_ENTITIES: Set[str] = {"NATO", "Russia"}
PRIMARY_ONLY_FOR_KPIS: bool = True

DEFAULT_ENTITY_DIR = Path(__file__).resolve().parent.parent / "data" / "entities"

DEFAULT_PRINCIPAL_CONFIG_PATH = DEFAULT_ENTITY_DIR / "principal_entities.json"
DEFAULT_RELATED_CONFIG_PATH = DEFAULT_ENTITY_DIR / "related_entities.json"

NATO_COUNTRIES = {
    "united states",
    "usa",
    "us",
    "united kingdom",
    "uk",
    "canada",
    "france",
    "germany",
    "italy",
    "spain",
    "poland",
    "romania",
    "lithuania",
    "latvia",
    "estonia",
    "norway",
    "sweden",
    "finland",
    "denmark",
    "netherlands",
    "belgium",
    "czech republic",
    "slovakia",
    "hungary",
    "croatia",
    "montenegro",
    "north macedonia",
    "slovenia",
    "bulgaria",
    "turkey",
    "greece",
    "portugal",
    "albania",
    "iceland",
    "luxembourg",
    "ukraine",
}

RUSSIA_COUNTRIES = {
    "russia",
    "russian federation",
    "belarus",
    "crimea",
    "donetsk",
    "luhansk",
    "lugansk",
    "dagestan",
    "chechnya",
    "syria",
}

_FALLBACK_PRINCIPAL_CONFIG: Dict[str, Dict[str, Any]] = {
    "NATO": {
        "weight": 8.0,
        "aliases": [
            {"alias": "NATO", "weight": 8.5},
            {"alias": "OTAN", "weight": 8.3},
            {"alias": "N.A.T.O", "weight": 7.5},
            {"alias": "North Atlantic Treaty Organization", "weight": 7.2},
            {"alias": "North Atlantic Alliance", "weight": 7.0},
            {"alias": "Organización del Tratado del Atlántico Norte", "weight": 7.0},
            {"alias": "Организация Североатлантического договора", "weight": 6.8},
            {"alias": "Організація Північноатлантичного договору", "weight": 6.5},
            {"alias": "НАТО", "weight": 8.0},
        ],
    },
    "Russia": {
        "weight": 8.0,
        "aliases": [
            {"alias": "Russia", "weight": 8.5},
            {"alias": "Rusia", "weight": 8.3},
            {"alias": "Federación Rusa", "weight": 7.5},
            {"alias": "Russian Federation", "weight": 7.5},
            {"alias": "Russian Fed", "weight": 6.8},
            {"alias": "Российская Федерация", "weight": 7.8},
            {"alias": "Российская федерация", "weight": 7.8},
            {"alias": "Российской Федерации", "weight": 7.6},
            {"alias": "России", "weight": 7.3},
            {"alias": "Россия", "weight": 7.3},
            {"alias": "РФ", "weight": 7.9},
            {"alias": "россия", "weight": 7.0},
            {"alias": "российская федерация", "weight": 7.2},
        ],
    },
}


def _fallback_principal_specs() -> List[EntitySpec]:
    specs: List[EntitySpec] = []
    for norm, config in _FALLBACK_PRINCIPAL_CONFIG.items():
        metadata = {"source": "fallback_principal"}
        metadata.update(config.get("metadata") or {})
        specs.append(
            EntitySpec(
                name=config.get("name") or norm,
                entity_norm=norm,
                type="principal",
                weight=config.get("weight"),
                aliases=config.get("aliases"),
                metadata=metadata,
            )
        )
    return specs


@dataclass(frozen=True)
class AliasMatcher:
    alias: str
    entity_norm: str
    entity_label: str
    entity_type: str
    weight: float
    priority: float
    pattern: re.Pattern
    metadata: Dict[str, Any]
    linked_principal: Optional[str]


@dataclass
class EntityDefinition:
    norm: str
    label: str
    entity_type: Optional[str]
    aliases: Dict[str, float] = field(default_factory=dict)
    alias_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    weight: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityDictionary:
    matchers: List[AliasMatcher] = field(default_factory=list)
    entity_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    alias_lookup: Dict[str, AliasMatcher] = field(default_factory=dict)

    def entity_type(self, entity_norm: str) -> Optional[str]:
        info = self.entity_meta.get(entity_norm)
        return info.get("type") if info else None


def _normalize_entity_type(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    norm = str(value).strip().lower()
    if not norm:
        return None
    if norm in {"principal", "primary"}:
        return "principal"
    if norm in {"related", "secondary", "context"}:
        return "related"
    return norm


def _default_entity_weight(entity_type: Optional[str], base_weight: Optional[float]) -> float:
    if base_weight is not None:
        try:
            return float(base_weight)
        except (TypeError, ValueError):
            pass
    return 8.0 if entity_type == "principal" else 1.0


def _alias_priority(weight: float, entity_type: Optional[str], alias: str, metadata: Optional[Dict[str, Any]]) -> float:
    priority = float(weight)
    if entity_type == "principal":
        priority += 50.0
    priority += min(len(alias), 64) * 0.05
    if metadata:
        extra = metadata.get("priority")
        if extra is not None:
            try:
                priority += float(extra)
            except (TypeError, ValueError):
                pass
    return priority


def _resolve_regex_flags(case_sensitive: bool, extra_flags: Optional[Any]) -> int:
    flags = re.UNICODE
    if not case_sensitive:
        flags |= re.IGNORECASE
    if not extra_flags:
        return flags
    if isinstance(extra_flags, str):
        flag_names: Iterable[str] = [part.strip() for part in extra_flags.split("|") if part.strip()]
    elif isinstance(extra_flags, Iterable):
        flag_names = [str(part).strip() for part in extra_flags if str(part).strip()]
    else:
        return flags
    for name in flag_names:
        attr = getattr(re, name, None)
        if isinstance(attr, int):
            flags |= attr
    return flags


def _compile_alias_pattern(alias: str, metadata: Optional[Dict[str, Any]]) -> re.Pattern:
    metadata = metadata or {}
    case_sensitive = bool(metadata.get("case_sensitive"))
    flags = _resolve_regex_flags(case_sensitive, metadata.get("flags"))
    pattern_text = metadata.get("pattern") or metadata.get("regex")
    if pattern_text:
        return re.compile(pattern_text, flags)
    boundary = metadata.get("boundary", True)
    if boundary:
        pattern_text = rf"(?<!\\w){re.escape(alias)}(?!\\w)"
    else:
        pattern_text = re.escape(alias)
    return re.compile(pattern_text, flags)


def _infer_linked_principal(definition: EntityDefinition) -> Optional[str]:
    if definition.entity_type == "principal" and definition.norm in PRIMARY_ENTITIES:
        return definition.norm

    meta = dict(definition.metadata or {})
    direct = meta.get("linked_principal")
    if isinstance(direct, str):
        direct_clean = direct.strip()
        if direct_clean in PRIMARY_ENTITIES:
            return direct_clean

    category = str(meta.get("category", "")).lower()
    if "nato" in category or "ally" in category or "partner" in category:
        return "NATO"
    if "russia" in category or "kremlin" in category:
        return "Russia"

    country = str(meta.get("country", "")).lower()
    if country in NATO_COUNTRIES:
        return "NATO"
    if country in RUSSIA_COUNTRIES:
        return "Russia"

    alliance = str(meta.get("alliance", "")).lower()
    if "nato" in alliance:
        return "NATO"
    if "russia" in alliance:
        return "Russia"

    return None


def _merge_definition(definition: EntityDefinition, spec: EntitySpec) -> None:
    spec_type = _normalize_entity_type(spec.type)
    if spec_type == "principal":
        definition.entity_type = "principal"
    elif spec_type and not definition.entity_type:
        definition.entity_type = spec_type

    if spec.name:
        name_clean = str(spec.name).strip()
        if name_clean and definition.label == definition.norm:
            definition.label = name_clean

    if spec.weight is not None:
        try:
            candidate_weight = float(spec.weight)
        except (TypeError, ValueError):
            candidate_weight = None
        if candidate_weight is not None:
            if definition.weight is None or candidate_weight > definition.weight:
                definition.weight = candidate_weight

    definition.metadata.update(spec.metadata or {})

    base_weight = _default_entity_weight(definition.entity_type, definition.weight)
    for alias in spec.aliases or []:
        alias_text = str(alias).strip()
        if not alias_text:
            continue
        alias_weight = spec.alias_weights.get(alias_text)
        if alias_weight is None:
            alias_weight = base_weight
        else:
            try:
                alias_weight = float(alias_weight)
            except (TypeError, ValueError):
                alias_weight = base_weight
        current = definition.aliases.get(alias_text)
        if current is None or alias_weight > current:
            definition.aliases[alias_text] = alias_weight
        alias_meta = spec.alias_metadata.get(alias_text)
        if alias_meta:
            definition.alias_metadata.setdefault(alias_text, {}).update(alias_meta)


def _default_principal_entity_specs(logger: Optional[logging.Logger] = None) -> List[EntitySpec]:
    logger = logger or logging.getLogger(__name__)
    if DEFAULT_PRINCIPAL_CONFIG_PATH.exists():
        try:
            file_specs = load_entities(None, str(DEFAULT_PRINCIPAL_CONFIG_PATH))
            normalized: List[EntitySpec] = []
            for spec in file_specs:
                spec.type = spec.type or "principal"
                spec.entity_norm = spec.entity_norm or spec.name
                normalized.append(spec)
            if normalized:
                logger.debug(
                    "Cargando entidades principales desde %s (%d entradas)",
                    DEFAULT_PRINCIPAL_CONFIG_PATH,
                    len(normalized),
                )
                return normalized
            logger.warning(
                "El fichero %s no contiene entidades válidas; se usará el fallback",
                DEFAULT_PRINCIPAL_CONFIG_PATH,
            )
        except Exception as exc:  # pragma: no cover - logging path
            logger.warning(
                "No se pudo cargar %s (%s); se usará el fallback",
                DEFAULT_PRINCIPAL_CONFIG_PATH,
                exc,
            )
    return _fallback_principal_specs()


def _default_related_entity_specs(logger: Optional[logging.Logger] = None) -> List[EntitySpec]:
    logger = logger or logging.getLogger(__name__)
    if DEFAULT_RELATED_CONFIG_PATH.exists():
        try:
            file_specs = load_entities(None, str(DEFAULT_RELATED_CONFIG_PATH))
            normalized: List[EntitySpec] = []
            for spec in file_specs:
                spec.type = spec.type or "related"
                spec.entity_norm = spec.entity_norm or spec.name
                normalized.append(spec)
            if normalized:
                logger.debug(
                    "Cargando entidades relacionadas desde %s (%d entradas)",
                    DEFAULT_RELATED_CONFIG_PATH,
                    len(normalized),
                )
                return normalized
        except Exception as exc:  # pragma: no cover - logging path
            logger.warning(
                "No se pudo cargar %s (%s); entidades relacionadas por defecto no disponibles",
                DEFAULT_RELATED_CONFIG_PATH,
                exc,
            )
    return []


def build_entity_dictionary(
    entities: Sequence[EntitySpec],
    *,
    ensure_principal: bool = True,
    logger: Optional[logging.Logger] = None,
) -> EntityDictionary:
    logger = logger or logging.getLogger(__name__)
    specs: List[EntitySpec] = list(entities)
    if ensure_principal:
        specs.extend(_default_principal_entity_specs(logger=logger))
    specs.extend(_default_related_entity_specs(logger=logger))

    definitions: Dict[str, EntityDefinition] = {}
    for spec in specs:
        norm_candidate = spec.entity_norm or spec.name
        if not norm_candidate:
            continue
        norm_clean = str(norm_candidate).strip()
        if not norm_clean:
            continue
        key = norm_clean.casefold()
        definition = definitions.get(key)
        if definition is None:
            label = str(spec.name).strip() if spec.name else norm_clean
            definition = EntityDefinition(
                norm=norm_clean,
                label=label or norm_clean,
                entity_type=_normalize_entity_type(spec.type),
            )
            definitions[key] = definition
        _merge_definition(definition, spec)

    if not definitions:
        return EntityDictionary()

    alias_lookup: Dict[str, AliasMatcher] = {}
    entity_meta: Dict[str, Dict[str, Any]] = {}

    for definition in definitions.values():
        if definition.entity_type == "principal" and definition.norm not in PRIMARY_ENTITIES:
            if logger:
                logger.warning(
                    "Entity '%s' marked as principal but not in PRIMARY_ENTITIES; demoting to related.",
                    definition.norm,
                )
            definition.entity_type = "related"

        entity_type = definition.entity_type or "related"
        base_weight = _default_entity_weight(entity_type, definition.weight)

        linked_principal = _infer_linked_principal(definition)
        if entity_type == "principal":
            linked_principal = definition.norm

        if definition.norm not in definition.aliases:
            definition.aliases[definition.norm] = base_weight
        if definition.label and definition.label not in definition.aliases:
            definition.aliases.setdefault(definition.label, base_weight)

        alias_names = sorted(definition.aliases.keys(), key=lambda a: a.casefold())
        entity_meta[definition.norm] = {
            "type": entity_type,
            "label": definition.label,
            "aliases": alias_names,
            "metadata": dict(definition.metadata),
            "base_weight": base_weight,
            "linked_principal": linked_principal,
        }

        for alias, weight in definition.aliases.items():
            alias_meta = definition.alias_metadata.get(alias, {})
            effective_weight = float(weight) if weight is not None else base_weight
            pattern = _compile_alias_pattern(alias, alias_meta)
            priority = _alias_priority(effective_weight, entity_type, alias, alias_meta)
            matcher = AliasMatcher(
                alias=alias,
                entity_norm=definition.norm,
                entity_label=definition.label,
                entity_type=entity_type,
                weight=effective_weight,
                priority=priority,
                pattern=pattern,
                metadata=dict(alias_meta),
                linked_principal=linked_principal,
            )

            alias_key = alias.casefold()
            existing = alias_lookup.get(alias_key)
            if existing is None:
                alias_lookup[alias_key] = matcher
                continue

            replace = False
            if matcher.priority > existing.priority:
                replace = True
            elif matcher.priority == existing.priority:
                if matcher.weight > existing.weight:
                    replace = True
                elif matcher.weight == existing.weight and matcher.entity_type == "principal" and existing.entity_type != "principal":
                    replace = True

            if replace:
                if logger and matcher.entity_norm != existing.entity_norm:
                    logger.debug(
                        "Alias '%s' reasignado de %s a %s", alias, existing.entity_norm, matcher.entity_norm
                    )
                alias_lookup[alias_key] = matcher

    matchers = sorted(
        alias_lookup.values(),
        key=lambda m: (-m.priority, -len(m.alias), m.alias.casefold()),
    )

    return EntityDictionary(
        matchers=matchers,
        entity_meta=entity_meta,
        alias_lookup=alias_lookup,
    )


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _default_zero(value: Optional[float]) -> float:
    return float(value) if value is not None else 0.0


def _stance_bucket(value: float) -> str:
    if value > 0.15:
        return "positive"
    if value < -0.15:
        return "negative"
    return "neutral"


def _stance_code(label: str) -> str:
    return {"positive": "pos", "negative": "neg"}.get(label, "neu")


def _max_emotion_prob(emotion_scores: object) -> float:
    if isinstance(emotion_scores, dict) and emotion_scores:
        try:
            return max(float(v) for v in emotion_scores.values())
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _impact_score_from_row(row: pd.Series) -> float:
    stance_val = float(row.get("stance_value") or 0.0)
    sentiment_conf = float(row.get("sentiment_score") or 0.0)
    if sentiment_conf <= 0.0 or stance_val == 0.0:
        return 0.0

    topic_score = row.get("topic_score")
    topic_weight = 1.0
    if topic_score not in (None, "", "nan"):
        try:
            topic_f = float(topic_score)
            if not math.isnan(topic_f):
                topic_weight = 0.6 + 0.4 * max(0.0, min(1.0, topic_f))
        except (TypeError, ValueError):
            topic_weight = 1.0

    engagement_total = _default_zero(row.get("engagement"))
    reach_total = _default_zero(row.get("reach"))
    engagement_weight = 1.0 + math.log1p(max(0.0, engagement_total) + max(0.0, reach_total) * 0.25)

    emotion_peak = _max_emotion_prob(row.get("emotion_scores"))
    emotion_weight = 0.7 + 0.3 * max(0.0, min(1.0, emotion_peak))

    raw = stance_val * sentiment_conf * topic_weight * emotion_weight * engagement_weight
    # Bound the tail so extremely viral posts do not explode the scale
    return float(max(min(raw, 25.0), -25.0))


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
    engagement: Optional[float]
    reach: Optional[float]
    likes: Optional[float]
    shares: Optional[float]
    replies: Optional[float]
    quotes: Optional[float]
    views: Optional[float]
    entity_norm: Optional[str] = None
    entity_type: Optional[str] = None
    entity_label: Optional[str] = None
    alias_weight: Optional[float] = None
    matched_text: Optional[str] = None
    confidence: Optional[float] = None
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    matched_alias: Optional[str] = None
    detector: str = "alias"
    text_source: Optional[str] = None
    ner_group: Optional[str] = None
    linked_principal: Optional[str] = None

def _snippet(text: str, needle: str, window: int = 160) -> str:
    """
    Extrae ventana de texto alrededor de la mención; fallback al texto completo si no se encuentra.
    """
    clean = text.strip()
    if not clean:
        return clean
    target = (needle or "").strip()
    if not target:
        return clean
    text_lower = clean.lower()
    needle_lower = target.lower()
    idx = text_lower.find(needle_lower)
    if idx == -1:
        return clean
    start = max(0, idx - window)
    end = min(len(clean), idx + len(target) + window)
    snip = clean[start:end].strip()
    if start > 0:
        snip = "... " + snip
    if end < len(clean):
        snip = snip + " ..."
    return snip


def _normalize_optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8", errors="ignore").strip()
        except Exception:
            text = str(value).strip()
        return text or None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def _normalize_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(candidate):
        return None
    try:
        return int(candidate)
    except (OverflowError, ValueError):
        return None


def _span_overlaps(span: Tuple[int, int], existing: List[Tuple[int, int]]) -> bool:
    start, end = span
    for estart, eend in existing:
        if start < eend and end > estart:
            return True
    return False


def extract_entity_mentions(
    df: pd.DataFrame,
    entities: Sequence[EntitySpec],
    *,
    text_col: str = "text_clean",
    id_col: str = "item_id",
    topic_id_col: str = "topic_id",
    topic_label_col: str = "topic_label",
    context_window: int = 160,
    entity_dict: Optional[EntityDictionary] = None,
    ensure_principal: bool = True,
    logger: Optional[logging.Logger] = None,
    use_ner: bool = False,
    ner_model: str = "Davlan/xlm-roberta-base-ner-hrl",
    ner_device: Optional[int] = None,
    ner_aggregation: str = "simple",
    ner_allowed_groups: Optional[Sequence[str]] = None,
    ner_min_score: float = 0.6,
    ner_allow_unmatched: bool = False,
    caption_col: Optional[str] = None,
    summary_col: Optional[str] = None,
    caption_weight: float = 0.8,
    summary_weight: float = 0.2,
) -> List[MentionCandidate]:
    mentions: List[MentionCandidate] = []
    if df.empty:
        return mentions

    logger = logger or logging.getLogger(__name__)
    entity_list = list(entities) if entities else []

    if entity_dict is None:
        entity_dict = build_entity_dictionary(entity_list, ensure_principal=ensure_principal, logger=logger)

    if not entity_dict.matchers:
        return mentions

    ner_pipeline = None
    ner_inference_failed = False
    allowed_groups_set: Optional[Set[str]] = None
    if use_ner:
        if ner_allowed_groups:
            allowed_groups_set = {str(group).upper() for group in ner_allowed_groups if str(group).strip()}
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore import

            ner_pipeline = hf_pipeline(
                "ner",
                model=ner_model,
                aggregation_strategy=ner_aggregation,
                device=ner_device,
            )
            logger.info(
                "NER activado con el modelo %s (aggregation=%s)", ner_model, ner_aggregation
            )
        except Exception as exc:  # pragma: no cover - runtime env dependent
            logger.warning(
                "No se pudo inicializar el modelo NER '%s': %s. Se continuará sin NER.",
                ner_model,
                exc,
            )
            ner_pipeline = None

    if caption_col is None and summary_col is None and text_col not in df.columns:
        raise ValueError(f"No existe la columna de texto '{text_col}' en el DataFrame.")

    if caption_col and caption_col not in df.columns:
        raise ValueError(f"No existe la columna '{caption_col}' en el DataFrame.")
    if summary_col and summary_col not in df.columns:
        raise ValueError(f"No existe la columna '{summary_col}' en el DataFrame.")

    text_source_specs: List[Tuple[str, float, str]] = []
    if caption_col:
        text_source_specs.append((caption_col, caption_weight, "caption"))
    if summary_col:
        text_source_specs.append((summary_col, summary_weight, "summary"))
    if not text_source_specs:
        text_source_specs.append((text_col, 1.0, text_col))

    for _, row in df.iterrows():
        available_sources: List[Tuple[str, float, str]] = []
        for col_name, base_weight, source_tag in text_source_specs:
            raw_value = row.get(col_name)
            text_value = str(raw_value) if raw_value is not None else ""
            if text_value.strip():
                available_sources.append((text_value, base_weight, source_tag))

        if not available_sources:
            continue

        total_weight = sum(weight for _, weight, _ in available_sources)
        if total_weight <= 0.0:
            total_weight = float(len(available_sources))
        normalized_sources = [
            (text_value, (weight / total_weight) if total_weight > 0 else (1.0 / len(available_sources)), source_tag)
            for text_value, weight, source_tag in available_sources
        ]

        item_id_value = (
            _normalize_optional_str(row.get(id_col))
            or _normalize_optional_str(row.get("item_id"))
            or ""
        )
        source_val = _normalize_optional_str(row.get("source"))
        timestamp_val = _normalize_optional_str(row.get("timestamp"))
        lang_val = _normalize_optional_str(row.get("lang"))
        topic_label_val = _normalize_optional_str(row.get(topic_label_col))
        topic_id_val = _normalize_optional_int(row.get(topic_id_col))
        topic_score_val = _coerce_float(row.get("topic_score"))

        likes_val = _default_zero(_coerce_float(row.get("likes")))
        shares_val = sum(
            _default_zero(_coerce_float(row.get(col)))
            for col in (
                "retweets",
                "telegram_forwards",
                "forward_count",
                "forwards",
                "shares",
            )
        )
        replies_val = _default_zero(_coerce_float(row.get("replies")))
        quotes_val = _default_zero(_coerce_float(row.get("quotes")))
        engagement_val = _coerce_float(row.get("engagement"))
        if engagement_val is None:
            engagement_val = likes_val + shares_val + replies_val + quotes_val
        views_val = _coerce_float(row.get("views"))
        reach_val = _coerce_float(row.get("reach"))
        if reach_val is None:
            reach_val = views_val

        for text, normalized_weight, source_tag in normalized_sources:
            text_lower = text.casefold()
            spans_by_entity: Dict[str, List[Tuple[int, int]]] = {}

            for matcher in entity_dict.matchers:
                if not matcher.metadata.get("pattern") and matcher.alias.casefold() not in text_lower:
                    continue
                matches = list(matcher.pattern.finditer(text))
                if not matches:
                    continue
                entity_spans = spans_by_entity.setdefault(matcher.entity_norm, [])
                for match in matches:
                    span = (match.start(), match.end())
                    if _span_overlaps(span, entity_spans):
                        continue
                    entity_spans.append(span)
                    matched_text = match.group(0)
                    snippet = _snippet(text, matched_text, window=context_window)

                    mentions.append(
                        MentionCandidate(
                            item_id=item_id_value,
                            entity=matcher.entity_norm,
                            alias=matcher.alias,
                            source=source_val,
                            timestamp=timestamp_val,
                            topic_id=topic_id_val,
                            topic_label=topic_label_val,
                            topic_score=topic_score_val,
                            lang=lang_val,
                            full_text=text,
                            snippet=snippet,
                            engagement=engagement_val,
                            reach=reach_val,
                            likes=likes_val,
                            shares=shares_val,
                            replies=replies_val,
                            quotes=quotes_val,
                            views=views_val,
                            entity_norm=matcher.entity_norm,
                            entity_type=matcher.entity_type,
                            entity_label=matcher.entity_label,
                            alias_weight=matcher.weight,
                            matched_text=matched_text,
                            confidence=matcher.weight * normalized_weight,
                            span_start=span[0],
                            span_end=span[1],
                            matched_alias=matcher.alias,
                            detector="alias",
                            text_source=source_tag,
                            linked_principal=matcher.linked_principal,
                        )
                    )

            if ner_pipeline and not ner_inference_failed:
                try:
                    ner_results = ner_pipeline(text)
                except Exception as exc:  # pragma: no cover - runtime env dependent
                    logger.warning("Error ejecutando NER: %s. Se deshabilita para el resto del proceso.", exc)
                    ner_inference_failed = True
                    ner_results = []
                for ner_match in ner_results:
                    score_val = _coerce_float(ner_match.get("score")) or 0.0
                    if score_val < ner_min_score:
                        continue
                    group_label_raw = ner_match.get("entity_group") or ner_match.get("entity")
                    group_label = str(group_label_raw).upper() if group_label_raw else None
                    if allowed_groups_set and group_label and group_label not in allowed_groups_set:
                        continue
                    matched_text = _normalize_optional_str(ner_match.get("word") or ner_match.get("text"))
                    if not matched_text:
                        continue
                    start_raw = ner_match.get("start")
                    end_raw = ner_match.get("end")
                    span_start = int(start_raw) if isinstance(start_raw, (int, float)) else None
                    span_end = int(end_raw) if isinstance(end_raw, (int, float)) else None
                    if span_start is None or span_end is None or span_end <= span_start:
                        idx = text.lower().find(matched_text.lower())
                        if idx != -1:
                            span_start, span_end = idx, idx + len(matched_text)
                    entity_norm_val: Optional[str] = None
                    entity_type_val: str = "related"
                    entity_label_val: str = matched_text
                    alias_value = matched_text
                    alias_weight_val = score_val

                    linked_principal_val: Optional[str] = None
                    alias_matcher = entity_dict.alias_lookup.get(matched_text.casefold())
                    if alias_matcher:
                        entity_norm_val = alias_matcher.entity_norm
                        entity_type_val = alias_matcher.entity_type
                        entity_label_val = alias_matcher.entity_label
                        alias_value = alias_matcher.alias
                        alias_weight_val = alias_matcher.weight
                        linked_principal_val = alias_matcher.linked_principal
                    elif not ner_allow_unmatched:
                        continue
                    else:
                        entity_norm_val = matched_text
                        linked_principal_val = entity_dict.entity_meta.get(
                            entity_norm_val, {}
                        ).get("linked_principal")

                    entity_key = entity_norm_val or matched_text
                    entity_spans = spans_by_entity.setdefault(entity_key, [])
                    span_tuple: Optional[Tuple[int, int]] = None
                    if span_start is not None and span_end is not None and span_end > span_start:
                        span_tuple = (span_start, span_end)
                        if _span_overlaps(span_tuple, entity_spans):
                            continue
                        entity_spans.append(span_tuple)

                    mentions.append(
                        MentionCandidate(
                            item_id=item_id_value,
                            entity=entity_norm_val or matched_text,
                            alias=alias_value,
                            source=source_val,
                            timestamp=timestamp_val,
                            topic_id=topic_id_val,
                            topic_label=topic_label_val,
                            topic_score=topic_score_val,
                            lang=lang_val,
                            full_text=text,
                            snippet=_snippet(text, matched_text, window=context_window),
                            engagement=engagement_val,
                            reach=reach_val,
                            likes=likes_val,
                            shares=shares_val,
                            replies=replies_val,
                            quotes=quotes_val,
                            views=views_val,
                            entity_norm=entity_norm_val or matched_text,
                            entity_type=entity_type_val,
                            entity_label=entity_label_val,
                            alias_weight=alias_weight_val,
                            matched_text=matched_text,
                            confidence=score_val,
                            span_start=span_start,
                            span_end=span_end,
                            matched_alias=alias_value,
                            detector="ner",
                            ner_group=group_label,
                            text_source=source_tag,
                            linked_principal=linked_principal_val,
                        )
                    )
    return mentions


def mentions_to_dataframe(mentions: Sequence[MentionCandidate]) -> pd.DataFrame:
    """Convierte la lista de menciones detectadas en un DataFrame normalizado."""

    base_columns = [
        "item_id",
        "entity_norm",
        "entity_type",
        "entity_label",
        "matched_alias",
        "matched_text",
        "alias_weight",
        "confidence",
        "detector",
        "ner_group",
        "span_start",
        "span_end",
        "source",
        "lang",
        "timestamp",
        "topic_id",
        "topic_label",
        "topic_score",
        "engagement",
        "reach",
        "likes",
        "shares",
        "replies",
        "quotes",
        "views",
        "full_text",
        "snippet",
        "text_source",
        "linked_principal",
    ]

    if not mentions:
        return pd.DataFrame(columns=base_columns)

    df = pd.DataFrame([m.__dict__ for m in mentions])

    if "entity_norm" not in df.columns or df["entity_norm"].isna().all():
        df["entity_norm"] = df.get("entity")
    if "entity" not in df.columns:
        df["entity"] = df["entity_norm"]
    else:
        df["entity"] = df["entity_norm"].fillna(df["entity"])

    if "entity_type" not in df.columns:
        df["entity_type"] = "related"
    df["entity_type"] = (
        df["entity_type"].fillna("related").astype(str).str.strip().str.lower().replace({"": "related"})
    )

    df["matched_alias"] = df.get("matched_alias", df.get("alias"))
    if "matched_alias" not in df.columns or df["matched_alias"].isna().all():
        df["matched_alias"] = df.get("alias")

    if "alias_weight" in df.columns:
        df["alias_weight"] = df["alias_weight"].astype(float, errors="ignore")
    else:
        df["alias_weight"] = df.get("confidence")

    if "confidence" not in df.columns or df["confidence"].isna().all():
        df["confidence"] = df["alias_weight"]

    df["detector"] = df.get("detector", "alias").fillna("alias").astype(str)
    df["ner_group"] = df.get("ner_group")
    if "text_source" not in df.columns:
        df["text_source"] = None

    df["matched_text"] = df.get("matched_text")
    df["span_start"] = df.get("span_start")
    df["span_end"] = df.get("span_end")

    for col in base_columns:
        if col not in df.columns:
            df[col] = None

    df = df[base_columns]
    return df


def _select_unique_mentions(
    mentions_df: pd.DataFrame,
    *,
    include_topic: bool = False,
) -> pd.DataFrame:
    """Selecciona una única fila por entidad-item (y tópico opcionalmente)."""

    if mentions_df.empty:
        return mentions_df.copy()

    working = mentions_df.copy()
    if "alias_weight" in working.columns:
        working["alias_weight"] = working["alias_weight"].fillna(0.0)
    else:
        working["alias_weight"] = 0.0

    sort_cols = ["entity_type", "entity_norm", "item_id", "alias_weight"]
    ascending = [True, True, True, False]
    if include_topic:
        sort_cols.insert(3, "topic_id")
        ascending.insert(3, True)

    working = working.sort_values(by=sort_cols, ascending=ascending)

    subset_cols = ["entity_type", "entity_norm", "item_id"]
    if include_topic:
        subset_cols.append("topic_id")

    return working.drop_duplicates(subset=subset_cols, keep="first")


def _infer_period_bounds(posts_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Intenta inferir el rango temporal (YYYY-MM-DD) disponible en el dataset."""

    candidate_cols = [
        "date",
        "timestamp",
        "published_at",
        "datetime",
        "created_at",
    ]

    for col in candidate_cols:
        if col not in posts_df.columns:
            continue
        series = pd.to_datetime(posts_df[col], errors="coerce", utc=True)
        if series.notna().any():
            start = series.min().date().isoformat()
            end = series.max().date().isoformat()
            return start, end
    return None, None


def _group_weighted_mean(
    grouped: pd.core.groupby.generic.DataFrameGroupBy, value_col: str, weight_col: str
) -> pd.Series:
    def _calc(sub_df: pd.DataFrame) -> float:
        if value_col not in sub_df.columns:
            return float("nan")
        weights = sub_df[weight_col].fillna(0.0)
        values = sub_df[value_col]
        mask = values.notna() & weights.notna()
        if not mask.any():
            return float("nan")
        weights = weights[mask]
        values = values[mask]
        total = float(weights.sum())
        if total <= 0:
            return float("nan")
        return float((values * weights).sum() / total)

    return grouped.apply(_calc)


def compute_entity_summary(
    posts_df: pd.DataFrame,
    mentions_df: pd.DataFrame,
    *,
    weight_col: str = "engagement",
    min_weight: float = 1.0,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Calcula métricas KPI para entidades principales."""

    columns = [
        "entity_norm",
        "freq",
        "sentiment_mean",
        "sentiment_mean_weighted",
        "related_entities",
        "top_emotion_weighted",
        "period_start",
        "period_end",
    ]

    logger = logger or logging.getLogger(__name__)

    if mentions_df.empty:
        return pd.DataFrame(columns=columns)

    work = mentions_df.copy()
    if "entity_type" not in work.columns:
        work["entity_type"] = "related"
    work["entity_type"] = work["entity_type"].astype(str).str.lower()

    target_mentions = work[work["entity_type"] == "principal"] if PRIMARY_ONLY_FOR_KPIS else work
    if target_mentions.empty:
        return pd.DataFrame(columns=columns)

    unique_mentions = _select_unique_mentions(target_mentions, include_topic=False)

    joined = unique_mentions.merge(
        posts_df,
        on="item_id",
        how="left",
        suffixes=("", "_post"),
        indicator=True,
    )

    if joined.shape[0] != unique_mentions.shape[0]:
        raise AssertionError("Unexpected row growth after joining mentions with posts by item_id")

    missing_posts = int(joined["_merge"].value_counts().get("left_only", 0))
    if missing_posts:
        logger.warning("%s menciones carecen de fila en el dataset principal tras el join", missing_posts)

    joined = joined.drop(columns=["_merge"])

    for col in ["topic_id", "topic_label", "lang", "timestamp"]:
        post_col = f"{col}_post"
        if post_col in joined.columns:
            joined[col] = joined[col].where(joined[col].notna(), joined[post_col])
            joined = joined.drop(columns=[post_col])

    if "sentiment_score" not in joined.columns:
        raise KeyError("sentiment_score column required in posts_df to compute entity summary")

    weight_source = weight_col if weight_col in joined.columns else None
    if weight_source is None or joined[weight_source].isna().all():
        if "impact_score" in joined.columns and not joined["impact_score"].isna().all():
            weight_source = "impact_score"
            logger.info("Usando 'impact_score' como peso al no disponer de '%s'", weight_col)
        else:
            weight_source = None
            logger.info("No se encontró columna de peso válida; se utilizará peso=1")

    if weight_source:
        weight_series = joined[weight_source]
        missing_weights = int(weight_series.isna().sum())
        if missing_weights:
            logger.info("%d mensajes con %s nulo; peso forzado a %s", missing_weights, weight_source, min_weight)
        weights = weight_series.fillna(0.0).abs()
        weights = weights.clip(lower=min_weight)
    else:
        weights = pd.Series(min_weight, index=joined.index)

    joined["__weight__"] = weights

    grouped = joined.groupby("entity_norm", dropna=False)

    freq = grouped["item_id"].nunique().astype(int)
    sentiment_mean = grouped["sentiment_score"].mean()
    sentiment_weighted = _group_weighted_mean(grouped, "sentiment_score", "__weight__")

    summary = pd.DataFrame({
        "entity_norm": freq.index,
        "freq": freq.values,
        "sentiment_mean": sentiment_mean.values,
        "sentiment_mean_weighted": sentiment_weighted.values,
    })

    related_lists: Dict[str, List[str]] = {}
    if "linked_principal" in work.columns and "entity_norm" in work.columns:
        related_mentions = work[work["entity_type"] == "related"].copy()
        if not related_mentions.empty:
            related_mentions["linked_principal"] = related_mentions["linked_principal"].astype(str).str.strip()
            filtered = related_mentions[related_mentions["linked_principal"].isin(PRIMARY_ENTITIES)]
            if not filtered.empty:
                grouped_related = filtered.groupby("linked_principal")["entity_norm"].agg(
                    lambda s: sorted({val for val in s if isinstance(val, str) and val.strip()})
                )
                for principal, names in grouped_related.items():
                    related_lists[principal] = [f"{name} - {principal}" for name in names]

    summary["related_entities"] = summary["entity_norm"].map(
        lambda key: json.dumps(related_lists.get(key, []), ensure_ascii=False)
    )

    emotion_cols = [col for col in joined.columns if col.startswith("emotion_prob_")]
    for col in emotion_cols:
        emotion_key = col.replace("emotion_prob_", "").lower()
        mean_col = f"mean_emotion_{emotion_key}"
        wmean_col = f"wmean_emotion_{emotion_key}"
        summary[mean_col] = summary["entity_norm"].map(grouped[col].mean())
        summary[wmean_col] = summary["entity_norm"].map(
            _group_weighted_mean(grouped, col, "__weight__")
        )

    weighted_cols = [col for col in summary.columns if col.startswith("wmean_emotion_")]

    def _top_emotion(row: pd.Series) -> Optional[str]:
        if not weighted_cols:
            return None
        values = {col.replace("wmean_emotion_", ""): row[col] for col in weighted_cols if pd.notna(row[col])}
        if not values:
            return None
        return max(values, key=values.get)

    summary["top_emotion_weighted"] = summary.apply(_top_emotion, axis=1)

    if period_start is None or period_end is None:
        inferred_start, inferred_end = _infer_period_bounds(posts_df)
        period_start = period_start or inferred_start
        period_end = period_end or inferred_end

    summary["period_start"] = period_start or ""
    summary["period_end"] = period_end or ""

    summary = summary[summary["entity_norm"].isin(PRIMARY_ENTITIES)]
    summary = summary.sort_values("entity_norm").reset_index(drop=True)

    ordered_cols = [
        "entity_norm",
        "freq",
        "sentiment_mean",
        "sentiment_mean_weighted",
        "related_entities",
    ]
    emotion_mean_cols = sorted([col for col in summary.columns if col.startswith("mean_emotion_")])
    emotion_wmean_cols = sorted([col for col in summary.columns if col.startswith("wmean_emotion_")])
    ordered_cols.extend(emotion_mean_cols)
    ordered_cols.extend(emotion_wmean_cols)
    ordered_cols.extend(["top_emotion_weighted", "period_start", "period_end"])

    for col in ordered_cols:
        if col not in summary.columns:
            summary[col] = None

    summary = summary[ordered_cols]
    return summary


def compute_entity_topic(
    posts_df: pd.DataFrame,
    mentions_df: pd.DataFrame,
    *,
    include_related: bool = True,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Genera tabla de co-ocurrencia entidad ↔ tópico."""

    columns = ["entity_norm", "entity_type", "topic_id", "topic_label", "msgs", "share"]

    logger = logger or logging.getLogger(__name__)

    if mentions_df.empty:
        return pd.DataFrame(columns=columns)

    work = mentions_df.copy()
    work["entity_type"] = work["entity_type"].astype(str).str.lower()
    if not include_related:
        work = work[work["entity_type"] == "principal"]

    if work.empty:
        return pd.DataFrame(columns=columns)

    join_subset = posts_df[[c for c in ["item_id", "topic_id", "topic_label"] if c in posts_df.columns]]
    joined = work.merge(join_subset, on="item_id", how="left", suffixes=("", "_post"), indicator=True)

    if joined.shape[0] != work.shape[0]:
        raise AssertionError("Unexpected row growth after joining mentions with posts for topic table")

    missing_posts = int(joined["_merge"].value_counts().get("left_only", 0))
    if missing_posts:
        logger.warning("%s menciones carecen de tópico tras el join", missing_posts)

    joined = joined.drop(columns=["_merge"])

    for col in ["topic_id", "topic_label"]:
        post_col = f"{col}_post"
        if post_col in joined.columns:
            joined[col] = joined[col].where(joined[col].notna(), joined[post_col])
            joined = joined.drop(columns=[post_col])

    unique_mentions = _select_unique_mentions(joined, include_topic=True)

    grouped = unique_mentions.groupby(
        ["entity_norm", "entity_type", "topic_id", "topic_label"], dropna=False
    )
    msgs = grouped["item_id"].nunique().reset_index(name="msgs")

    msgs["entity_norm"] = msgs["entity_norm"].astype(str)
    msgs["entity_type"] = msgs["entity_type"].astype(str).str.lower()

    totals = msgs.groupby(["entity_norm", "entity_type"])  # type: ignore[arg-type]
    total_msgs = totals["msgs"].transform("sum")
    msgs["share"] = msgs["msgs"] / total_msgs.replace(0, pd.NA)
    msgs["share"] = msgs["share"].fillna(0.0)

    msgs = msgs.sort_values(
        ["entity_type", "entity_norm", "msgs"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    for col in columns:
        if col not in msgs.columns:
            msgs[col] = None

    return msgs[columns]


def analyze_entities(
    posts_df: pd.DataFrame,
    entities: Sequence[EntitySpec],
    *,
    text_col: str = "text_clean",
    id_col: str = "item_id",
    topic_id_col: str = "topic_id",
    topic_label_col: str = "topic_label",
    context_window: int = 160,
    ensure_principal: bool = True,
    weight_col: str = "engagement",
    include_related_topics: bool = True,
    use_ner: bool = False,
    ner_model: str = "Davlan/xlm-roberta-base-ner-hrl",
    ner_device: Optional[int] = None,
    ner_aggregation: str = "simple",
    ner_allowed_groups: Optional[Sequence[str]] = None,
    ner_min_score: float = 0.6,
    ner_allow_unmatched: bool = False,
    entity_dict: Optional[EntityDictionary] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, pd.DataFrame]:
    """Pipeline principal para detección y agregación de entidades."""

    logger = logger or logging.getLogger(__name__)

    active_dict = entity_dict or build_entity_dictionary(
        entities,
        ensure_principal=ensure_principal,
        logger=logger,
    )

    mentions_list = extract_entity_mentions(
        posts_df,
        entities,
        text_col=text_col,
        id_col=id_col,
        topic_id_col=topic_id_col,
        topic_label_col=topic_label_col,
        context_window=context_window,
        entity_dict=active_dict,
        ensure_principal=ensure_principal,
        logger=logger,
        use_ner=use_ner,
        ner_model=ner_model,
        ner_device=ner_device,
        ner_aggregation=ner_aggregation,
        ner_allowed_groups=ner_allowed_groups,
        ner_min_score=ner_min_score,
        ner_allow_unmatched=ner_allow_unmatched,
    )

    mentions_df = mentions_to_dataframe(mentions_list)

    summary_df = compute_entity_summary(
        posts_df,
        mentions_df,
        weight_col=weight_col,
        logger=logger,
    )

    topic_df = compute_entity_topic(
        posts_df,
        mentions_df,
        include_related=include_related_topics,
        logger=logger,
    )

    return {
        "mentions": mentions_df,
        "entity_summary": summary_df,
        "entity_topic": topic_df,
    }


def export_entity_tables(
    entity_summary: pd.DataFrame,
    entity_topic: pd.DataFrame,
    output_dir: str,
    *,
    summary_filename: str = "entity_summary.csv",
    topic_filename: str = "entity_topic.csv",
    sep: str = ";",
    encoding: str = "utf-8-sig",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """Exporta las tablas de entidades en formato listo para Tableau."""

    logger = logger or logging.getLogger(__name__)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / summary_filename
    topic_path = output_path / topic_filename

    summary_df = entity_summary.copy()
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=[
            "entity_norm",
            "freq",
            "sentiment_mean",
            "sentiment_mean_weighted",
            "top_emotion_weighted",
            "period_start",
            "period_end",
        ])

    summary_cols_order = [
        "entity_norm",
        "freq",
        "sentiment_mean",
        "sentiment_mean_weighted",
    ]
    summary_cols_order.extend(sorted([c for c in summary_df.columns if c.startswith("mean_emotion_")]))
    summary_cols_order.extend(sorted([c for c in summary_df.columns if c.startswith("wmean_emotion_")]))
    summary_cols_order.extend(["top_emotion_weighted", "period_start", "period_end"])

    for col in summary_cols_order:
        if col not in summary_df.columns:
            summary_df[col] = None
    summary_df = summary_df[summary_cols_order]
    summary_df.to_csv(summary_path, index=False, sep=sep, encoding=encoding)
    logger.info("entity_summary exportado en %s", summary_path)

    topic_df = entity_topic.copy()
    topic_cols = ["entity_norm", "entity_type", "topic_id", "topic_label", "msgs", "share"]
    for col in topic_cols:
        if col not in topic_df.columns:
            topic_df[col] = None
    topic_df = topic_df[topic_cols]
    topic_df.to_csv(topic_path, index=False, sep=sep, encoding=encoding)
    logger.info("entity_topic exportado en %s", topic_path)

    return {
        "entity_summary": str(summary_path),
        "entity_topic": str(topic_path),
    }


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
                "lang", "full_text", "snippet", "text_source", "sentiment_label", "sentiment_score",
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

    for start in tqdm(
        range(0, len(snippets), sentiment_batch_size),
        desc="Sentiment mentions",
        unit="batch",
        leave=False,
    ):
        batch = snippets[start:start + sentiment_batch_size]
        results = sentiment_scorer.score_batch(batch)
        for lab, conf, dist in results:
            sentiment_labels.append(lab)
            sentiment_scores.append(conf)
            sentiment_dists.append(dist)

    mentions_df["sentiment_label"] = sentiment_labels
    mentions_df["sentiment_score"] = sentiment_scores
    mentions_df["sentiment_dist"] = sentiment_dists
    mentions_df["stance"] = mentions_df["sentiment_label"].apply(_stance_code)
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

    for start in tqdm(
        range(0, len(snippets), emotion_batch_size),
        desc="Emotion mentions",
        unit="batch",
        leave=False,
    ):
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
    mentions_df["impact_score"] = mentions_df.apply(_impact_score_from_row, axis=1)
    return mentions_df


def summarize_entity_mentions(mentions_df: pd.DataFrame) -> pd.DataFrame:
    if mentions_df.empty:
        return pd.DataFrame(
            columns=[
                "entity", "topic_id", "topic_label", "n_posts", "n_mentions",
                "mean_sentiment_score", "mean_topic_score",
                "stance_index", "stance_label",
                "sentiment_prob_positive", "sentiment_prob_negative", "sentiment_prob_neutral",
                "impact_score_sum", "impact_score_mean",
                "engagement_mean", "reach_mean",
                "top_emotion", "sentiment_label_mode", "emotion_label_mode",
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
    enriched["stance_weight"] = enriched["sentiment_score"].fillna(0.0).astype(float) + 0.3

    group_cols = ["entity", "topic_id", "topic_label"]
    summary = enriched.groupby(group_cols, dropna=False).agg(
        n_posts=("item_id", "nunique"),
        n_mentions=("item_id", "count"),
        mean_sentiment_score=("sentiment_score", "mean"),
        mean_topic_score=("topic_score", "mean"),
        sentiment_prob_positive=("sentiment_prob_positive", "mean"),
        sentiment_prob_negative=("sentiment_prob_negative", "mean"),
        sentiment_prob_neutral=("sentiment_prob_neutral", "mean"),
        impact_score_sum=("impact_score", "sum"),
        impact_score_mean=("impact_score", "mean"),
        engagement_mean=("engagement", "mean"),
        reach_mean=("reach", "mean"),
    )

    emotion_cols = [c for c in enriched.columns if c.startswith("emotion_prob_")]
    for col in emotion_cols:
        summary[col] = enriched.groupby(group_cols, dropna=False)[col].mean()

    def _mode(series: pd.Series) -> Optional[str]:
        mode = series.mode(dropna=True)
        return str(mode.iloc[0]) if not mode.empty else None

    summary["sentiment_label_mode"] = enriched.groupby(group_cols, dropna=False)["sentiment_label"].apply(_mode)
    summary["emotion_label_mode"] = enriched.groupby(group_cols, dropna=False)["emotion_label"].apply(_mode)

    def _weighted_stance(sub_df: pd.DataFrame) -> float:
        weights = sub_df["stance_weight"].sum()
        if weights <= 0:
            return 0.0
        return float((sub_df["stance_value"] * sub_df["stance_weight"]).sum() / weights)

    stance_series = enriched.groupby(group_cols, dropna=False).apply(_weighted_stance)
    summary["stance_index"] = stance_series

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


def aggregate_mentions_per_item(mentions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce per-item aggregates so we can enrich posts with the entity-conditioned metrics.
    """
    if mentions_df.empty:
        return pd.DataFrame(
            columns=[
                "item_id",
                "n_entity_mentions",
                "entities_detected",
                "sentiment_score",
                "sentiment_label",
                "sentiment_dist",
                "stance_value",
                "stance",
                "emotion_label",
                "emotion_scores",
                "impact_score",
                "impact_score_mean",
                "engagement_mean",
                "reach_mean",
                "related_entities",
            ]
        )

    work = mentions_df.copy()
    work["_weight"] = work["sentiment_score"].fillna(0.0).astype(float) + 0.3
    work["_weight"] = work["_weight"].clip(lower=0.1)

    group = work.groupby("item_id", dropna=False)
    aggregated = group.agg(
        n_entity_mentions=("entity", "count"),
        sentiment_score=("sentiment_score", "mean"),
        topic_score_mean=("topic_score", "mean"),
        impact_score=("impact_score", "sum"),
        impact_score_mean=("impact_score", "mean"),
        engagement_mean=("engagement", "mean"),
        reach_mean=("reach", "mean"),
    )

    aggregated["entities_detected"] = group["entity"].apply(
        lambda s: sorted({val for val in s.dropna() if isinstance(val, str) and val.strip()})
    )

    def _collect_related(sub_df: pd.DataFrame) -> List[str]:
        principal_present = {
            str(val).strip()
            for val, etype in zip(sub_df["entity_norm"], sub_df["entity_type"])
            if isinstance(val, str) and val.strip() and str(etype).lower() == "principal"
        }
        related_pairs: Set[str] = set()
        for _, row in sub_df.iterrows():
            if str(row.get("entity_type", "")).lower() != "related":
                continue
            linked = str(row.get("linked_principal") or "").strip()
            if not linked or linked not in principal_present:
                continue
            related_name = str(row.get("entity_norm") or "").strip()
            if not related_name:
                continue
            related_pairs.add(f"{related_name} - {linked}")
        return sorted(related_pairs)

    related_series = group.apply(_collect_related)
    aggregated["related_entities"] = related_series.reindex(aggregated.index).apply(
        lambda values: json.dumps(values, ensure_ascii=False)
    )

    def _weighted_stance(sub_df: pd.DataFrame) -> float:
        total_weight = sub_df["_weight"].sum()
        if total_weight <= 0:
            return 0.0
        return float((sub_df["stance_value"] * sub_df["_weight"]).sum() / total_weight)

    aggregated["stance_value"] = group.apply(_weighted_stance)
    aggregated["stance_bucket"] = aggregated["stance_value"].apply(_stance_bucket)
    aggregated["stance"] = aggregated["stance_bucket"].apply(_stance_code)

    weight_sum = group["_weight"].sum()

    sent_probs = pd.json_normalize(work["sentiment_dist"]).fillna(0.0)
    sentiment_keys = ["positive", "negative", "neutral"]
    if sent_probs.empty:
        aggregated["sentiment_dist"] = [{} for _ in range(len(aggregated))]
    else:
        for key in sentiment_keys:
            if key not in sent_probs.columns:
                sent_probs[key] = 0.0
        sent_probs = sent_probs[sentiment_keys].mul(work["_weight"], axis=0)
        sent_weighted = sent_probs.groupby(work["item_id"]).sum()
        sent_weighted = sent_weighted.div(weight_sum, axis=0).reindex(aggregated.index, fill_value=0.0)
        aggregated["sentiment_dist"] = sent_weighted.apply(
            lambda row: {k: float(row[k]) for k in sentiment_keys}, axis=1
        )

    aggregated["sentiment_label"] = aggregated["sentiment_dist"].apply(
        lambda dist: max(dist, key=dist.get) if dist else None
    )
    if aggregated["sentiment_label"].isna().any():
        mask = aggregated["sentiment_label"].isna()
        aggregated.loc[mask, "sentiment_label"] = aggregated.loc[mask, "stance_bucket"]

    emo_probs = pd.json_normalize(work["emotion_scores"]).fillna(0.0)
    emotion_cols = sorted(set(emo_probs.columns) | {lbl.lower() for lbl in DEFAULT_EMOTIONS})
    if not emotion_cols:
        aggregated["emotion_scores"] = [{} for _ in range(len(aggregated))]
        aggregated["emotion_label"] = None
    else:
        for col in emotion_cols:
            if col not in emo_probs.columns:
                emo_probs[col] = 0.0
        emo_probs = emo_probs[emotion_cols].mul(work["_weight"], axis=0)
        emo_weighted = emo_probs.groupby(work["item_id"]).sum()
        emo_weighted = emo_weighted.div(weight_sum, axis=0).reindex(aggregated.index, fill_value=0.0)
        aggregated["emotion_scores"] = emo_weighted.apply(
            lambda row: {k: float(row[k]) for k in emotion_cols}, axis=1
        )
        aggregated["emotion_label"] = aggregated["emotion_scores"].apply(
            lambda dist: max(dist, key=dist.get) if dist else None
        )

    aggregated = aggregated.drop(columns=["stance_bucket"]).reset_index()
    return aggregated


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
