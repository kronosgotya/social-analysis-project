from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, date
import sys
import re
import ast
from typing import Dict, Optional, List, Set, Any, Iterable, Tuple
from collections import Counter

# --- Fix ruta para importar src/*
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import joblib  # noqa: E402
from transformers.utils.logging import set_verbosity_warning  # noqa: E402
set_verbosity_warning()  # menos ruido en consola

from src.utils import (  # noqa: E402
    ensure_dirs, normalize_source, add_engagement, add_dominant_emotion,
    export_tableau_csv, emotions_to_long
)
from src.preprocessing import load_telegram, load_x, unify_frames  # noqa: E402
# SentenceTransformer para clasificador embeddings
from sentence_transformers import SentenceTransformer  # noqa: E402
from src.network import (  # noqa: E402
    build_x_graph, graph_metrics, export_gexf,
    edges_from_x, nodes_metrics_df
)
from src.topics_bertopic import fit_topics, summarize_topics  # noqa: E402
from src.entities_runtime import (  # noqa: E402
    load_entities,
    NATO_ALIASES,
    RUSSIA_ALIASES,
    RELATED_TO_PRINCIPAL,
    DROP_UNKNOWN,
)
from src.entity_analysis import (  # noqa: E402
    extract_entity_mentions,
    score_entity_mentions,
    summarize_entity_mentions,
    serialize_mentions_for_export,
    aggregate_mentions_per_item,
    mentions_to_dataframe,
    explode_entity_mentions,
    ENTITY_EMOTION_DIMENSIONS,
    _normalize_optional_str,
    _normalize_optional_int,
    _default_zero,
    _coerce_float,
    _span_overlaps,
    _snippet,
    MentionCandidate,
)


def _ensure_json_array(value):
    if isinstance(value, (list, tuple)):
        try:
            return json.dumps(list(value), ensure_ascii=False)
        except TypeError:
            return json.dumps(list(value), ensure_ascii=False, default=str)
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "[]"
        if stripped.startswith("[") or stripped.startswith("{"):
            return stripped
        return json.dumps(stripped, ensure_ascii=False)
    if pd.isna(value):
        return "[]"
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def _parse_json_list(raw: Any) -> List[Any]:
    if isinstance(raw, list):
        return raw
    if raw in (None, "", "[]"):
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return []
    return []


GEO_OUTPUT_COLUMNS = ["item_id", "country", "weight", "method", "source", "date", "lang"]
ENTITY_OUTPUT_COLUMNS = [
    "item_id",
    "source",
    "date",
    "lang",
    "principal_entity",
    "related_entity",
    "alias",
    "stance",
    "stance_value",
    "sentiment_label",
    "sentiment_score",
    "emotion_label",
] + ENTITY_EMOTION_DIMENSIONS + [
    "impact_score",
    "engagement",
    "manual_label_topic",
    "manual_label_subtopic",
    "snippet",
]


def _float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_item_date(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    if not text:
        return ""
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if match:
        return match.group(0)
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.date().isoformat()


def _build_geo_table(base_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    geo_rows: List[Dict[str, Any]] = []
    posts_with_geo = 0

    for _, row in base_df.iterrows():
        item_date = _normalize_item_date(row.get("timestamp") or row.get("date"))
        geo_distribution = [
            entry for entry in _parse_json_list(row.get("geo_country_distribution"))
            if isinstance(entry, dict)
        ]
        if geo_distribution:
            posts_with_geo += 1
        for entry in geo_distribution:
            country = str(entry.get("country") or "").strip()
            if not country:
                continue
            geo_rows.append(
                {
                    "item_id": str(row.get("item_id") or "").strip(),
                    "country": country,
                    "weight": _float_or_default(entry.get("weight")),
                    "method": str(entry.get("method") or "").strip(),
                    "source": str(row.get("source") or "").strip(),
                    "date": item_date,
                    "lang": str(row.get("lang") or "").strip(),
                }
            )

    geo_df = pd.DataFrame(geo_rows, columns=GEO_OUTPUT_COLUMNS)
    if not geo_df.empty:
        geo_df["weight"] = pd.to_numeric(geo_df["weight"], errors="coerce").fillna(0.0)
        geo_df = geo_df.drop_duplicates(subset=["item_id", "country"], keep="first")
    return geo_df, posts_with_geo


def _build_entity_table(
    base_df: pd.DataFrame,
    alias_maps,
    *,
    drop_unknown: bool,
) -> Tuple[pd.DataFrame, int, int]:
    entity_rows: List[Dict[str, Any]] = []
    raw_entity_rows = 0
    posts_with_entities = 0

    for _, row in base_df.iterrows():
        mentions_list = [
            m for m in _parse_json_list(row.get("entity_mentions")) if isinstance(m, dict)
        ]
        raw_entity_rows += len(mentions_list)
        exploded = explode_entity_mentions(row, alias_maps, drop_unknown=drop_unknown)
        if exploded:
            posts_with_entities += 1
            entity_rows.extend(exploded)

    entity_df = pd.DataFrame(entity_rows)
    if entity_df.empty:
        entity_df = pd.DataFrame(columns=ENTITY_OUTPUT_COLUMNS)
    else:
        for emo in ENTITY_EMOTION_DIMENSIONS:
            entity_df[emo] = entity_df["emotion_scores"].apply(
                lambda d, key=emo: float(d.get(key, 0.0)) if isinstance(d, dict) else 0.0
            )
        entity_df = entity_df.drop(columns=["emotion_scores", "topic_label"], errors="ignore")
        entity_df = entity_df.drop_duplicates(
            subset=["item_id", "principal_entity", "related_entity", "stance"],
            keep="first",
        )
        for col in ["stance_value", "sentiment_score", "impact_score", "engagement"]:
            if col in entity_df.columns:
                entity_df[col] = pd.to_numeric(entity_df[col], errors="coerce").fillna(0.0)
        for col in ENTITY_OUTPUT_COLUMNS:
            if col not in entity_df.columns:
                if col in ENTITY_EMOTION_DIMENSIONS or col in {"stance_value", "sentiment_score", "impact_score", "engagement"}:
                    entity_df[col] = 0.0
                else:
                    entity_df[col] = ""
        if "stance" in entity_df.columns:
            entity_df["stance"] = entity_df["stance"].fillna("neu").replace("", "neu")
        for col in [
            "item_id",
            "manual_label_topic",
            "manual_label_subtopic",
            "alias",
            "related_entity",
            "snippet",
            "date",
            "source",
            "lang",
            "principal_entity",
            "sentiment_label",
        ]:
            if col in entity_df.columns:
                entity_df[col] = entity_df[col].fillna("").astype(str)
        entity_df = entity_df[ENTITY_OUTPUT_COLUMNS]
    return entity_df, raw_entity_rows, posts_with_entities


ENCODING = "utf-8-sig"
SEP = ";"
TOPIC_CLASSIFIER_PATH = Path("models") / "topic_classifier" / "topic_classifier.joblib"
TOPIC_MANUAL_PATH = Path("data") / "ground_truth" / "topics_manual_labels.csv"
DEFAULT_CLASSIFIER_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_EMBEDDER_CACHE: Dict[str, SentenceTransformer] = {}

_NATO_FLAG_EMOJIS = "🇺🇸🇬🇧🇫🇷🇩🇪🇮🇹🇪🇸🇵🇹🇳🇱🇧🇪🇱🇺🇨🇦🇳🇴🇩🇰🇸🇪🇫🇮🇮🇸🇵🇱🇨🇿🇸🇰🇭🇺🇷🇴🇧🇬🇬🇷🇹🇷🇸🇮🇭🇷🇲🇪🇲🇰🇦🇱🇪🇪🇱🇻🇱🇹🇦🇺🇳🇿🇨🇭"
_RUSSIA_FLAG_EMOJIS = "🇷🇺🇧🇾"
_UKRAINE_FLAG_EMOJI = "🇺🇦"

_NATO_AIR_ASSET_HASHTAGS = [
    r"f[- ]?35", r"f[- ]?16", r"eurofighter", r"typhoon", r"rafale", r"gripen",
    r"f[- ]?15", r"f[ah]-?18", r"ef-?18", r"fa-?50", r"e-?3a", r"awacs",
    r"e-?3\s?sentry", r"e-?7", r"wedgetail", r"mq-?9", r"reaper", r"bayraktar",
    r"tb-?2", r"scaneagle", r"rq-?20", r"puma", r"raven", r"blackhornet",
    r"vector", r"phoenixghost", r"uj-?22", r"r-?18", r"uav", r"drone", r"uas",
    r"natoairasset", r"ukraineairasset"
]

_RUSSIA_AIR_ASSET_HASHTAGS = [
    r"su-?57", r"su-?35", r"su-?30", r"su-?27", r"su-?24", r"su-?25",
    r"mig-?31", r"mig-?29", r"mig-?35", r"mig-?23", r"mig-?21", r"tu-?95",
    r"tu-?22m3", r"tu-?160", r"il-?76", r"il-?78", r"a-?50u?", r"awacsruso",
    r"beriev", r"orlan-?10", r"orion", r"forpost", r"shahed", r"geran",
    r"lancet", r"zala", r"kub", r"supercam", r"eleron", r"okhotnik", r"s-?70",
    r"uavruso", r"droneruso", r"russianairasset", r"vks"
]

_FALLBACK_PATTERNS: Dict[str, Dict[str, object]] = {
    "NATO": {
        "patterns": [
            {"regex": re.compile(r"(?iu)\b(?:nato|otan|нато|north atlantic treaty|north atlantic alliance)\b")},
            {"regex": re.compile(r"(?iu)\bshape\b")},
            {"regex": re.compile(r"(?iu)supreme headquarters allied powers europe")},
            {"regex": re.compile(r"(?iu)allied\s+(?:air|joint|combined|command)")},
            {"regex": re.compile(rf"[{_NATO_FLAG_EMOJIS}]+"), "entity_type": "related", "alias": "NATO flag"},
            {"regex": re.compile(rf"(?iu)#?(?:{'|'.join(_NATO_AIR_ASSET_HASHTAGS)})\b"), "entity_type": "related"},
        ],
        "entity_type": "principal",
        "linked_principal": "NATO",
    },
    "Russia": {
        "patterns": [
            {"regex": re.compile(
                r"(?iu)\b(?:russia|russian(?:s)?|rusia|rusija|rusijos|rusiją|rusijoje|rusko|rusku|rusk[aáéý][\w-]*|"
                r"rusos|rusové|rusland|venemaa|venāja?|venäjä|venäl[äa]is[\w-]*|"
                r"россия|россией|россии|россию|российск[а-яё]+|русск[а-яё]+|рф|росія|росії|росію|"
                r"rosyjsk[a-zęó]+|rosjan|rosję|rosji|"
                r"kriev[\w-]+|krievu|krievijas|"
                r"droni\s+russ[iia]+|drones?\s+russes?|"
                r"rusya|rusya'nın|rusya'ya|rusya'dan|rus hava|rus uça|rus dron|rus ordusu|rusya saldır|"
                r"rus\w*(?:havan|drone|jet|uçak|hava|füze))\b"
            )},
            {"regex": re.compile(rf"[{_RUSSIA_FLAG_EMOJIS}]+"), "entity_type": "related", "alias": "Russian flag"},
            {"regex": re.compile(r"(?iu)\b(?:geran(?:ium)?[-\s]?2|shahed[-\s]?136|fab[-\s]?(?:100|250|500|1000)|gerbera|gerań)\b"), "entity_type": "related"},
            {"regex": re.compile(r"(?iu)\b(?:putin|kremlin|moskva|moscow)\b")},
            {"regex": re.compile(rf"(?iu)#?(?:{'|'.join(_RUSSIA_AIR_ASSET_HASHTAGS)})\b"), "entity_type": "related"},
        ],
        "entity_type": "principal",
        "linked_principal": "Russia",
    },
    "Ukraine": {
        "patterns": [
            {"regex": re.compile(r"(?iu)\b(?:ukraine|ucrania|ucraina|ukraina|ukrainu|ukraiņu|ukrayna|ukrainsk)\b")},
            {"regex": re.compile(r"(?iu)\bукраїн[а-яіїєґ]+\b")},
            {"regex": re.compile(r"(?iu)\bукраин[а-яё]+\b")},
            {"regex": re.compile(rf"[{_UKRAINE_FLAG_EMOJI}]"), "entity_type": "related", "alias": "Ukraine flag"},
        ],
        "entity_type": "related",
        "linked_principal": "NATO",
    },
}

def _get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDER_CACHE[model_name]


def _load_topic_classifier() -> Optional[Dict[str, Any]]:
    if not TOPIC_CLASSIFIER_PATH.exists():
        return None
    try:
        model_bundle = joblib.load(TOPIC_CLASSIFIER_PATH)
    except Exception as exc:
        print(f"ⓘ Could not load topic classifier ({exc}).")
        return None
    keys = set(model_bundle.keys())
    kind = model_bundle.get("kind")
    if kind == "sentence-transformer":
        required = {"topic_clf", "subtopic_clf", "topic_label_encoder", "subtopic_label_encoder", "embedding_model_name"}
        if not required.issubset(keys):
            print("ⓘ Topic classifier (embeddings) missing expected components; ignoring.")
            return None
    elif kind == "sentence-transformer-multitask":
        required = {
            "manual_topic_clf",
            "manual_subtopic_clf",
            "manual_topic_label_encoder",
            "manual_subtopic_label_encoder",
            "embedding_model_name",
        }
        if not required.issubset(keys):
            # compatibilidad con bundles antiguos
            legacy_required = {
                "topic_id_clf",
                "manual_topic_clf",
                "manual_subtopic_clf",
                "topic_id_label_encoder",
                "manual_topic_label_encoder",
                "manual_subtopic_label_encoder",
                "embedding_model_name",
            }
            if legacy_required.issubset(keys):
                # dejamos pasar; las rutas que usan topic_id_clf lo ignorarán
                pass
            else:
                print("ⓘ Topic classifier (multitask) missing expected components; ignoring.")
                return None
    else:
        required_keys = {"vectorizer", "topic_clf", "subtopic_clf"}
        if not required_keys.issubset(keys):
            print("ⓘ Topic classifier file is missing expected components; ignoring.")
            return None
        model_bundle.setdefault("kind", "tfidf")
    return model_bundle


def _load_manual_topic_labels() -> pd.DataFrame:
    if not TOPIC_MANUAL_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(TOPIC_MANUAL_PATH, sep=SEP, encoding=ENCODING)
    df["topic_id"] = df["topic_id"].astype(str)
    for col in ["manual_label_topic", "manual_label_subtopic"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)
    return df[["topic_id", "manual_label_topic", "manual_label_subtopic"]]


def _topic_terms_to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(term) for term in value if term)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return " ".join(str(term) for term in parsed if term)
            except json.JSONDecodeError:
                pass
        return text.replace(",", " ").replace("[", " ").replace("]", " ")
    return str(value)


def _normalize_label(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _majority_label(series: pd.Series) -> str:
    values = [_normalize_label(val) for val in series if _normalize_label(val)]
    if not values:
        return ""
    counter = Counter(values)
    most_common = counter.most_common()
    top_count = most_common[0][1]
    candidates = sorted([label for label, count in most_common if count == top_count])
    return candidates[0] if candidates else ""


def _first_non_empty(seq: Iterable[object]) -> object:
    for val in seq:
        if isinstance(val, list) and val:
            return val
        if isinstance(val, str) and val.strip():
            return val
    return []


_SERIES_LINE_PATTERN = re.compile(r"^\s*\d+\s+\[(.*)\]\s*$")


def _parse_series_style_terms(text: str) -> List[str]:
    cleaned = text.replace("\\n", "\n")
    tokens: List[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or "Name:" in line or "dtype" in line:
            continue
        match = _SERIES_LINE_PATTERN.match(line)
        if not match:
            continue
        inside = match.group(1)
        tokens.extend([tok.strip().strip("'\"") for tok in inside.split(",") if tok.strip()])
    return tokens


def _serialize_terms(value: object) -> str:
    """
    Serializa términos a JSON garantizando una lista plana de strings legibles.
    Maneja valores anidados (list[list[str]], tuplas...) y strings con
    representaciones de listas.
    """

    if isinstance(value, pd.Series):
        return _serialize_terms(value.tolist())

    def _maybe_parse_container(text: str) -> object:
        stripped = text.strip()
        if not stripped:
            return []
        if not (stripped.startswith("[") and stripped.endswith("]")):
            series_tokens = _parse_series_style_terms(stripped)
            if series_tokens:
                return series_tokens
            return text
        try:
            parsed = json.loads(stripped)
            return parsed
        except Exception:
            try:
                parsed = ast.literal_eval(stripped)
                return parsed
            except Exception:
                series_tokens = _parse_series_style_terms(stripped)
                if series_tokens:
                    return series_tokens
                return text

    def _flatten_terms(obj: object) -> Iterable[str]:
        if obj is None:
            return
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                yield from _flatten_terms(item)
            return
        if isinstance(obj, str):
            parsed = _maybe_parse_container(obj)
            if parsed is obj:
                token = obj.strip()
                if token:
                    yield token
            else:
                yield from _flatten_terms(parsed)
            return
        token = str(obj).strip()
        if token:
            yield token

    seen: Set[str] = set()
    ordered: List[str] = []
    for term in _flatten_terms(value):
        if term not in seen:
            seen.add(term)
            ordered.append(term)
    formatted = ", ".join(f"'{tok}'" for tok in ordered)
    return f"[{formatted}]"


def _build_subtopic_parent_map(df: pd.DataFrame, topic_col: str, subtopic_col: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    conflicts: Set[str] = set()
    if df is None or df.empty:
        return mapping
    for _, row in df.iterrows():
        topic = _normalize_label(row.get(topic_col))
        subtopic = _normalize_label(row.get(subtopic_col))
        if not subtopic or not topic:
            continue
        if subtopic not in mapping:
            mapping[subtopic] = topic
        elif mapping[subtopic] != topic:
            conflicts.add(subtopic)
    for sub in conflicts:
        mapping.pop(sub, None)
    return mapping


def _enforce_subtopic_hierarchy(df: pd.DataFrame, topic_col: str, subtopic_col: str, mapping: Dict[str, str]) -> None:
    if df is None or df.empty or not mapping:
        return
    if topic_col not in df.columns or subtopic_col not in df.columns:
        return
    subtopics = df[subtopic_col].astype(str).str.strip()
    replacement = subtopics.map(mapping)
    mask = subtopics != ""
    mask &= replacement.notna()
    if mask.any():
        df.loc[mask, topic_col] = replacement[mask]


def _apply_classifier_predictions(
    classifier: Optional[Dict[str, Any]],
    texts: List[str],
    assignments: List[Dict[str, Any]],
) -> None:
    if classifier is None or not assignments or not texts:
        return
    kind = classifier.get("kind")
    if kind not in {"sentence-transformer", "sentence-transformer-multitask"}:
        return

    embed_name = classifier.get("embedding_model_name", DEFAULT_CLASSIFIER_EMBED_MODEL)
    normalize_embeddings = classifier.get("normalize_embeddings", True)
    batch_size = int(classifier.get("inference_batch_size", 128))
    embedder = _get_embedder(embed_name)
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize_embeddings,
    )

    manual_topic_preds: List[str]
    manual_subtopic_preds: List[str]

    if kind == "sentence-transformer-multitask":
        manual_topic_encoder = classifier.get("manual_topic_label_encoder")
        manual_subtopic_encoder = classifier.get("manual_subtopic_label_encoder")
        manual_topic_clf = classifier.get("manual_topic_clf")
        manual_subtopic_clf = classifier.get("manual_subtopic_clf")
        if (
            manual_topic_encoder is None
            or manual_subtopic_encoder is None
            or manual_topic_clf is None
            or manual_subtopic_clf is None
        ):
            return

        manual_topic_preds = manual_topic_encoder.inverse_transform(manual_topic_clf.predict(embeddings))
        manual_subtopic_preds = manual_subtopic_encoder.inverse_transform(manual_subtopic_clf.predict(embeddings))

        for idx, assign in enumerate(assignments):
            pred_topic = manual_topic_preds[idx] if idx < len(manual_topic_preds) else ""
            if pred_topic and not _normalize_label(assign.get("manual_label_topic")):
                assign["manual_label_topic"] = pred_topic

            pred_subtopic = manual_subtopic_preds[idx] if idx < len(manual_subtopic_preds) else ""
            if pred_subtopic and not _normalize_label(assign.get("manual_label_subtopic")):
                assign["manual_label_subtopic"] = pred_subtopic

    elif kind == "sentence-transformer":
        manual_topic_encoder = classifier.get("topic_label_encoder")
        manual_subtopic_encoder = classifier.get("subtopic_label_encoder")
        manual_topic_clf = classifier.get("topic_clf")
        manual_subtopic_clf = classifier.get("subtopic_clf")
        if manual_topic_encoder is None or manual_subtopic_encoder is None:
            return
        manual_topic_preds = manual_topic_encoder.inverse_transform(manual_topic_clf.predict(embeddings))
        manual_subtopic_preds = manual_subtopic_encoder.inverse_transform(manual_subtopic_clf.predict(embeddings))

        for idx, assign in enumerate(assignments):
            pred_topic = manual_topic_preds[idx] if idx < len(manual_topic_preds) else ""
            if pred_topic and not _normalize_label(assign.get("manual_label_topic")):
                assign["manual_label_topic"] = pred_topic

            pred_subtopic = manual_subtopic_preds[idx] if idx < len(manual_subtopic_preds) else ""
            if pred_subtopic and not _normalize_label(assign.get("manual_label_subtopic")):
                assign["manual_label_subtopic"] = pred_subtopic

def _derive_facts_posts_tableau(
    input_path: Path,
    *,
    output_filename: str = "facts_posts_tableau.csv",
    sep: str = ";",
    encoding: str = "utf-8-sig",
) -> None:
    if not input_path.exists():
        print(f"ⓘ Could not find {input_path.name}; skipping Tableau derivative.")
        return

    df = pd.read_csv(input_path, sep=sep, encoding=encoding, dtype=str)

    alias_maps = (NATO_ALIASES, RUSSIA_ALIASES, RELATED_TO_PRINCIPAL)
    entity_df, raw_mentions, posts_with_entities = _build_entity_table(
        df,
        alias_maps,
        drop_unknown=DROP_UNKNOWN,
    )
    output_path = input_path.parent / output_filename
    export_tableau_csv(entity_df, str(output_path))

    print(
        f"✔ {output_filename} generated → {len(entity_df)} rows | posts with entities: "
        f"{posts_with_entities} | mentions processed: {raw_mentions}"
    )


def _fallback_extract_mentions(
    df: pd.DataFrame,
    *,
    context_window: int,
    existing_item_ids: Set[str],
) -> List[MentionCandidate]:
    if df.empty:
        return []

    fallback_mentions: List[MentionCandidate] = []
    debug_counter: Dict[Tuple[str, str], int] = {}
    sample_matches: Dict[Tuple[str, str], List[str]] = {}

    skipped_existing = 0
    for _, row in df.iterrows():
        item_id_value = (
            _normalize_optional_str(row.get("item_id"))
            or _normalize_optional_str(row.get("tweet_id"))
            or ""
        )
        if not item_id_value:
            continue
        if item_id_value in existing_item_ids:
            skipped_existing += 1
            continue

        text_value = str(row.get("text_clean") or "").strip()
        if not text_value:
            continue

        source_val = _normalize_optional_str(row.get("source"))
        timestamp_val = _normalize_optional_str(row.get("timestamp"))
        lang_val = _normalize_optional_str(row.get("lang"))
        topic_label_val = _normalize_optional_str(row.get("topic_label"))
        topic_id_val = _normalize_optional_int(row.get("topic_id"))
        topic_score_val = _coerce_float(row.get("topic_score"))

        likes_val = _default_zero(_coerce_float(row.get("likes")))
        shares_val = sum(
            _default_zero(_coerce_float(row.get(col)))
            for col in ("retweets", "telegram_forwards", "forward_count", "forwards", "shares")
        )
        replies_val = _default_zero(_coerce_float(row.get("replies")))
        quotes_val = _default_zero(_coerce_float(row.get("quotes")))

        engagement_val = _coerce_float(row.get("engagement"))
        if engagement_val is None:
            engagement_val = likes_val + shares_val + replies_val + quotes_val

        views_val = _coerce_float(row.get("views"))
        reach_val = _coerce_float(row.get("reach")) or views_val

        spans_by_entity: Dict[str, List[Tuple[int, int]]] = {}

        for entity_norm, cfg in _FALLBACK_PATTERNS.items():
            patterns = cfg.get("patterns", [])
            entity_type_default = cfg.get("entity_type", "related")
            linked_principal = cfg.get("linked_principal")
            entity_spans = spans_by_entity.setdefault(entity_norm, [])

            for pattern_info in patterns:
                if isinstance(pattern_info, dict):
                    regex = pattern_info["regex"]
                    entity_type = pattern_info.get("entity_type", entity_type_default)
                    alias_override = pattern_info.get("alias")
                else:
                    regex = pattern_info
                    entity_type = entity_type_default
                    alias_override = None

                for match in regex.finditer(text_value):
                    span = match.span()
                    if _span_overlaps(span, entity_spans):
                        continue
                    entity_spans.append(span)

                    matched_text = match.group(0)
                    snippet = _snippet(text_value, matched_text, window=context_window)

                    key = (str(source_val or ""), entity_norm)
                    debug_counter[key] = debug_counter.get(key, 0) + 1
                    sample_matches.setdefault(key, []).append(matched_text)

                    fallback_mentions.append(
                        MentionCandidate(
                            item_id=item_id_value,
                            entity=entity_norm,
                            alias=alias_override or matched_text,
                            source=source_val,
                            timestamp=timestamp_val,
                            topic_id=topic_id_val,
                            topic_label=topic_label_val,
                            topic_score=topic_score_val,
                            lang=lang_val,
                            full_text=text_value,
                            snippet=snippet,
                            engagement=engagement_val,
                            reach=reach_val,
                            likes=likes_val,
                            shares=shares_val,
                            replies=replies_val,
                            quotes=quotes_val,
                            views=views_val,
                            entity_norm=entity_norm,
                            entity_type=str(entity_type),
                            entity_label=entity_norm,
                            alias_weight=1.0,
                            matched_text=matched_text,
                            confidence=1.0,
                            span_start=span[0],
                            span_end=span[1],
                            matched_alias=matched_text,
                            detector="regex_fallback",
                            text_source="text_clean",
                            linked_principal=str(linked_principal) if linked_principal else None,
                        )
                    )

    if fallback_mentions:
        print("ⓘ Fallback summary:")
        for (src, ent), cnt in sorted(debug_counter.items()):
            samples = ", ".join(sample_matches[(src, ent)][:3])
            print(f"   - source={src or 'unknown'} entity={ent} matches={cnt} samples=[{samples}]")
    if skipped_existing:
        print(f"ⓘ Fallback skipped {skipped_existing} rows already containing mentions from the primary extractor.")

    return fallback_mentions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telegram", type=str, default="")
    ap.add_argument("--x", type=str, default="")
    ap.add_argument("--max_rows", type=int, default=0, help="0 = todos")
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="GPU id (0), CPU (-1), o MPS (-2 / mps) en Apple Silicon",
    )
    ap.add_argument("--emotion_model", type=str, default="joeddav/xlm-roberta-large-xnli",
                    help="Modelo zero-shot para emociones")
    ap.add_argument("--entities", type=str, default="OTAN,Rusia",
                    help="Lista separada por comas de entidades para análisis condicionado")
    ap.add_argument("--entities_file", type=str, default="",
                    help="Archivo YAML/JSON/TXT con entidades (opcional)")
    ap.add_argument("--entity_window", type=int, default=160,
                    help="Ventana en caracteres alrededor de la mención (contexto)")
    ap.add_argument("--skip_topics", action="store_true", help="Omitir el modelado de tópicos")
    ap.add_argument("--skip_emotions", action="store_true", help="Omitir el scoring de emociones")
    ap.add_argument(
        "--split-outputs",
        dest="split_outputs",
        action="store_true",
        default=True,
        help="Escribe geo_country_exploded y facts_posts_tableau (por defecto activado).",
    )
    ap.add_argument(
        "--no-split-outputs",
        dest="split_outputs",
        action="store_false",
        help="Desactiva la escritura de salidas separadas.",
    )
    args = ap.parse_args()

    raw_device = args.device
    if raw_device is None:
        coerced_device = None
    else:
        if isinstance(raw_device, str):
            token = raw_device.strip().lower()
            if token in {"", "none"}:
                coerced_device = None
            elif token in {"-1", "cpu"}:
                coerced_device = -1
            elif token in {"-2", "mps"}:
                coerced_device = -2
            else:
                try:
                    coerced_device = int(token)
                except Exception as exc:
                    raise ValueError(f"Valor de --device no reconocido: {raw_device}") from exc
        else:
            coerced_device = raw_device
    args.device = coerced_device

    ensure_dirs(
        "data/processed",
        "results/graphs",
        "results/charts",
        "results/topics",
        "models/bertopic/global",
    )

    # --- Telegram
    df_tg = None
    if args.telegram and Path(args.telegram).exists():
        df_tg = load_telegram(args.telegram)
        if args.max_rows > 0:
            df_tg = df_tg.head(args.max_rows)
        df_tg = add_engagement(df_tg)
        df_tg_export = df_tg.copy()
        if "geo_country_distribution" in df_tg_export.columns:
            df_tg_export["geo_country_distribution"] = df_tg_export["geo_country_distribution"].apply(
                _ensure_json_array
            )
        export_tableau_csv(df_tg_export, "data/processed/telegram_preprocessed.csv")
        print("✔ TG procesado → data/processed/telegram_preprocessed.csv")

    # --- X
    df_x = None
    if args.x and Path(args.x).exists():
        df_x = load_x(args.x)
        if args.max_rows > 0:
            df_x = df_x.head(args.max_rows)
        df_x = add_engagement(df_x)
        export_tableau_csv(df_x, "data/processed/x_preprocessed.csv")
        print("✔ X procesado → data/processed/x_preprocessed.csv")

        # Network (X)
        G = build_x_graph(df_x)
        metrics = nodes_metrics_df(G)
        export_tableau_csv(metrics, "results/graphs/x_nodes_metrics.csv")
        export_gexf(G, "results/graphs/x_interactions.gexf")
        # Edges para Tableau/Gephi
        edges = edges_from_x(df_x)
        if not edges.empty:
            export_tableau_csv(edges, "data/processed/x_edges.csv")
        print("✔ Red X → .gexf, x_nodes_metrics.csv, x_edges.csv")

    if df_tg is None and df_x is None:
        print("No input CSV files found. Use --telegram and/or --x to provide sources.")
        return

    entities = load_entities(args.entities, args.entities_file)

    # --- Unificado
    df_all = unify_frames(df_tg, df_x)
    if not df_all.empty:
        df_all = normalize_source(df_all)
        df_all = add_engagement(df_all)

        # Topic modeling (BERTopic) sobre todo el corpus disponible
        topic_text_col = "text_topic" if "text_topic" in df_all.columns else "text_clean"
        if topic_text_col == "text_topic":
            non_empty_ratio = (
                df_all["text_topic"].astype(str).str.strip() != ""
            ).mean()
            if non_empty_ratio < 0.7:
                print(
                    f"ⓘ text_topic sólo está disponible en {non_empty_ratio:.1%} de los posts;"
                    " usando text_clean para BERTopic."
                )
                topic_text_col = "text_clean"

        topic_mask = df_all[topic_text_col].astype(str).str.strip() != ""
        if topic_mask.any() and not args.skip_topics:
            topic_df = df_all[topic_mask].copy()
            lang_series = topic_df.get("lang", pd.Series([""] * len(topic_df), index=topic_df.index)).astype(str).str.strip().str.lower()

            topic_classifier_bundle = _load_topic_classifier()

            cyrillic_langs = {"ru", "uk", "bg", "be", "sr", "mk"}
            western_core_langs = {"en", "pl", "de", "fr", "it", "es", "pt", "nl"}
            western_scandi_langs = {"sv", "fi", "da", "no"}
            western_baltic_langs = {"lv", "lt", "et"}
            western_central_langs = {"cs", "sk", "hu", "ro", "tr"}

            group_configs = [
                {
                    "name": "cyrillic",
                    "mask": lang_series.isin(cyrillic_langs),
                    "fit_kwargs": {
                        "min_topic_size": 4,
                        "umap_kwargs": {"n_neighbors": 45},
                        "hdbscan_kwargs": {"min_cluster_size": 4, "min_samples": 1},
                    },
                },
                {
                    "name": "western_core",
                    "mask": lang_series.isin(western_core_langs),
                    "fit_kwargs": {
                        "min_topic_size": 7,
                        "umap_kwargs": {"n_neighbors": 32},
                        "hdbscan_kwargs": {"min_cluster_size": 7, "min_samples": 1},
                    },
                },
                {
                    "name": "western_scandi",
                    "mask": lang_series.isin(western_scandi_langs),
                    "fit_kwargs": {
                        "min_topic_size": 5,
                        "umap_kwargs": {"n_neighbors": 38},
                        "hdbscan_kwargs": {"min_cluster_size": 5, "min_samples": 1},
                    },
                },
                {
                    "name": "western_baltic",
                    "mask": lang_series.isin(western_baltic_langs),
                    "fit_kwargs": {
                        "min_topic_size": 3,
                        "umap_kwargs": {"n_neighbors": 36},
                        "hdbscan_kwargs": {"min_cluster_size": 3, "min_samples": 1},
                    },
                },
                {
                    "name": "western_central",
                    "mask": lang_series.isin(western_central_langs),
                    "fit_kwargs": {
                        "min_topic_size": 6,
                        "umap_kwargs": {"n_neighbors": 35},
                        "hdbscan_kwargs": {"min_cluster_size": 6, "min_samples": 1},
                    },
                },
                {
                    "name": "other",
                    "mask": pd.Series([True] * len(topic_df), index=topic_df.index),
                    "fit_kwargs": {
                        "min_topic_size": 5,
                        "umap_kwargs": {"n_neighbors": 35},
                        "hdbscan_kwargs": {"min_cluster_size": 5, "min_samples": 1},
                    },
                },
            ]

            topic_assignments: List[Dict[str, object]] = []
            summary_frames: List[pd.DataFrame] = []
            topics_tables: List[pd.DataFrame] = []
            global_topic_counter = 0
            used_mask = pd.Series([False] * len(topic_df), index=topic_df.index)

            group_summaries: List[str] = []
            group_metrics: List[Dict[str, object]] = []

            for group_cfg in group_configs:
                group_name = group_cfg["name"]
                group_mask = group_cfg["mask"] & ~used_mask
                if not group_mask.any():
                    continue
                used_mask |= group_mask
                group_df = topic_df[group_mask].copy()

                docs = group_df[topic_text_col].astype(str).tolist()
                ids = group_df["item_id"].astype(str).tolist()
                dts = group_df["timestamp"].astype(str).tolist()
                lang_codes = group_df["lang"].astype(str).tolist() if "lang" in group_df.columns else None
                docs_clean = group_df["text_clean"].astype(str).tolist() if "text_clean" in group_df.columns else docs

                cache_dir = (Path("models") / "bertopic" / f"group_{group_name}").as_posix()

                print(f"▶ BERTopic grupo {group_name} ({len(docs)} documentos)")

                topic_out = fit_topics(
                    docs,
                    ids=ids,
                    cache_dir=cache_dir,
                    min_topic_size=group_cfg["fit_kwargs"]["min_topic_size"],
                    n_gram_range=(1, 2),
                    seed=42,
                    lang_codes=lang_codes,
                    apply_reduce_outliers=group_cfg.get("apply_reduce_outliers", False),
                    reduce_outliers_kwargs=group_cfg.get("reduce_outliers_kwargs"),
                    umap_kwargs=group_cfg["fit_kwargs"].get("umap_kwargs"),
                    hdbscan_kwargs=group_cfg["fit_kwargs"].get("hdbscan_kwargs"),
                )

                assignments = topic_out["assignments"]
                _apply_classifier_predictions(topic_classifier_bundle, docs_clean, assignments)

                summary = summarize_topics(
                    topic_out["model"],
                    docs,
                    ids,
                    dts,
                    assignments=assignments,
                )

                mapping: Dict[int, int] = {}
                for original_tid in sorted({a["topic_id"] for a in assignments if int(a["topic_id"]) >= 0}):
                    mapping[int(original_tid)] = global_topic_counter
                    global_topic_counter += 1

                for assign in assignments:
                    original_tid = int(assign["topic_id"])
                    assign["topic_id_original"] = original_tid
                    assign["lang_group"] = group_name
                    if original_tid >= 0:
                        assign["topic_id"] = mapping.get(original_tid, original_tid)

                if "topic_id" in summary.columns:
                    summary["topic_id_original"] = summary["topic_id"].astype(int)
                    summary["topic_id"] = summary["topic_id"].astype(int).map(lambda tid: mapping.get(tid, tid))
                else:
                    summary["topic_id_original"] = -1
                summary["lang_group"] = group_name

                topic_assignments.extend(assignments)
                summary_frames.append(summary)

                topics_table_group = topic_out.get("model_info", {}).get("topics_table")
                if topics_table_group is not None:
                    topics_table_group = topics_table_group.copy()
                    if "Topic" in topics_table_group.columns:
                        topics_table_group["Topic_original"] = topics_table_group["Topic"].astype(int)
                        topics_table_group["Topic"] = topics_table_group["Topic"].astype(int).map(
                            lambda tid: mapping.get(tid, tid)
                        )
                    topics_table_group["lang_group"] = group_name
                    topics_tables.append(topics_table_group)

                model_info = topic_out.get("model_info", {})
                n_outliers = model_info.get("n_outliers", "n/a")
                n_docs = model_info.get("n_documents", len(docs))
                group_summaries.append(
                    f"{group_name}: docs={n_docs}, outliers={n_outliers}"
                )
                group_metrics.append({
                    "lang_group": group_name,
                    "n_documents": n_docs,
                    "n_outliers": n_outliers,
                    "outlier_ratio": model_info.get("outlier_ratio"),
                    "n_topics": model_info.get("n_topics"),
                })

            if group_summaries:
                print("ⓘ Resumen por grupo -> " + "; ".join(group_summaries))
            if group_metrics:
                metrics_df = pd.DataFrame(group_metrics)
                export_tableau_csv(metrics_df, "results/topics/topic_group_metrics.csv")

            # Reordenar tópicos globalmente por volumen para evitar bloques por idioma
            new_topic_id_map: Dict[int, int] = {}
            if topics_tables:
                combined_table_preview = pd.concat(topics_tables, ignore_index=True)
                positive_table = combined_table_preview[combined_table_preview["Topic"] >= 0].copy()
                if not positive_table.empty:
                    positive_table = (
                        positive_table[["Topic", "Count"]]
                        .groupby("Topic", as_index=False)["Count"]
                        .max()
                        .sort_values(["Count", "Topic"], ascending=[False, True])
                    )
                    new_topic_id_map = {
                        int(row.Topic): idx for idx, row in enumerate(positive_table.itertuples(index=False))
                    }

            if new_topic_id_map:
                for assign in topic_assignments:
                    tid = int(assign.get("topic_id", -1))
                    if tid >= 0 and tid in new_topic_id_map:
                        assign["topic_id"] = new_topic_id_map[tid]
                for summary in summary_frames:
                    if "topic_id" in summary.columns:
                        topic_vals = pd.to_numeric(summary["topic_id"], errors="coerce")
                        mask = topic_vals.notna() & (topic_vals.astype(int) >= 0)
                        if mask.any():
                            current_ids = topic_vals.loc[mask].astype(int)
                            mapped_ids = current_ids.map(new_topic_id_map)
                            mapped_ids = mapped_ids.fillna(current_ids)
                            summary.loc[mask, "topic_id"] = mapped_ids.astype(int)
                for table in topics_tables:
                    topic_vals = pd.to_numeric(table["Topic"], errors="coerce")
                    mask = topic_vals.notna() & (topic_vals.astype(int) >= 0)
                    if mask.any():
                        current_ids = topic_vals.loc[mask].astype(int)
                        mapped_ids = current_ids.map(new_topic_id_map)
                        mapped_ids = mapped_ids.fillna(current_ids)
                        table.loc[mask, "Topic"] = mapped_ids.astype(int)

            uid_to_topic = {a["uid"]: a for a in topic_assignments}

            df_all["topic_id"] = df_all["item_id"].astype(str).map(
                lambda uid: uid_to_topic.get(uid, {}).get("topic_id")
            )
            df_all["topic_label"] = df_all["item_id"].astype(str).map(
                lambda uid: uid_to_topic.get(uid, {}).get("label")
            )
            df_all["topic_score"] = df_all["item_id"].astype(str).map(
                lambda uid: uid_to_topic.get(uid, {}).get("score")
            )
            df_all["topic_terms"] = df_all["item_id"].astype(str).map(
                lambda uid: uid_to_topic.get(uid, {}).get("terms")
            )

            assignments_df = pd.DataFrame(topic_assignments)
            assignments_df = assignments_df.rename(
                columns={"uid": "item_id", "label": "topic_label", "score": "topic_score", "terms": "topic_terms"}
            )

            summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
            topics_table = pd.concat(topics_tables, ignore_index=True) if topics_tables else pd.DataFrame()
            if not topics_table.empty and "Count" in topics_table.columns:
                topics_table = topics_table.sort_values(["Count", "Topic"], ascending=[False, True]).reset_index(drop=True)

            manual_topics = _load_manual_topic_labels()
            manual_topic_map = {}
            manual_subtopic_map = {}
            manual_hierarchy_map: Dict[str, str] = {}
            if not manual_topics.empty:
                manual_topic_map = manual_topics.set_index("topic_id")["manual_label_topic"].to_dict()
                manual_subtopic_map = manual_topics.set_index("topic_id")["manual_label_subtopic"].to_dict()
                manual_hierarchy_map = _build_subtopic_parent_map(
                    manual_topics,
                    "manual_label_topic",
                    "manual_label_subtopic",
                )

            summary_df["topic_id_str"] = summary_df.get("topic_id", pd.NA).astype(str)
            summary_df["manual_label_topic"] = summary_df["topic_id_str"].map(manual_topic_map).fillna("")
            summary_df["manual_label_subtopic"] = summary_df["topic_id_str"].map(manual_subtopic_map).fillna("")
            _enforce_subtopic_hierarchy(summary_df, "manual_label_topic", "manual_label_subtopic", manual_hierarchy_map)

            if topic_classifier_bundle is None:
                topic_classifier_bundle = _load_topic_classifier()
            if (
                topic_classifier_bundle is not None
                and topic_classifier_bundle.get("kind") == "sentence-transformer"
                and not summary_df.empty
            ):
                if "top_terms" in summary_df.columns:
                    term_source = summary_df["top_terms"].apply(_topic_terms_to_text)
                elif "topic_terms" in summary_df.columns:
                    term_source = summary_df["topic_terms"].apply(_topic_terms_to_text)
                else:
                    term_source = pd.Series([""] * len(summary_df), index=summary_df.index)
                term_source = term_source.fillna("").astype(str)

                topic_clf = topic_classifier_bundle["topic_clf"]
                subtopic_clf = topic_classifier_bundle["subtopic_clf"]

                if topic_classifier_bundle.get("kind") == "sentence-transformer":
                    embed_name = topic_classifier_bundle.get("embedding_model_name", DEFAULT_CLASSIFIER_EMBED_MODEL)
                    normalize_embeddings = topic_classifier_bundle.get("normalize_embeddings", True)
                    batch_size = int(topic_classifier_bundle.get("inference_batch_size", 128))
                    embedder = _get_embedder(embed_name)
                    embeddings = embedder.encode(
                        term_source.tolist(),
                        batch_size=batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=normalize_embeddings,
                    )
                    topic_encoder = topic_classifier_bundle["topic_label_encoder"]
                    subtopic_encoder = topic_classifier_bundle["subtopic_label_encoder"]
                    topic_preds = pd.Series(
                        topic_encoder.inverse_transform(topic_clf.predict(embeddings)),
                        index=summary_df.index,
                    )
                    subtopic_preds = pd.Series(
                        subtopic_encoder.inverse_transform(subtopic_clf.predict(embeddings)),
                        index=summary_df.index,
                    )
                else:
                    vectorizer = topic_classifier_bundle.get("vectorizer")
                    if vectorizer is None:
                        topic_preds = pd.Series(["" for _ in range(len(summary_df))], index=summary_df.index)
                        subtopic_preds = pd.Series(["" for _ in range(len(summary_df))], index=summary_df.index)
                    else:
                        X_features = vectorizer.transform(term_source.tolist())
                        topic_preds = pd.Series(topic_clf.predict(X_features), index=summary_df.index)
                        subtopic_preds = pd.Series(subtopic_clf.predict(X_features), index=summary_df.index)

                mask_topic_missing = summary_df["manual_label_topic"].astype(str).str.strip() == ""
                summary_df.loc[mask_topic_missing, "manual_label_topic"] = topic_preds[mask_topic_missing]

                mask_subtopic_missing = summary_df["manual_label_subtopic"].astype(str).str.strip() == ""
                summary_df.loc[mask_subtopic_missing, "manual_label_subtopic"] = subtopic_preds[mask_subtopic_missing]

            hierarchy_map = _build_subtopic_parent_map(summary_df, "manual_label_topic", "manual_label_subtopic")
            _enforce_subtopic_hierarchy(summary_df, "manual_label_topic", "manual_label_subtopic", hierarchy_map)

            final_topic_map = summary_df.set_index("topic_id_str")["manual_label_topic"].to_dict()
            final_subtopic_map = summary_df.set_index("topic_id_str")["manual_label_subtopic"].to_dict()

            if not assignments_df.empty:
                assignments_df["manual_label_topic"] = (
                    assignments_df["topic_id"].astype(str).map(final_topic_map).fillna("")
                )
                assignments_df["manual_label_subtopic"] = (
                    assignments_df["topic_id"].astype(str).map(final_subtopic_map).fillna("")
                )
                _enforce_subtopic_hierarchy(assignments_df, "manual_label_topic", "manual_label_subtopic", hierarchy_map)

                required_topic_cols = {"Count", "manual_label_topic", "manual_label_subtopic", "top_terms"}
                for col in required_topic_cols:
                    if col not in topics_table.columns:
                        topics_table[col] = pd.NA

                topics_counts = assignments_df.groupby("topic_id", as_index=False).agg(
                    Count=("item_id", "count"),
                    manual_label_topic=("manual_label_topic", _majority_label),
                    manual_label_subtopic=("manual_label_subtopic", _majority_label),
                    topic_terms=("topic_terms", _first_non_empty),
                )

                if not topics_table.empty and "Topic" in topics_table.columns:
                    topics_table["Topic"] = pd.to_numeric(topics_table["Topic"], errors="coerce")
                    topics_table = topics_table.dropna(subset=["Topic"])
                    topics_table["Topic"] = topics_table["Topic"].astype(int)
                    topics_table = topics_table.drop_duplicates("Topic", keep="first").set_index("Topic")
                else:
                    topics_table = pd.DataFrame().set_index(pd.Index([], name="Topic"))

                for _, row in topics_counts.iterrows():
                    tid = int(row["topic_id"])
                    if tid not in topics_table.index:
                        topics_table.loc[tid, "Count"] = 0
                    topics_table.at[tid, "Count"] = row["Count"]
                    topics_table.at[tid, "manual_label_topic"] = row["manual_label_topic"]
                    topics_table.at[tid, "manual_label_subtopic"] = row["manual_label_subtopic"]
                    topic_terms_value = _serialize_terms(row["topic_terms"])
                    topics_table.at[tid, "top_terms"] = topic_terms_value

                topics_table = topics_table.reset_index()
                topics_table = topics_table.sort_values(["Count", "Topic"], ascending=[False, True]).reset_index(drop=True)

            if "topic_terms" in assignments_df.columns:
                assignments_df["topic_terms"] = assignments_df["topic_terms"].apply(_serialize_terms)
            export_tableau_csv(assignments_df, "data/processed/topics_assignments.csv")

            if "top_terms" in summary_df.columns:
                summary_df["top_terms"] = summary_df["top_terms"].apply(_serialize_terms)
            summary_df = summary_df.drop(columns=["topic_id_str"])
            export_tableau_csv(summary_df, "data/processed/topics_summary_daily.csv")

            if "TopTerms" in topics_table.columns:
                topics_table = topics_table.rename(columns={"TopTerms": "top_terms"})
            if "topic_terms" in topics_table.columns:
                topics_table = topics_table.rename(columns={"topic_terms": "top_terms"})
            if "Representation" in topics_table.columns:
                topics_table["top_terms"] = topics_table["Representation"]
            elif "top_terms" in topics_table.columns:
                topics_table["top_terms"] = topics_table["top_terms"].apply(_serialize_terms)

            topic_id_col = None
            for candidate in ["Topic", "topic_id"]:
                if candidate in topics_table.columns:
                    topic_id_col = candidate
                    break
            if topic_id_col:
                topics_table["manual_label_topic"] = (
                    topics_table[topic_id_col].astype(str).map(final_topic_map).fillna("")
                )
                topics_table["manual_label_subtopic"] = (
                    topics_table[topic_id_col].astype(str).map(final_subtopic_map).fillna("")
                )
                _enforce_subtopic_hierarchy(topics_table, "manual_label_topic", "manual_label_subtopic", hierarchy_map)
            if "Count" in topics_table.columns:
                topics_table["Count"] = pd.to_numeric(topics_table["Count"], errors="coerce").fillna(0).astype(int)
            topics_table = topics_table.loc[:, ~topics_table.columns.duplicated()]
            preferred_cols = [
                "Topic",
                "topic_id",
                "Count",
                "Name",
                "Representation",
                "top_terms",
                "manual_label_topic",
                "manual_label_subtopic",
            ]
            ordered = [col for col in preferred_cols if col in topics_table.columns]
            topics_table = topics_table[ordered + [col for col in topics_table.columns if col not in ordered]]
            export_tableau_csv(topics_table, "results/topics/topic_info.csv")

            df_all["topic_id_str"] = df_all["topic_id"].astype(str)
            df_all["manual_label_topic"] = df_all["topic_id_str"].map(final_topic_map).fillna("")
            df_all["manual_label_subtopic"] = df_all["topic_id_str"].map(final_subtopic_map).fillna("")
            mask_topic_override = df_all["manual_label_topic"].astype(str).str.strip() != ""
            df_all.loc[mask_topic_override, "topic_label"] = df_all.loc[mask_topic_override, "manual_label_topic"]
            _enforce_subtopic_hierarchy(df_all, "manual_label_topic", "manual_label_subtopic", hierarchy_map)
            df_all = df_all.drop(columns=["topic_id_str"])
        else:
            df_all["topic_id"] = pd.NA
            df_all["topic_label"] = pd.NA
            df_all["topic_score"] = pd.NA

        # Entity-conditioned sentimiento + fallback
        if entities:
            caption_col = "text_caption_clean" if "text_caption_clean" in df_all.columns else None
            summary_col = "text_summary_clean" if "text_summary_clean" in df_all.columns else None
            mentions = extract_entity_mentions(
                df_all,
                entities,
                text_col="text_clean",
                id_col="item_id",
                topic_id_col="topic_id",
                topic_label_col="topic_label",
                context_window=args.entity_window,
                caption_col=caption_col,
                summary_col=summary_col,
                caption_weight=0.8,
                summary_weight=0.2,
            )

            source_series = df_all["source"] if "source" in df_all.columns else pd.Series(dtype=str)
            sources_present = {
                str(src).strip()
                for src in source_series.dropna().unique().tolist()
                if str(src).strip()
            }
            print("ⓘ Sources present before entity extraction:", sorted(sources_present))
            print(
                "ⓘ Sample counts by source:",
                {
                    src: int((df_all["source"].astype(str).str.strip() == src).sum())
                    for src in sources_present
                },
            )

            detected_sources = Counter(
                (getattr(m, "source", "") or "").strip() or "unknown" for m in mentions
            )
            existing_item_ids = {
                str(getattr(m, "item_id", "")).strip()
                for m in mentions
                if getattr(m, "item_id", None)
            }
            print("ⓘ Detected sources (initial):", dict(detected_sources))

            fallback_mentions = _fallback_extract_mentions(
                df_all,
                context_window=args.entity_window,
                existing_item_ids=existing_item_ids,
            )
            if fallback_mentions:
                mentions.extend(fallback_mentions)
                detected_sources.update(
                    (getattr(m, "source", "") or "").strip() or "unknown"
                    for m in fallback_mentions
                )
                fallback_sources = sorted({(getattr(m, "source", "") or "").strip() or "unknown" for m in fallback_mentions})
                print(
                    "✔ Fallback entity detection added"
                    f" {len(fallback_mentions)} mentions for sources: {', '.join(fallback_sources)}"
                )

            if mentions:
                formatted_counts = ", ".join(
                    f"{src or 'unknown'}: {count}"
                    for src, count in sorted(detected_sources.items(), key=lambda kv: kv[0])
                )
                print(
                    f"✔ Entity mentions detected before scoring ({len(mentions)} total) → {formatted_counts}"
                )

            if args.skip_emotions:
                mentions_df = mentions_to_dataframe(mentions)
            else:
                mentions_df = score_entity_mentions(
                    mentions,
                    sentiment_device=args.device,
                    emotion_device=args.device,
                    emotion_model=args.emotion_model,
                )

            if not mentions_df.empty:
                mentions_export = serialize_mentions_for_export(mentions_df)
                export_tableau_csv(mentions_export, "data/processed/entity_mentions.csv")
                if not args.skip_emotions:
                    summary_mentions = summarize_entity_mentions(mentions_df)
                    export_tableau_csv(summary_mentions, "data/processed/entity_topic_summary.csv")
                    item_summary = aggregate_mentions_per_item(mentions_df)
                    if not item_summary.empty:
                        df_all = df_all.merge(item_summary, on="item_id", how="left")

                item_to_mentions = {}
                for rec in mentions_df.to_dict(orient="records"):
                    payload = {
                        "entity": rec.get("entity"),
                        "alias": rec.get("alias"),
                        "stance": rec.get("stance"),
                        "stance_value": rec.get("stance_value"),
                        "sentiment_label": rec.get("sentiment_label"),
                        "sentiment_score": rec.get("sentiment_score"),
                        "emotion_label": rec.get("emotion_label"),
                        "topic_id": rec.get("topic_id"),
                        "topic_label": rec.get("topic_label"),
                        "topic_score": rec.get("topic_score"),
                        "impact_score": rec.get("impact_score"),
                        "engagement": rec.get("engagement"),
                        "reach": rec.get("reach"),
                        "snippet": rec.get("snippet"),
                        "text_source": rec.get("text_source"),
                        "sentiment_dist": rec.get("sentiment_dist"),
                        "emotion_scores": rec.get("emotion_scores"),
                    }
                    item_id_key = str(rec.get("item_id"))
                    item_to_mentions.setdefault(item_id_key, []).append(payload)

                df_all["entity_mentions"] = df_all["item_id"].astype(str).apply(
                    lambda uid: json.dumps(item_to_mentions.get(uid, []), ensure_ascii=False)
                )
                mention_source_series = (
                    mentions_df["source"] if "source" in mentions_df.columns else pd.Series(dtype=str)
                )
                sources_with_mentions = {
                    str(src).strip()
                    for src in mention_source_series.dropna().unique().tolist()
                    if str(src).strip()
                }
                missing_after = sorted(s for s in sources_present if s not in sources_with_mentions)
                if missing_after:
                    print(
                        "⚠️ No entity mentions extracted for sources: "
                        + ", ".join(missing_after)
                    )
            else:
                print("ⓘ No mentions found for the configured entities.")
                df_all["entity_mentions"] = df_all["item_id"].astype(str).apply(lambda _: "[]")
        else:
            print("ⓘ Análisis de entidades omitido (sin entidades configuradas).")
            df_all["entity_mentions"] = df_all["item_id"].astype(str).apply(lambda _: "[]")

        numeric_defaults = {
            "impact_score": 0.0,
            "impact_score_mean": 0.0,
            "n_entity_mentions": 0,
            "stance_value": 0.0,
        }
        for col, default in numeric_defaults.items():
            if col in df_all.columns:
                df_all[col] = df_all[col].fillna(default)
            else:
                df_all[col] = default

        if "stance" in df_all.columns:
            df_all["stance"] = df_all["stance"].fillna("neu")
        else:
            df_all["stance"] = "neu"

        if "entities_detected" in df_all.columns:
            df_all["entities_detected"] = df_all["entities_detected"].apply(
                lambda v: list(v) if isinstance(v, (list, tuple, set)) else []
            )
        else:
            df_all["entities_detected"] = [[] for _ in range(len(df_all))]

        if "sentiment_dist" in df_all.columns:
            df_all["sentiment_dist"] = df_all["sentiment_dist"].apply(
                lambda v: v if isinstance(v, dict) else {}
            )
        else:
            df_all["sentiment_dist"] = [{} for _ in range(len(df_all))]

        if "emotion_scores" in df_all.columns:
            df_all["emotion_scores"] = df_all["emotion_scores"].apply(
                lambda v: v if isinstance(v, dict) else {}
            )
        else:
            df_all["emotion_scores"] = [{} for _ in range(len(df_all))]

        if "related_entities" in df_all.columns:
            df_all["related_entities"] = df_all["related_entities"].apply(
                lambda v: v if isinstance(v, str) else json.dumps(v or [], ensure_ascii=False)
            )
        else:
            df_all["related_entities"] = [json.dumps([], ensure_ascii=False) for _ in range(len(df_all))]

        if "entity_sentiment_polarity" not in df_all.columns:
            df_all["entity_sentiment_polarity"] = [json.dumps([], ensure_ascii=False) for _ in range(len(df_all))]

        if "topic_terms" in df_all.columns:
            df_all["topic_terms"] = df_all["topic_terms"].apply(
                lambda v: list(v) if isinstance(v, (list, tuple))
                else ([] if v in (None, "") else [str(v)])
            )
        else:
            df_all["topic_terms"] = [[] for _ in range(len(df_all))]

        df_all = add_dominant_emotion(df_all)

        alias_maps = (NATO_ALIASES, RUSSIA_ALIASES, RELATED_TO_PRINCIPAL)
        total_posts = len(df_all)
        geo_df, posts_with_geo = _build_geo_table(df_all)
        entity_df, raw_entity_rows, posts_with_entities = _build_entity_table(
            df_all,
            alias_maps,
            drop_unknown=DROP_UNKNOWN,
        )
        geo_rows_count = len(geo_df)
        entity_rows_after = len(entity_df)

        if args.split_outputs:
            export_tableau_csv(geo_df, "data/processed/geo_country_exploded.csv")
            export_tableau_csv(entity_df, "data/processed/facts_posts_tableau.csv")
            print("✔ Generated split outputs (geo_country_exploded.csv, facts_posts_tableau.csv)")

        print("Pipeline stats:")
        print(f"  Total posts: {total_posts}")
        print(f"  Posts with geo data: {posts_with_geo}")
        print(f"  Posts with entities: {posts_with_entities}")
        print(f"  Geo rows exploded: {geo_rows_count}")
        print(f"  Entity rows exploded (raw/deduped): {raw_entity_rows}/{entity_rows_after}")

        # Base de hechos para Tableau
        facts_cols = [
            "source","timestamp","author","author_location","item_id","lang","geolocation","geo_country_distribution",
            "stance","stance_value",
            "impact_score","impact_score_mean","n_entity_mentions","entities_detected",
            "emoji_count","text_clean","text_topic","topic_terms","link","likes","retweets","replies","quotes","views",
            "topic_id","topic_label","topic_score","manual_label_topic","manual_label_subtopic","related_entities","entity_sentiment_polarity","entity_mentions"
        ]
        facts_cols = [c for c in facts_cols if c in df_all.columns]
        facts = df_all[facts_cols].copy()

        def _to_dict_safe(val):
            if isinstance(val, dict):
                return val
            if isinstance(val, str) and val.strip().startswith("{"):
                try:
                    return json.loads(val)
                except Exception:
                    return {}
            return {}

        if "impact_score" in facts.columns:
            facts["impact_score"] = facts["impact_score"].fillna(0.0)
        if "impact_score_mean" in facts.columns:
            facts["impact_score_mean"] = facts["impact_score_mean"].fillna(0.0)
        if "stance" in facts.columns:
            facts["stance"] = facts["stance"].fillna("neu")
        if "entities_detected" in facts.columns:
            facts["entities_detected"] = facts["entities_detected"].apply(
                lambda v: json.dumps(list(v), ensure_ascii=False)
                if isinstance(v, (list, tuple, set))
                else ("[]" if pd.isna(v) or v == "" else str(v))
            )
        if "entity_mentions" in facts.columns:
            facts["entity_mentions"] = facts["entity_mentions"].fillna("[]")
        if "topic_terms" in facts.columns:
            facts["topic_terms"] = facts["topic_terms"].apply(
                lambda v: json.dumps(list(v), ensure_ascii=False)
                if isinstance(v, (list, tuple))
                else ("[]" if pd.isna(v) or v == "" else str(v))
            )
        if "geo_country_distribution" in facts.columns:
            facts["geo_country_distribution"] = facts["geo_country_distribution"].apply(
                _ensure_json_array
            )
        # engagement coherente (0 si falta)
        if set(["likes","retweets","replies","quotes"]).issubset(facts.columns):
            facts["engagement"] = facts[["likes","retweets","replies","quotes"]].fillna(0).astype(float).sum(axis=1)
        else:
            facts["engagement"] = 0

        export_tableau_csv(facts, "data/processed/facts_posts.csv")
        if not args.split_outputs:
            _derive_facts_posts_tableau(Path("data/processed/facts_posts.csv"))

        # Emotions long
        emo_long = emotions_to_long(df_all)
        if not emo_long.empty:
            export_tableau_csv(emo_long, "data/processed/emotions_long.csv")

        if "entities_detected" in df_all.columns:
            df_all["entities_detected"] = df_all["entities_detected"].apply(
                lambda v: json.dumps(list(v), ensure_ascii=False)
                if isinstance(v, (list, tuple, set))
                else ("[]" if pd.isna(v) or v == "" else str(v))
            )
        if "sentiment_dist" in df_all.columns:
            df_all["sentiment_dist"] = df_all["sentiment_dist"].apply(
                lambda v: json.dumps(v, ensure_ascii=False)
                if isinstance(v, dict)
                else ("{}" if pd.isna(v) or v == "" else str(v))
            )
        if "emotion_scores" in df_all.columns:
            df_all["emotion_scores"] = df_all["emotion_scores"].apply(
                lambda v: json.dumps(v, ensure_ascii=False)
                if isinstance(v, dict)
                else ("{}" if pd.isna(v) or v == "" else str(v))
            )
        if "topic_terms" in df_all.columns:
            df_all["topic_terms"] = df_all["topic_terms"].apply(
                lambda v: json.dumps(list(v), ensure_ascii=False)
                if isinstance(v, (list, tuple))
                else ("[]" if pd.isna(v) or v == "" else str(v))
            )
        if "geo_country_distribution" in df_all.columns:
            df_all["geo_country_distribution"] = df_all["geo_country_distribution"].apply(
                _ensure_json_array
            )

        # Unificado completo
        export_tableau_csv(df_all, "data/processed/all_platforms.csv")
        print("✔ facts_posts.csv, emotions_long.csv, all_platforms.csv generados")

if __name__ == "__main__":
    main()


# === Finalize manual and classifier labels on processed CSVs ===

import atexit

def _normalize_topic_id_value(v):
    try:
        import pandas as _pd
    except Exception:
        # Minimal normalization w/o pandas
        if v is None:
            return ""
        s = str(v).strip()
        try:
            f = float(s)
            i = int(f)
            return str(i) if f == float(i) else s
        except Exception:
            return s
    if _pd.isna(v):
        return ""
    s = str(v).strip()
    try:
        f = float(s)
        i = int(f)
        return str(i) if f == float(i) else s
    except Exception:
        return s

def _load_manual_label_table():
    import pandas as pd
    p = Path("data") / "ground_truth" / "topics_manual_labels.csv"
    if not p.exists():
        print("ⓘ Ground truth topics_manual_labels.csv not found; skipping manual merge.")
        return pd.DataFrame()
    df = pd.read_csv(p, sep=";", encoding="utf-8-sig", low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    for col in ("topic_id","manual_label_topic","manual_label_subtopic"):
        if col not in df.columns:
            print("ⓘ Ground truth missing columns; skipping manual merge.")
            return pd.DataFrame()
    df["topic_id_norm"] = df["topic_id"].apply(_normalize_topic_id_value)
    return df[["topic_id_norm","manual_label_topic","manual_label_subtopic"]].drop_duplicates()

def _load_classifier_bundle():
    p = Path("models") / "topic_classifier" / "topic_classifier.joblib"
    if not p.exists():
        return None
    try:
        import joblib  # type: ignore
        bundle = joblib.load(p)
        return bundle
    except Exception as e:
        print(f"ⓘ Could not load topic classifier: {e}")
        return None

def _predict_manual_labels(bundle, texts_series):
    kind = bundle.get("kind")
    model_name = bundle.get("embedding_model_name")
    if not model_name:
        raise RuntimeError("Classifier bundle missing 'embedding_model_name'.")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(f"sentence-transformers unavailable: {e}")
    embedder = SentenceTransformer(model_name)
    texts = texts_series.fillna("").astype(str).tolist()
    embs = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    import numpy as np
    X = np.array(embs)

    if kind == "sentence-transformer":
        topic_clf = bundle["topic_clf"]
        sub_clf = bundle["subtopic_clf"]
        le_topic = bundle["topic_label_encoder"]
        le_sub = bundle["subtopic_label_encoder"]
        t_idx = topic_clf.predict(X)
        s_idx = sub_clf.predict(X)
        import pandas as pd
        t = pd.Series(le_topic.inverse_transform(t_idx), index=texts_series.index)
        s = pd.Series(le_sub.inverse_transform(s_idx), index=texts_series.index)
        return t, s
    elif kind == "sentence-transformer-multitask":
        manual_topic_clf = bundle["manual_topic_clf"]
        manual_subtopic_clf = bundle["manual_subtopic_clf"]
        manual_topic_encoder = bundle["manual_topic_label_encoder"]
        manual_subtopic_encoder = bundle["manual_subtopic_label_encoder"]
        t_idx = manual_topic_clf.predict(X)
        s_idx = manual_subtopic_clf.predict(X)
        import pandas as pd
        t = pd.Series(manual_topic_encoder.inverse_transform(t_idx), index=texts_series.index)
        s = pd.Series(manual_subtopic_encoder.inverse_transform(s_idx), index=texts_series.index)
        return t, s
    else:
        raise RuntimeError(f"Unknown classifier kind: {kind}")

def _export_with_bom(df, path_str):
    # Use project export helper if available to keep ; and utf-8-sig
    try:
        export_tableau_csv(df, path_str)  # from src.utils
    except Exception:
        import pandas as pd
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path_str, sep=";", encoding="utf-8-sig", index=False)

def _apply_labels_to_dataframe(df, name, manuals, bundle):
    import pandas as pd
    if df is None or df.empty:
        return df

    cols = {c.lower(): c for c in df.columns}
    topic_col = cols.get("topic_id")
    if not topic_col:
        return df

    if topic_col != "topic_id":
        df = df.rename(columns={topic_col: "topic_id"})

    manual_topic_col = cols.get("manual_label_topic")
    if manual_topic_col and manual_topic_col != "manual_label_topic":
        df = df.rename(columns={manual_topic_col: "manual_label_topic"})
    manual_subtopic_col = cols.get("manual_label_subtopic")
    if manual_subtopic_col and manual_subtopic_col != "manual_label_subtopic":
        df = df.rename(columns={manual_subtopic_col: "manual_label_subtopic"})

    if "manual_label_topic" not in df.columns:
        df["manual_label_topic"] = pd.NA
    if "manual_label_subtopic" not in df.columns:
        df["manual_label_subtopic"] = pd.NA

    df["topic_id_norm"] = df["topic_id"].apply(_normalize_topic_id_value)

    before_t = df["manual_label_topic"].notna().sum()
    before_s = df["manual_label_subtopic"].notna().sum()

    if manuals is not None and not manuals.empty:
        df = df.merge(manuals, on="topic_id_norm", how="left", suffixes=("", "_manual_ref"))
        for col in ("manual_label_topic", "manual_label_subtopic"):
            ref = f"{col}_manual_ref"
            if ref in df.columns:
                df[col] = df[col].where(df[col].notna() & (df[col].astype(str).str.strip() != ""), df[ref])
                df.drop(columns=[ref], inplace=True, errors="ignore")

    text_col = cols.get("text") or cols.get("text_clean")
    if bundle is not None and (text_col or "text" in df.columns):
        working_text_col = "text"
        if text_col and text_col != "text" and text_col in df.columns:
            df = df.rename(columns={text_col: working_text_col})
        if working_text_col in df.columns:
            try:
                mask = df["manual_label_topic"].isna() | (df["manual_label_topic"].astype(str).str.strip() == "")
                if mask.any():
                    t_pred, s_pred = _predict_manual_labels(bundle, df.loc[mask, working_text_col])
                    df.loc[mask, "manual_label_topic"] = df.loc[mask, "manual_label_topic"].fillna(t_pred)
                    df.loc[mask, "manual_label_subtopic"] = df.loc[mask, "manual_label_subtopic"].fillna(s_pred)
            except Exception as e:
                print(f"ⓘ Classifier skipped for {name}: {e}")
        if text_col and text_col != "text":
            df = df.rename(columns={working_text_col: text_col})

    after_t = df["manual_label_topic"].notna().sum()
    after_s = df["manual_label_subtopic"].notna().sum()
    df.drop(columns=["topic_id_norm"], inplace=True, errors="ignore")

    if topic_col and topic_col != "topic_id":
        df = df.rename(columns={"topic_id": topic_col})
    if manual_topic_col and manual_topic_col != "manual_label_topic":
        df = df.rename(columns={"manual_label_topic": manual_topic_col})
    if manual_subtopic_col and manual_subtopic_col != "manual_label_subtopic":
        df = df.rename(columns={"manual_label_subtopic": manual_subtopic_col})

    print(f"✔ {name}: topics {before_t}→{after_t} / subtopics {before_s}→{after_s}")
    return df

def _finalize_processed_topic_labels():
    import pandas as pd
    processed = Path("data") / "processed"
    if not processed.exists():
        return

    manuals = _load_manual_label_table()
    bundle = _load_classifier_bundle()

    # Fixed set we know about + autodiscovery
    candidates = [
        "facts_posts.csv",
        "facts_posts_tableau.csv",
        "all_platforms.csv",
        "entity_mentions.csv",
        "entity_topic_summary.csv",
    ]
    # Add any other csvs with topic_id
    for p in processed.glob("*.csv"):
        if p.name not in candidates:
            try:
                df_probe = pd.read_csv(p, sep=";", encoding="utf-8-sig", nrows=5, low_memory=False)
            except Exception:
                try:
                    df_probe = pd.read_csv(p, sep=",", encoding="utf-8", nrows=5, low_memory=False)
                except Exception:
                    continue
            if "topic_id" in [c.lower() for c in df_probe.columns]:
                candidates.append(p.name)

    for name in sorted(set(candidates)):
        p = processed / name
        if not p.exists():
            continue
        try:
            try:
                df = pd.read_csv(p, sep=";", encoding="utf-8-sig", low_memory=False)
            except Exception:
                df = pd.read_csv(p, sep=",", encoding="utf-8", low_memory=False)
            df2 = _apply_labels_to_dataframe(df, name, manuals, bundle)
            _export_with_bom(df2, str(p))
        except Exception as e:
            print(f"ⓘ Skipped {name}: {e}")

# Register finalize step to run after the script completes its normal pipeline
atexit.register(_finalize_processed_topic_labels)
