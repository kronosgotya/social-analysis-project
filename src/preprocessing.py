# -*- coding: utf-8 -*-
"""
Preprocesamiento para Telegram y X (fecha simple):
- Lector CSV tolerante (UTF-8 con/sin BOM, autodetección de separador, salta líneas malas)
- Normalización de FECHA (YYYY-MM-DD) sin horas/zonas
- Extracción de emojis (si está `emoji`)
- Limpieza de espacios
- X: rellena tweet_id desde URL si falta, o crea ID sustituto estable
- Unificación de esquemas (item_id común)
"""

from __future__ import annotations
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import csv
import pandas as pd
from stopwordsiso import stopwords as stopwords_iso
from simplemma import simple_tokenizer, lemmatize

try:  # langdetect es opcional; lo usamos si está disponible
    from langdetect import detect_langs, DetectorFactory, LangDetectException  # type: ignore
    DetectorFactory.seed = 42
    _LANGDETECT_AVAILABLE = True
except Exception:  # pragma: no cover - fallback cuando no está instalado
    detect_langs = None  # type: ignore
    LangDetectException = Exception  # type: ignore
    _LANGDETECT_AVAILABLE = False

_LANG_FALLBACK = {
    "en": "en",
    "es": "es",
    "en-gb": "en",
    "en-us": "en",
    "en_uk": "en",
    "ru": "ru",
    "uk": "uk",
    "ua": "uk",
    "pl": "pl",
    "cs": "cs",
    "sk": "sk",
    "ro": "ro",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "tr": "tr",
    "sv": "sv",
    "fi": "fi",
    "lt": "lt",
    "lv": "lv",
    "et": "et",
    "da": "da",
    "no": "no",
    "bg": "bg",
    "sr": "sr",
    "mk": "mk",
    "hu": "hu",
    "ar": "ar",
}

_GLOBAL_STOPWORDS: Dict[str, set] = {}
_DEFAULT_LANG = "en"
_LANG_UNKNOWN = "und"
_TOKEN_MIN_LENGTH = 3
_TOPIC_STOPWORDS_EXTRA = {"https", "http", "amp", "rt", "t.me"}
_LANG_MISSING_TOKENS = {
    "",
    "und",
    "unk",
    "unknown",
    "none",
    "null",
    "nan",
    "n/a",
    "na",
    "?",
    "??",
}
_LANGDETECT_MIN_LENGTH = 20
_LANGDETECT_MAX_CHARS = 1000

_LANG_COUNTRY_LOOKUP_CACHE: Optional[Dict[str, List[Dict[str, float]]]] = None
_LANG_COUNTRY_LOOKUP_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "lang_country_lookup.csv"
)
_HANDLE_ALLOWED_RE = re.compile(r"^[A-Za-z0-9_]{1,30}$")
_HANDLE_URL_RE = re.compile(r"(?:twitter|x)\.com/([^/\s]+)/", flags=re.IGNORECASE)

def _normalize_lang_code(lang: Optional[str]) -> str:
    if not isinstance(lang, str) or not lang.strip():
        return _DEFAULT_LANG
    token = lang.strip().lower().replace("_", "-")
    token = token.split("-")[0]
    if token in _LANG_FALLBACK:
        token = _LANG_FALLBACK[token]
    return token if token else _DEFAULT_LANG

def _stopwords_for(lang: str) -> set:
    if lang not in _GLOBAL_STOPWORDS:
        try:
            sw = set(stopwords_iso(lang))
        except Exception:
            sw = set()
        if lang != _DEFAULT_LANG and not sw:
            sw = _stopwords_for(_DEFAULT_LANG)
        _GLOBAL_STOPWORDS[lang] = sw | _TOPIC_STOPWORDS_EXTRA
    return _GLOBAL_STOPWORDS[lang]

def preprocess_for_topics(text: str, lang: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    lowered = text.strip().lower()
    if lowered in {"", "nan", "none", "null"}:
        return ""
    lang_code = _normalize_lang_code(lang)
    stopwords = _stopwords_for(lang_code)
    processed: List[str] = []
    for token in simple_tokenizer(text):
        token_lower = token.lower()
        if len(token_lower) < _TOKEN_MIN_LENGTH and not token_lower.isdigit():
            continue
        if token_lower in stopwords:
            continue
        lemma = token_lower
        try:
            lemma_candidate = lemmatize(token_lower, lang=lang_code)
            if lemma_candidate:
                lemma = lemma_candidate.lower()
        except Exception:
            pass
        if len(lemma) < _TOKEN_MIN_LENGTH and not lemma.isdigit():
            continue
        if lemma in stopwords:
            continue
        processed.append(lemma)
    return " ".join(processed)

def _prepare_topic_series(texts: Iterable[str], langs: Iterable[str]) -> List[str]:
    return [preprocess_for_topics(txt, lang) for txt, lang in zip(texts, langs)]


def _load_lang_country_lookup() -> Dict[str, List[Dict[str, float]]]:
    global _LANG_COUNTRY_LOOKUP_CACHE
    if _LANG_COUNTRY_LOOKUP_CACHE is not None:
        return _LANG_COUNTRY_LOOKUP_CACHE

    table: Dict[str, List[Dict[str, float]]] = {}
    if _LANG_COUNTRY_LOOKUP_PATH.exists():
        try:
            lookup_df = pd.read_csv(_LANG_COUNTRY_LOOKUP_PATH)
            required_cols = {"lang", "country", "weight"}
            if required_cols.issubset(set(lookup_df.columns)):
                for lang_code, group in lookup_df.groupby("lang"):
                    lang_token = str(lang_code).strip().lower()
                    if not lang_token:
                        continue
                    normalized_lang = _normalize_lang_code(lang_token)
                    records: List[Dict[str, float]] = []
                    for _, row in group.iterrows():
                        country = str(row.get("country") or "").strip()
                        if not country:
                            continue
                        weight_raw = row.get("weight")
                        try:
                            weight_val = float(weight_raw)
                        except (TypeError, ValueError):
                            weight_val = 0.0
                        if weight_val < 0:
                            weight_val = 0.0
                        records.append({"country": country, "weight": weight_val})
                    total_weight = sum(entry["weight"] for entry in records)
                    if total_weight > 0:
                        normalized_records = [
                            {"country": entry["country"], "weight": entry["weight"] / total_weight}
                            for entry in records
                        ]
                    else:
                        normalized_records = []
                    if normalized_records:
                        table[normalized_lang] = normalized_records
                        raw_key = lang_token
                        if raw_key != normalized_lang:
                            table[raw_key] = [dict(entry) for entry in normalized_records]
        except Exception:
            table = {}
    _LANG_COUNTRY_LOOKUP_CACHE = table
    return table


def _parse_geolocation_tokens(value: str) -> List[str]:
    if not isinstance(value, str):
        return []
    value = value.strip()
    if not value:
        return []
    tokens = [token.strip() for token in re.split(r"[;,/|]", value) if token.strip()]
    if tokens:
        return tokens
    return [value]


def _extract_handle_candidate(value: object) -> str:
    """
    Devuelve un posible handle (sin '@') si cumple el patrón estándar.
    """
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return ""
        if token.startswith("@"):
            token = token[1:]
        if _HANDLE_ALLOWED_RE.match(token):
            return token
        match = _HANDLE_URL_RE.search(token)
        if match:
            candidate = match.group(1).strip("@")
            if _HANDLE_ALLOWED_RE.match(candidate):
                return candidate
    return ""


def _extract_handle_from_url(value: object) -> str:
    if not isinstance(value, str):
        return ""
    token = value.strip()
    if not token:
        return ""
    match = _HANDLE_URL_RE.search(token)
    if not match:
        return ""
    candidate = match.group(1).split("?")[0].split("#")[0].strip("@")
    return candidate if _HANDLE_ALLOWED_RE.match(candidate) else ""


def _parse_mentions_field(value: object) -> Tuple[List[str], List[str]]:
    """
    Devuelve dos listas: handles con arroba y sin arroba.
    """
    with_prefix: List[str] = []
    plain: List[str] = []

    if value is None:
        return with_prefix, plain

    tokens: List[str]
    if isinstance(value, list):
        tokens = [str(item) for item in value]
    else:
        text = str(value).strip()
        if not text:
            return with_prefix, plain
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                tokens = [str(item) for item in parsed]
            else:
                tokens = [text]
        except Exception:
            tokens = [tok for tok in re.split(r"[;\s,|]+", text) if tok]

    for raw in tokens:
        candidate = str(raw).strip()
        if not candidate:
            continue
        if candidate.startswith("@"):
            bare = candidate[1:].strip()
        else:
            bare = candidate.strip("@")
            candidate = f"@{bare}" if bare else ""
        if not bare or not _HANDLE_ALLOWED_RE.match(bare):
            continue
        if candidate and candidate not in with_prefix:
            with_prefix.append(candidate)
        if bare not in plain:
            plain.append(bare)
    return with_prefix, plain


def _geo_country_distribution(geo_value: str, lang_value: str) -> List[Dict[str, object]]:
    tokens = _parse_geolocation_tokens(geo_value)
    if tokens:
        weight = 1.0 / len(tokens)
        return [
            {"country": token, "weight": weight, "method": "reported"}
            for token in tokens
        ]

    lang_lookup = _load_lang_country_lookup()
    lang_key = _normalize_lang_code(lang_value)
    records = lang_lookup.get(lang_key) or lang_lookup.get(str(lang_value).strip().lower(), [])
    if records:
        return [
            {"country": entry["country"], "weight": float(entry["weight"]), "method": "lang_lookup"}
            for entry in records
        ]
    return []

# =========================
# Lector CSV robusto
# =========================
def _read_csv_utf8(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
            sep=None,  # autodetect
            engine="python",
            quotechar='"',
            doublequote=True,
            escapechar="\\",
            on_bad_lines="skip",
        )
    except Exception:
        pass

    for sep in [";", ",", "\t", "|"]:
        try:
            return pd.read_csv(
                path,
                encoding="utf-8-sig",
                dtype=str,
                keep_default_na=False,
                sep=sep,
                engine="python",
                quotechar='"',
                doublequote=True,
                escapechar="\\",
                on_bad_lines="skip",
            )
        except Exception:
            continue

    return pd.read_csv(
        path,
        encoding="latin-1",
        dtype=str,
        keep_default_na=False,
        sep=None,
        engine="python",
        on_bad_lines="skip",
    )


def _read_telegram_csv(path: str) -> pd.DataFrame:
    """Robust reader for Telegram exports with embedded newlines/semicolons."""

    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter=";", quotechar="\"", doublequote=True)
            rows = list(reader)
        if not rows:
            return pd.DataFrame()
        header = rows[0]
        width = len(header)
        data: List[List[str]] = []
        for row in rows[1:]:
            if not row:
                continue
            if len(row) < width:
                row = row + [""] * (width - len(row))
            elif len(row) > width:
                row = row[:width]
            data.append(row)
        df = pd.DataFrame(data, columns=header)
        return df
    except Exception:
        return _read_csv_utf8(path)

# =========================
# Helpers
# =========================
def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r"\s+", " ", text).strip()


def _clean_lang_token(value: object) -> str:
    if isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return ""
        token = token.replace("_", "-")
        token = re.split(r"[;,|]", token)[0].strip()
        if "/" in token:
            token = token.split("/")[0].strip()
        if token in _LANG_MISSING_TOKENS:
            return ""
        return token
    return ""


def _detect_language_candidate(text: object) -> str:
    if not _LANGDETECT_AVAILABLE:
        return ""
    if not isinstance(text, str):
        return ""
    sample = normalize_whitespace(text)
    if len(sample) < _LANGDETECT_MIN_LENGTH:
        return ""
    sample = sample[:_LANGDETECT_MAX_CHARS]
    try:
        guesses = detect_langs(sample)  # type: ignore[operator]
    except LangDetectException:  # type: ignore
        return ""
    except Exception:
        return ""
    if not guesses:
        return ""
    best = max(guesses, key=lambda cand: getattr(cand, "prob", 0.0))
    prob = getattr(best, "prob", 0.0)
    if prob < 0.6:
        return ""
    lang_code = getattr(best, "lang", "")
    if not isinstance(lang_code, str):
        return ""
    return lang_code.strip().lower()


def _enrich_language_series(texts: pd.Series, langs: pd.Series) -> pd.Series:
    texts_series = texts.fillna("").astype(str)
    lang_series = langs.fillna("").apply(_clean_lang_token).astype(str)
    detect_mask = lang_series.isin(_LANG_MISSING_TOKENS) | (lang_series == "")
    if detect_mask.any():
        if _LANGDETECT_AVAILABLE:
            detected = texts_series.loc[detect_mask].apply(_detect_language_candidate)
            for idx, detected_lang in detected.items():
                if isinstance(detected_lang, str) and detected_lang.strip():
                    lang_series.loc[idx] = detected_lang.strip().lower()
    normalized_values: List[str] = []
    for raw in lang_series:
        token = str(raw or "").strip().lower()
        if not token:
            normalized_values.append(_LANG_UNKNOWN)
            continue
        normalized_values.append(_normalize_lang_code(token) or _LANG_UNKNOWN)
    return pd.Series(normalized_values, index=langs.index, dtype="object")


def extract_emojis(text: str) -> List[str]:
    try:
        import emoji  # type: ignore
        if not isinstance(text, str) or not text:
            return []
        return [e["emoji"] for e in emoji.emoji_list(text)]
    except Exception:
        return []

_TWEET_ID_RE = re.compile(r"/status/([0-9]{8,25})")

def _extract_tweet_id_from_url(url: str) -> Optional[str]:
    if not isinstance(url, str) or not url:
        return None
    m = _TWEET_ID_RE.search(url)
    if m:
        return m.group(1)
    m2 = re.search(r"([0-9]{8,25})", url)
    return m2.group(1) if m2 else None

def _surrogate_tweet_id(author: str, text_clean: str) -> str:
    base = f"{author}|{text_clean}"
    return "X_" + hashlib.md5(base.encode("utf-8", errors="ignore")).hexdigest()[:16]

def _normalize_date_series(series: pd.Series) -> pd.Series:
    """
    Devuelve sólo la FECHA en formato 'YYYY-MM-DD' a partir de:
    - ISO (YYYY-MM-DD[ T]HH:MM:SS[.fff]...)
    - Day-first (dd/mm/yyyy o dd-mm-yyyy con hora opcional y espacios múltiples)
    No intenta epochs (evita falsos positivos tipo 2034).
    """
    s = series.astype(str)
    # Normaliza espacios (incluye NBSP) y colapsa múltiples
    s = s.str.replace("\u00A0", " ", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    out = pd.Series("", index=s.index, dtype="object")

    # 1) ISO: empieza por YYYY-MM-DD
    iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    if iso_mask.any():
        out.loc[iso_mask] = s[iso_mask].str.extract(r"^(\d{4}-\d{2}-\d{2})", expand=False)

    # 2) Day-first dd/mm/yyyy o dd-mm-yyyy (con hora opcional)
    rem = out.eq("")
    df_mask = rem & s.str.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(\s+\d{1,2}:\d{2}(:\d{2})?)?$", na=False)
    if df_mask.any():
        # extrae solo la parte fecha (dd/mm/yyyy o dd-mm-yyyy)
        only_date = s[df_mask].str.extract(r"^(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", expand=False)
        parsed = pd.to_datetime(only_date, errors="coerce", dayfirst=True)
        out.loc[df_mask] = parsed.dt.strftime("%Y-%m-%d")

    # 3) Fallback: si aún vacío y contiene separadores, intenta parseo genérico (dayfirst True)
    rem = out.eq("")
    cand = rem & s.str.contains(r"[/-]", regex=True)
    if cand.any():
        parsed = pd.to_datetime(s[cand], errors="coerce", dayfirst=True)
        out.loc[cand] = parsed.dt.strftime("%Y-%m-%d")

    # Limpia NaT
    out = out.replace("NaT", "")
    return out

def _first_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lower = {c.strip().lower(): c for c in df.columns}
    for nm in names:
        key = nm.strip().lower()
        if key in lower:
            return lower[key]
    return None

# =========================
# Telegram
# =========================
def load_telegram(path: str) -> pd.DataFrame:
    """
    Lee el CSV de Telegram y homologa los campos analíticos con los de X.
    - Detecta alias comunes (timestamp, texto, métricas) para tolerar cambios de esquema.
    - Normaliza fecha a YYYY-MM-DD y añade columnas vacías si faltan.
    """
    df = _read_telegram_csv(path).copy()
    if df.empty:
        return pd.DataFrame()

    mask = df.apply(lambda row: any(str(val).strip() for val in row), axis=1)
    df = df.loc[mask].reset_index(drop=True)

    timestamp_aliases = ["timestamp", "date", "datetime", "created_at", "dt"]
    channel_aliases = ["channel", "channel_name", "chat", "chat_name", "author", "from"]
    message_aliases = ["messageId", "message_id", "id", "msg_id"]
    uid_aliases = ["uid", "message_uid", "item_uid", "permalink_uid"]
    kind_aliases = ["kind", "media_kind", "type"]
    text_aliases = ["text", "message", "content", "body"]
    caption_aliases = ["caption", "media_caption", "message_caption", "text_caption"]
    summary_aliases = ["summary", "gemini_summary", "ai_summary"]
    link_aliases = ["link", "permalink", "url", "message_link"]
    lang_aliases = ["lang", "language", "locale"]
    geo_aliases = [
        "geolocation",
        "geo_location",
        "geo",
        "location",
        "location_name",
        "country",
        "countries",
    ]

    c_timestamp = _first_col(df, timestamp_aliases)
    c_channel = _first_col(df, channel_aliases)
    c_message = _first_col(df, message_aliases)
    c_uid = _first_col(df, uid_aliases)
    c_kind = _first_col(df, kind_aliases)
    c_text = _first_col(df, text_aliases)
    c_caption = _first_col(df, caption_aliases)
    c_summary = _first_col(df, summary_aliases)
    c_link = _first_col(df, link_aliases)
    c_lang = _first_col(df, lang_aliases)
    c_geo = _first_col(df, geo_aliases)

    text_source_col = c_text or c_caption or c_summary
    missing = [("timestamp", c_timestamp), ("channel", c_channel), ("messageId", c_message), ("text", text_source_col)]
    missing = [field for field, col in missing if col is None]
    if missing:
        raise ValueError(f"Telegram CSV missing required columns: {missing}")

    # Normalización básica de strings
    def _as_str(col_name: str) -> pd.Series:
        if col_name and (col_name in df.columns):
            series = df[col_name].fillna("")
            return series.astype(str)
        return pd.Series([""] * len(df), index=df.index, dtype="object")

    out = pd.DataFrame(index=df.index)
    out["source"] = "Telegram"
    out["messageId"] = _as_str(c_message)
    out["author"] = _as_str(c_channel)
    out["kind"] = _as_str(c_kind)
    out["lang"] = _as_str(c_lang)
    base_series = _as_str(text_source_col)
    caption_series = _as_str(c_caption)
    summary_series = _as_str(c_summary)

    primary_text = caption_series.copy()
    mask_caption_empty = primary_text.astype(str).str.strip() == ""
    primary_text.loc[mask_caption_empty] = summary_series.loc[mask_caption_empty]
    mask_still_empty = primary_text.astype(str).str.strip() == ""
    primary_text.loc[mask_still_empty] = base_series.loc[mask_still_empty]

    out["text"] = primary_text
    out["text_caption"] = caption_series
    out["text_summary"] = summary_series
    out["link"] = _as_str(c_link)
    out["geolocation"] = _as_str(c_geo)
    out["geolocation"] = out["geolocation"].astype(str).str.strip()

    if c_uid:
        out["uid"] = _as_str(c_uid)
    else:
        out["uid"] = (out["author"].str.strip() + ":" + out["messageId"].str.strip()).str.strip(":")

    # Texto limpio + emojis
    out["text_caption_clean"] = out["text_caption"].apply(normalize_whitespace)
    out["text_summary_clean"] = out["text_summary"].apply(normalize_whitespace)
    out["text_clean"] = out["text"].apply(normalize_whitespace)
    out["emojis"] = out["text"].apply(extract_emojis)
    out["lang_raw"] = out["lang"].astype(str)
    out["lang"] = _enrich_language_series(out["text_clean"], out["lang_raw"])
    out["text_topic"] = _prepare_topic_series(out["text_clean"], out["lang"])
    out["emoji_count"] = out["emojis"].apply(len)

    # Fecha normalizada (sólo día)
    out["timestamp"] = _normalize_date_series(_as_str(c_timestamp))

    # Métricas: mapeamos a las mismas columnas que X
    def _metric(name_list: List[str]) -> pd.Series:
        col = _first_col(df, name_list)
        if col and col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return pd.Series([0] * len(df), index=df.index)

    out["likes"] = _metric(["likes", "reactions", "reaction_count", "reactions_count", "reactions_total"])
    out["retweets"] = _metric(["forwards", "forward_count", "shares", "shares_count"])
    out["replies"] = _metric(["replies", "reply_count", "comments", "comments_count"])
    out["quotes"] = _metric(["quotes", "quote_count"])
    out["views"] = _metric(["views", "view_count", "views_count"])

    # Para trazabilidad adicional guardamos métricas originales si existen
    for original, alias_list in {
        "telegram_forwards": ["forwards", "forward_count", "shares", "shares_count"],
        "telegram_reactions": ["reactions", "reaction_count", "reactions_count", "reactions_total"],
    }.items():
        col = _first_col(df, alias_list)
        if col and col in df.columns and original not in out.columns:
            out[original] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    out["geo_country_distribution"] = [
        _geo_country_distribution(geo, lang)
        for geo, lang in zip(out["geolocation"], out["lang"])
    ]

    base_cols = [
        "source","timestamp","author","messageId","uid","kind",
        "text","text_clean","link","lang","emojis","emoji_count",
        "likes","retweets","replies","quotes","views",
    ]
    extras = [c for c in out.columns if c not in base_cols]
    return out[base_cols + extras]

# =========================
# X / Twitter
# =========================
def load_x(path: str) -> pd.DataFrame:
    df = _read_csv_utf8(path).copy()

    id_aliases = ["Tweet ID","tweet id","id","status_id","tweet_id","TweetId"]
    date_aliases = ["Date","date","created_at","Created At","Datetime","Timestamp","Posted at","posted_at","time","tweet_date","created_at_x","Created at"]
    text_aliases = ["Content","content","text","Tweet text","Tweet"]
    lang_aliases = ["Language","language","lang"]
    author_id_aliases = ["Author ID","author id","user_id","User ID"]
    author_aliases = ["Author Name","author name","username","screen_name","Author","Display name"]
    location_aliases = [
        "Author Location",
        "author location",
        "author_location",
        "location",
        "user_location",
        "user location",
        "place",
        "place_full_name",
        "place name",
    ]
    url_aliases = ["URL","url","Tweet URL","Tweet link","Link"]
    handle_aliases = ["Author Username","author username","author_username","username","screen_name","Author Handle","author handle","handle"]
    profile_url_aliases = ["Author Profile URL","author profile url","profile_url","profile url"]
    reply_id_aliases = ["Reply to userid","reply to userid","reply_to_userid","in_reply_to_userid","in_reply_to_user_id"]
    reply_username_aliases = ["Reply to username","reply to username","reply_to_username","in_reply_to_screen_name","in_reply_to_username"]
    mention_aliases = ["MentionedUsers","mentionedusers","Mentioned Users","mentions","Mentioned Handles","mentioned_handles"]
    retweeted_handle_aliases = ["Retweeted Username","retweeted username","retweet_username","Retweeted Handle","retweeted_handle","retweet handle"]
    retweeted_url_aliases = ["Retweeted X URL","retweeted x url","Retweeted Tweet URL","retweet url","Retweeted URL"]
    retweeted_id_aliases = ["Retweeted X ID","retweeted x id","Retweeted Tweet ID","retweet id","Retweeted ID"]
    quoted_handle_aliases = ["Quoted Username","quoted username","quote_username","Quoted Handle","quoted_handle","quote handle"]
    quoted_url_aliases = ["Quoted Tweet URL","quoted tweet url","Quoted X URL","quote url","Quoted URL"]
    quoted_id_aliases = ["Quoted X ID","quoted x id","Quoted Tweet ID","quote id","Quoted ID"]
    conversation_aliases = ["Conversation ID","conversation id","conversation_id"]
    is_reply_aliases = ["Is Reply","is reply","is_reply"]

    c_id = _first_col(df, id_aliases)
    c_date = _first_col(df, date_aliases)
    c_text = _first_col(df, text_aliases)
    c_lang = _first_col(df, lang_aliases) or "lang"
    c_author_id = _first_col(df, author_id_aliases) or "author_id"
    c_author = _first_col(df, author_aliases) or "author"
    c_url = _first_col(df, url_aliases)
    c_author_location = _first_col(df, location_aliases)
    c_author_handle = _first_col(df, handle_aliases)
    c_profile_url = _first_col(df, profile_url_aliases)
    c_reply_id = _first_col(df, reply_id_aliases)
    c_reply_username = _first_col(df, reply_username_aliases)
    c_mentions = _first_col(df, mention_aliases)
    c_retweeted_handle = _first_col(df, retweeted_handle_aliases)
    c_retweeted_url = _first_col(df, retweeted_url_aliases)
    c_retweeted_id = _first_col(df, retweeted_id_aliases)
    c_quoted_handle = _first_col(df, quoted_handle_aliases)
    c_quoted_url = _first_col(df, quoted_url_aliases)
    c_quoted_id = _first_col(df, quoted_id_aliases)
    c_conversation = _first_col(df, conversation_aliases)
    c_is_reply = _first_col(df, is_reply_aliases)

    if c_text is None:
        raise ValueError("X CSV missing tweet text column (e.g., 'Content'/'text').")
    if c_lang not in df.columns:
        df[c_lang] = ""
    if c_author_id not in df.columns:
        df[c_author_id] = ""
    if c_author not in df.columns:
        df[c_author] = ""

    def _series_str(col_name: Optional[str], *, strip: bool = True) -> pd.Series:
        if col_name and col_name in df.columns:
            series = df[col_name].fillna("").astype(str)
            if strip:
                series = series.str.strip()
            return series
        return pd.Series([""] * len(df), index=df.index, dtype="object")

    out = pd.DataFrame(index=df.index)
    out["source"] = "X"

    # tweet_id: columna → URL → sustituto
    if c_id and c_id in df.columns:
        base_tid = df[c_id].astype(str).str.extract(r"([0-9]{8,25})", expand=False).fillna("")
    else:
        base_tid = pd.Series([""] * len(df), index=df.index, dtype="object")

    if c_url and c_url in df.columns:
        url_tid = _series_str(c_url, strip=False).apply(lambda v: _extract_tweet_id_from_url(v) or "")
        mask_empty = base_tid.str.strip() == ""
        base_tid.loc[mask_empty] = url_tid.loc[mask_empty]

    author_series = _series_str(c_author, strip=False)
    text_series = _series_str(c_text, strip=False)
    missing_mask = base_tid.str.strip() == ""
    if missing_mask.any():
        fallback_ids = [
            _surrogate_tweet_id(author or "", text or "")
            for author, text in zip(author_series.loc[missing_mask], text_series.loc[missing_mask])
        ]
        base_tid.loc[missing_mask] = fallback_ids

    out["tweet_id"] = base_tid.fillna("").astype(str)

    # FECHA (YYYY-MM-DD)
    if c_date and c_date in df.columns:
        out["timestamp"] = _normalize_date_series(_series_str(c_date, strip=False))
    else:
        out["timestamp"] = pd.Series([""] * len(df), index=df.index, dtype="object")

    # Resto
    out["author_id"] = _series_str(c_author_id)
    out["author"] = author_series
    out["text"] = text_series
    out["text_clean"] = out["text"].apply(normalize_whitespace)
    lang_series_raw = _series_str(c_lang, strip=False)
    out["lang_raw"] = lang_series_raw
    out["lang"] = _enrich_language_series(out["text_clean"], lang_series_raw)
    out["text_topic"] = _prepare_topic_series(out["text_clean"], out["lang"])
    out["emojis"] = out["text"].apply(extract_emojis)
    out["emoji_count"] = out["emojis"].apply(len)
    link_series = _series_str(c_url)
    out["link"] = link_series
    out["author_location"] = _series_str(c_author_location)
    out["geolocation"] = out["author_location"]

    handle_series = _series_str(c_author_handle).apply(_extract_handle_candidate)
    if c_profile_url and c_profile_url in df.columns:
        profile_handles = _series_str(c_profile_url, strip=False).apply(_extract_handle_from_url)
        handle_series = handle_series.where(handle_series != "", profile_handles)
    fallback_handles = author_series.apply(_extract_handle_candidate)
    handle_series = handle_series.where(handle_series != "", fallback_handles)
    out["author_handle"] = handle_series.fillna("")

    out["reply_to_userid"] = _series_str(c_reply_id)
    reply_username_series = _series_str(c_reply_username).apply(_extract_handle_candidate)
    out["reply_to_username"] = reply_username_series.fillna("")

    out["conversation_id"] = _series_str(c_conversation)
    is_reply_series = _series_str(c_is_reply).str.upper()
    out["is_reply"] = is_reply_series.replace({"TRUE": "TRUE", "FALSE": "FALSE"})

    retweeted_handle_series = _series_str(c_retweeted_handle).apply(_extract_handle_candidate)
    if c_retweeted_url and c_retweeted_url in df.columns:
        retweeted_handle_series = retweeted_handle_series.where(
            retweeted_handle_series != "",
            _series_str(c_retweeted_url, strip=False).apply(_extract_handle_from_url),
        )
    out["retweeted_handle"] = retweeted_handle_series.fillna("")
    out["retweeted_tweet_id"] = _series_str(c_retweeted_id)

    quoted_handle_series = _series_str(c_quoted_handle).apply(_extract_handle_candidate)
    if c_quoted_url and c_quoted_url in df.columns:
        quoted_handle_series = quoted_handle_series.where(
            quoted_handle_series != "",
            _series_str(c_quoted_url, strip=False).apply(_extract_handle_from_url),
        )
    out["quoted_handle"] = quoted_handle_series.fillna("")
    out["quoted_tweet_id"] = _series_str(c_quoted_id)

    if c_mentions and c_mentions in df.columns:
        mentions_raw = df[c_mentions]
    else:
        mentions_raw = pd.Series([[] for _ in range(len(df))], index=df.index)
    parsed_mentions = mentions_raw.apply(_parse_mentions_field)
    out["mentioned_handles"] = parsed_mentions.apply(lambda pair: pair[0])
    out["mentioned_users"] = parsed_mentions.apply(lambda pair: pair[1])

    # Engagement tolerante
    def _to_int(series_name_list: List[str]) -> pd.Series:
        name = _first_col(df, series_name_list)
        if not name or name not in df.columns:
            return pd.Series([0] * len(df), index=df.index)
        return pd.to_numeric(df[name], errors="coerce").fillna(0).astype(int)

    out["likes"] = _to_int(["Likes","likes"])
    out["retweets"] = _to_int(["Retweets","retweets"])
    out["replies"] = _to_int(["Replies","replies"])
    out["quotes"] = _to_int(["Quotes","quotes"])
    out["views"]  = _to_int(["Views","views","impressions"])

    out["geo_country_distribution"] = [
        _geo_country_distribution(geo, lang)
        for geo, lang in zip(out["geolocation"], out["lang"])
    ]

    column_order = [
        "source",
        "timestamp",
        "author",
        "author_id",
        "author_handle",
        "author_location",
        "geolocation",
        "tweet_id",
        "text",
        "text_clean",
        "text_topic",
        "lang",
        "lang_raw",
        "emojis",
        "emoji_count",
        "link",
        "likes",
        "retweets",
        "replies",
        "quotes",
        "views",
        "reply_to_userid",
        "reply_to_username",
        "conversation_id",
        "is_reply",
        "mentioned_handles",
        "mentioned_users",
        "retweeted_handle",
        "retweeted_tweet_id",
        "quoted_handle",
        "quoted_tweet_id",
        "geo_country_distribution",
    ]
    # Aseguramos que todas las columnas existen antes de reordenar
    for col in column_order:
        if col not in out.columns:
            if col in {"mentioned_handles", "mentioned_users"}:
                out[col] = [[] for _ in range(len(out))]
            else:
                out[col] = ""
    return out[column_order]

# =========================
# Unificación
# =========================
def unify_frames(df_tg: pd.DataFrame | None, df_x: pd.DataFrame | None) -> pd.DataFrame:
    frames = []
    if df_tg is not None and not df_tg.empty:
        tg = df_tg.rename(columns={"messageId": "item_id"}).copy()
        tg["source"] = "Telegram"
        frames.append(tg)
    if df_x is not None and not df_x.empty:
        xx = df_x.rename(columns={"tweet_id": "item_id"}).copy()
        xx["source"] = "X"
        xx["item_id"] = xx["item_id"].fillna("").astype(str).str.strip()
        missing_mask = xx["item_id"].str.strip() == ""
        if missing_mask.any():
            fallback_ids = [
                _surrogate_tweet_id(author or "", text or "")
                for author, text in zip(
                    xx.loc[missing_mask, "author"].astype(str),
                    xx.loc[missing_mask, "text_clean"].astype(str),
                )
            ]
            xx.loc[missing_mask, "item_id"] = fallback_ids
        frames.append(xx)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)

    base_cols = ["source","timestamp","author","item_id","text","text_clean","text_topic","lang","emojis","emoji_count","link"]
    for col in base_cols:
        if col not in df.columns:
            df[col] = None

    metric_cols = ["likes","retweets","replies","quotes","views"]
    for col in metric_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    drop_globals = [
        "sentiment_label",
        "sentiment_score",
        "emotion_label",
        "emotion_scores",
        "emotion_prob_positive",
        "emotion_prob_negative",
        "emotion_prob_neutral",
    ]
    drop_existing = [col for col in drop_globals if col in df.columns]
    if drop_existing:
        df = df.drop(columns=drop_existing)

    return df[base_cols + [c for c in df.columns if c not in base_cols]]
