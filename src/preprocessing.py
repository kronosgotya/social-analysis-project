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
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import pandas as pd
from stopwordsiso import stopwords as stopwords_iso
from simplemma import simple_tokenizer, lemmatize

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
_TOKEN_MIN_LENGTH = 3
_TOPIC_STOPWORDS_EXTRA = {"https", "http", "amp", "rt", "t.me"}

_LANG_COUNTRY_LOOKUP_CACHE: Optional[Dict[str, List[Dict[str, float]]]] = None
_LANG_COUNTRY_LOOKUP_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "lang_country_lookup.csv"
)

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

# =========================
# Helpers
# =========================
def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r"\s+", " ", text).strip()

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
    df = _read_csv_utf8(path).copy()

    timestamp_aliases = ["timestamp", "date", "datetime", "created_at", "dt"]
    channel_aliases = ["channel", "channel_name", "chat", "chat_name", "author", "from"]
    message_aliases = ["messageId", "message_id", "id", "msg_id"]
    uid_aliases = ["uid", "message_uid", "item_uid", "permalink_uid"]
    kind_aliases = ["kind", "media_kind", "type"]
    caption_aliases = ["text", "message", "content", "body"]
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
    c_caption = _first_col(df, caption_aliases)
    c_summary = _first_col(df, summary_aliases)
    c_text = c_caption or c_summary
    c_link = _first_col(df, link_aliases)
    c_lang = _first_col(df, lang_aliases)
    c_geo = _first_col(df, geo_aliases)

    missing = [("timestamp", c_timestamp), ("channel", c_channel), ("messageId", c_message), ("text", c_text)]
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
    caption_series = _as_str(c_caption)
    summary_series = _as_str(c_summary)

    out["text_caption"] = caption_series
    out["text_summary"] = summary_series

    out["text"] = caption_series
    if c_summary:
        mask_empty_caption = out["text"].str.strip() == ""
        out.loc[mask_empty_caption, "text"] = summary_series.loc[mask_empty_caption]
    out["link"] = _as_str(c_link)
    out["geolocation"] = _as_str(c_geo)
    out["geolocation"] = out["geolocation"].astype(str).str.strip()

    if c_uid:
        out["uid"] = _as_str(c_uid)
    else:
        out["uid"] = (out["author"].str.strip() + ":" + out["messageId"].str.strip()).str.strip(":")

    # Texto limpio + emojis
    out["text_clean"] = out["text"].apply(normalize_whitespace)
    out["text_caption_clean"] = out["text_caption"].apply(normalize_whitespace)
    out["text_summary_clean"] = out["text_summary"].apply(normalize_whitespace)
    out["emojis"] = out["text"].apply(extract_emojis)
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

    c_id = _first_col(df, id_aliases)
    c_date = _first_col(df, date_aliases)
    c_text = _first_col(df, text_aliases)
    c_lang = _first_col(df, lang_aliases) or "lang"
    c_author_id = _first_col(df, author_id_aliases) or "author_id"
    c_author = _first_col(df, author_aliases) or "author"
    c_url = _first_col(df, url_aliases)
    c_author_location = _first_col(df, location_aliases)

    if c_text is None:
        raise ValueError("X CSV missing tweet text column (e.g., 'Content'/'text').")
    if c_lang not in df.columns:
        df[c_lang] = ""
    if c_author_id not in df.columns:
        df[c_author_id] = ""
    if c_author not in df.columns:
        df[c_author] = ""

    out = pd.DataFrame()
    out["source"] = "X"

    # tweet_id: columna → URL → sustituto
    if c_id and c_id in df.columns:
        base_tid = df[c_id].astype(str).str.extract(r"([0-9]{8,25})", expand=False)
    else:
        base_tid = pd.Series([None] * len(df))

    if (base_tid is None) or (base_tid.isna().all()):
        if c_url and c_url in df.columns:
            base_tid = df[c_url].astype(str).apply(_extract_tweet_id_from_url)
        else:
            base_tid = pd.Series([None] * len(df))

    tmp_text = df[c_text].astype(str).fillna("")
    tmp_author = df[c_author].astype(str).fillna("")
    filled_tid = pd.Series(
        [t if isinstance(t, str) and t.strip() else _surrogate_tweet_id(a, t2)
         for t, a, t2 in zip(base_tid.fillna(""), tmp_author, tmp_text)],
        index=df.index,
    )
    out["tweet_id"] = filled_tid

    # FECHA (YYYY-MM-DD)
    if c_date and c_date in df.columns:
        out["timestamp"] = _normalize_date_series(df[c_date])
    else:
        out["timestamp"] = ""  # sin fecha usable, no forzamos epochs

    # Resto
    out["author_id"] = df[c_author_id].astype(str)
    out["author"] = df[c_author].fillna("").astype(str)
    out["lang"] = df[c_lang].fillna("").astype(str)
    out["text"] = df[c_text].fillna("").astype(str)
    out["text_clean"] = out["text"].apply(normalize_whitespace)
    out["text_topic"] = _prepare_topic_series(out["text_clean"], out["lang"])
    out["emojis"] = out["text"].apply(extract_emojis)
    out["emoji_count"] = out["emojis"].apply(len)
    out["link"] = df[c_url].astype(str) if c_url and (c_url in df.columns) else ""
    if c_author_location and c_author_location in df.columns:
        out["author_location"] = df[c_author_location].fillna("").astype(str).str.strip()
    else:
        out["author_location"] = ""

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

    return out[[
        "source","timestamp","author","author_location","tweet_id","text","text_clean","text_topic","lang","emojis","emoji_count","link",
        "likes","retweets","replies","quotes","views"
    ]]

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
        xx["item_id"] = xx["item_id"].apply(
            lambda v: v if isinstance(v, str) and v.strip()
            else _surrogate_tweet_id(xx.get("author", ""), xx.get("text_clean", ""))
        )
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

    return df[base_cols + [c for c in df.columns if c not in base_cols]]
