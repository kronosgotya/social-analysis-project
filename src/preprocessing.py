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
from typing import List, Optional
import pandas as pd

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
    df = _read_csv_utf8(path)
    required = ["timestamp", "channel", "messageId", "uid", "kind", "summary", "link", "lang"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Telegram CSV missing required columns: {missing}")

    df = df.copy()
    df = df.rename(columns={"summary": "text", "channel": "author"})
    df["source"] = "Telegram"
    df["text_clean"] = df["text"].apply(normalize_whitespace)
    df["emojis"] = df["text"].apply(extract_emojis)
    df["emoji_count"] = df["emojis"].apply(len)

    # FECHA normalizada (YYYY-MM-DD)
    df["timestamp"] = _normalize_date_series(df["timestamp"])

    # Tipos string
    for c in ("timestamp", "lang", "messageId", "uid", "kind", "author", "link"):
        if c in df.columns:
            df[c] = df[c].astype(str)

    out_cols = [
        "source","timestamp","author","messageId","uid","kind",
        "text","text_clean","link","lang","emojis","emoji_count",
    ]
    return df[out_cols]

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
    url_aliases = ["URL","url","Tweet URL","Tweet link","Link"]

    c_id = _first_col(df, id_aliases)
    c_date = _first_col(df, date_aliases)
    c_text = _first_col(df, text_aliases)
    c_lang = _first_col(df, lang_aliases) or "lang"
    c_author_id = _first_col(df, author_id_aliases) or "author_id"
    c_author = _first_col(df, author_aliases) or "author"
    c_url = _first_col(df, url_aliases)

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
    out["author"] = df[c_author].astype(str)
    out["lang"] = df[c_lang].astype(str)
    out["text"] = df[c_text].astype(str)
    out["text_clean"] = out["text"].apply(normalize_whitespace)
    out["emojis"] = out["text"].apply(extract_emojis)
    out["emoji_count"] = out["emojis"].apply(len)
    out["link"] = df[c_url].astype(str) if c_url and (c_url in df.columns) else ""

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
        "source","timestamp","author","tweet_id","text","text_clean","lang","emojis","emoji_count","link",
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

    base_cols = ["source","timestamp","author","item_id","text","text_clean","lang","emojis","emoji_count","link"]
    for col in base_cols:
        if col not in df.columns:
            df[col] = None

    return df[base_cols + [c for c in df.columns if c not in base_cols]]
