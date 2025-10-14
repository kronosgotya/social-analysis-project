# -*- coding: utf-8 -*-
"""
geoloc.py
Heurística: tz_message -> tz_author -> lang (+ topónimos opcional).
Retorna (country_iso2 | None, conf (0-1), basis str|None)
"""
from __future__ import annotations
from typing import Optional, Tuple

# mapas mínimos; en producción, inyecta tus tablas completas
TZ_TO_COUNTRY = {
    "Europe/Moscow": "RU",
    "Europe/Warsaw": "PL",
    "Europe/Madrid": "ES",
    "America/Caracas": "VE",
}
LANG_DEFAULT_COUNTRY = {
    "ru": "RU", "pl": "PL", "es": "ES", "en": None
}

def infer_country(tz_message: Optional[str], tz_author: Optional[str],
                  lang: Optional[str], text: Optional[str] = None) -> Tuple[Optional[str], float, Optional[str]]:
    if tz_message and tz_message in TZ_TO_COUNTRY:
        return TZ_TO_COUNTRY[tz_message], 0.9, "tz_message"
    if tz_author and tz_author in TZ_TO_COUNTRY:
        return TZ_TO_COUNTRY[tz_author], 0.7, "tz_author"
    if lang:
        ctry = LANG_DEFAULT_COUNTRY.get(lang.lower())
        if ctry:
            return ctry, 0.6, "lang"
        # ejemplo específico: español + Caracas ⇒ Venezuela
        if lang.lower() == "es" and tz_author == "America/Caracas":
            return "VE", 0.7, "lang"
    # TODO: topónimos (bonus)
    return None, 0.0, None