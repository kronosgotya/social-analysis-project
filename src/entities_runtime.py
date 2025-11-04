# -*- coding: utf-8 -*-
"""
entities_runtime.py
Capa de selección de ENTIDADES runtime (sin hardcode).
- Entrada: --entities "OTAN,Rusia" o --entities-file path.{yml|yaml|json|txt}
- Salida: lista de entidades [{name:str, type:str|None, aliases:list[str]}]
Además expone mapas de alias básicos utilizados durante la explosión de entidades.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json, os

NATO_ALIASES = {"NATO", "OTAN", "НАТО"}
RUSSIA_ALIASES = {"Russia", "Rusia", "Россия"}
RELATED_TO_PRINCIPAL = {
    "Ukraine": "NATO",
    "Ucrania": "NATO",
    "Украина": "NATO",
    "Україна": "NATO",
}
DROP_UNKNOWN = True

try:
    import yaml  # opcional: PyYAML en requirements
except Exception:
    yaml = None

def _normalize_alias_entries(raw_aliases: Optional[Any]) -> Tuple[List[str], Dict[str, float], Dict[str, Dict[str, Any]]]:
    alias_list: List[str] = []
    alias_weights: Dict[str, float] = {}
    alias_metadata: Dict[str, Dict[str, Any]] = {}

    if raw_aliases is None:
        return alias_list, alias_weights, alias_metadata

    if isinstance(raw_aliases, dict):
        for alias_value, maybe_weight in raw_aliases.items():
            alias_text = str(alias_value).strip() if alias_value is not None else ""
            if not alias_text:
                continue
            alias_list.append(alias_text)
            if maybe_weight is not None:
                try:
                    alias_weights[alias_text] = float(maybe_weight)
                except (TypeError, ValueError):
                    pass
        return alias_list, alias_weights, alias_metadata

    for entry in raw_aliases:
        alias_value: Optional[Any]
        alias_weight: Optional[Any] = None
        metadata: Dict[str, Any] = {}
        if isinstance(entry, dict):
            alias_value = entry.get("alias") or entry.get("value") or entry.get("text")
            alias_weight = entry.get("weight")
            metadata = {k: v for k, v in entry.items() if k not in {"alias", "value", "text", "weight"}}
        else:
            alias_value = entry
        alias_text = str(alias_value).strip() if alias_value is not None else ""
        if not alias_text:
            continue
        alias_list.append(alias_text)
        if alias_weight is not None:
            try:
                alias_weights[alias_text] = float(alias_weight)
            except (TypeError, ValueError):
                pass
        if metadata:
            alias_metadata.setdefault(alias_text, {}).update(metadata)

    return alias_list, alias_weights, alias_metadata


@dataclass
class EntitySpec:
    name: str
    type: Optional[str] = None
    aliases: Optional[Any] = None
    entity_norm: Optional[str] = None
    weight: Optional[float] = None
    alias_weights: Dict[str, float] = field(default_factory=dict)
    alias_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        alias_list, parsed_weights, parsed_metadata = _normalize_alias_entries(self.aliases)

        alias_weights: Dict[str, float] = {}
        for alias, weight in dict(self.alias_weights).items():
            if alias:
                try:
                    alias_weights[alias] = float(weight)
                except (TypeError, ValueError):
                    continue
        alias_weights.update(parsed_weights)

        alias_metadata: Dict[str, Dict[str, Any]] = {}
        for alias, meta in dict(self.alias_metadata).items():
            if alias:
                alias_metadata[alias] = dict(meta)
        for alias, meta in parsed_metadata.items():
            alias_metadata.setdefault(alias, {}).update(meta)

        canonical_name = str(self.name).strip() if self.name is not None else ""
        if canonical_name:
            alias_list.append(canonical_name)

        if self.entity_norm is not None:
            norm_value = str(self.entity_norm).strip()
        else:
            norm_value = canonical_name
        if norm_value:
            alias_list.append(norm_value)
        self.entity_norm = norm_value or None

        unique_aliases = list(dict.fromkeys(a for a in alias_list if a))
        self.aliases = unique_aliases or None

        default_weight: Optional[float] = None
        if self.weight is not None:
            try:
                default_weight = float(self.weight)
            except (TypeError, ValueError):
                default_weight = None

        for alias in unique_aliases:
            if alias not in alias_weights:
                alias_weights[alias] = default_weight if default_weight is not None else 1.0

        self.alias_weights = {
            alias: float(weight) for alias, weight in alias_weights.items() if alias
        }
        self.alias_metadata = {alias: meta for alias, meta in alias_metadata.items() if alias}
        self.metadata = dict(self.metadata or {})

def _from_simple_list(names: List[str]) -> List[EntitySpec]:
    out = []
    for n in [x.strip() for x in names if x.strip()]:
        out.append(EntitySpec(name=n, type=None, aliases=[n]))
    return out

def _load_from_file(path: str) -> List[EntitySpec]:
    ext = os.path.splitext(path)[-1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in [".yml", ".yaml"] and yaml:
            data = yaml.safe_load(f)
        elif ext == ".json":
            data = json.load(f)
        else:
            # .txt u otros: una entidad por línea
            data = [line.strip() for line in f if line.strip()]

    if isinstance(data, list):
        # lista de strings o dicts
        specs = []
        for item in data:
            if isinstance(item, str):
                specs.append(EntitySpec(name=item, aliases=[item]))
            elif isinstance(item, dict):
                name = item.get("name") or item.get("entity_norm")
                aliases = item.get("aliases")
                specs.append(EntitySpec(
                    name=name,
                    type=item.get("type"),
                    aliases=aliases,
                    entity_norm=item.get("entity_norm"),
                    weight=item.get("weight"),
                    alias_weights=item.get("alias_weights") or {},
                    alias_metadata=item.get("alias_metadata") or {},
                    metadata=item.get("metadata") or {},
                ))
        return specs
    elif isinstance(data, dict):
        # dict con clave 'entities'
        ents = data.get("entities", [])
        if ents and isinstance(ents[0], str):
            return _from_simple_list(ents)
        specs: List[EntitySpec] = []
        for e in ents:
            if not isinstance(e, dict):
                continue
            name = e.get("name") or e.get("entity_norm")
            specs.append(EntitySpec(
                name=name,
                type=e.get("type"),
                aliases=e.get("aliases"),
                entity_norm=e.get("entity_norm"),
                weight=e.get("weight"),
                alias_weights=e.get("alias_weights") or {},
                alias_metadata=e.get("alias_metadata") or {},
                metadata=e.get("metadata") or {},
            ))
        return specs
    else:
        raise ValueError("Formato de fichero de entidades no reconocido")

def load_entities(entities_csv: Optional[str], entities_file: Optional[str]) -> List[EntitySpec]:
    if entities_file:
        return _load_from_file(entities_file)
    if entities_csv:
        return _from_simple_list(entities_csv.split(","))
    return []
