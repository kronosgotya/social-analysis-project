# -*- coding: utf-8 -*-
"""
entities_runtime.py
Capa de selección de ENTIDADES runtime (sin hardcode).
- Entrada: --entities "OTAN,Rusia" o --entities-file path.{yml|yaml|json|txt}
- Salida: lista de entidades [{name:str, type:str|None, aliases:list[str]}]
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional
import json, os

try:
    import yaml  # opcional: PyYAML en requirements
except Exception:
    yaml = None

@dataclass
class EntitySpec:
    name: str
    type: Optional[str] = None
    aliases: Optional[List[str]] = None

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
                specs.append(EntitySpec(
                    name=item.get("name"),
                    type=item.get("type"),
                    aliases=item.get("aliases") or [item.get("name")]
                ))
        return specs
    elif isinstance(data, dict):
        # dict con clave 'entities'
        ents = data.get("entities", [])
        return _from_simple_list(ents) if ents and isinstance(ents[0], str) else [
            EntitySpec(
                name=e.get("name"),
                type=e.get("type"),
                aliases=e.get("aliases") or [e.get("name")]
            ) for e in ents
        ]
    else:
        raise ValueError("Formato de fichero de entidades no reconocido")

def load_entities(entities_csv: Optional[str], entities_file: Optional[str]) -> List[EntitySpec]:
    if entities_file:
        return _load_from_file(entities_file)
    if entities_csv:
        return _from_simple_list(entities_csv.split(","))
    return []