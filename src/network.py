from __future__ import annotations
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Iterable, Dict, Any, Optional

# ================== Helpers ==================

def _node_key(author_id: Optional[str], author_handle: Optional[str]) -> str:
    """
    Usa handle si está disponible, en formato '@handle'.
    Si no, usa el author_id como string.
    """
    if isinstance(author_handle, str) and author_handle.strip():
        h = author_handle.strip()
        if not h.startswith("@"):
            h = "@" + h
        return h
    return str(author_id) if author_id is not None else ""

def _label_for_node(node_key: str, fallback_label: Optional[str]) -> str:
    if node_key:
        return node_key
    return str(fallback_label) if fallback_label is not None else ""

def _add_edge(G: nx.DiGraph, src: str, dst: str, kind: str):
    if not src or not dst or src == dst:
        return
    w = G.get_edge_data(src, dst, {}).get("weight", 0) + 1
    G.add_edge(src, dst, kind=kind, weight=w)

# ================== Build graph ==================

def build_x_graph(df_x: pd.DataFrame) -> nx.DiGraph:
    """
    Construye el grafo dirigido de X con menciones, replies,
    y (si están disponibles) retweets/quotes.
    Usa '@handle' cuando es posible para facilitar lectura en Gephi.
    """
    G = nx.DiGraph()

    # Mapa de id->handle para enriquecer replies
    id2handle: Dict[str, str] = {}
    for _, r in df_x.iterrows():
        aid = str(r.get("author_id") or "").strip()
        ah = str(r.get("author_handle") or "").strip()
        if aid and ah:
            id2handle[aid] = ah

    # Añade nodos de autores (con etiqueta humana)
    for _, r in df_x.iterrows():
        src_key = _node_key(r.get("author_id"), r.get("author_handle"))
        label = _label_for_node(src_key, r.get("author"))
        if src_key and not G.has_node(src_key):
            G.add_node(src_key, label=label)

    # Menciones detectadas/normalizadas
    for _, r in df_x.iterrows():
        src_key = _node_key(r.get("author_id"), r.get("author_handle"))
        for h in (r.get("mentioned_handles") or r.get("mentioned_users") or []):
            if not isinstance(h, str) or not h.strip():
                continue
            dst_key = h.strip()
            if not dst_key.startswith("@"):
                dst_key = "@" + dst_key
            if not G.has_node(dst_key):
                G.add_node(dst_key, label=dst_key)
            _add_edge(G, src_key, dst_key, kind="mention")

    # Replies (id->handle si es posible)
    for _, r in df_x.iterrows():
        src_key = _node_key(r.get("author_id"), r.get("author_handle"))
        reply_to = str(r.get("reply_to_userid") or "").strip()
        if reply_to:
            dst_key = _node_key(reply_to, id2handle.get(reply_to))
            if not G.has_node(dst_key):
                G.add_node(dst_key, label=dst_key)
            _add_edge(G, src_key, dst_key, kind="reply")

    # Retweets / Quotes (si se pudieron extraer handles del URL)
    for _, r in df_x.iterrows():
        src_key = _node_key(r.get("author_id"), r.get("author_handle"))
        rt_h = str(r.get("retweeted_handle") or "").strip()
        if rt_h:
            dst_key = "@" + rt_h if not rt_h.startswith("@") else rt_h
            if not G.has_node(dst_key):
                G.add_node(dst_key, label=dst_key)
            _add_edge(G, src_key, dst_key, kind="retweet")

        qt_h = str(r.get("quoted_handle") or "").strip()
        if qt_h:
            dst_key = "@" + qt_h if not qt_h.startswith("@") else qt_h
            if not G.has_node(dst_key):
                G.add_node(dst_key, label=dst_key)
            _add_edge(G, src_key, dst_key, kind="quote")

    return G

# ================== Metrics & Export ==================

def graph_metrics(G: nx.DiGraph) -> pd.DataFrame:
    # Grados ponderados por weight
    deg = dict(G.degree(weight="weight"))
    indeg = dict(G.in_degree(weight="weight"))
    outdeg = dict(G.out_degree(weight="weight"))

    # Centralidades
    bc = nx.betweenness_centrality(G)  # sin peso (recomendado para estructura)
    try:
        ec = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    except Exception:
        ec = {n: 0.0 for n in G.nodes}
    pr = nx.pagerank(G, weight="weight")

    df = pd.DataFrame({
        "node": list(G.nodes),
        "degree": [deg.get(n, 0) for n in G.nodes],
        "in_degree": [indeg.get(n, 0) for n in G.nodes],
        "out_degree": [outdeg.get(n, 0) for n in G.nodes],
        "betweenness": [bc.get(n, 0.0) for n in G.nodes],
        "eigenvector": [ec.get(n, 0.0) for n in G.nodes],
        "pagerank": [pr.get(n, 0.0) for n in G.nodes],
        "label": [G.nodes[n].get("label") for n in G.nodes]
    }).sort_values("pagerank", ascending=False)
    return df

def export_gexf(G: nx.DiGraph, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, path)

# ================== Edges table for Tableau/Gephi ==================

def edges_from_x(df_x: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Mapa id->handle para replies
    id2handle: Dict[str, str] = {}
    for _, r in df_x.iterrows():
        aid = str(r.get("author_id") or "").strip()
        ah = str(r.get("author_handle") or "").strip()
        if aid and ah:
            id2handle[aid] = ah

    def key_from_id_or_handle(id_val: str | None, handle_val: str | None) -> str:
        if handle_val and isinstance(handle_val, str) and handle_val.strip():
            h = handle_val.strip()
            return "@" + h if not h.startswith("@") else h
        if id_val:
            return str(id_val)
        return ""

    # Menciones
    for _, r in df_x.iterrows():
        src = key_from_id_or_handle(r.get("author_id"), r.get("author_handle"))
        ts = r.get("timestamp")
        for h in (r.get("mentioned_handles") or r.get("mentioned_users") or []):
            if not isinstance(h, str) or not h.strip():
                continue
            dst = h if h.startswith("@") else "@" + h
            if src and dst and src != dst:
                rows.append({"src_user": src, "dst_user": dst, "edge_kind": "mention", "weight": 1, "timestamp": ts})

    # Replies
    for _, r in df_x.iterrows():
        src = key_from_id_or_handle(r.get("author_id"), r.get("author_handle"))
        dst = key_from_id_or_handle(str(r.get("reply_to_userid") or "").strip(), id2handle.get(str(r.get("reply_to_userid") or "").strip()))
        ts = r.get("timestamp")
        if src and dst and src != dst:
            rows.append({"src_user": src, "dst_user": dst, "edge_kind": "reply", "weight": 1, "timestamp": ts})

    # Retweets / Quotes (si hay handle)
    for _, r in df_x.iterrows():
        src = key_from_id_or_handle(r.get("author_id"), r.get("author_handle"))
        ts = r.get("timestamp")
        rt_h = str(r.get("retweeted_handle") or "").strip()
        if rt_h:
            dst = "@" + rt_h if not rt_h.startswith("@") else rt_h
            if src and dst and src != dst:
                rows.append({"src_user": src, "dst_user": dst, "edge_kind": "retweet", "weight": 1, "timestamp": ts})
        qt_h = str(r.get("quoted_handle") or "").strip()
        if qt_h:
            dst = "@" + qt_h if not qt_h.startswith("@") else qt_h
            if src and dst and src != dst:
                rows.append({"src_user": src, "dst_user": dst, "edge_kind": "quote", "weight": 1, "timestamp": ts})

    if not rows:
        return pd.DataFrame(columns=["src_user","dst_user","edge_kind","weight","first_ts","last_ts"])

    edges = pd.DataFrame(rows)
    agg = (edges.groupby(["src_user","dst_user","edge_kind"])
                 .agg(weight=("weight","sum"),
                      first_ts=("timestamp","min"),
                      last_ts=("timestamp","max"))
                 ).reset_index()
    return agg

def nodes_metrics_df(G: nx.DiGraph) -> pd.DataFrame:
    return graph_metrics(G)
