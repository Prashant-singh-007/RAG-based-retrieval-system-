"""Shared utilities for the GraphRAG knowledge graph.

Kept separate from graph_builder.py (construction) and graph_retriever.py
(query-time traversal) so both can share the same node-ID scheme and
persistence format without duplicating logic.
"""

import json
import os
import re

import networkx as nx

GRAPH_CHECKPOINT_FILENAME = "graph_checkpoint.json"


def normalize(name):
    """Case/whitespace-insensitive key used for matching entity names
    (e.g. so "TPS54331" and "tps54331" resolve to the same node)."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def entity_node_id(name, ntype="ENTITY"):
    """Deterministic node id for an entity, namespaced by type so two
    different entity types that happen to share a normalized name (rare,
    but possible) don't collide."""
    return f"entity::{ntype}::{normalize(name)}"


def chunk_node_id(source, page, chunk_id):
    """Deterministic node id for a chunk, namespaced so chunk nodes never
    collide with entity nodes in the same graph."""
    return f"chunk::{source}::{page}::{chunk_id}"


def is_entity_node(G, node_id):
    return G.has_node(node_id) and G.nodes[node_id].get("node_type") == "entity"


def is_chunk_node(G, node_id):
    return G.has_node(node_id) and G.nodes[node_id].get("node_type") == "chunk"


def save_graph(G, persist_dir):
    """Persist the graph as node-link JSON alongside the vector store
    (same persist_dir Chroma/vectordb.py already uses)."""
    os.makedirs(persist_dir, exist_ok=True)
    path = os.path.join(persist_dir, GRAPH_CHECKPOINT_FILENAME)
    data = nx.node_link_data(G)
    with open(path, "w") as f:
        json.dump(data, f)


def load_graph(persist_dir):
    """Load the persisted graph. Returns None if no checkpoint exists yet
    (e.g. build_db.py hasn't been (re)run since GraphRAG was added)."""
    path = os.path.join(persist_dir, GRAPH_CHECKPOINT_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return nx.node_link_graph(data)
