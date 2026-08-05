"""Knowledge-graph construction for GraphRAG.

Builds a single NetworkX MultiDiGraph containing two kinds of nodes:

  - CHUNK nodes: one per chunk, storing source, page, chunk_id, the chunk's
    full text, and its image_refs string - everything needed to reconstruct
    a retrievable langchain Document directly from the graph (see
    graph_retriever.py), with no separate corpus file to keep in sync.
  - ENTITY nodes: one per distinct entity extracted anywhere in the corpus
    (component names, part numbers, manufacturers, interfaces, etc.),
    deduplicated case-insensitively across all chunks.

Edges:
  - entity --MENTIONED_IN--> chunk, added whenever that entity was extracted
    from that chunk (this is what lets graph_retriever.py map traversed
    entities back to retrievable chunks).
  - entity --RELATION--> entity, for relations extracted within a chunk
    (edge carries the originating chunk_id and a relation label, e.g.
    "CO_OCCURS_WITH" from entity_extractor.py).
"""

import networkx as nx

from entity_extractor import extract_entities_relations
from graph_utils import entity_node_id, chunk_node_id, normalize, save_graph


def build_knowledge_graph(chunks, verbose=True):
    """chunks: the list of langchain Documents produced by chunker.py - the
    SAME list (same order) passed to build_vectordb()/BM25Index, since each
    chunk's position in this list becomes its chunk_id. Consistency across
    build_db.py's steps is what lets graph nodes and vector-store entries
    refer to "the same chunk" unambiguously.
    """
    G = nx.MultiDiGraph()
    total = len(chunks)

    for idx, chunk in enumerate(chunks):
        if verbose and idx % 50 == 0:
            print(f"  Graph extraction: chunk {idx}/{total}")

        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        chunk_id = str(idx)
        c_node = chunk_node_id(source, page, chunk_id)

        G.add_node(
            c_node,
            node_type="chunk",
            source=source,
            page=page,
            chunk_id=chunk_id,
            text=chunk.page_content,
            image_refs=chunk.metadata.get("image_refs", ""),
        )

        try:
            result = extract_entities_relations(chunk.page_content)
        except Exception as e:
            print(f"  Entity extraction failed on chunk {idx}: {e}")
            continue

        # Map this chunk's locally extracted entity names -> their graph
        # node ids, so relations (which reference entities by name) can be
        # resolved to the correct (deduplicated, corpus-wide) node.
        local_ids = {}
        for ent in result.get("entities", []):
            e_node = entity_node_id(ent["name"], ent.get("type", "ENTITY"))
            if not G.has_node(e_node):
                G.add_node(e_node, node_type="entity", name=ent["name"], type=ent.get("type", "ENTITY"))
            G.add_edge(e_node, c_node, relation="MENTIONED_IN", chunk_id=chunk_id)
            local_ids[normalize(ent["name"])] = e_node

        for rel in result.get("relations", []):
            src_node = local_ids.get(normalize(rel.get("source", "")))
            tgt_node = local_ids.get(normalize(rel.get("target", "")))
            if src_node and tgt_node and src_node != tgt_node:
                G.add_edge(
                    src_node, tgt_node,
                    relation=rel.get("relation", "RELATED_TO"),
                    chunk_id=chunk_id,
                )

    if verbose:
        n_chunk_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "chunk")
        n_entity_nodes = G.number_of_nodes() - n_chunk_nodes
        print(
            f"  Graph built: {n_chunk_nodes} chunk nodes, "
            f"{n_entity_nodes} entity nodes, {G.number_of_edges()} edges."
        )

    return G


def build_and_save_graph(chunks, persist_dir, verbose=True):
    """Convenience wrapper used by build_db.py: build the graph, persist it
    to persist_dir (same directory as the Chroma store / BM25 corpus), and
    return it."""
    G = build_knowledge_graph(chunks, verbose=verbose)
    save_graph(G, persist_dir)
    return G
