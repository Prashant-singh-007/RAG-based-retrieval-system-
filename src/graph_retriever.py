"""Query-time graph traversal retrieval for GraphRAG.

Implements the "Query -> Query Entity Extraction -> Graph Traversal (2-hop)
-> Retrieve associated chunks" stage of rag.py's retrieve() pipeline.

Traversal walks only entity<->entity edges (both directions) so hop count
means "N relationships away from an entity mentioned in the query" - chunk
nodes are not part of the hop count, they're the leaves collected via
MENTIONED_IN edges from whichever entities the traversal reached.
"""

from collections import deque

from langchain_core.documents import Document

from entity_extractor import extract_entities_relations
from graph_utils import normalize, is_entity_node, is_chunk_node

DEFAULT_HOPS = 2
DEFAULT_MAX_SEEDS = 8


def extract_query_entities(query, G, max_seeds=DEFAULT_MAX_SEEDS):
    """Find entity nodes in G that match the query.

    Two passes, so a graph built without spaCy recognizing every relevant
    term still gets reasonable seeds:
      1. Run the same spaCy/regex extractor used at build time on the
         query, and match extracted names against entity node names.
      2. Substring fallback: any entity node whose name appears literally
         in the query (covers cases like a query mentioning a part number
         spaCy's NER didn't tag as an entity but the regex pass elsewhere
         in the corpus did).
    """
    seeds = set()
    query_norm = query.lower()

    entity_names = {
        normalize(data.get("name", "")): node
        for node, data in G.nodes(data=True)
        if data.get("node_type") == "entity"
    }

    try:
        result = extract_entities_relations(query)
        for ent in result.get("entities", []):
            node = entity_names.get(normalize(ent["name"]))
            if node:
                seeds.add(node)
    except Exception:
        pass  # fall through to substring matching below

    if len(seeds) < max_seeds:
        for name_key, node in entity_names.items():
            if name_key and name_key in query_norm:
                seeds.add(node)
                if len(seeds) >= max_seeds:
                    break

    return list(seeds)[:max_seeds]


def traverse_entities(seed_nodes, G, n_hops=DEFAULT_HOPS):
    """BFS strictly over entity<->entity edges, up to n_hops.
    Returns {entity_node_id: hop_distance}."""
    visited = {n: 0 for n in seed_nodes if G.has_node(n)}
    frontier = deque(visited.keys())

    while frontier:
        node = frontier.popleft()
        dist = visited[node]
        if dist >= n_hops:
            continue
        neighbors = set(G.successors(node)) | set(G.predecessors(node))
        for neighbor in neighbors:
            if not is_entity_node(G, neighbor):
                continue  # don't traverse through/into chunk nodes
            if neighbor not in visited:
                visited[neighbor] = dist + 1
                frontier.append(neighbor)

    return visited


def _mentioned_chunks(entity_node, G):
    """Chunk nodes this entity is MENTIONED_IN (outgoing edges only, since
    graph_builder.py always adds these edges entity -> chunk)."""
    for neighbor in G.successors(entity_node):
        if not is_chunk_node(G, neighbor):
            continue
        edge_data = G.get_edge_data(entity_node, neighbor)
        if any(e.get("relation") == "MENTIONED_IN" for e in edge_data.values()):
            yield neighbor


def _chunk_node_to_document(node_id, G):
    """Reconstruct a langchain Document straight from chunk-node metadata -
    no separate corpus lookup needed."""
    data = G.nodes[node_id]
    return Document(
        page_content=data.get("text", ""),
        metadata={
            "source": data.get("source", "unknown"),
            "page": data.get("page", "?"),
            "chunk_id": data.get("chunk_id", ""),
            "image_refs": data.get("image_refs", ""),
        },
    )


def graph_search(query, G, n_hops=DEFAULT_HOPS, top_k=20):
    """Full graph-retrieval path: entity extraction -> 2-hop entity
    traversal -> chunk nodes reached (via MENTIONED_IN) from any visited
    entity, ranked by the minimum hop distance among entities that reach it.

    Returns a list of Document, closest match first. Returns [] if there's
    no graph, or the query doesn't match any entity in it (dense retrieval
    still covers that case in rag.py).
    """
    if G is None:
        return []

    seeds = extract_query_entities(query, G)
    if not seeds:
        return []

    visited_entities = traverse_entities(seeds, G, n_hops=n_hops)

    best_hop_per_chunk = {}
    for entity_node, hop in visited_entities.items():
        for chunk_node in _mentioned_chunks(entity_node, G):
            if chunk_node not in best_hop_per_chunk or hop < best_hop_per_chunk[chunk_node]:
                best_hop_per_chunk[chunk_node] = hop

    ranked = sorted(best_hop_per_chunk.items(), key=lambda item: item[1])
    return [_chunk_node_to_document(node_id, G) for node_id, _ in ranked[:top_k]]
