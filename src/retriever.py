"""Iterative Retrieval and Ranking Engine (Section 3.3.2). Standard vector
search returns individual chunks, often several from the same document. To
ensure diversity, the Reference Vector Database is queried and results are
filtered down to unique components (source datasheets); if the initial batch
yields fewer than k unique components, the batch size is expanded until the
target is met or a safety limit is reached. Candidates are then ranked with
a boosted cosine-equivalent similarity score.
"""

DEFAULT_BATCH = 20
MAX_BATCH = 200
EXPLICIT_CONSTRAINT_BOOST = 0.25  # Eq. (2): e.g. a specific manufacturer match
CONTEXTUAL_BOOST = 0.10  # Eq. (2): components contextually related to the input


def _component_id(doc):
    """A 'component' is one source datasheet; page/chunk-level results are
    collapsed to their source document for diversity-aware retrieval."""
    return doc.metadata.get("source", "unknown")


def cosine_equivalent_similarity(distance):
    """Eq. (1): for L2-normalized embeddings, converts Euclidean distance d_i
    into a cosine-equivalent similarity s_i = 1 - d_i^2 / 2."""
    return 1.0 - (distance ** 2) / 2.0


def apply_boost(similarity, matches_explicit_constraint=False, is_contextually_related=False):
    """Eq. (2): bounded additive boost s'_i = min(1.0, s_i + beta_i)."""
    beta = 0.0
    if matches_explicit_constraint:
        beta = EXPLICIT_CONSTRAINT_BOOST
    elif is_contextually_related:
        beta = CONTEXTUAL_BOOST
    return min(1.0, similarity + beta)


def iterative_retrieve(vectordb, query, k=4, constraints=None):
    """Query the Reference Vector Database, expanding the search batch until
    at least k unique components are found (or MAX_BATCH is reached), then
    rank them by boosted cosine-equivalent similarity.

    constraints: optional dict of explicit user constraints, e.g.
    {"manufacturer": "Texas Instruments"}, used for the 0.25 boost. A looser
    "contextually related" boost (0.10) is left to the caller to set, since it
    depends on comparison semantics beyond raw metadata matching.

    Returns a list of (Document, boosted_similarity) tuples, best match first.
    """
    constraints = constraints or {}
    batch_size = DEFAULT_BATCH

    while True:
        results = vectordb.similarity_search_with_score(query, k=batch_size)

        best_per_component = {}
        for doc, distance in results:
            cid = _component_id(doc)
            similarity = cosine_equivalent_similarity(distance)
            if cid not in best_per_component or similarity > best_per_component[cid][1]:
                best_per_component[cid] = (doc, similarity)

        if len(best_per_component) >= k or batch_size >= MAX_BATCH:
            break
        batch_size = min(batch_size * 2, MAX_BATCH)

    ranked = []
    for doc, similarity in best_per_component.values():
        matches_explicit = any(
            str(value).lower() in doc.page_content.lower()
            or str(value).lower() == str(doc.metadata.get(field, "")).lower()
            for field, value in constraints.items()
        )
        boosted = apply_boost(similarity, matches_explicit_constraint=matches_explicit)
        ranked.append((doc, boosted))

    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked[:k]
