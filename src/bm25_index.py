"""Sparse lexical retrieval (BM25) and Reciprocal Rank Fusion (RRF), used to
combine sparse and dense result lists for the Hybrid BM25 + Vector Search
strategy (Section 5.2.2).

Requires: pip install rank_bm25
"""

from rank_bm25 import BM25Okapi

RRF_K = 60  # standard RRF constant used in the paper


def _tokenize(text):
    return text.lower().split()


class BM25Index:
    """Lexical index over the same chunk corpus used for dense retrieval,
    built once at pipeline startup."""

    def __init__(self, documents):
        self.documents = documents
        self._corpus_tokens = [_tokenize(d.page_content) for d in documents]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def search(self, query, k=20):
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.documents[i] for i in ranked_idx[:k]]


def _doc_key(doc):
    return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])


def reciprocal_rank_fusion(ranked_lists, k=RRF_K, top_k=20):
    """RRF(d) = sum over rankings r of 1 / (k + rank_r(d)), per Cormack et al.
    (2009) as used in Section 5.2.2. Does not require score normalization
    between dense and sparse rankings.
    """
    scores = {}
    doc_lookup = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_lookup[key] = doc

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_lookup[key] for key, _ in fused[:top_k]]
