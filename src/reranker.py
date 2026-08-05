from sentence_transformers import CrossEncoder

# Paper (Section 5.2.3): cross-encoder reranking with ms-marco-MiniLM-L-6-v2,
# trained on MS MARCO passage ranking [48,49].
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker():
    return CrossEncoder(MODEL_NAME)


def rerank(reranker, query, docs, top_k):
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
