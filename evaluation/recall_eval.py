"""Recall@k evaluation (Section 4.3 / Table 4), adapted to a
question + reference_answer-only dataset.csv (no ground-truth source/chunk
labels).

IMPORTANT - this is a proxy, not the paper's exact metric:
The paper's Recall@k checks whether the ground-truth supporting text unit
appears in the top-k retrieved set. Without a per-question ground-truth
chunk/source label, that check isn't directly computable. This module
instead estimates a **groundedness proxy**: for each k, does the top-k
retrieved context contain enough of the reference answer's content to
plausibly have produced it?

Two proxy implementations:
  - lexical_recall_at_k: fast, no LLM calls. Fraction of the reference
    answer's informative (non-stopword) tokens found verbatim in the
    concatenated top-k context. Cheap but brittle to paraphrasing -
    treat absolute numbers as directional, not precise.
  - llm_groundedness_at_k: one LLM call per (question, k), asking whether
    the reference answer is supported by the context. Slower, more
    semantically robust, closer in spirit to the paper's check.

If you can add ground-truth labels to dataset.csv (e.g. a `gt_source` and
`gt_page` column identifying the chunk that actually answers each
question), use strict_recall_at_k instead - it IS the paper's exact metric,
with no proxy involved.
"""
import re

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on",
    "for", "and", "or", "with", "this", "that", "it", "as", "by", "be",
    "can", "will", "you", "your", "if", "at", "from", "which", "these",
    "those", "into", "using", "use", "used", "not", "should", "must",
    "it's", "its", "their", "there", "than", "then", "also",
}


def _content_tokens(text):
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def lexical_recall_at_k(reference_answer, retrieved_docs_by_k):
    """retrieved_docs_by_k: {k: [Document, ...]}, one entry per k value,
    already sliced to the top-k for that k. Returns {k: recall_fraction}
    for this single question."""
    ref_tokens = _content_tokens(reference_answer)
    if not ref_tokens:
        return {k: 0.0 for k in retrieved_docs_by_k}

    results = {}
    for k, docs in retrieved_docs_by_k.items():
        context_tokens = _content_tokens(" ".join(d.page_content for d in docs))
        hit = ref_tokens & context_tokens
        results[k] = len(hit) / len(ref_tokens)
    return results


_GROUNDEDNESS_MODEL = "llama3.2"
_GROUNDEDNESS_PROMPT = """Given the CONTEXT below, judge whether the REFERENCE ANSWER's
information could plausibly have been produced from this context alone.

Answer with exactly one word: "yes" or "no".

CONTEXT:
{context}

REFERENCE ANSWER:
{reference_answer}

Answer:
"""


def llm_groundedness_at_k(reference_answer, retrieved_docs_by_k, llm=None):
    """Same shape/contract as lexical_recall_at_k, but each k's score is a
    binary 0/1 groundedness judgment from an LLM rather than a token-overlap
    fraction. Costs one LLM call per k per question."""
    llm = llm or ChatOllama(model=_GROUNDEDNESS_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(_GROUNDEDNESS_PROMPT)

    results = {}
    for k, docs in retrieved_docs_by_k.items():
        context = "\n\n".join(d.page_content for d in docs)
        message = prompt.format(context=context, reference_answer=reference_answer)
        answer = llm.invoke(message).content.strip().lower()
        results[k] = 1.0 if answer.startswith("y") else 0.0
    return results


def strict_recall_at_k(ground_truth_key, retrieved_docs_by_k, doc_key_fn):
    """The paper's exact Recall@k - use this INSTEAD of the two proxies
    above if you add ground-truth labels to dataset.csv.

    ground_truth_key: the key doc_key_fn would produce for the true
    supporting chunk, e.g. (source, page) built the same way as
    bm25_index.py's _doc_key / rag.py's _doc_key.
    doc_key_fn: a function(Document) -> key, applied to each retrieved doc
    (reuse the same _doc_key already defined in rag.py or bm25_index.py so
    keys match exactly).
    """
    results = {}
    for k, docs in retrieved_docs_by_k.items():
        keys = {doc_key_fn(d) for d in docs}
        results[k] = 1.0 if ground_truth_key in keys else 0.0
    return results
