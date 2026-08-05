import os
import sys
import json

sys.path.append(os.path.dirname(__file__))

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from embeddings import get_embedding_model
from vectordb import load_vectordb, PERSIST_DIR
from reranker import get_reranker, rerank
from retriever import iterative_retrieve
from graph_utils import load_graph
from graph_retriever import graph_search, DEFAULT_HOPS
from loader import load_pdf
from summarizer import generate_structured_summary, summary_to_text
from image_utils import collect_image_paths, build_multimodal_content

# Model B (Section 3.3.3 / Figure 3): comparative-reasoning and answer
# generation model. Vision-capable so it can reason over actual pinouts,
# block diagrams, and schematics, not just [IMAGE_REF_n] text tags.
OLLAMA_MODEL = "llama3.2"

QA_PROMPT_TEMPLATE = """
Answer the question using ONLY the provided context.

Rules:
- If the question asks for one specific fact (a number, date, name, amount,
  or duration), give that fact clearly, copying figures and qualifiers
  (e.g. "Up to", "At least", "Within") exactly as written in the context.
  A short supporting clause is fine (e.g. "Up to 12 GHz, for the RF input
  frequency range").
- If the question is broader or descriptive (e.g. "tell me about X",
  "what does X do", "summarize X"), synthesize the relevant information from
  the context into a coherent answer of a few sentences to a short
  paragraph, covering what the component is, its key specs, and notable
  features. Do not pad with filler; every sentence should carry information
  from the context.
- Do not restate the question verbatim, and do not mention sources or page
  numbers in the answer text.
- If the answer is not found in the context, say so directly rather than
  guessing.

Context:
{context}

Question:
{question}

Answer:
"""

# Section 3.3.3: "the prompt aggregates the input datasheet with the k
# retrieved reference datasheets and injects a structured system instruction
# requiring a detailed analysis of similarities and differences", further
# augmented by any user-defined instruction from the GUI.
COMPARISON_PROMPT_TEMPLATE = """
You are an electronics engineer comparing a target component against candidate
replacement/alternative components, using ONLY the provided context.

Target component summary:
{target_summary}

Candidate components:
{context}

{user_instruction}

Provide a structured comparison covering, for each candidate: overall
similarity to the target, key matching specifications, key differences, and
whether it is a plausible drop-in or near replacement. Base every claim
strictly on the provided context; do not invent specifications.
"""


def _doc_key(doc):
    """Dedup key for merging graph + dense results: same (source, page,
    text-prefix) identity used elsewhere in this project (bm25_index.py's
    reciprocal_rank_fusion used the same shape of key). Not chunk_id,
    because dense-retrieval Documents coming straight out of Chroma don't
    carry a chunk_id in their metadata - only graph-reconstructed Documents
    do (see graph_retriever.py's _chunk_node_to_document) - so we need a key
    both sources can produce identically.
    """
    return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])


def merge_candidates(graph_docs, dense_docs):
    """Merge graph-traversal and dense-retrieval candidate lists into one
    deduplicated list. Graph results are listed first (they're grounded in
    an explicit extracted relationship to the query's entities, which is
    exactly the signal dense similarity search can't see), followed by any
    dense hits not already covered. The cross-encoder reranker (next stage)
    does the real fine-grained ordering, so this merge only needs to dedupe,
    not rank.
    """
    seen = set()
    merged = []
    for doc in graph_docs + dense_docs:
        key = _doc_key(doc)
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return merged


class RAGPipeline:
    def __init__(self, top_k=4, retrieve_k=20, use_rerank=True,
                 use_graph=True, graph_hops=DEFAULT_HOPS):
        """
        top_k: number of unique components to return (k in the paper's FRR@k).
        retrieve_k: candidate pool size before final ranking/reranking.
        use_rerank: apply cross-encoder reranking to the merged candidate
                    set (Section 5.2.3).
        use_graph: run the graph-traversal retrieval stage (query entity
                   extraction -> 2-hop entity traversal -> linked chunks)
                   and merge its results with dense retrieval before
                   reranking. Requires build_db.py to have built and saved
                   a knowledge graph (graph_builder.py); if no graph
                   checkpoint is found this prints a notice and the
                   pipeline runs on dense retrieval alone.
        graph_hops: max BFS hop distance for entity-to-entity graph
                    traversal (2 = the requested default).
        """
        self.top_k = top_k
        self.retrieve_k = retrieve_k
        self.use_rerank = use_rerank
        self.use_graph = use_graph
        self.graph_hops = graph_hops

        self.embedding_model = get_embedding_model()
        self.vectordb = load_vectordb(self.embedding_model)
        self.reranker = get_reranker() if use_rerank else None

        self.G = load_graph(PERSIST_DIR) if use_graph else None
        if use_graph and self.G is None:
            print("No graph checkpoint found at", PERSIST_DIR,
                  "- graph retrieval disabled for this run "
                  "(run build_db.py to build and save the knowledge graph).")

        self.llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
        self.qa_prompt = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)
        self.comparison_prompt = ChatPromptTemplate.from_template(COMPARISON_PROMPT_TEMPLATE)

    # ---- Section 3.3.2: Iterative Retrieval and Ranking Engine ----

    def _dense_candidates(self, query_text, k):
        ranked = iterative_retrieve(self.vectordb, query_text, k=k)
        return [doc for doc, _ in ranked]

    def _graph_candidates(self, query_text, k):
        if self.G is None:
            return []
        return graph_search(query_text, self.G, n_hops=self.graph_hops, top_k=k)

    def retrieve(self, query_text, k=None):
        """GraphRAG retrieval pipeline:

            query_text
              -> query entity extraction + 2-hop graph traversal (_graph_candidates)
              -> dense vector retrieval (_dense_candidates)
              -> merge graph + dense results, deduplicated (merge_candidates)
              -> cross-encoder reranking (rerank), if enabled
              -> top-k Documents

        Replaces the previous BM25+RRF hybrid stage: graph traversal now
        supplies the "second signal" alongside dense similarity, surfacing
        chunks connected to the query's entities by an explicit extracted
        relationship even when they aren't lexically or semantically close
        to the query text itself.
        """
        k = k or self.top_k

        graph_docs = self._graph_candidates(query_text, self.retrieve_k)
        dense_docs = self._dense_candidates(query_text, self.retrieve_k)
        candidates = merge_candidates(graph_docs, dense_docs)

        if self.use_rerank and self.reranker:
            return rerank(self.reranker, query_text, candidates, k)
        return candidates[:k]

    # ---- shared multimodal invocation ----

    def _invoke(self, prompt_text, image_paths=None):
        """Invoke Model B with plain text, or text+images (multimodal
        HumanMessage) when the retrieved chunks reference extracted figures."""
        if image_paths:
            content = build_multimodal_content(prompt_text, image_paths)
            response = self.llm.invoke([HumanMessage(content=content)])
        else:
            response = self.llm.invoke(prompt_text)
        return response.content

    # ---- Section 3.3.1: Direct Question Answering mode ----

    def answer(self, query, use_images=True):
        """The user's natural-language question is embedded directly to form
        the query vector (Section 3.3.1)."""
        docs = self.retrieve(query)
        context = "\n\n".join(
            f"[{d.metadata.get('source', 'unknown')} p.{d.metadata.get('page', '?')}] {d.page_content}"
            for d in docs
        )
        message = self.qa_prompt.format(context=context, question=query)
        image_paths = collect_image_paths(docs) if use_images else []
        return self._invoke(message, image_paths), docs

    # ---- Section 3.2/3.3.1: Comparative Analysis mode ----

    def compare_datasheet(self, pdf_path, user_instruction="", direct_comparison_mode=False, k=None):
        """Runs the on-demand multimodal-style inference pipeline (Figure 2):
        the input datasheet is processed without chunking to preserve global
        context, summarized by Model A, and the summary embedding is used as
        the query vector for similarity retrieval. Model B then produces a
        structured comparative report (Figure 3).

        direct_comparison_mode: if True, retrieves and passes the FULL text
        of each candidate datasheet to Model B (higher cost, more context);
        if False, uses only the retrieved chunks (Section 3.3.3).
        """
        k = k or self.top_k

        input_docs = load_pdf(pdf_path)
        full_text = "\n\n".join(d.page_content for d in input_docs)

        # Model A analyzes the input datasheet's text AND figures jointly
        # (Figure 2: "Text and Figures" -> Summary generation).
        input_image_paths = collect_image_paths(input_docs)
        structured_summary = generate_structured_summary(
            full_text, image_paths=input_image_paths
        )
        query_text = summary_to_text(structured_summary)

        retrieved = self.retrieve(query_text, k=k)

        if direct_comparison_mode:
            sources = {d.metadata.get("source") for d in retrieved}
            context_parts = []
            for source in sources:
                full = self.vectordb.get(where={"source": source})
                text = "\n\n".join(full.get("documents", []))
                context_parts.append(f"=== {source} ===\n{text}")
            context = "\n\n".join(context_parts)
        else:
            context = "\n\n".join(
                f"[{d.metadata.get('source', 'unknown')} p.{d.metadata.get('page', '?')}] {d.page_content}"
                for d in retrieved
            )

        # Model B's comparison sees the target's figures plus the candidates'
        # figures (Section 3.3.3: "complete textual content and associated
        # visual data (images and schematics) for the top-k selected
        # datasheets" in direct comparison mode).
        comparison_image_paths = input_image_paths + collect_image_paths(retrieved)

        instruction = f"Specific question/instruction: {user_instruction}" if user_instruction else ""
        target_summary_text = (
            json.dumps(structured_summary, ensure_ascii=False)
            if isinstance(structured_summary, dict)
            else structured_summary
        )
        message = self.comparison_prompt.format(
            target_summary=target_summary_text,
            context=context,
            user_instruction=instruction,
        )
        report = self._invoke(message, comparison_image_paths)
        return report, retrieved, structured_summary


def main():
    pipeline = RAGPipeline()
    print("RAG ready.")
    print("Ask a question directly, or type 'compare <path-to-pdf>' to run a comparative analysis, or 'exit'.")
    while True:
        query = input("\n> ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        if query.lower().startswith("compare "):
            pdf_path = query[len("compare "):].strip()
            if not os.path.exists(pdf_path):
                print(f"File not found: {pdf_path}")
                continue
            report, docs, summary = pipeline.compare_datasheet(pdf_path)
            print(f"\nInput summary:\n{summary}")
            print(f"\nComparative Report:\n{report}")
            print("\nRetrieved components:")
            for d in docs:
                print(f" - {d.metadata.get('source')} (page {d.metadata.get('page')})")
            continue

        answer, docs = pipeline.answer(query)
        print(f"\nAnswer:\n{answer}")
        print("\nSources:")
        for d in docs:
            print(f" - {d.metadata.get('source')} (page {d.metadata.get('page')})")


def query_rag(question):
    pipeline = RAGPipeline()
    answer, docs = pipeline.answer(question)
    contexts = [doc.page_content for doc in docs]
    return {
        "answer": answer,
        "contexts": contexts,
    }


if __name__ == "__main__":
    main()
