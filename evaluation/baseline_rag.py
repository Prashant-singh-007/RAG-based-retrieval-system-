"""VanillaRAG baseline pipeline (Section 4.2): plain dense retrieval over a
512-char-chunk vector store, no graph traversal, no BM25/RRF, and no
cross-encoder reranking. Used as the comparison arm for the LLM judge and
for the retrieval Recall@k table, mirroring the paper's baseline setup so
your HSG-style pipeline (src/rag.py) can be evaluated against something
equivalent to what the paper calls "VanillaRAG".

Requires baseline_build_db.py to have been run first.
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from embeddings import get_embedding_model
from baseline_build_db import BASELINE_PERSIST_DIR, BASELINE_COLLECTION

# Same generation model as rag.py's Model B, so any quality difference in
# the judge comparison reflects the retrieval/index strategy, not the LLM.
OLLAMA_MODEL = "llama3.2"

# Deliberately plain - no fact/synthesis branching like rag.py's
# QA_PROMPT_TEMPLATE, since VanillaRAG in the paper is the minimal
# "conventional RAG method" comparison point.
VANILLA_QA_PROMPT = """
Answer the question using ONLY the provided context. If the answer is not
in the context, say so directly rather than guessing.

Context:
{context}

Question:
{question}

Answer:
"""


class VanillaRAGPipeline:
    def __init__(self, top_k=4):
        self.top_k = top_k
        self.embedding_model = get_embedding_model()
        self.vectordb = Chroma(
            collection_name=BASELINE_COLLECTION,
            embedding_function=self.embedding_model,
            persist_directory=BASELINE_PERSIST_DIR,
        )
        self.llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
        self.qa_prompt = ChatPromptTemplate.from_template(VANILLA_QA_PROMPT)

    def retrieve(self, query, k=None):
        """Plain dense top-k similarity search - no diversity expansion,
        no boosting, no reranking (unlike retriever.py's iterative_retrieve
        used by the main pipeline)."""
        k = k or self.top_k
        return self.vectordb.similarity_search(query, k=k)

    def answer(self, query):
        docs = self.retrieve(query)
        context = "\n\n".join(
            f"[{d.metadata.get('source', 'unknown')} p.{d.metadata.get('page', '?')}] {d.page_content}"
            for d in docs
        )
        message = self.qa_prompt.format(context=context, question=query)
        response = self.llm.invoke(message)
        return response.content, docs
