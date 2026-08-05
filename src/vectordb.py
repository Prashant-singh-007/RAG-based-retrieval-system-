import os
import shutil
import pickle
from langchain_chroma import Chroma

PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
COLLECTION_NAME = "rag_documents"
BM25_CORPUS_FILENAME = "bm25_corpus.pkl"


def build_vectordb(chunks, embedding_model, persist_dir=PERSIST_DIR):
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )

    # BM25 (Section 5.2.2) needs the raw chunk text/metadata, not embeddings,
    # so the same chunk corpus is persisted alongside the vector store.
    with open(os.path.join(persist_dir, BM25_CORPUS_FILENAME), "wb") as f:
        pickle.dump(chunks, f)

    return vectordb


def load_vectordb(embedding_model, persist_dir=PERSIST_DIR):
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=persist_dir,
    )


def load_bm25_corpus(persist_dir=PERSIST_DIR):
    """Load the raw chunk corpus used to build the sparse BM25 index."""
    path = os.path.join(persist_dir, BM25_CORPUS_FILENAME)
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        return pickle.load(f)
