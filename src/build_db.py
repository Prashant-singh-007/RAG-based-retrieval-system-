import os
import sys

sys.path.append(os.path.dirname(__file__))

from loader import load_pdfs_from_dir
from chunker import chunk_documents
from embeddings import get_embedding_model
from vectordb import build_vectordb, PERSIST_DIR
from graph_builder import build_and_save_graph

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")


def main():
    print(f"Loading PDFs from {DATA_DIR} ...")
    # extract_images=True: identifies visual elements (block diagrams, pinout
    # schematics, etc.) and replaces them with [IMAGE_REF_n] placeholders
    # linked to saved image files via metadata (Section 3.1, Figure 1).
    documents = load_pdfs_from_dir(DATA_DIR, extract_images=True)
    print(f"Loaded {len(documents)} pages.")

    print("Chunking documents ...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks (1000/200 semantic chunking).")

    print("Loading embedding model (nomic-embed-text v1.5 via Ollama) ...")
    embedding_model = get_embedding_model()

    print(f"Building vector store at {PERSIST_DIR} ...")
    build_vectordb(chunks, embedding_model)
    print("Vector store and BM25 corpus are ready.")

    # IMPORTANT: pass the exact same `chunks` list used above - each chunk's
    # position in this list becomes its chunk_id, stored directly on its
    # graph node (graph_builder.py), so it must match 1:1 with what's in the
    # vector store / BM25 corpus.
    print("Building knowledge graph (entity/relation extraction via spaCy) ...")
    build_and_save_graph(chunks, PERSIST_DIR)
    print(f"Done. Vector store, BM25 corpus, and knowledge graph are ready at {PERSIST_DIR}.")


if __name__ == "__main__":
    main()
