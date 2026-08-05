"""VanillaRAG baseline index builder - mirrors the paper's baseline config
(Section 4.2): chunk_size=512, dense-only retrieval, no graph, no BM25,
no reranking. Built as a *separate* vector store so it doesn't interfere
with the main HSG-style pipeline's chroma_db/graph_checkpoint.

Run this once before using baseline_rag.py or run_evaluation.py.
"""
import os
import shutil
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from loader import load_pdfs_from_dir
from chunker import chunk_documents
from embeddings import get_embedding_model
from langchain_chroma import Chroma

# Adjust if your project layout differs - this assumes eval/ sits next to
# Data/ and src/, matching build_db.py's DATA_DIR convention.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
BASELINE_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db_baseline")
BASELINE_COLLECTION = "vanilla_rag_documents"

# Paper (Section 4.2): "we adopted a widely used configuration by setting
# the chunk size to 512" for VanillaRAG. The paper doesn't specify overlap;
# 50 is a conventional default kept small relative to the 512 chunk size -
# change BASELINE_CHUNK_OVERLAP below if you want to test sensitivity to it.
BASELINE_CHUNK_SIZE = 512
BASELINE_CHUNK_OVERLAP = 50


def main():
    print(f"Loading PDFs from {DATA_DIR} ...")
    # No image extraction needed for the baseline - VanillaRAG here is
    # text-only, matching the paper's baseline (no multimodal component).
    documents = load_pdfs_from_dir(DATA_DIR, extract_images=False)
    print(f"Loaded {len(documents)} pages.")

    print(f"Chunking documents ({BASELINE_CHUNK_SIZE}/{BASELINE_CHUNK_OVERLAP}, VanillaRAG baseline setting) ...")
    chunks = chunk_documents(
        documents,
        chunk_size=BASELINE_CHUNK_SIZE,
        chunk_overlap=BASELINE_CHUNK_OVERLAP,
    )
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model (same model as the main pipeline, for a fair comparison) ...")
    embedding_model = get_embedding_model()

    if os.path.exists(BASELINE_PERSIST_DIR):
        shutil.rmtree(BASELINE_PERSIST_DIR)
    os.makedirs(BASELINE_PERSIST_DIR, exist_ok=True)

    print(f"Building baseline vector store at {BASELINE_PERSIST_DIR} ...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=BASELINE_COLLECTION,
        persist_directory=BASELINE_PERSIST_DIR,
    )
    print("VanillaRAG baseline vector store ready.")


if __name__ == "__main__":
    main()
