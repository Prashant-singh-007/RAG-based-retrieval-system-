"""Streamlit frontend for the RAG pipeline (src/rag.py).

Run with:
    streamlit run frontend/app.py

Three modes:
  - Ask a question      -> RAGPipeline.answer() against your indexed corpus
  - Upload a PDF & ask   -> ad-hoc, in-memory index of just the uploaded file
  - Compare a datasheet  -> RAGPipeline.compare_datasheet()
"""
import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rag import RAGPipeline  # noqa: E402

st.set_page_config(page_title="Datasheet RAG", page_icon="📟", layout="wide")


# ---- Pipeline loading (cached so it's built once per session, not per rerun) ----

@st.cache_resource(show_spinner="Loading pipeline (embeddings, vector store, graph, reranker)...")
def load_pipeline(top_k, retrieve_k, use_rerank, use_graph):
    return RAGPipeline(
        top_k=top_k,
        retrieve_k=retrieve_k,
        use_rerank=use_rerank,
        use_graph=use_graph,
    )


# ---- Sidebar: settings + mode switch ----

st.sidebar.title("📟 Datasheet RAG")
mode = st.sidebar.radio("Mode", ["Ask a question", "Upload a PDF & ask", "Compare a datasheet"])

st.sidebar.divider()
st.sidebar.subheader("Retrieval settings")
top_k = st.sidebar.slider("top_k (components returned)", 1, 10, 4)
retrieve_k = st.sidebar.slider("retrieve_k (candidate pool)", 5, 50, 20)
use_rerank = st.sidebar.checkbox("Cross-encoder reranking", value=True)
use_graph = st.sidebar.checkbox("Graph traversal retrieval", value=True)

pipeline = load_pipeline(top_k, retrieve_k, use_rerank, use_graph)

# ---- Diagnostics: confirm the pipeline actually loaded real data ----
# If you're seeing "not found in context" for everything, check this panel
# first - "0 chunks indexed" means the vector store path resolved to an
# empty/wrong directory (usually a working-directory mismatch), not a real
# retrieval failure.
with st.sidebar.expander("🔍 Diagnostics", expanded=True):
    try:
        from vectordb import PERSIST_DIR
        st.caption(f"Vector store path:\n`{PERSIST_DIR}`")
        st.caption(f"Exists on disk: {os.path.isdir(PERSIST_DIR)}")
    except Exception as e:
        st.error(f"Could not resolve PERSIST_DIR: {e}")

    try:
        count = pipeline.vectordb._collection.count()
        if count == 0:
            st.error(f"0 chunks indexed — vector store is empty. Run build_db.py, "
                      f"or check the path above points at your real chroma_db.")
        else:
            st.success(f"{count} chunks indexed")
    except Exception as e:
        st.warning(f"Could not read chunk count: {e}")

    if use_graph:
        if pipeline.G is None:
            st.error("Graph enabled but no graph_checkpoint.json found at that path.")
        else:
            st.success(f"Graph loaded: {pipeline.G.number_of_nodes()} nodes, {pipeline.G.number_of_edges()} edges")


def render_sources(docs):
    """Shared source/image renderer used by both modes."""
    if not docs:
        return
    with st.expander(f"📄 Sources ({len(docs)})", expanded=False):
        for d in docs:
            source = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "?")
            st.markdown(f"**{source}** — page {page}")
            st.caption(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))
            st.divider()

    try:
        from image_utils import collect_image_paths
        image_paths = collect_image_paths(docs)
    except Exception:
        image_paths = []
    if image_paths:
        with st.expander(f"🖼️ Referenced figures ({len(image_paths)})", expanded=False):
            cols = st.columns(min(len(image_paths), 4))
            for i, path in enumerate(image_paths):
                if os.path.exists(path):
                    cols[i % len(cols)].image(path, use_container_width=True)


# ---- Mode 1: Ask a question (chat interface) ----

if mode == "Ask a question":
    st.title("Ask a question")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant" and turn.get("docs"):
                render_sources(turn["docs"])

    query = st.chat_input("Ask about a component, e.g. 'What is the operating voltage of the ADMV1455?'")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                answer, docs = pipeline.answer(query)
            st.markdown(answer)
            render_sources(docs)

        st.session_state.chat_history.append({"role": "assistant", "content": answer, "docs": docs})

    if st.session_state.chat_history:
        st.sidebar.divider()
        if st.sidebar.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()


# ---- Mode 2: Upload a PDF & ask (chat scoped to just that document) ----

elif mode == "Upload a PDF & ask":
    st.title("Upload a PDF and ask questions about it")
    st.caption(
        "Runs your actual GraphRAG pipeline on just this file: entity/relation "
        "extraction, an in-memory knowledge graph, and graph-guided + dense "
        "retrieval — same as build_db.py, just not written to disk."
    )

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key="qa_pdf_upload")

    if uploaded:
        # Only re-index if this is a new/different file - entity extraction
        # is the slow part, so we don't want to redo it on every chat turn.
        file_bytes = uploaded.getvalue()
        file_key = f"{uploaded.name}_{len(file_bytes)}"

        if st.session_state.get("uploaded_pdf_key") != file_key:
            tmp_path = os.path.join("/tmp", uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(file_bytes)

            from loader import load_pdf
            from chunker import chunk_documents
            from langchain_chroma import Chroma
            from graph_builder import build_knowledge_graph

            with st.spinner("Reading and chunking the PDF..."):
                docs = load_pdf(tmp_path, extract_images=True)
                chunks = chunk_documents(docs)

            with st.spinner(f"Building dense index over {len(chunks)} chunks..."):
                # No persist_directory -> in-memory only. Never written to
                # disk, never collides with your real chroma_db.
                doc_vectordb = Chroma.from_documents(
                    documents=chunks,
                    embedding=pipeline.embedding_model,
                    collection_name=f"upload_{abs(hash(file_key))}",
                )

            doc_graph = None
            if use_graph:
                with st.spinner(
                    "Extracting entities/relations and building the knowledge graph "
                    "(the slow step - scales with document length)..."
                ):
                    # Same construction graph_builder.build_and_save_graph uses,
                    # just kept in memory instead of saved to a checkpoint file.
                    doc_graph = build_knowledge_graph(chunks, verbose=False)

            st.session_state.uploaded_pdf_key = file_key
            st.session_state.uploaded_pdf_vectordb = doc_vectordb
            st.session_state.uploaded_pdf_graph = doc_graph
            st.session_state.uploaded_pdf_chat = []

            msg = f"Indexed {len(chunks)} chunks from {uploaded.name}"
            if doc_graph is not None:
                msg += f" — graph: {doc_graph.number_of_nodes()} nodes, {doc_graph.number_of_edges()} edges"
            st.success(msg)

        doc_vectordb = st.session_state.uploaded_pdf_vectordb
        doc_graph = st.session_state.uploaded_pdf_graph
        chat_key = "uploaded_pdf_chat"

        for turn in st.session_state.get(chat_key, []):
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn["role"] == "assistant" and turn.get("docs"):
                    render_sources(turn["docs"])

        query = st.chat_input("Ask a question about this PDF")
        if query:
            st.session_state[chat_key].append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Graph traversal + dense retrieval..."):
                    from graph_retriever import graph_search
                    from retriever import iterative_retrieve
                    from rag import merge_candidates
                    from reranker import rerank as rerank_fn

                    graph_docs = (
                        graph_search(query, doc_graph, n_hops=pipeline.graph_hops, top_k=pipeline.retrieve_k)
                        if doc_graph is not None else []
                    )
                    dense_docs = [
                        d for d, _ in iterative_retrieve(doc_vectordb, query, k=pipeline.retrieve_k)
                    ]
                    candidates = merge_candidates(graph_docs, dense_docs)

                    if use_rerank and pipeline.reranker:
                        docs = rerank_fn(pipeline.reranker, query, candidates, top_k)
                    else:
                        docs = candidates[:top_k]

                    context = "\n\n".join(
                        f"[{d.metadata.get('source', 'unknown')} p.{d.metadata.get('page', '?')}] {d.page_content}"
                        for d in docs
                    )
                    message = pipeline.qa_prompt.format(context=context, question=query)
                    from image_utils import collect_image_paths
                    image_paths = collect_image_paths(docs)
                    answer = pipeline._invoke(message, image_paths)

                st.markdown(answer)
                st.caption(f"Retrieved via: {len(graph_docs)} graph hits, {len(dense_docs)} dense hits → {len(docs)} after merge/rerank")
                render_sources(docs)

            st.session_state[chat_key].append({"role": "assistant", "content": answer, "docs": docs})

        if st.session_state.get(chat_key):
            st.sidebar.divider()
            if st.sidebar.button("Clear this PDF's conversation"):
                st.session_state[chat_key] = []
                st.rerun()
    else:
        st.info("Upload a PDF above to start asking questions about it.")


# ---- Mode 3: Compare a datasheet ----

else:
    st.title("Compare a datasheet")
    st.caption("Upload a component datasheet to find and compare similar/alternative components in the knowledge base.")

    uploaded = st.file_uploader("Upload a datasheet (PDF)", type=["pdf"])
    user_instruction = st.text_input(
        "Optional instruction (e.g. 'focus on power consumption differences')", value=""
    )
    direct_mode = st.checkbox(
        "Direct comparison mode (pass full candidate datasheets, higher cost)", value=False
    )

    if uploaded and st.button("Run comparison", type="primary"):
        tmp_path = os.path.join("/tmp", uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Summarizing input datasheet and retrieving candidates..."):
            report, retrieved, structured_summary = pipeline.compare_datasheet(
                tmp_path,
                user_instruction=user_instruction,
                direct_comparison_mode=direct_mode,
            )

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Input datasheet summary")
            st.json(structured_summary) if isinstance(structured_summary, dict) else st.markdown(structured_summary)
        with col2:
            st.subheader("Retrieved candidates")
            for d in retrieved:
                st.markdown(f"- **{d.metadata.get('source', 'unknown')}** (page {d.metadata.get('page', '?')})")

        st.subheader("Comparative report")
        st.markdown(report)
        render_sources(retrieved)