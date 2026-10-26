"""DatasheetIQ - GraphRAG for electronics datasheets (Streamlit frontend).

Run with:
    streamlit run frontend/app.py

Pages:
  - Overview             -> project summary, architecture, tech stack
  - Ask a question       -> RAGPipeline.answer() against the indexed corpus
  - Upload a PDF & ask   -> ad-hoc in-memory GraphRAG index of one uploaded file
  - Compare a datasheet  -> RAGPipeline.compare_datasheet()
"""
import os
import sys
import tempfile

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rag import RAGPipeline  # noqa: E402

# --------------------------------------------------------------------------
# Personalize these for your resume / portfolio
# --------------------------------------------------------------------------
APP_NAME = "DatasheetIQ"
APP_TAGLINE = "Ask questions, find alternatives, and compare electronic components, grounded in the datasheets themselves."
AUTHOR_NAME = "Your Name"
GITHUB_URL = "https://github.com/your-username/your-repo"
LINKEDIN_URL = "https://www.linkedin.com/in/your-profile"

st.set_page_config(
    page_title=f"{APP_NAME} | GraphRAG for Datasheets",
    page_icon="📟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, button, input, textarea {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Clean up default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.5rem; padding-bottom: 4rem; max-width: 1180px; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #0ea5e9 100%);
    border-radius: 20px;
    padding: 2.6rem 2.8rem;
    color: #ffffff;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px rgba(14, 165, 233, 0.18);
}
.hero h1 {
    font-size: 2.6rem; font-weight: 800; margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em; color: #ffffff;
}
.hero p { font-size: 1.1rem; opacity: 0.92; margin: 0 0 1.2rem 0; max-width: 760px; line-height: 1.55; }
.badge {
    display: inline-block; padding: 0.28rem 0.75rem; margin: 0 0.4rem 0.4rem 0;
    border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.28);
    color: #ffffff;
}

/* Page headers */
.page-title { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.1rem; }
.page-sub { opacity: 0.7; margin-bottom: 1.2rem; }

/* Cards */
.card {
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 16px; padding: 1.3rem 1.4rem; height: 100%;
    background: rgba(148, 163, 184, 0.06);
    transition: transform .15s ease, box-shadow .15s ease;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(15, 23, 42, 0.12); }
.card .icon { font-size: 1.7rem; margin-bottom: 0.4rem; }
.card h4 { margin: 0 0 0.35rem 0; font-size: 1.05rem; font-weight: 700; }
.card p { margin: 0; font-size: 0.92rem; opacity: 0.8; line-height: 1.5; }

/* Metrics */
div[data-testid="stMetric"] {
    border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 14px;
    padding: 0.9rem 1.1rem; background: rgba(148, 163, 184, 0.06);
}
div[data-testid="stMetricValue"] { font-weight: 800; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px; font-weight: 600; padding: 0.5rem 1.1rem;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1e3a8a, #0ea5e9); border: none; color: white;
}

/* Chat */
div[data-testid="stChatMessage"] {
    border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 14px;
    padding: 0.8rem 1rem; margin-bottom: 0.7rem;
}

/* Sidebar */
section[data-testid="stSidebar"] { border-right: 1px solid rgba(148, 163, 184, 0.25); }
.brand { font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.1rem; }
.brand-sub { font-size: 0.78rem; opacity: 0.65; margin-bottom: 1rem; }

/* Footer */
.site-footer {
    margin-top: 3rem; padding-top: 1.2rem; text-align: center; font-size: 0.85rem;
    opacity: 0.7; border-top: 1px solid rgba(148, 163, 184, 0.3);
}
.site-footer a { color: inherit; font-weight: 600; text-decoration: none; margin: 0 0.5rem; }
.site-footer a:hover { text-decoration: underline; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Pipeline loading (cached so it's built once per session, not per rerun)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pipeline (embeddings, vector store, graph, reranker)...")
def load_pipeline(top_k, retrieve_k, use_rerank, use_graph):
    return RAGPipeline(
        top_k=top_k,
        retrieve_k=retrieve_k,
        use_rerank=use_rerank,
        use_graph=use_graph,
    )


def save_upload_to_temp(uploaded_file):
    """Write an uploaded file to the OS temp directory (cross-platform)."""
    path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return path


def get_stats(pipeline):
    """Collect index/graph statistics for display. Never raises."""
    stats = {"chunks": None, "nodes": None, "edges": None, "sources": None}
    try:
        stats["chunks"] = pipeline.vectordb._collection.count()
    except Exception:
        pass
    try:
        if pipeline.G is not None:
            stats["nodes"] = pipeline.G.number_of_nodes()
            stats["edges"] = pipeline.G.number_of_edges()
    except Exception:
        pass
    try:
        metas = pipeline.vectordb.get(include=["metadatas"]).get("metadatas", [])
        stats["sources"] = len({m.get("source") for m in metas if m})
    except Exception:
        pass
    return stats


def fmt(value):
    return f"{value:,}" if isinstance(value, int) else "-"


# --------------------------------------------------------------------------
# Sidebar: navigation + settings
# --------------------------------------------------------------------------
st.sidebar.markdown(f'<div class="brand">📟 {APP_NAME}</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="brand-sub">GraphRAG for electronics datasheets</div>', unsafe_allow_html=True)

PAGES = ["🏠  Overview", "💬  Ask a question", "📄  Upload a PDF & ask", "⚖️  Compare a datasheet"]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.divider()
with st.sidebar.expander("⚙️ Retrieval settings", expanded=False):
    top_k = st.slider("top_k (results returned)", 1, 10, 4)
    retrieve_k = st.slider("retrieve_k (candidate pool)", 5, 50, 20)
    use_rerank = st.checkbox("Cross-encoder reranking", value=True)
    use_graph = st.checkbox("Graph traversal retrieval", value=True)

try:
    pipeline = load_pipeline(top_k, retrieve_k, use_rerank, use_graph)
except Exception as e:
    st.error(
        "The retrieval pipeline failed to start. Make sure Ollama is running and "
        f"the vector store has been built with build_db.py.\n\n`{e}`"
    )
    st.stop()

stats = get_stats(pipeline)

with st.sidebar.expander("🔍 System status", expanded=False):
    if stats["chunks"] is None:
        st.warning("Could not read the vector store.")
    elif stats["chunks"] == 0:
        st.error("Vector store is empty. Run build_db.py or check the store path.")
    else:
        st.success(f"{stats['chunks']:,} chunks indexed")

    if use_graph:
        if pipeline.G is None:
            st.error("Graph enabled but no graph_checkpoint.json was found.")
        else:
            st.success(f"Graph: {stats['nodes']:,} nodes, {stats['edges']:,} edges")

    try:
        from vectordb import PERSIST_DIR
        st.caption(f"Store path: `{PERSIST_DIR}`")
    except Exception:
        pass

st.sidebar.markdown(
    f"""
<div style="font-size:0.8rem; opacity:0.7; margin-top:1rem;">
Built by <b>{AUTHOR_NAME}</b><br>
<a href="{GITHUB_URL}" target="_blank">GitHub</a> ·
<a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>
</div>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Shared UI helpers
# --------------------------------------------------------------------------
def page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def feature_card(icon, title, body):
    return f"""
<div class="card">
  <div class="icon">{icon}</div>
  <h4>{title}</h4>
  <p>{body}</p>
</div>
"""


def render_footer():
    st.markdown(
        f"""
<div class="site-footer">
  {APP_NAME} · Built by {AUTHOR_NAME} ·
  <a href="{GITHUB_URL}" target="_blank">GitHub</a>
  <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sources(docs):
    """Shared source/figure renderer used by all pages."""
    if not docs:
        return
    with st.expander(f"📄 Sources ({len(docs)})", expanded=False):
        for d in docs:
            source = d.metadata.get("source", "unknown")
            page_no = d.metadata.get("page", "?")
            st.markdown(f"**{source}** · page {page_no}")
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


# ==========================================================================
# PAGE: Overview
# ==========================================================================
if page == PAGES[0]:
    st.markdown(
        f"""
<div class="hero">
  <h1>{APP_NAME}</h1>
  <p>{APP_TAGLINE}</p>
  <span class="badge">GraphRAG</span>
  <span class="badge">Hybrid Retrieval</span>
  <span class="badge">Cross-Encoder Reranking</span>
  <span class="badge">Multimodal LLM</span>
  <span class="badge">100% Local / Offline</span>
</div>
""",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Datasheets indexed", fmt(stats["sources"]))
    m2.metric("Text chunks", fmt(stats["chunks"]))
    m3.metric("Graph entities", fmt(stats["nodes"]))
    m4.metric("Graph relations", fmt(stats["edges"]))

    st.markdown("### What it does")
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        feature_card(
            "💬", "Grounded Q&A",
            "Ask natural-language questions about any indexed component and get answers "
            "taken strictly from the datasheet text, with page-level sources.",
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        feature_card(
            "📄", "Chat with any PDF",
            "Upload a datasheet and a knowledge graph plus dense index is built in memory "
            "for that file, so you can query it instantly without touching the main corpus.",
        ),
        unsafe_allow_html=True,
    )
    c3.markdown(
        feature_card(
            "⚖️", "Component comparison",
            "Upload a datasheet and the system finds similar parts, then writes a structured "
            "report on matching specs, differences, and drop-in suitability.",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("&nbsp;")
    st.markdown("### How it works")
    st.graphviz_chart(
        """
digraph G {
  rankdir=LR;
  bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#1e3a8a", fontcolor="white",
        color="#0ea5e9", fontname="Helvetica", fontsize=11, margin="0.18,0.1"];
  edge [color="#94a3b8"];

  Q  [label="User query"];
  E  [label="Entity extraction\\n(spaCy + regex)"];
  GT [label="2-hop graph\\ntraversal"];
  DV [label="Dense vector\\nsearch (Chroma)"];
  M  [label="Merge &\\ndedupe"];
  R  [label="Cross-encoder\\nrerank"];
  L  [label="LLM answer\\n(text + figures)", fillcolor="#0ea5e9"];

  Q -> E -> GT -> M;
  Q -> DV -> M;
  M -> R -> L;
}
""",
        use_container_width=True,
    )

    st.markdown("### Tech stack")
    t1, t2, t3, t4 = st.columns(4)
    t1.markdown("**Retrieval**\n\n- ChromaDB\n- NetworkX knowledge graph\n- Cross-encoder reranker")
    t2.markdown("**Models**\n\n- nomic-embed-text v1.5\n- Llama 3.2 (vision)\n- ms-marco MiniLM")
    t3.markdown("**NLP & Parsing**\n\n- spaCy NER\n- PyPDFLoader\n- pdfplumber figure extraction")
    t4.markdown("**Application**\n\n- Python\n- LangChain\n- Streamlit\n- Ollama (local inference)")

    st.info("👈 Pick a tool from the sidebar to try it out.")
    render_footer()


# ==========================================================================
# PAGE: Ask a question
# ==========================================================================
elif page == PAGES[1]:
    page_header("Ask a question", "Query the indexed datasheet corpus in plain English.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    EXAMPLES = [
        "What is the operating voltage of the ADMV1455?",
        "What interfaces does this component support?",
        "Summarize the key features of the ADMV1455.",
    ]

    if not st.session_state.chat_history:
        st.markdown("**Try one of these:**")
        ex_cols = st.columns(len(EXAMPLES))
        for i, ex in enumerate(EXAMPLES):
            if ex_cols[i].button(ex, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_query = ex
                st.rerun()

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant" and turn.get("docs"):
                render_sources(turn["docs"])

    typed = st.chat_input("Ask about a component, e.g. 'What is the operating voltage of the ADMV1455?'")
    query = typed or st.session_state.pop("pending_query", None)

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
        if st.sidebar.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ==========================================================================
# PAGE: Upload a PDF & ask
# ==========================================================================
elif page == PAGES[2]:
    page_header(
        "Upload a PDF and ask questions",
        "Runs the full GraphRAG pipeline on just this file: entity extraction, an in-memory "
        "knowledge graph, and graph-guided plus dense retrieval. Nothing is written to your main index.",
    )

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key="qa_pdf_upload")

    if uploaded:
        # Only re-index if this is a new/different file - entity extraction
        # is the slow part, so we don't redo it on every chat turn.
        file_bytes = uploaded.getvalue()
        file_key = f"{uploaded.name}_{len(file_bytes)}"

        if st.session_state.get("uploaded_pdf_key") != file_key:
            tmp_path = save_upload_to_temp(uploaded)

            from loader import load_pdf
            from chunker import chunk_documents
            from langchain_chroma import Chroma
            from graph_builder import build_knowledge_graph

            with st.status("Indexing your PDF...", expanded=True) as status:
                st.write("Reading and chunking the PDF...")
                docs = load_pdf(tmp_path, extract_images=True)
                chunks = chunk_documents(docs)

                st.write(f"Building dense index over {len(chunks)} chunks...")
                # No persist_directory -> in-memory only. Never written to
                # disk, never collides with your real chroma_db.
                doc_vectordb = Chroma.from_documents(
                    documents=chunks,
                    embedding=pipeline.embedding_model,
                    collection_name=f"upload_{abs(hash(file_key))}",
                )

                doc_graph = None
                if use_graph:
                    st.write("Extracting entities/relations and building the knowledge graph "
                             "(slowest step, scales with document length)...")
                    doc_graph = build_knowledge_graph(chunks, verbose=False)

                status.update(label="PDF indexed", state="complete", expanded=False)

            st.session_state.uploaded_pdf_key = file_key
            st.session_state.uploaded_pdf_vectordb = doc_vectordb
            st.session_state.uploaded_pdf_graph = doc_graph
            st.session_state.uploaded_pdf_chat = []
            st.session_state.uploaded_pdf_chunks = len(chunks)

        doc_vectordb = st.session_state.uploaded_pdf_vectordb
        doc_graph = st.session_state.uploaded_pdf_graph
        chat_key = "uploaded_pdf_chat"

        s1, s2, s3 = st.columns(3)
        s1.metric("Chunks", fmt(st.session_state.get("uploaded_pdf_chunks")))
        s2.metric("Graph entities", fmt(doc_graph.number_of_nodes() if doc_graph is not None else None))
        s3.metric("Graph relations", fmt(doc_graph.number_of_edges() if doc_graph is not None else None))

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
                    from image_utils import collect_image_paths

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
                    image_paths = collect_image_paths(docs)
                    answer = pipeline._invoke(message, image_paths)

                st.markdown(answer)
                st.caption(
                    f"Retrieved via {len(graph_docs)} graph hits + {len(dense_docs)} dense hits "
                    f"→ {len(docs)} after merge/rerank"
                )
                render_sources(docs)

            st.session_state[chat_key].append({"role": "assistant", "content": answer, "docs": docs})

        if st.session_state.get(chat_key):
            st.sidebar.divider()
            if st.sidebar.button("🗑️ Clear this PDF's chat", use_container_width=True):
                st.session_state[chat_key] = []
                st.rerun()
    else:
        st.info("Upload a PDF above to start asking questions about it.")


# ==========================================================================
# PAGE: Compare a datasheet
# ==========================================================================
else:
    page_header(
        "Compare a datasheet",
        "Upload a component datasheet to find similar or alternative parts in the knowledge base "
        "and get a structured comparison report.",
    )

    uploaded = st.file_uploader("Upload a datasheet (PDF)", type=["pdf"])
    user_instruction = st.text_input(
        "Optional instruction",
        value="",
        placeholder="e.g. focus on power consumption differences",
    )
    direct_mode = st.checkbox(
        "Direct comparison mode (pass full candidate datasheets, higher cost)", value=False
    )

    if uploaded and st.button("Run comparison", type="primary"):
        tmp_path = save_upload_to_temp(uploaded)

        with st.spinner("Summarizing input datasheet and retrieving candidates..."):
            report, retrieved, structured_summary = pipeline.compare_datasheet(
                tmp_path,
                user_instruction=user_instruction,
                direct_comparison_mode=direct_mode,
            )

        st.markdown("### Comparative report")
        st.markdown(report)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Input datasheet summary")
            if isinstance(structured_summary, dict):
                st.json(structured_summary)
            else:
                st.markdown(structured_summary)
        with col2:
            st.markdown("#### Retrieved candidates")
            for d in retrieved:
                st.markdown(f"- **{d.metadata.get('source', 'unknown')}** (page {d.metadata.get('page', '?')})")

        render_sources(retrieved)
    elif not uploaded:
        st.info("Upload a datasheet PDF to begin.")