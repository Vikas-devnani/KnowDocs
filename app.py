"""
app.py  ─  Streamlit UI for KnowDocs
─────────────────────────────────────────────────────────────────
Run with:  streamlit run app.py
─────────────────────────────────────────────────────────────────
"""

import streamlit as st

# Page config MUST be first Streamlit command
st.set_page_config(
    page_title="KnowDocs",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from loguru import logger
from src.rag_pipeline import ingest_documents, query, get_pipeline_status
from src.config import settings

# ─────────────────────────────────────────────
#  Custom CSS  (clean, dark-accent theme)
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Global ── */
    body { font-family: 'Segoe UI', sans-serif; }

    /* ── Header strip ── */
    .header-box {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 1.4rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .header-box h1 { font-size: 2rem; margin: 0; }
    .header-box p  { opacity: 0.8; margin: 0.3rem 0 0 0; }

    /* ── Source cards ── */
    
    .source-card {
        background: #1f2937;          /* dark background */
        border-left: 4px solid #60a5fa;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
        color: #e5e7eb;
    }

    .source-meta {
        font-weight: 600;
        color: #60a5fa;
        margin-bottom: 0.35rem;
    }

    /* ── Answer box ── */
    .answer-box {
        background: #f8fff8;
        border: 1px solid #2dc653;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
        color: #111111 !important;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* ── Status badge ── */
    .badge-green { color: #2dc653; font-weight: 700; }
    .badge-red   { color: #e63946; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────

st.markdown(
    """
    <div class="header-box">
        <h1>KnowDocs</h1>
        <p>Upload technology documents · Ask questions · Get grounded answers</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
#  Sidebar — Config & Upload
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    # API key input (if not set in .env)
    api_key_input = st.text_input(
        "Groq API Key",
        value=settings.groq_api_key or "",
        type="password",
        help="Get a free key at https://console.groq.com/keys",
    )
    if api_key_input:
        settings.groq_api_key = api_key_input

    model_choice = st.selectbox(
        "LLM Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0,
    )
    settings.groq_model = model_choice

    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, settings.top_k_results)
    settings.top_k_results = top_k

    st.divider()

    # ── Index status ───────────────────────────────────────────
    st.subheader("📊 Index Status")
    status = get_pipeline_status()
    if status["indexed"]:
        st.markdown(
            f"<span class='badge-green'>✅ Index ready</span> "
            f"— {status['vector_count']} vectors",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span class='badge-red'>⚠️ No index yet</span> — upload a PDF",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── PDF Upload ────────────────────────────────────────────
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more technology PDF documents.",
    )

    reset_index = st.checkbox(
        "Reset existing index",
        value=False,
        help="Tick to wipe the current index and rebuild from scratch.",
    )

    if st.button("🚀 Process Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing documents …"):
                file_bytes_list = [f.read() for f in uploaded_files]
                filenames = [f.name for f in uploaded_files]
                result = ingest_documents(
                    sources=file_bytes_list,
                    filenames=filenames,
                    reset_index=reset_index,
                )

            if result["status"] == "success":
                st.success(
                    f"✅ Indexed {result['chunks_created']} chunks "
                    f"from {result['files_processed']} document(s)!"
                )
                st.rerun()  # refresh index status badge
            else:
                st.error(f"Ingestion failed: {result.get('message', 'unknown error')}")


# ─────────────────────────────────────────────
#  Main Panel — Q&A
# ─────────────────────────────────────────────

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("💬 Ask a Question")

    # Suggested questions
    sample_qs = [
        "What is the main purpose of this document?",
        "Summarise the key technical specifications.",
        "What are the system requirements?",
        "Explain the architecture described in the document.",
        "What are the known limitations or issues mentioned?",
    ]
    with st.expander("💡 Sample questions (click to expand)"):
        for q in sample_qs:
            if st.button(q, key=f"sample_{q[:20]}"):
                st.session_state["prefill_question"] = q

    # Query input
    default_q = st.session_state.pop("prefill_question", "")
    user_question = st.text_area(
        "Your question",
        value=default_q,
        height=100,
        placeholder="e.g. What does this document say about network security?",
    )

    ask_btn = st.button("🔍 Get Answer", type="primary", use_container_width=True)

    if ask_btn:
        if not settings.groq_api_key:
            st.error("🔑 Please enter your Groq API key in the sidebar.")
        elif not user_question.strip():
            st.warning("Please type a question.")
        elif not get_pipeline_status()["indexed"]:
            st.warning("No documents indexed yet — please upload a PDF first.")
        else:
            with st.spinner("Searching documents and generating answer …"):
                response = query(
                    question=user_question,
                    groq_api_key=settings.groq_api_key,
                    groq_model=settings.groq_model,
                    top_k=top_k,
                )

            # ── Answer ─────────────────────────────────────────
            if response.get("error") and response["error"] != "None":
                st.error(f"Error: {response['error']}")
            else:
                st.markdown("### 📝 Answer")
                st.markdown(
                    f"<div class='answer-box' style='color:#111111;'>{response['answer']}</div>",
                    unsafe_allow_html=True,
                )

                # ── Source chunks (transparency) ──────────────
                sources = response.get("sources", [])
                if sources:
                    st.markdown("---")
                    st.markdown("### 📚 Source Chunks Used")
                    st.caption(
                        f"Retrieved {len(sources)} most relevant chunks "
                        f"(lower score = more similar)"
                    )

                    for src in sources:
                        score_bar = min(int((1 - src.get('score', 0) / 2) * 100), 100)
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-meta">
                                    [{src['index']}] {src['source']} — Page {src['page']}
                                    &nbsp;|&nbsp; Score: {src.get('score', 'N/A')}
                                </div>
                                {src['content']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # ── Show raw context (collapsible) ────────────
                with st.expander("🔬 View raw injected context (debug)"):
                    st.code(response.get("context", ""), language="text")


with col_right:
    st.subheader("ℹ️ How it Works")
    st.markdown(
        """
        **RAG Pipeline:**

        1. **Upload PDF** — your document is loaded page by page
        2. **Chunking** — text split into ~500-char overlapping segments
        3. **Embeddings** — each chunk → 384-dim vector
           (`all-MiniLM-L6-v2`, runs locally)
        4. **FAISS Index** — vectors stored for fast search
        5. **Query** — your question is embedded and compared
           to all chunks (cosine similarity)
        6. **Top-k retrieval** — best matching chunks are returned
        7. **Groq LLM** — answer generated using *only* those chunks
           (no hallucination)

        ---
        **Anti-hallucination guarantee:**
        The prompt explicitly instructs the LLM to answer
        *only* from retrieved context.  If the answer isn't
        in the document, the model says so.

        ---
        **Models available:**
        - `llama3-8b-8192` — fast, free
        - `llama3-70b-8192` — smarter
        - `mixtral-8x7b-32768` — 32k context
        """
    )


# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────

st.divider()
st.caption(
    "Built with 🐍 Python · 🦜 LangChain · ⚡ Groq API · 🔍 FAISS · "
    "🤗 sentence-transformers · 🌊 Streamlit"
)
