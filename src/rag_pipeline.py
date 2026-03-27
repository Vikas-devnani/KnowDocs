# -----------------------------
# OFFLINE MODE (prevents HF retry loops)
# -----------------------------
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


# -----------------------------
# IMPORTS
# -----------------------------
import tempfile
import shutil
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from groq import Groq


# -----------------------------
# CONFIG
# -----------------------------
VECTOR_PATH = "vector_store"


@st.cache_resource
def load_embeddings():
    """
    Load embedding model once per Streamlit session.
    Prevents repeated downloads and startup delays.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="models",
        model_kwargs={"device": "cpu"}
    )


embeddings = load_embeddings()

vector_store = None


# -----------------------------
# INGEST DOCUMENTS
# -----------------------------
def ingest_documents(sources=None, filenames=None, reset_index=False, **kwargs):

    global vector_store

    if reset_index and os.path.exists(VECTOR_PATH):
        shutil.rmtree(VECTOR_PATH)

    documents = []

    if not sources:
        return {
            "status": "error",
            "message": "No documents provided",
            "vector_count": 0
        }

    for file in sources:

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        documents.extend(docs)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    # Create FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_PATH)

    return {
        "status": "success",
        "chunks_created": len(chunks),
        "vector_count": len(chunks),
        "files_processed": len(sources)
    }


# -----------------------------
# QUERY FUNCTION
# -----------------------------
def query(question, groq_api_key, groq_model="llama3-8b-8192", top_k=4):

    global vector_store

    # Load FAISS index if not already loaded
    if vector_store is None:

        if os.path.exists(VECTOR_PATH):
            vector_store = FAISS.load_local(
                VECTOR_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            return {
                "answer": "No documents indexed yet.",
                "sources": []
            }

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    client = Groq(api_key=groq_api_key)

    response = client.chat.completions.create(
        model=groq_model,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    sources = []
    for i, doc in enumerate(docs):

        sources.append({
            "index": i + 1,
            "source": doc.metadata.get("source", "document"),
            "page": doc.metadata.get("page", "N/A"),
            "content": doc.page_content[:500]
        })

    return {
        "answer": answer,
        "sources": sources
    }


# -----------------------------
# PIPELINE STATUS
# -----------------------------
def get_pipeline_status():

    if os.path.exists(VECTOR_PATH):

        try:
            store = FAISS.load_local(
                VECTOR_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )

            vector_count = store.index.ntotal

            return {
                "indexed": True,
                "message": "Vector database ready",
                "vector_count": vector_count
            }

        except Exception:

            return {
                "indexed": False,
                "message": "Vector database exists but failed to load",
                "vector_count": 0
            }

    return {
        "indexed": False,
        "message": "No documents indexed yet",
        "vector_count": 0
    }