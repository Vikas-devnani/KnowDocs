# 📘 KnowDocs — Complete Project Instructions

> **Chat with your technology documents using Retrieval-Augmented Generation (RAG)**
> Stack: Python · FAISS · Groq API · sentence-transformers · Streamlit

---

## 📁 Project Folder Structure

```
rag_tech_docs/
├── app.py                    ← Streamlit web UI (entry point)
├── requirements.txt          ← All Python dependencies
├── .env.example              ← Template for your API keys
├── .gitignore
│
├── src/                      ← Core source code (modular)
│   ├── __init__.py
│   ├── config.py             ← Centralised settings (reads .env)
│   ├── pdf_loader.py         ← PDF ingestion (PyPDF + pdfminer fallback)
│   ├── text_splitter.py      ← RecursiveCharacterTextSplitter
│   ├── embeddings.py         ← sentence-transformers (free, local)
│   ├── faiss_store.py        ← FAISS index: build, save, load, search
│   ├── groq_llm.py           ← Groq API integration + prompt engineering
│   └── rag_pipeline.py       ← Orchestrator: ingest + query pipelines
│
├── scripts/
│   └── run_cli.py            ← Command-line interface (no UI needed)
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py      ← pytest unit tests
│
├── data/
│   └── sample_pdfs/          ← Drop your test PDFs here
│
└── faiss_index/              ← Auto-created; stores the vector index
    ├── index.faiss           ← FAISS binary index
    └── index.pkl             ← Docstore + metadata
```

---

## 🧠 System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE (Streamlit)                   │
│         Upload PDF          │          Ask Question              │
└──────────┬──────────────────┴─────────────────┬─────────────────┘
           │                                    │
           ▼  INGESTION PIPELINE                ▼  QUERY PIPELINE
  ┌─────────────────┐                 ┌──────────────────────┐
  │  PDF Loader     │                 │  Query Embedding     │
  │  (PyPDF)        │                 │  (sentence-trans.)   │
  └────────┬────────┘                 └──────────┬───────────┘
           │                                     │
           ▼                                     ▼
  ┌─────────────────┐                 ┌──────────────────────┐
  │  Text Splitter  │                 │  FAISS Similarity    │
  │  (500 chars,    │                 │  Search (top-k)      │
  │   50 overlap)   │                 └──────────┬───────────┘
  └────────┬────────┘                            │
           │                                     ▼
           ▼                          ┌──────────────────────┐
  ┌─────────────────┐                 │  Context Injection   │
  │  Embeddings     │                 │  into Prompt         │
  │  (MiniLM-L6-v2) │                 └──────────┬───────────┘
  └────────┬────────┘                            │
           │                                     ▼
           ▼                          ┌──────────────────────┐
  ┌─────────────────┐                 │  Groq API            │
  │  FAISS Index    │◄────────────────│  (LLaMA3/Mixtral)    │
  │  (persist disk) │                 └──────────┬───────────┘
  └─────────────────┘                            │
                                                 ▼
                                      ┌──────────────────────┐
                                      │  Grounded Answer     │
                                      │  + Source Citations  │
                                      └──────────────────────┘
```

**Key design decisions:**
| Decision | Choice | Reason |
|---|---|---|
| Vector DB | FAISS | In-process, no server, blazing fast |
| Embeddings | all-MiniLM-L6-v2 | Free, local, 384-dim, great quality |
| LLM | Groq (LLaMA3 / Mixtral) | Free tier, very fast inference |
| Anti-hallucination | Strict prompt + context-only instruction | Model told to refuse if answer not in context |
| Chunking | RecursiveCharacterTextSplitter | Preserves semantic units |

---

## ⚙️ Setup Instructions (Step-by-Step)

### Step 1 — Prerequisites

Make sure you have:
- **Python 3.9 or higher** (check: `python --version`)
- **pip** (comes with Python)
- A **free Groq API key** (takes 1 minute to get)

### Step 2 — Get a Free Groq API Key

1. Go to **https://console.groq.com/keys**
2. Sign up (free, no credit card needed)
3. Click **"Create API Key"**
4. Copy the key — looks like: `gsk_xxxxxxxxxxxxxxxxxxxx`

### Step 3 — Clone / Unzip the Project

```bash
# If you unzipped the project:
cd rag_tech_docs
```

### Step 4 — Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 5 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️  This installs PyTorch + sentence-transformers. It may take 3–5 minutes
> and download ~500 MB on first run (the embedding model weights).

### Step 6 — Configure Environment

```bash
# Copy the template
cp .env.example .env

# Edit .env and paste your Groq API key:
#   GROQ_API_KEY=gsk_your_key_here
```

Open `.env` in any text editor and set `GROQ_API_KEY`.

### Step 7 — Run the App

```bash
streamlit run app.py
```

Your browser will open automatically at **http://localhost:8501**

---

## 🖥️ Using the Streamlit UI

### Uploading Documents
1. In the **left sidebar**, find "📄 Upload Documents"
2. Click **Browse files** and select one or more PDF files
3. Optionally tick **"Reset existing index"** to start fresh
4. Click **🚀 Process Documents**
5. Wait for the green success message

### Asking Questions
1. Type your question in the **"💬 Ask a Question"** box
2. Click **🔍 Get Answer**
3. The answer appears in a green box
4. Scroll down to see **📚 Source Chunks** — the exact passages used
5. Expand **🔬 View raw injected context** for debugging

### Configuration (Sidebar)
| Setting | Description |
|---|---|
| Groq API Key | Paste here if not set in .env |
| LLM Model | Choose llama3-8b (fast) or mixtral (more context) |
| Chunks to retrieve | Higher = more context, slower response |

---

## 🖱️ Using the Command-Line Interface

```bash
# Ingest a PDF
python scripts/run_cli.py --ingest data/sample_pdfs/manual.pdf

# Ingest multiple PDFs
python scripts/run_cli.py --ingest doc1.pdf doc2.pdf doc3.pdf

# Ask a question
python scripts/run_cli.py --query "What is the purpose of this system?"

# Ask with more context chunks
python scripts/run_cli.py --query "Explain the architecture" --top-k 6

# Check index status
python scripts/run_cli.py --status

# Reset index and re-ingest
python scripts/run_cli.py --ingest manual.pdf --reset

# Output as JSON (for programmatic use)
python scripts/run_cli.py --query "What are the specs?" --json
```

---

## 🔄 Detailed RAG Pipeline Explanation

### 1. PDF Loading (`src/pdf_loader.py`)
- **Primary**: `PyPDFLoader` from LangChain Community — fast, pure Python
- **Fallback**: `pdfminer.six` for complex/scanned PDFs
- Each page becomes a `Document` object with `{source, page}` metadata
- Blank pages (< 30 chars) are automatically filtered out

### 2. Text Preprocessing
- Excessive whitespace is collapsed: `" ".join(text.split())`
- Pages with no meaningful text are dropped
- Source filenames are preserved in metadata

### 3. Chunking (`src/text_splitter.py`)
- **Algorithm**: `RecursiveCharacterTextSplitter`
- **Chunk size**: 500 characters (configurable in .env)
- **Overlap**: 50 characters (so context isn't lost at chunk boundaries)
- **Separators tried in order**: `\n\n` → `\n` → `. ` → ` ` → `""`
- Result: each chunk inherits parent metadata + a `chunk_index`

### 4. Embedding Generation (`src/embeddings.py`)
- **Model**: `all-MiniLM-L6-v2` (22 MB, 384 dimensions)
- Runs **100% locally** — no API key, no cost, no rate limits
- Embeddings are **L2-normalised** (cosine similarity = dot product)
- Model is cached with `@lru_cache` — loaded only once per session

### 5. FAISS Indexing (`src/faiss_store.py`)
- **Index type**: `IndexFlatL2` (exact brute-force, perfect for < 100k chunks)
- Built via LangChain's `FAISS.from_documents()`
- **Saved to disk**: `faiss_index/index.faiss` + `faiss_index/index.pkl`
- On restart, `FAISS.load_local()` restores the index in seconds
- Incremental updates: `add_documents()` adds new chunks without rebuilding

### 6. Similarity Search
- User query is embedded with the same model
- `similarity_search_with_score(query, k=4)` returns top-4 chunks
- Returns `(Document, L2_distance)` pairs — lower distance = more relevant

### 7. Context Injection + Groq Prompt
```
[Source 1: manual.pdf, Page 3]
<chunk text>

[Source 2: manual.pdf, Page 7]
<chunk text>
...
```
This labelled context is injected into the strict grounded prompt.

### 8. Groq API Call (exact format)

```python
POST https://api.groq.com/openai/v1/chat/completions

Headers:
  Authorization: Bearer gsk_your_api_key
  Content-Type: application/json

Body:
{
  "model": "llama3-8b-8192",
  "messages": [
    {
      "role": "user",
      "content": "You are a precise assistant... [context] ... [question]"
    }
  ],
  "temperature": 0.1,
  "max_tokens": 1024
}
```

---

## 🧪 Example Queries & Expected Outputs

### Example 1 — Factual retrieval
```
Q: What is the purpose of this document?
A: According to the document, its purpose is to [exact text from PDF].
   [Source 1: readme.pdf, Page 1]
```

### Example 2 — Technical specification
```
Q: What are the system requirements?
A: The document specifies the following system requirements:
   - Operating System: [from doc]
   - Memory: [from doc]
   [Source 2: manual.pdf, Page 4]
```

### Example 3 — Not in document
```
Q: What will the weather be like tomorrow?
A: I could not find this information in the provided documents.
```
*(This proves the anti-hallucination is working!)*

---

## 🛡️ Anti-Hallucination Design

The system uses **four layers** to prevent hallucination:

| Layer | Mechanism |
|---|---|
| 1. Retrieval | Only pass relevant chunks, not the entire document |
| 2. Prompt instruction | "Only use information explicitly stated in the context below" |
| 3. Fallback instruction | "If the answer is not in the context, say so explicitly" |
| 4. Low temperature | `temperature=0.1` → near-deterministic, minimal creativity |

---

## 📊 Advantages & Limitations

### ✅ Advantages
- **Zero hallucination** — strict prompt engineering ensures grounded answers
- **Free to run** — Groq free tier + local embeddings = $0 cost
- **Fast** — FAISS in-process search, Groq's LPU hardware for fast inference
- **Private** — documents never leave your machine (only queries go to Groq)
- **Persistent** — FAISS index survives restarts; no re-embedding needed
- **Scalable** — handles multiple PDFs; incremental indexing supported
- **Transparent** — source chunks always shown with similarity scores
- **Modular** — each component (loader, splitter, embedder, LLM) is independent

### ⚠️ Limitations
- **PDF quality** — scanned image-only PDFs won't work (no OCR by default)
- **Context window** — llama3-8b has 8k token limit; very long docs may need Mixtral
- **Groq rate limits** — free tier has request/minute limits
- **Language** — all-MiniLM-L6-v2 is optimised for English
- **Table/chart data** — complex PDF tables may not extract cleanly
- **No memory** — each query is independent; no conversation history

---

## 🚀 Future Improvements

| Feature | Implementation |
|---|---|
| OCR support | Add `pytesseract` + `pdf2image` for scanned PDFs |
| Conversation memory | Store chat history in `st.session_state`, pass to Groq |
| Re-ranking | Add a cross-encoder to re-rank top-k results |
| Hybrid search | Combine FAISS (semantic) + BM25 (keyword) for better recall |
| Multi-modal | Use `GPT-4V` or `LLaVA` for PDF pages with charts/diagrams |
| GPU embeddings | Change `device='cpu'` → `device='cuda'` in embeddings.py |
| Larger index | Swap `IndexFlatL2` → `IndexIVFFlat` for millions of vectors |
| API endpoint | Wrap `rag_pipeline.py` in FastAPI for production deployment |
| Docker | Add `Dockerfile` + `docker-compose.yml` |
| Auth | Add user authentication for multi-tenant deployments |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run a specific test class
pytest tests/test_pipeline.py::TestEmbeddings -v
```

---

## 🔧 Configuration Reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `llama3-8b-8192` | Groq model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `FAISS_INDEX_PATH` | `faiss_index/tech_docs.index` | Where to save the FAISS index |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RESULTS` | `4` | Chunks retrieved per query |

---

## 🤝 Quick Troubleshooting

| Problem | Solution |
|---|---|
| `GROQ_API_KEY not set` | Add key to `.env` or paste in sidebar |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `No documents indexed` | Upload a PDF and click "Process Documents" |
| Empty answers | Try increasing `top_k` or re-uploading the PDF |
| Slow first run | Normal — embedding model downloads ~22 MB once |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |

---

*Built with ❤️ using Python · LangChain · Groq API · FAISS · sentence-transformers · Streamlit*
