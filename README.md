# KnowDocs – AI Powered Document Question Answering System

KnowDocs is a **Retrieval Augmented Generation (RAG)** based system that enables users to upload PDF documents and ask questions about their contents. The system processes documents, creates semantic embeddings, stores them in a vector database, and retrieves relevant information to generate accurate answers using a Large Language Model.

The project demonstrates how modern Generative AI systems combine **vector databases, embeddings, and LLMs** to build intelligent document assistants.

---

# Project Overview

Traditional search systems rely on keyword matching, which often fails to capture the semantic meaning of user queries. KnowDocs solves this limitation by implementing a **semantic retrieval pipeline**.

The system converts document content into vector embeddings and stores them in a vector database. When a user asks a question, the system retrieves the most relevant document chunks and provides them as context to a Large Language Model, which generates an accurate answer grounded in the document.

---

# Key Features

- **PDF Document Upload**  
  Users can upload one or multiple PDF documents.

- **Document Parsing**  
  PDF files are parsed and converted into structured text.

- **Text Chunking**  
  Documents are split into smaller text segments to improve retrieval accuracy.

- **Semantic Embeddings**  
  Text chunks are converted into vector embeddings using a transformer-based model.

- **Vector Database Storage**  
  Embeddings are stored in a FAISS vector database for efficient similarity search.

- **Semantic Retrieval**  
  User queries are embedded and compared with stored vectors to retrieve relevant content.

- **LLM Answer Generation**  
  Retrieved context is passed to a Large Language Model to generate answers.

- **Source Attribution**  
  The system displays document source references and page numbers.

---

# System Architecture

The system follows a **Retrieval Augmented Generation (RAG)** architecture.

```
User Query
     │
     ▼
Streamlit Interface
     │
     ▼
PDF Upload
     │
     ▼
Document Parsing
     │
     ▼
Text Chunking
     │
     ▼
Embedding Model
(Sentence Transformers)
     │
     ▼
Vector Database (FAISS)
     │
     ▼
Retriever
     │
     ▼
Groq Large Language Model
     │
     ▼
Generated Answer + Sources
```

---

# Technology Stack

| Component | Technology |
|----------|------------|
| Programming Language | Python |
| Frontend | Streamlit |
| Embedding Model | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database | FAISS |
| LLM Provider | Groq |
| Document Processing | LangChain + PyPDF |
| Logging | Loguru |
| Environment Management | Python Virtual Environment |

---

# Project Structure

```
KnowDocs
│
├── app.py
├── requirements.txt
├── .env
├── README.md
│
└── src
    │
    ├── __init__.py
    ├── config.py
    └── rag_pipeline.py
```

### app.py
Contains the Streamlit interface and handles user interactions such as document uploads and question input.

### config.py
Manages environment variables and application configuration.

### rag_pipeline.py
Implements the core RAG pipeline including:

- document ingestion
- text chunking
- embedding generation
- vector storage
- retrieval
- LLM interaction

---

# Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/knowdocs.git
cd knowdocs
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux / Mac

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Configuration

Create a `.env` file in the root directory.

Example configuration:

```
GROQ_API_KEY=your_groq_api_key

GROQ_MODEL=llama3-8b-8192

EMBEDDING_MODEL=all-MiniLM-L6-v2

FAISS_INDEX_PATH=faiss_index/tech_docs.index

CHUNKS_STORE_PATH=faiss_index/chunks.pkl

CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=4
```

---

# Running the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

The application will run at:

```
http://localhost:8501
```

---

# How the System Works

### Step 1 – Document Upload
Users upload PDF documents through the interface.

### Step 2 – Text Extraction
Text is extracted from the uploaded PDF documents.

### Step 3 – Text Chunking
Documents are divided into smaller segments to improve retrieval efficiency.

### Step 4 – Embedding Generation
Each chunk is converted into a vector embedding using a transformer model.

### Step 5 – Vector Storage
Embeddings are stored in a FAISS vector database.

### Step 6 – Query Processing
User queries are converted into embeddings.

### Step 7 – Semantic Retrieval
The system retrieves the most relevant document chunks from the vector database.

### Step 8 – Answer Generation
The retrieved context is sent to the Groq Large Language Model to generate an answer.

### Step 9 – Source Display
The system displays the document source and page number associated with the answer.

---

# Example Workflow

1. Upload a document containing technical information.  
2. Ask a question related to the document.  
3. The system retrieves relevant sections from the document.  
4. The LLM generates an answer based on the retrieved context.

---

# Future Improvements

Possible enhancements include:

- Highlighting answer sections directly within PDFs
- Support for additional document formats such as DOCX and TXT
- Hybrid search combining keyword and semantic retrieval
- Conversational memory for multi-turn interactions
- Multi-document summarization

---

# Learning Outcomes

This project demonstrates practical implementation of:

- Retrieval Augmented Generation systems
- Vector similarity search
- Embedding-based semantic retrieval
- Large Language Model integration
- Interactive AI applications using Streamlit

---

