# ⚡ FastAPI RAG Chatbot

A high-performance, modular Retrieval-Augmented Generation (RAG) chatbot built with **Python**, **FastAPI**, **ChromaDB**, and the **OpenAI API**. Features an interactive dark-mode glassmorphic Web UI, multi-format document ingestion (PDF, Markdown, TXT, CSV, JSON), vector similarity search, and auditable source citations.

---

## 🌟 Key Highlights

- **⚡ Modern Async FastAPI Backend**: High-throughput REST API with automatic OpenAPI / Swagger documentation (`/docs`).
- **🧠 Vector Similarity & Retrieval**: Powered by ChromaDB with persistent local storage and an automatic in-memory cosine fallback for zero-dependency execution.
- **🔍 OpenAI Integration**: Dense embeddings generated using `text-embedding-3-small` (1536 dims) and contextual synthesis with `gpt-4o-mini` (or `gpt-4o`).
- **📚 Multi-Format Ingestion**: Chunk, embed, and index `.pdf`, `.txt`, `.md`, `.csv`, and `.json` documents with configurable sliding-window overlap.
- **🎨 Glassmorphic Chat UI**: Built-in responsive web interface with live markdown formatting, source attribution cards, document dropzone, and presets.
- **🔒 Auditable Provenance**: Every response provides citations displaying matching document names, chunk indexes, and similarity confidence scores.
- **🧪 100% Tested**: Complete unit and integration test suite with `pytest` and `fastapi.testclient`.

---

## 🏗️ Architecture & Workflow

```
[ User Upload / Prompt ]
         │
         ▼
┌──────────────────┐
│ Document Parser  │ ◄── PDF, TXT, MD, CSV, JSON
└────────┬─────────┘
         │ Text Extraction & Sliding-Window Chunking
         ▼
┌──────────────────┐
│ OpenAI Embedding │ ◄── text-embedding-3-small
└────────┬─────────┘
         │ 1536-dim Vectors
         ▼
┌──────────────────┐
│ ChromaDB / Store │ ◄── Persistent HNSW / Cosine Index
└────────┬─────────┘
         │
         │ Top-K Similarity Search
         ▼
┌──────────────────┐
│ Context Assembly │ ◄── Augments system prompt with citations
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ OpenAI Chat API  │ ◄── gpt-4o-mini / gpt-4o
└────────┬─────────┘
         │
         ▼
[ Grounded Answer + Citations ]
```

---

## 📁 Repository Structure

```
rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Pydantic environment & app configuration
│   ├── main.py                   # FastAPI routes, lifecycle & static mounts
│   ├── data/
│   │   ├── sample_knowledge.txt  # Pre-seeded sample knowledge base
│   │   └── chroma_db/            # ChromaDB local persistence directory
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_processor.py # PDF/text extraction & recursive chunker
│   │   ├── vector_store.py       # ChromaDB client & cosine search interface
│   │   └── engine.py             # Orchestrates embeddings, retrieval & LLM
│   └── static/
│       ├── index.html            # Single-page glassmorphism chat web application
│       ├── style.css             # Dark theme design system & animations
│       └── app.js                # Frontend state, API client & marked.js renderer
├── tests/
│   ├── __init__.py
│   ├── test_processor.py         # Chunker & text cleaner unit tests
│   ├── test_vector_store.py      # Vector similarity & CRUD unit tests
│   └── test_api.py               # FastAPI endpoint integration tests
├── .env.example                  # Environment configuration template
├── .gitignore                    # Python, environment & database ignores
├── Dockerfile                    # Containerization configuration
├── docker-compose.yml            # One-click multi-container orchestration
├── requirements.txt              # Production & testing dependencies
└── README.md                     # Comprehensive documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Dheerajkumar129/rag-chatbot.git
cd rag-chatbot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and insert your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=800
CHUNK_OVERLAP=150
TOP_K_RESULTS=4
```

> **Note**: If `OPENAI_API_KEY` is not provided, the application runs in **Offline Demo Mode**, demonstrating vector search, chunking, and source citation workflows using deterministic pseudo-embeddings!

### 3. Run the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser:
- **Web UI**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🐳 Docker Deployment

Run with Docker Compose:

```bash
# Set your key and launch
OPENAI_API_KEY=your_key_here docker compose up --build
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Server status, active vector store, chunk counts, and model info |
| `POST` | `/api/chat` | Submit question with history; returns answer and source citations |
| `POST` | `/api/ingest/file` | Upload `.pdf`, `.txt`, `.md`, `.csv`, `.json` to chunk and embed |
| `POST` | `/api/ingest/text` | Directly index raw text string into the knowledge base |
| `GET` | `/api/documents` | List all indexed document sources and chunk counts |
| `DELETE`| `/api/documents/{source}` | Remove document and its vector embeddings |
| `POST` | `/api/clear` | Purge entire vector index |
| `GET` | `/api/config` | Inspect current chunking and retrieval parameters |

### Example: Chat Query via `curl`

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the primary difference between ChromaDB and FAISS?",
    "history": [],
    "top_k": 4
  }'
```

---

## 🧪 Running Tests

Execute the automated test suite with `pytest`:

```bash
pytest tests/ -v
```

---

## 🛡️ License

MIT License. Free for personal, academic, and commercial use.
