import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.rag.engine import RAGEngine
from app.rag.vector_store import VectorStore

# Initialize global RAG Engine instance
vector_store = VectorStore(
    persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
    collection_name=settings.CHROMA_COLLECTION_NAME
)
rag_engine = RAGEngine(vector_store=vector_store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Seed sample knowledge base if vector store is currently empty."""
    if vector_store.count() == 0:
        sample_path = Path(__file__).resolve().parent / "data" / "sample_knowledge.txt"
        if sample_path.exists():
            try:
                sample_text = sample_path.read_text(encoding="utf-8")
                rag_engine.ingest_text(sample_text, source_name="sample_knowledge.txt")
                print(f"[RAG] Automatically seeded sample knowledge ({vector_store.count()} chunks indexed).")
            except Exception as e:
                print(f"[RAG] Warning: Failed to seed sample knowledge: {e}")
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-ready small RAG Chatbot powered by FastAPI, ChromaDB / Cosine Vector Store, and OpenAI API.",
    lifespan=lifespan
)

# Enable CORS for local testing & web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static web directory
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Check multipart availability for file upload route
try:
    import multipart  # noqa: F401
    HAS_MULTIPART = True
except ImportError:
    HAS_MULTIPART = False

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Pydantic Request/Response Models
class MessageTurn(BaseModel):
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user query or prompt")
    history: Optional[List[MessageTurn]] = Field(default=[], description="Previous conversation turns")
    top_k: Optional[int] = Field(default=None, description="Number of context chunks to retrieve")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "What is the difference between ChromaDB and FAISS?",
                "history": [],
                "top_k": 4
            }
        }
    }


class Citation(BaseModel):
    citation_id: int
    source: str
    chunk_index: int
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    context_chunks_count: int
    model: str
    error: Optional[str] = None


class IngestTextRequest(BaseModel):
    text: str = Field(..., description="Raw text content to index into the RAG vector store")
    source_name: Optional[str] = Field(default="manual_entry.txt", description="Identifier for provenance tracking")


class IngestResponse(BaseModel):
    source: str
    chunks_added: int
    total_chunks: int
    backend: str
    status: str = "success"


class HealthResponse(BaseModel):
    status: str
    version: str
    total_chunks: int
    sources_count: int
    vector_backend: str
    openai_configured: bool
    chat_model: str
    embedding_model: str


# Routes
@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the single-page chat user interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(
        content={"message": "FastAPI RAG Chatbot API is operational. Visit /docs for OpenAPI documentation."}
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Return health status, current index stats, and vector backend information."""
    sources = vector_store.get_sources()
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        total_chunks=vector_store.count(),
        sources_count=len(sources),
        vector_backend=vector_store.backend_name,
        openai_configured=bool(settings.OPENAI_API_KEY.strip()),
        chat_model=settings.OPENAI_CHAT_MODEL,
        embedding_model=settings.OPENAI_EMBEDDING_MODEL
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["RAG Chat"])
def chat_endpoint(request: ChatRequest):
    """Ask a question to the RAG chatbot; retrieves relevant context and synthesizes an answer."""
    if not request.message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty.")

    history_dicts = [{"role": h.role, "content": h.content} for h in request.history] if request.history else []
    result = rag_engine.chat(
        message=request.message,
        history=history_dicts,
        top_k=request.top_k
    )
    return ChatResponse(**result)


if HAS_MULTIPART:
    @app.post("/api/ingest/file", response_model=IngestResponse, tags=["Knowledge Ingestion"])
    async def ingest_file_endpoint(file: UploadFile = File(...)):
        """Upload a file (.txt, .md, .pdf, .csv, .json) to chunk, embed, and index into the vector store."""
        try:
            content = await file.read()
            filename = file.filename or "uploaded_document.txt"
            result = rag_engine.ingest_file(filename=filename, content=content)
            return IngestResponse(**result)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to ingest file: {str(e)}"
            )
else:
    @app.post("/api/ingest/file", tags=["Knowledge Ingestion"])
    async def ingest_file_endpoint_fallback():
        """Fallback when python-multipart is not installed."""
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="File upload requires python-multipart. Install with: pip install python-multipart"
        )


@app.post("/api/ingest/text", response_model=IngestResponse, tags=["Knowledge Ingestion"])
def ingest_text_endpoint(payload: IngestTextRequest):
    """Directly submit raw text or documentation notes into the vector store."""
    if not payload.text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text cannot be empty.")

    try:
        result = rag_engine.ingest_text(text=payload.text, source_name=payload.source_name)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest text: {str(e)}"
        )


@app.get("/api/documents", tags=["Knowledge Ingestion"])
def list_documents():
    """List all indexed document sources along with chunk counts."""
    sources = vector_store.get_sources()
    return {
        "total_documents": len(sources),
        "total_chunks": vector_store.count(),
        "documents": sources
    }


@app.delete("/api/documents/{source_name}", tags=["Knowledge Ingestion"])
def delete_document(source_name: str):
    """Delete a document and all its associated vector embeddings from the store."""
    deleted_count = vector_store.delete_by_source(source_name)
    if deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No chunks found for document '{source_name}'."
        )
    return {
        "status": "success",
        "message": f"Successfully deleted {deleted_count} chunks for source '{source_name}'.",
        "deleted_chunks": deleted_count,
        "remaining_chunks": vector_store.count()
    }


@app.post("/api/clear", tags=["Knowledge Ingestion"])
def clear_knowledge_base():
    """Purge all vector embeddings and documents from the collection."""
    vector_store.clear()
    return {
        "status": "success",
        "message": "Knowledge base has been completely purged.",
        "remaining_chunks": 0
    }


@app.get("/api/config", tags=["Configuration"])
def get_config():
    """Inspect current RAG parameters (safe public view)."""
    return {
        "chat_model": settings.OPENAI_CHAT_MODEL,
        "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "top_k_results": settings.TOP_K_RESULTS,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
        "vector_backend": vector_store.backend_name,
        "openai_configured": bool(settings.OPENAI_API_KEY.strip())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
