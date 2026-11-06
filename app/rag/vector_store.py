import math
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.rag.document_processor import DocumentChunk


class SearchResult(BaseModel):
    """Result returned from vector similarity search."""
    id: str
    text: str
    source: str
    chunk_index: int
    score: float = Field(description="Cosine similarity score (0.0 to 1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict)


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two numeric vectors."""
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """Vector database interface supporting ChromaDB with an in-memory cosine fallback."""

    def __init__(self, persist_directory: str = "./data/chroma_db", collection_name: str = "rag_documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chroma_client = None
        self.collection = None
        self._use_chroma = False

        # In-memory fallback storage
        self._memory_chunks: Dict[str, DocumentChunk] = {}
        self._memory_embeddings: Dict[str, List[float]] = {}

        self._initialize_backend()

    def _initialize_backend(self):
        """Attempt to initialize ChromaDB; fallback to in-memory store if unavailable."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            # cosine distance collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self._use_chroma = True
        except Exception:
            # Fallback to internal cosine similarity vector store
            self._use_chroma = False

    def add(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> int:
        """Add chunks and corresponding embeddings into the vector store."""
        if not chunks:
            return 0

        ids = [chunk.id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "char_count": chunk.char_count,
                **chunk.metadata
            }
            for chunk in chunks
        ]

        if self._use_chroma and self.collection is not None:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            for chunk, emb in zip(chunks, embeddings):
                self._memory_chunks[chunk.id] = chunk
                self._memory_embeddings[chunk.id] = emb

        return len(chunks)

    def search(self, query_embedding: List[float], top_k: int = 4) -> List[SearchResult]:
        """Perform top-k cosine similarity search."""
        if self._use_chroma and self.collection is not None:
            total_items = self.collection.count()
            if total_items == 0:
                return []
            
            actual_k = min(top_k, total_items)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_k,
                include=["documents", "metadatas", "distances"]
            )

            hits: List[SearchResult] = []
            if results and results.get("ids") and results["ids"][0]:
                res_ids = results["ids"][0]
                res_docs = results["documents"][0] if results.get("documents") else []
                res_metas = results["metadatas"][0] if results.get("metadatas") else []
                res_dists = results["distances"][0] if results.get("distances") else []

                for i in range(len(res_ids)):
                    dist = res_dists[i] if i < len(res_dists) else 1.0
                    # For cosine distance: distance = 1 - similarity => similarity = 1 - distance
                    similarity = max(0.0, min(1.0, 1.0 - dist))
                    meta = res_metas[i] if i < len(res_metas) else {}
                    hits.append(
                        SearchResult(
                            id=res_ids[i],
                            text=res_docs[i] if i < len(res_docs) else "",
                            source=meta.get("source", "unknown"),
                            chunk_index=int(meta.get("chunk_index", 0)),
                            score=round(similarity, 4),
                            metadata=meta
                        )
                    )
            return hits
        else:
            # Memory fallback cosine search
            if not self._memory_embeddings:
                return []

            scored: List[tuple[float, str]] = []
            for cid, emb in self._memory_embeddings.items():
                sim = cosine_similarity(query_embedding, emb)
                scored.append((sim, cid))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_matches = scored[:top_k]

            hits = []
            for sim, cid in top_matches:
                chunk = self._memory_chunks[cid]
                hits.append(
                    SearchResult(
                        id=chunk.id,
                        text=chunk.text,
                        source=chunk.source,
                        chunk_index=chunk.chunk_index,
                        score=round(sim, 4),
                        metadata=chunk.metadata
                    )
                )
            return hits

    def get_sources(self) -> List[Dict[str, Any]]:
        """List distinct sources and their chunk counts."""
        source_counts: Dict[str, int] = {}

        if self._use_chroma and self.collection is not None:
            data = self.collection.get(include=["metadatas"])
            for meta in (data.get("metadatas") or []):
                src = meta.get("source", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1
        else:
            for chunk in self._memory_chunks.values():
                source_counts[chunk.source] = source_counts.get(chunk.source, 0) + 1

        return [{"source": k, "chunks_count": v} for k, v in sorted(source_counts.items())]

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks originating from a specific source."""
        deleted = 0
        if self._use_chroma and self.collection is not None:
            existing = self.collection.get(where={"source": source})
            ids_to_del = existing.get("ids", [])
            if ids_to_del:
                self.collection.delete(ids=ids_to_del)
                deleted = len(ids_to_del)
        else:
            to_remove = [cid for cid, chunk in self._memory_chunks.items() if chunk.source == source]
            for cid in to_remove:
                self._memory_chunks.pop(cid, None)
                self._memory_embeddings.pop(cid, None)
            deleted = len(to_remove)

        return deleted

    def clear(self) -> None:
        """Clear all indexed documents from the collection."""
        if self._use_chroma and self.collection is not None:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        self._memory_chunks.clear()
        self._memory_embeddings.clear()

    def count(self) -> int:
        """Return total number of chunks currently indexed."""
        if self._use_chroma and self.collection is not None:
            return self.collection.count()
        return len(self._memory_chunks)

    @property
    def backend_name(self) -> str:
        """Return which backend is currently actively storing vectors."""
        return "ChromaDB" if self._use_chroma else "In-Memory Cosine Store"
