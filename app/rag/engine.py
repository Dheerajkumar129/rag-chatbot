import hashlib
import math
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from app.config import settings
from app.rag.document_processor import DocumentProcessor, DocumentChunk
from app.rag.vector_store import VectorStore, SearchResult


class RAGEngine:
    """End-to-end Retrieval-Augmented Generation engine coordinating ingestion, retrieval, and synthesis."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.settings = settings
        self.processor = DocumentProcessor(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP
        )
        self.vector_store = vector_store or VectorStore(
            persist_directory=self.settings.CHROMA_PERSIST_DIRECTORY,
            collection_name=self.settings.CHROMA_COLLECTION_NAME
        )
        self._openai_client: Optional[OpenAI] = None

    @property
    def client(self) -> Optional[Any]:
        """Lazy initialization of OpenAI client."""
        if OpenAI is None:
            return None
        if self._openai_client is None and self.settings.OPENAI_API_KEY:
            self._openai_client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
        return self._openai_client

    def _generate_mock_embedding(self, text: str, dim: int = 1536) -> List[float]:
        """Generate deterministic, pseudo-random normalized embedding for offline/testing mode."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Seed values from digest
        raw = []
        for i in range(dim):
            byte_val = h[i % len(h)]
            raw.append(math.sin(byte_val + i))
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw] if norm > 0 else raw

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings using OpenAI or mock fallback."""
        if not texts:
            return []

        if self.client:
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.settings.OPENAI_EMBEDDING_MODEL
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                # If network or API quota error, gracefully fallback if in debug/test mode
                if self.settings.DEBUG or not self.settings.OPENAI_API_KEY:
                    return [self._generate_mock_embedding(t) for t in texts]
                raise e
        else:
            # Deterministic mock embeddings for offline testing
            return [self._generate_mock_embedding(t) for t in texts]

    def ingest_text(self, text: str, source_name: str) -> Dict[str, Any]:
        """Process, chunk, embed, and store raw text into the vector database."""
        chunks = self.processor.chunk_text(text, source=source_name)
        if not chunks:
            return {"source": source_name, "chunks_added": 0, "total_chunks": self.vector_store.count()}

        # Generate embeddings in batches of 32
        all_embeddings: List[List[float]] = []
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = [chunk.text for chunk in chunks[i:i + batch_size]]
            all_embeddings.extend(self.create_embeddings(batch))

        added_count = self.vector_store.add(chunks, all_embeddings)
        return {
            "source": source_name,
            "chunks_added": added_count,
            "total_chunks": self.vector_store.count(),
            "backend": self.vector_store.backend_name
        }

    def ingest_file(self, filename: str, content: bytes) -> Dict[str, Any]:
        """Extract text from file and ingest into RAG knowledge base."""
        text = self.processor.extract_text_from_bytes(filename, content)
        return self.ingest_text(text=text, source_name=filename)

    def retrieve(self, query: str, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[SearchResult]:
        """Embed user query and retrieve top matching context chunks."""
        k = top_k or self.settings.TOP_K_RESULTS
        min_score = threshold if threshold is not None else self.settings.SIMILARITY_THRESHOLD

        if self.vector_store.count() == 0:
            return []

        query_emb = self.create_embeddings([query])[0]
        results = self.vector_store.search(query_emb, top_k=k)

        # Filter by similarity threshold
        filtered = [res for res in results if res.score >= min_score]
        return filtered or results[:1] if results else []

    def chat(self, message: str, history: Optional[List[Dict[str, str]]] = None, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Synthesize augmented response using retrieved context and conversation history."""
        # 1. Retrieve relevant knowledge chunks
        relevant_chunks = self.retrieve(message, top_k=top_k)

        # 2. Build context prompt
        context_blocks = []
        citations = []
        for i, chunk in enumerate(relevant_chunks, start=1):
            snippet = chunk.text[:180] + "..." if len(chunk.text) > 180 else chunk.text
            context_blocks.append(
                f"[{i}] Source: {chunk.source} (Chunk {chunk.chunk_index}, Relevance: {chunk.score:.2f})\n{chunk.text}"
            )
            citations.append({
                "citation_id": i,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "score": chunk.score,
                "snippet": snippet
            })

        context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant context found in the knowledge base."

        system_prompt = (
            "You are a helpful, accurate, and concise AI knowledge assistant powered by Retrieval-Augmented Generation (RAG).\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the user query using the provided context information whenever relevant.\n"
            "2. Cite your sources directly using format [1], [2], or (Source: filename.ext) when stating facts derived from context.\n"
            "3. If the context does not contain enough information to answer truthfully, state clearly: "
            "'I do not have enough information in my uploaded knowledge base to answer this question.', but you may answer from general knowledge if explicitly helpful, clearly noting that it is outside the provided documents.\n"
            "4. Maintain a polite, professional, and clear tone.\n\n"
            f"=== CONTEXT DOCUMENTS ===\n{context_str}\n=== END CONTEXT ==="
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Include prior conversation turns if provided
        if history:
            for turn in history[-6:]:  # Keep recent context window
                if turn.get("role") in ("user", "assistant") and turn.get("content"):
                    messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content": message})

        # 3. Call OpenAI Chat Completions API
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.OPENAI_CHAT_MODEL,
                    messages=messages,
                    temperature=0.3
                )
                answer = response.choices[0].message.content or ""
                return {
                    "answer": answer,
                    "citations": citations,
                    "context_chunks_count": len(relevant_chunks),
                    "model": self.settings.OPENAI_CHAT_MODEL
                }
            except Exception as e:
                return {
                    "answer": f"Error communicating with OpenAI API: {str(e)}",
                    "citations": citations,
                    "context_chunks_count": len(relevant_chunks),
                    "model": self.settings.OPENAI_CHAT_MODEL,
                    "error": str(e)
                }
        else:
            # Standalone demo answer when no API key is configured
            return {
                "answer": (
                    f"**[Demo Mode - OpenAI API Key Not Set]**\n\n"
                    f"Successfully retrieved **{len(relevant_chunks)} relevant chunk(s)** from your knowledge base:\n\n"
                    + "\n\n".join([f"- **[{c['citation_id']}] {c['source']}** (Score: {c['score']:.2f}):\n> _{c['snippet']}_" for c in citations])
                    + "\n\nTo enable full generative answers, provide your `OPENAI_API_KEY` in the `.env` file or environment variables."
                ),
                "citations": citations,
                "context_chunks_count": len(relevant_chunks),
                "model": "offline-demo-mode"
            }
