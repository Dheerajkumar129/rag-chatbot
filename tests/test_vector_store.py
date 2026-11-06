import pytest
from app.rag.document_processor import DocumentChunk
from app.rag.vector_store import VectorStore, cosine_similarity


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v2), 0.001) == 1.0

    v3 = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v3), 0.001) == 0.0


def test_vector_store_operations(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path / "chroma_test"), collection_name="test_col")
    store.clear()
    assert store.count() == 0

    chunks = [
        DocumentChunk(
            id="c1",
            text="Apples and bananas are nutritious fruits.",
            source="fruits.txt",
            chunk_index=0,
            char_count=42
        ),
        DocumentChunk(
            id="c2",
            text="Cars, trucks, and trains are vehicles.",
            source="vehicles.txt",
            chunk_index=0,
            char_count=38
        )
    ]

    # Orthogonal dummy embeddings
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]

    store.add(chunks, embeddings)
    assert store.count() == 2

    # Search for fruits
    search_hits = store.search(query_embedding=[0.9, 0.1, 0.0], top_k=2)
    assert len(search_hits) == 2
    assert search_hits[0].source == "fruits.txt"
    assert search_hits[0].score > 0.8

    # Check sources
    sources = store.get_sources()
    assert len(sources) == 2

    # Delete one source
    deleted = store.delete_by_source("fruits.txt")
    assert deleted == 1
    assert store.count() == 1

    # Clear all
    store.clear()
    assert store.count() == 0
