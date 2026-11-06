from app.main import (
    health_check,
    get_config,
    clear_knowledge_base,
    ingest_text_endpoint,
    list_documents,
    chat_endpoint,
    delete_document,
    ChatRequest,
    IngestTextRequest
)


def test_health_endpoint():
    res = health_check()
    assert res.status == "healthy"
    assert res.version == "1.0.0"
    assert res.vector_backend in ("ChromaDB", "In-Memory Cosine Store")


def test_config_endpoint():
    res = get_config()
    assert "chat_model" in res
    assert "embedding_model" in res
    assert res["chunk_size"] == 800


def test_full_rag_lifecycle():
    # 1. Clear database
    clear_res = clear_knowledge_base()
    assert clear_res["remaining_chunks"] == 0

    # 2. Ingest custom text
    payload = IngestTextRequest(
        text="The James Webb Space Telescope (JWST) is an infrared space observatory launched in December 2021. It orbits the Sun-Earth L2 Lagrange point.",
        source_name="jwst_mission.txt"
    )
    ingest_res = ingest_text_endpoint(payload)
    assert ingest_res.chunks_added >= 1
    assert ingest_res.source == "jwst_mission.txt"

    # 3. Check document list
    docs_res = list_documents()
    assert docs_res["total_documents"] >= 1
    sources = [d["source"] for d in docs_res["documents"]]
    assert "jwst_mission.txt" in sources

    # 4. Chat query with context retrieval
    chat_req = ChatRequest(
        message="When was the James Webb Space Telescope launched and what is its orbit?",
        history=[],
        top_k=2
    )
    chat_res = chat_endpoint(chat_req)
    assert chat_res.answer != ""
    assert len(chat_res.citations) >= 1
    assert chat_res.citations[0].source == "jwst_mission.txt"

    # 5. Delete the document
    del_res = delete_document("jwst_mission.txt")
    assert del_res["status"] == "success"

    # 6. Verify deletion
    docs_res2 = list_documents()
    sources2 = [d["source"] for d in docs_res2["documents"]]
    assert "jwst_mission.txt" not in sources2
