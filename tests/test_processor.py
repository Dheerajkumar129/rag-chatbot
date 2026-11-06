import pytest
from app.rag.document_processor import DocumentProcessor, DocumentChunk


def test_clean_text():
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    raw = "Hello   world!\r\n\r\nThis has    extra spaces.\n\n\n\nNew paragraph."
    cleaned = processor.clean_text(raw)
    assert "Hello world!" in cleaned
    assert "\r\n" not in cleaned
    assert "\n\n\n" not in cleaned


def test_chunking_short_text():
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    text = "Short single paragraph test string for testing chunking."
    chunks = processor.chunk_text(text, source="test.txt")
    assert len(chunks) == 1
    assert chunks[0].source == "test.txt"
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == text


def test_chunking_multi_paragraph():
    processor = DocumentProcessor(chunk_size=150, chunk_overlap=30)
    text = (
        "Paragraph 1 contains important background details regarding machine learning and artificial intelligence.\n\n"
        "Paragraph 2 discusses vector databases including ChromaDB, FAISS, and Qdrant for storing embeddings.\n\n"
        "Paragraph 3 concludes with best practices for Retrieval-Augmented Generation."
    )
    chunks = processor.chunk_text(text, source="multi.txt")
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.source == "multi.txt"
        assert len(chunk.text) > 0


def test_extract_text_from_bytes():
    processor = DocumentProcessor()
    content = b"Simple plain text file byte content."
    result = processor.extract_text_from_bytes("sample.txt", content)
    assert result == "Simple plain text file byte content."
