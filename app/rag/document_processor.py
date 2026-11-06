import re
import uuid
from io import BytesIO
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents an ingested text chunk with its provenance metadata."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source: str
    chunk_index: int
    char_count: int
    metadata: dict = Field(default_factory=dict)


class DocumentProcessor:
    """Extracts, cleans, and chunks text from diverse file types."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_bytes(self, filename: str, content: bytes) -> str:
        """Extract plain text from binary file contents based on extension."""
        ext = filename.lower().split(".")[-1] if "." in filename else ""

        if ext == "pdf":
            return self._extract_from_pdf(content)
        else:
            # Fallback to UTF-8 decoding with Latin-1 fallback
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                return content.decode("latin-1", errors="ignore")

    def _extract_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF bytes using pypdf if available."""
        try:
            import pypdf
            reader = pypdf.PdfReader(BytesIO(content))
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(text.strip())
            return "\n\n".join(pages_text)
        except ImportError:
            # Basic fallback if pypdf is not installed
            text = content.decode("utf-8", errors="ignore")
            # Filter ascii printable
            return "".join(c for c in text if c.isprintable() or c in "\n\r\t")

    def clean_text(self, text: str) -> str:
        """Normalize whitespaces and remove excessive empty lines."""
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def chunk_text(self, text: str, source: str) -> List[DocumentChunk]:
        """Split text into overlapping chunks respecting paragraph/sentence boundaries."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        chunks: List[DocumentChunk] = []

        # Split into paragraph-level segments first
        paragraphs = cleaned.split("\n\n")
        current_chunk = ""
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk_size and we already have content
            if current_chunk and len(current_chunk) + len(para) + 2 > self.chunk_size:
                chunks.append(
                    DocumentChunk(
                        text=current_chunk.strip(),
                        source=source,
                        chunk_index=chunk_idx,
                        char_count=len(current_chunk.strip()),
                        metadata={"source": source, "index": chunk_idx}
                    )
                )
                chunk_idx += 1

                # Carry over overlap from the end of current_chunk
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    # Break at space if possible
                    space_pos = overlap_text.find(" ")
                    if space_pos != -1:
                        overlap_text = overlap_text[space_pos:].strip()
                    current_chunk = overlap_text + "\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

            # If a single paragraph is larger than chunk_size, split by sentences or hard length
            while len(current_chunk) > self.chunk_size:
                split_point = self.chunk_size
                # Try finding a period or newline near split_point
                boundary = current_chunk.rfind(". ", 0, split_point)
                if boundary == -1 or boundary < split_point // 2:
                    boundary = current_chunk.rfind(" ", 0, split_point)
                if boundary == -1 or boundary < split_point // 2:
                    boundary = split_point

                chunk_str = current_chunk[:boundary].strip()
                if chunk_str:
                    chunks.append(
                        DocumentChunk(
                            text=chunk_str,
                            source=source,
                            chunk_index=chunk_idx,
                            char_count=len(chunk_str),
                            metadata={"source": source, "index": chunk_idx}
                        )
                    )
                    chunk_idx += 1

                # Next window starts with overlap
                start_next = max(0, boundary - self.chunk_overlap)
                current_chunk = current_chunk[start_next:].strip()

        # Add remaining tail
        if current_chunk.strip():
            chunks.append(
                DocumentChunk(
                    text=current_chunk.strip(),
                    source=source,
                    chunk_index=chunk_idx,
                    char_count=len(current_chunk.strip()),
                    metadata={"source": source, "index": chunk_idx}
                )
            )

        return chunks
