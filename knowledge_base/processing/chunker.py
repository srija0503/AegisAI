"""
knowledge_base/processing/chunker.py
───────────────────────────────────
Boundary-aware text chunking for security documents, CVE records,
and threat advisories. Preserves vulnerability context across chunks
to prevent fragmented semantic retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from knowledge_base.processing.metadata import DocumentMetadata


@dataclass
class DocumentChunk:
    """Represents an embedded piece of a larger document with attached metadata."""
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    total_chunks: int
    metadata: DocumentMetadata

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "metadata": self.metadata.to_dict(),
        }


class DocumentChunker:
    """
    Intelligent chunker that preserves security context headers (CVE, title, CVSS)
    and splits on paragraph / sentence boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 500,        # target character length per chunk
        chunk_overlap: int = 80,      # overlap character length
        preserve_header: bool = True,  # prepend vulnerability header to sub-chunks
    ) -> None:
        self.chunk_size = max(100, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))
        self.preserve_header = preserve_header

    def chunk_document(self, text: str, metadata: DocumentMetadata) -> list[DocumentChunk]:
        """
        Splits text into chunks while associating updated metadata.
        """
        if not text or not text.strip():
            return []

        clean_text = text.strip()
        doc_id = metadata.doc_id or "doc"

        # If document fits within single chunk, return immediately
        if len(clean_text) <= self.chunk_size:
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_c0",
                doc_id=doc_id,
                text=clean_text,
                chunk_index=0,
                total_chunks=1,
                metadata=metadata,
            )
            return [chunk]

        # Extract header prefix if requested (e.g. "VULNERABILITY ID: CVE-...")
        header_prefix = ""
        body_text = clean_text
        if self.preserve_header:
            header_lines = []
            if metadata.cve_id:
                header_lines.append(f"[{metadata.cve_id}] {metadata.title}")
            elif metadata.title:
                header_lines.append(f"[{metadata.title}]")
            if header_lines:
                header_prefix = " | ".join(header_lines) + "\n"

        effective_chunk_size = max(100, self.chunk_size - len(header_prefix))

        # Sentence / paragraph splitting
        sentences = self._split_into_sentences(body_text)
        raw_chunks: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > effective_chunk_size and current_chunk:
                raw_chunks.append(" ".join(current_chunk).strip())
                # Overlap logic: keep last sentence if feasible
                if self.chunk_overlap > 0 and len(current_chunk[-1]) < self.chunk_overlap:
                    current_chunk = [current_chunk[-1], sent]
                    current_len = len(current_chunk[0]) + sent_len + 1
                else:
                    current_chunk = [sent]
                    current_len = sent_len
            else:
                current_chunk.append(sent)
                current_len += sent_len + 1

        if current_chunk:
            raw_chunks.append(" ".join(current_chunk).strip())

        # If splitting failed to divide properly (e.g. huge unpunctuated block)
        if not raw_chunks:
            raw_chunks = self._fallback_char_split(body_text, effective_chunk_size)

        total_chunks = len(raw_chunks)
        chunks: list[DocumentChunk] = []

        for idx, chunk_str in enumerate(raw_chunks):
            final_text = (header_prefix + chunk_str).strip()
            chunk_obj = DocumentChunk(
                chunk_id=f"{doc_id}_c{idx}",
                doc_id=doc_id,
                text=final_text,
                chunk_index=idx,
                total_chunks=total_chunks,
                metadata=metadata,
            )
            chunks.append(chunk_obj)

        return chunks

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Splits on paragraph breaks and sentence punctuation."""
        paragraphs = text.split("\n\n")
        sentences = []
        for p in paragraphs:
            # Match sentence endings (.!?) followed by space or newline
            splits = re.split(r"(?<=[.!?])\s+", p.strip())
            for s in splits:
                if s.strip():
                    sentences.append(s.strip())
        return sentences

    def _fallback_char_split(self, text: str, size: int) -> list[str]:
        """Sliding character window fallback."""
        step = size - self.chunk_overlap
        if step <= 0:
            step = size
        chunks = []
        for i in range(0, len(text), step):
            c = text[i : i + size].strip()
            if c:
                chunks.append(c)
        return chunks
