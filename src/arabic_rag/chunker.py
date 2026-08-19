import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 240,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def _normalize_block(self, block: str) -> str:
        lines = [line.strip() for line in block.splitlines() if line.strip()]

        if not lines:
            return ""

        if any(line.startswith("•") or line.startswith("-") for line in lines):
            return "\n".join(lines)

        return " ".join(lines)

    def _is_heading(self, text: str) -> bool:
        stripped = text.strip()

        if not stripped:
            return False

        if len(stripped) > 55:
            return False

        if stripped.endswith(":"):
            return True

        if re.search(r"[\.!؟\?،]", stripped):
            return False

        token_count = len(re.findall(r"[\u0600-\u06FFA-Za-z0-9]+", stripped))
        return 1 <= token_count <= 8

    def _split_sentences(self, text: str) -> list[str]:
        bullet_normalized = text.replace("\n•", "\n• ")

        parts = re.split(
            r"(?<=[\.!؟\?])\s+|\n(?=•)|\n{2,}",
            bullet_normalized,
        )

        sentences = [part.strip() for part in parts if part.strip()]

        if not sentences:
            return [text.strip()] if text.strip() else []

        return sentences

    def _build_sections(self, text: str) -> list[dict[str, str]]:
        blocks = [
            self._normalize_block(block)
            for block in re.split(r"\n\s*\n", text)
        ]
        blocks = [block for block in blocks if block]

        sections: list[dict[str, str]] = []
        current_title = "النص"
        current_parts: list[str] = []

        def flush() -> None:
            if not current_parts:
                return

            sections.append(
                {
                    "title": current_title,
                    "text": "\n".join(current_parts).strip(),
                }
            )

        for block in blocks:
            if self._is_heading(block):
                flush()
                current_title = block
                current_parts = []
                continue

            current_parts.append(block)

        flush()

        if not sections and text.strip():
            sections.append(
                {
                    "title": "النص",
                    "text": text.strip(),
                }
            )

        return sections

    def _chunk_section(
        self,
        section_title: str,
        section_text: str,
        source_document: str,
        start_id: int,
    ) -> list[dict[str, Any]]:
        sentences = self._split_sentences(section_text)
        if not sentences:
            return []

        chunks: list[dict[str, Any]] = []
        current: list[str] = []
        chunk_id = start_id

        def emit() -> None:
            nonlocal chunk_id
            if not current:
                return

            chunk_text = " ".join(current).strip()
            if not chunk_text:
                return

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "section": section_title,
                    "page": None,
                    "source_document": source_document,
                    "text": chunk_text,
                }
            )
            chunk_id += 1

        for sentence in sentences:
            tentative = " ".join(current + [sentence]).strip()

            if current and len(tentative) > self.chunk_size:
                emit()

                # Preserve lightweight overlap with the last sentence only.
                overlap_seed = current[-1] if current else ""
                current = [overlap_seed] if overlap_seed else []

            current.append(sentence)

        emit()

        return chunks

    def _is_reference_noise(self, chunk: str) -> bool:
        lowered = chunk.lower()

        has_sources_heading = "المصادر" in chunk
        has_url = "http://" in lowered or "https://" in lowered or "www." in lowered
        url_tokens = re.findall(r"https?://\S+", lowered)

        arabic_tokens = re.findall(r"[\u0600-\u06FF]+", chunk)
        arabic_token_count = len(arabic_tokens)

        # Drop bibliography-style chunks dominated by links and reference items.
        if has_sources_heading and has_url:
            return True

        if len(url_tokens) >= 2 and arabic_token_count < 25:
            return True

        return False

    def _strip_reference_section(self, text: str) -> str:
        marker = re.search(r"(^|\n)\s*المصادر", text)

        if not marker:
            return text

        prefix = text[:marker.start()].strip()

        # Keep main body if it is non-trivial; otherwise leave original text.
        if len(prefix) >= 20:
            return prefix

        return text

    def split_with_metadata(
        self,
        text: str,
        source_document: str = "",
    ) -> list[dict[str, Any]]:
        candidate_text = self._strip_reference_section(text)
        sections = self._build_sections(candidate_text)

        chunk_records: list[dict[str, Any]] = []
        next_chunk_id = 1

        for section in sections:
            section_chunks = self._chunk_section(
                section_title=section["title"],
                section_text=section["text"],
                source_document=source_document,
                start_id=next_chunk_id,
            )

            for item in section_chunks:
                if self._is_reference_noise(item["text"]):
                    continue

                chunk_records.append(item)

            if section_chunks:
                next_chunk_id = section_chunks[-1]["chunk_id"] + 1

        return chunk_records

    def split(self, text: str) -> list[str]:
        chunk_records = self.split_with_metadata(text=text)
        return [item["text"] for item in chunk_records]
    