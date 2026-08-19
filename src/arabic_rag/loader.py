import re
import unicodedata

import pymupdf
from pypdf import PdfReader


class PDFLoader:

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def _clean_text(self, text: str) -> str:

        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Remove replacement characters
        text = text.replace("\ufffd", "")

        # Normalize spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Normalize excessive line breaks
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        return text.strip()

    def _extract_with_pymupdf(self) -> str:

        document = pymupdf.open(self.pdf_path)
        pages = []

        try:
            for page in document:
                # sort=False preserves content-stream order and avoids severe
                # reshuffling of RTL lines seen with sort=True on this PDF.
                page_text = page.get_text("text", sort=False)

                if page_text:
                    pages.append(page_text)
        finally:
            document.close()

        return "\n\n".join(pages)

    def _extract_with_pypdf(self) -> str:

        reader = PdfReader(self.pdf_path)
        pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ""

            if page_text:
                pages.append(page_text)

        return "\n\n".join(pages)

    def _is_allowed_char(self, char: str) -> bool:

        codepoint = ord(char)

        if char.isspace():
            return True

        # ASCII letters, numbers, punctuation.
        if 0x0020 <= codepoint <= 0x007E:
            return True

        # Arabic core/supplement/extended blocks.
        if 0x0600 <= codepoint <= 0x06FF:
            return True

        if 0x0750 <= codepoint <= 0x077F:
            return True

        if 0x08A0 <= codepoint <= 0x08FF:
            return True

        # Arabic presentation forms are common in PDF text layers.
        if 0xFB50 <= codepoint <= 0xFDFF:
            return True

        if 0xFE70 <= codepoint <= 0xFEFF:
            return True

        # Keep bidi control marks if present.
        if 0x200C <= codepoint <= 0x200F:
            return True

        # Preserve bullet points.
        if codepoint == 0x2022:
            return True

        return False

    def _quality_score(self, text: str) -> tuple[float, int, int]:

        nonspace_chars = [char for char in text if not char.isspace()]

        if not nonspace_chars:
            return (0.0, 0, 0)

        suspicious_count = sum(
            1
            for char in nonspace_chars
            if not self._is_allowed_char(char)
        )

        suspicious_ratio = suspicious_count / len(nonspace_chars)

        # Higher is better: fewer suspicious glyphs, then more content.
        return (-suspicious_ratio, len(nonspace_chars), -suspicious_count)

    def load(self) -> str:

        pypdf_text = self._clean_text(self._extract_with_pypdf())
        pymupdf_text = self._clean_text(self._extract_with_pymupdf())

        pypdf_score = self._quality_score(pypdf_text)
        pymupdf_score = self._quality_score(pymupdf_text)

        if pypdf_score >= pymupdf_score:
            return pypdf_text

        return pymupdf_text