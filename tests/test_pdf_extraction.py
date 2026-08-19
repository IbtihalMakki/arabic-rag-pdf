import re
import unicodedata
import unittest

import pymupdf

from src.arabic_rag.loader import PDFLoader


PDF_PATH = "data/pdfs/sample.pdf"


def legacy_clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufffd", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def legacy_reverse_arabic_lines(text: str) -> str:
    lines = text.splitlines()
    fixed_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            fixed_lines.append("")
            continue

        if len(re.findall(r"[\u0600-\u06FF]", line)) >= 3:
            line = line[::-1]

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def legacy_extract(pdf_path: str) -> str:
    document = pymupdf.open(pdf_path)
    pages = []

    try:
        for page in document:
            page_text = page.get_text("text", sort=True)

            if not page_text:
                continue

            page_text = legacy_reverse_arabic_lines(page_text)
            page_text = legacy_clean_text(page_text)
            pages.append(page_text)
    finally:
        document.close()

    return "\n\n".join(pages)


def suspicious_stats(text: str) -> tuple[int, int, float, int]:
    nonspace_chars = [ch for ch in text if not ch.isspace()]

    def is_allowed_char(ch: str) -> bool:
        cp = ord(ch)

        if 0x0020 <= cp <= 0x007E:
            return True
        if 0x0600 <= cp <= 0x06FF:
            return True
        if 0x0750 <= cp <= 0x077F:
            return True
        if 0x08A0 <= cp <= 0x08FF:
            return True
        if 0xFB50 <= cp <= 0xFDFF:
            return True
        if 0xFE70 <= cp <= 0xFEFF:
            return True
        if 0x200C <= cp <= 0x200F:
            return True
        if cp == 0x2022:
            return True

        return False

    suspicious = [ch for ch in nonspace_chars if not is_allowed_char(ch)]

    canadian = [
        ch
        for ch in nonspace_chars
        if (0x1400 <= ord(ch) <= 0x167F) or (0x18B0 <= ord(ch) <= 0x18FF)
    ]

    ratio = len(suspicious) / max(1, len(nonspace_chars))

    return len(suspicious), len(nonspace_chars), ratio, len(canadian)


class TestArabicPdfExtraction(unittest.TestCase):
    def test_loader_improves_arabic_quality(self) -> None:
        legacy_text = legacy_extract(PDF_PATH)
        fixed_text = PDFLoader(PDF_PATH).load()

        legacy_susp, legacy_nonspace, legacy_ratio, legacy_canadian = suspicious_stats(legacy_text)
        fixed_susp, fixed_nonspace, fixed_ratio, fixed_canadian = suspicious_stats(fixed_text)

        print("\n=== Baseline (legacy loader simulation) ===")
        print(f"nonspace={legacy_nonspace} suspicious={legacy_susp} ratio={legacy_ratio:.4f} canadian={legacy_canadian}")
        print(legacy_text[:500])

        print("\n=== New loader ===")
        print(f"nonspace={fixed_nonspace} suspicious={fixed_susp} ratio={fixed_ratio:.4f} canadian={fixed_canadian}")
        print(fixed_text[:500])

        # Must substantially reduce corrupted-script artifacts.
        self.assertLess(fixed_ratio, legacy_ratio * 0.5)

        # The new extraction should avoid Canadian-syllabics artifacts.
        self.assertEqual(fixed_canadian, 0)

        # Check for core semantic identity in Arabic.
        self.assertRegex(fixed_text, r"غازي")
        self.assertRegex(fixed_text, r"القصي")

        # Ensure mixed content survives (e.g., date appears in report).
        self.assertIn("1940", fixed_text)

        # Ensure we do not re-introduce global reversal side effects.
        self.assertNotIn("0491", fixed_text)
        self.assertNotIn("اغ يز ا يصقلبي", fixed_text)

        # Mixed Arabic/English/URL content should not be globally reversed.
        self.assertIn("https://ar.wikipedia.org/wiki/", fixed_text)
        self.assertNotIn("sptth", fixed_text)


if __name__ == "__main__":
    unittest.main()
