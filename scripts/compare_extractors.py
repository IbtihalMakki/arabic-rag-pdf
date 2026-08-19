import re
import sys
import unicodedata
from pathlib import Path

import pymupdf
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arabic_rag.loader import PDFLoader


def is_allowed_char(ch: str) -> bool:
    cp = ord(ch)

    if ch.isspace():
        return True

    # ASCII and common symbols
    if 0x0020 <= cp <= 0x007E:
        return True

    # Arabic blocks
    if 0x0600 <= cp <= 0x06FF:
        return True
    if 0x0750 <= cp <= 0x077F:
        return True
    if 0x08A0 <= cp <= 0x08FF:
        return True

    # Arabic presentation forms
    if 0xFB50 <= cp <= 0xFDFF:
        return True
    if 0xFE70 <= cp <= 0xFEFF:
        return True

    # Zero-width marks used in Arabic text occasionally
    if 0x200C <= cp <= 0x200F:
        return True

    # Arabic punctuation
    if cp in {0x060C, 0x061B, 0x061F, 0x066A, 0x066B, 0x066C, 0x06D4}:
        return True

    return False


def print_metrics(name: str, text: str) -> None:
    nonspace = [c for c in text if not c.isspace()]
    suspicious = [c for c in nonspace if not is_allowed_char(c)]
    canadian = [
        c for c in nonspace
        if (0x1400 <= ord(c) <= 0x167F) or (0x18B0 <= ord(c) <= 0x18FF)
    ]

    print(f"\n=== {name} ===")
    print("length:", len(text))
    print("nonspace:", len(nonspace))
    print("suspicious_count:", len(suspicious))
    print("suspicious_ratio:", round(len(suspicious) / max(1, len(nonspace)), 4))
    print("canadian_count:", len(canadian))
    print("replacement_count:", text.count("\ufffd"))
    print("gaz i hits:", len(re.findall(r"غازي|ﻏﺎزي", text)))
    print("qasibi-ish hits:", len(re.findall(r"القصيبي|اﻟﻘﺼﻴيبي|اﻟﻘﺼﻴيب", text)))

    unique_suspicious = sorted(set(suspicious))
    if unique_suspicious:
        print("suspicious_unique_sample:", "".join(unique_suspicious[:20]))
        for ch in unique_suspicious[:10]:
            print(" ", ch, f"U+{ord(ch):04X}", unicodedata.name(ch, "?"))

    print("sample:")
    print(text[:450])


def main() -> None:
    pdf = "data/pdfs/sample.pdf"

    doc = pymupdf.open(pdf)
    pymupdf_sort_true = "\n\n".join((p.get_text("text", sort=True) or "") for p in doc)
    pymupdf_sort_false = "\n\n".join((p.get_text("text", sort=False) or "") for p in doc)
    doc.close()

    reader = PdfReader(pdf)
    pypdf_text = "\n\n".join((p.extract_text() or "") for p in reader.pages)

    loader_text = PDFLoader(pdf).load()

    print_metrics("PyMuPDF text sort=True", pymupdf_sort_true)
    print_metrics("PyMuPDF text sort=False", pymupdf_sort_false)
    print_metrics("pypdf", pypdf_text)
    print_metrics("current loader output", loader_text)


if __name__ == "__main__":
    main()
