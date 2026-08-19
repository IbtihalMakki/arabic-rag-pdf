import unicodedata

import pymupdf
from pypdf import PdfReader


def show_codepoints(label: str, text: str, limit: int = 120) -> None:
    print(f"\n--- {label} codepoints (first {limit}) ---")
    for ch in text[:limit]:
        cp = ord(ch)
        print(f"{ch} U+{cp:04X} {unicodedata.name(ch, '?')}")


def main() -> None:
    pdf = "data/pdfs/sample.pdf"

    doc = pymupdf.open(pdf)
    page = doc[0]

    text = page.get_text("text", sort=True) or ""
    words = page.get_text("words", sort=True) or []
    dct = page.get_text("dict", sort=True) or {}
    raw = page.get_text("rawdict", sort=True) or {}

    print("--- PyMuPDF text page1 (first 900 chars) ---")
    print(text[:900])

    print("\n--- PyMuPDF words page1 first 30 tokens ---")
    for item in words[:30]:
        print(repr(item[4]))

    print("\n--- PyMuPDF dict spans page1 first 25 spans ---")
    spans = []
    for block in dct.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                if span_text.strip():
                    spans.append(
                        (
                            span_text,
                            span.get("font"),
                            span.get("flags"),
                            span.get("size"),
                        )
                    )

    for span_text, font, flags, size in spans[:25]:
        print(repr(span_text), "| font=", font, "| flags=", flags, "| size=", size)

    print("\n--- PyMuPDF rawdict chars page1 first 120 chars ---")
    chars = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    c = ch.get("c", "")
                    if c:
                        chars.append(c)

    for c in chars[:120]:
        cp = ord(c)
        print(f"{c} U+{cp:04X} {unicodedata.name(c, '?')}")

    weird = [c for c in chars if 0x1980 <= ord(c) <= 0x19FF]
    print("\n--- Weird chars in U+1980..U+19FF ---")
    print("count:", len(weird))
    print("unique:", sorted(set(weird)))

    show_codepoints("PyMuPDF text page1", text)

    # page 2 quick probe
    if len(doc) > 1:
        page2_text = doc[1].get_text("text", sort=True) or ""
        print("\n--- PyMuPDF text page2 (first 900 chars) ---")
        print(page2_text[:900])
        show_codepoints("PyMuPDF text page2", page2_text)

    doc.close()

    reader = PdfReader(pdf)
    pypdf_page1 = reader.pages[0].extract_text() or ""

    print("\n--- pypdf text page1 (first 900 chars) ---")
    print(pypdf_page1[:900])
    show_codepoints("pypdf text page1", pypdf_page1)

    if len(reader.pages) > 1:
        pypdf_page2 = reader.pages[1].extract_text() or ""
        print("\n--- pypdf text page2 (first 900 chars) ---")
        print(pypdf_page2[:900])
        show_codepoints("pypdf text page2", pypdf_page2)


if __name__ == "__main__":
    main()
