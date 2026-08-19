from src.arabic_rag.loader import PDFLoader
from src.arabic_rag.text_processor import ArabicTextProcessor
from src.arabic_rag.chunker import TextChunker
raw = PDFLoader("data/pdfs/sample.pdf").load()
repaired = ArabicTextProcessor().process(raw)
chunks = TextChunker().split_with_metadata(repaired, "data/pdfs/sample.pdf")
for c in chunks:
    cid = c["chunk_id"]
    print("=== CHUNK", cid, "IDX=", cid - 1, "===")
    print(c["text"])
