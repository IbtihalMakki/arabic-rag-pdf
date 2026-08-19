import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arabic_rag.loader import PDFLoader
from src.arabic_rag.text_processor import ArabicTextProcessor
from src.arabic_rag.chunker import TextChunker
from src.arabic_rag.embeddings import EmbeddingModel
from src.arabic_rag.vector_store import VectorStore
from src.arabic_rag.retriever import Retriever
from src.arabic_rag.generator import AnswerGenerator


QUESTION = "من هو غازي القصيبي؟"
PDF_PATH = "data/pdfs/sample.pdf"


@dataclass
class TraceData:
    raw_text: str
    normalized_text: str
    repaired_text: str
    repair_count: int
    repairs: list[dict]
    chunk_records: list[dict]
    results: list[dict]
    prompt_context: str
    raw_model_output: str
    decision: str
    payload: dict | None
    answer: str


def run_trace() -> TraceData:
    loader = PDFLoader(PDF_PATH)
    raw_text = loader.load()

    processor = ArabicTextProcessor()
    diagnostics = processor.process_with_diagnostics(raw_text)
    normalized_text = diagnostics["normalized_text"]
    repaired_text = diagnostics["repaired_text"]
    repairs = diagnostics["repairs"]
    repair_count = diagnostics["repair_count"]

    chunker = TextChunker()
    chunk_records = chunker.split_with_metadata(
        text=repaired_text,
        source_document=PDF_PATH,
    )
    chunks = [item["text"] for item in chunk_records]
    metadatas = [
        {
            "chunk_id": item["chunk_id"],
            "section": item["section"],
            "page": item["page"],
            "source_document": item["source_document"],
        }
        for item in chunk_records
    ]

    embedding_model = EmbeddingModel()
    embeddings = embedding_model.encode(chunks)

    vector_store = VectorStore(dimension=embeddings.shape[1])
    vector_store.add(embeddings, chunks, metadatas=metadatas)

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )
    dynamic_k = retriever.suggest_k(QUESTION)
    results = retriever.retrieve(QUESTION, k=dynamic_k)

    context = "\n\n".join(item["chunk"] for item in results)

    generator = AnswerGenerator()

    debug = generator.generate(
        question=QUESTION,
        context=context,
        retrieved_chunks=results,
        return_debug=True,
    )

    answer = debug["final_answer"]

    return TraceData(
        raw_text=raw_text,
        normalized_text=normalized_text,
        repaired_text=repaired_text,
        repair_count=repair_count,
        repairs=repairs,
        chunk_records=chunk_records,
        results=results,
        prompt_context=debug["prompt_context"],
        raw_model_output=debug["raw_model_output"],
        decision=debug["decision"],
        payload=debug["payload"],
        answer=answer,
    )


def main() -> None:
    trace = run_trace()

    print("=== RAW EXTRACTED TEXT (first 2000) ===")
    print(trace.raw_text[:2000])

    print("\n=== NORMALIZED TEXT (first 2000) ===")
    print(trace.normalized_text[:2000])

    print("\n=== REPAIRED TEXT (first 2000) ===")
    print(trace.repaired_text[:2000])

    print("\n=== REPAIR DIAGNOSTICS ===")
    print("repairs:", trace.repair_count)
    for idx, item in enumerate(trace.repairs, start=1):
        print(f"[Repair {idx}] reason={item['reason']} confidence={item['confidence']}")
        print("left:", item["left"])
        print("right:", item["right"])
        print("merged:", item["merged"])
        print("-" * 70)

    print("\n=== CHUNK COUNT ===")
    print(len(trace.chunk_records))

    print("\n=== CHUNKS ===")
    for item in trace.chunk_records:
        print(f"[Chunk {item['chunk_id']}] section={item['section']}")
        print(item["text"])
        print("-" * 70)

    print("\n=== RETRIEVED CHUNKS ===")
    for idx, result in enumerate(trace.results, start=1):
        print(
            f"[Retrieved {idx}] fused={result.get('fused_score', result['score']):.4f} "
            f"dense={result.get('dense_score', 0.0):.4f} "
            f"sparse={result.get('sparse_score', 0.0):.4f} "
            f"chunk_id={result.get('chunk_id')}"
        )
        bridge_hit = result.get("bridge_terms_hit", [])
        if bridge_hit:
            print(f"  [BRIDGE HIT: {bridge_hit}]")
        metadata = result.get("metadata", {})
        if metadata:
            print("metadata:", metadata)
        print(result["chunk"])
        print("-" * 70)

    print("\n=== PROMPT CONTEXT SENT TO GENERATOR ===")
    print(trace.prompt_context)

    print("\n=== RAW MODEL OUTPUT ===")
    print(trace.raw_model_output)

    print("\n=== GROUNDING DECISION ===")
    print(trace.decision)

    print("\n=== PARSED PAYLOAD ===")
    print(json.dumps(trace.payload, ensure_ascii=False, indent=2))

    print("\n=== GENERATED ANSWER ===")
    print(trace.answer)

    # JSON snapshot for reproducibility.
    snapshot = {
        "question": QUESTION,
        "chunk_count": len(trace.chunk_records),
        "retrieved": trace.results,
        "decision": trace.decision,
        "payload": trace.payload,
        "answer": trace.answer,
    }
    print("\n=== TRACE SUMMARY JSON ===")
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
