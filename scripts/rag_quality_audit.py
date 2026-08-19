import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arabic_rag.chunker import TextChunker
from src.arabic_rag.embeddings import EmbeddingModel
from src.arabic_rag.generator import AnswerGenerator
from src.arabic_rag.loader import PDFLoader
from src.arabic_rag.query_analysis import analyze_query
from src.arabic_rag.retriever import Retriever
from src.arabic_rag.text_processor import ArabicTextProcessor
from src.arabic_rag.vector_store import VectorStore


PDF_PATH = "data/pdfs/sample.pdf"
TOP_K = 5


@dataclass
class QuestionCase:
    question: str
    expected_class: str
    category: str
    note: str


def contains_bad_artifacts(text: str) -> dict[str, bool]:
    return {
        "has_canadian_syllabics": bool(re.search(r"[\u1400-\u167F\u18B0-\u18FF]", text)),
        "has_replacement_char": "\ufffd" in text,
        "has_legacy_corruption_phrase": "اغ يز ا يصقلبي" in text,
        "has_reversed_1940": "0491" in text,
    }


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def is_reference_chunk(chunk: str) -> bool:
    lowered = chunk.lower()
    return (
        "المصادر" in chunk
        or "https://" in lowered
        or "http://" in lowered
        or "www." in lowered
    )


def terms(text: str) -> set[str]:
    return set(re.findall(r"[\u0600-\u06FFA-Za-z0-9]+", text))


def lexical_coverage(question: str, chunk_text: str) -> float:
    q_terms = {token for token in terms(question) if len(token) >= 2}
    c_terms = {token for token in terms(chunk_text) if len(token) >= 2}

    if not q_terms:
        return 0.0

    return len(q_terms & c_terms) / len(q_terms)


def classify_behavior(answer: str) -> str:
    refusal = AnswerGenerator.INSUFFICIENT_INFO_MESSAGE
    if answer.strip() == refusal:
        return "refusal"

    if "لكن لا توفر الوثيقة" in answer:
        return "partial"

    return "answered"


def build_cases() -> list[QuestionCase]:
    cases = [
        # A. Direct factual questions
        QuestionCase("من هو غازي القصيبي؟", "A", "A-direct", "identity"),
        QuestionCase("من هو غازي عبد الرحمن القصيبي؟", "A", "A-direct", "identity variant"),

        # B. Date questions
        QuestionCase("متى وأين ولد غازي القصيبي؟", "B", "B-date", "birth date+location"),
        QuestionCase("في أي عام وُلد غازي القصيبي؟", "B", "B-date", "birth year"),
        QuestionCase("هل تذكر الوثيقة تاريخ وفاة غازي القصيبي؟", "B", "B-date", "death date partial"),

        # C. Location questions
        QuestionCase("أين ولد غازي القصيبي؟", "B", "C-location", "birthplace"),
        QuestionCase("في أي مدينة تشير الوثيقة إلى مولده؟", "B", "C-location", "birth city"),

        # D. List questions
        QuestionCase("ما المناصب الحكومية التي شغلها غازي القصيبي؟", "B", "D-list", "positions"),
        QuestionCase("اذكر الأعمال الأدبية المذكورة في الوثيقة.", "B", "D-list", "works list"),

        # E. Section questions
        QuestionCase("ماذا تذكر المقدمة عن غازي القصيبي؟", "B", "E-section", "intro section"),
        QuestionCase("ما الذي تذكره الخاتمة عن غازي القصيبي؟", "B", "E-section", "conclusion section"),

        # F. Multi-hop questions
        QuestionCase("لخص نشأته وتعليمه ومناصبه الحكومية.", "B", "F-multihop", "multi-aspect"),
        QuestionCase("أين ولد وماذا درس وما أبرز أعماله؟", "B", "F-multihop", "triple intent"),

        # G. Yes/no verification
        QuestionCase("هل كان غازي القصيبي طبيبًا؟", "C", "G-yesno", "unsupported"),
        QuestionCase("هل تصفه الوثيقة بأنه أديب؟", "B", "G-yesno", "supported/partial"),

        # H. Partially supported
        QuestionCase("أين ولد ومتى توفي؟", "B", "H-partial", "birth likely present, death partial"),
        QuestionCase("ما اسم والده وتاريخ وفاته؟", "C", "H-partial", "mostly unsupported"),

        # I. Unsupported questions
        QuestionCase("ما لون عيني غازي القصيبي؟", "C", "I-unsupported", "unsupported"),
        QuestionCase("ما اسم زوجته؟", "C", "I-unsupported", "unsupported"),

        # J. Adversarial hallucination questions
        QuestionCase("كم كان راتبه عندما كان وزيرًا؟", "C", "J-adversarial", "unsupported numeric"),
        QuestionCase("كم عدد أبنائه؟", "C", "J-adversarial", "unsupported numeric"),
        QuestionCase("ما سبب وفاته؟", "C", "J-adversarial", "unsupported"),

        # K. Arabic/English mixed questions
        QuestionCase("What does the document say about Ghazi Al-Gosaibi?", "B", "K-mixed", "English query"),
        QuestionCase("ما هي Government roles المذكورة؟", "B", "K-mixed", "mixed terms"),

        # L. Numeric questions
        QuestionCase("اذكر السنة المذكورة لميلاده.", "B", "L-numeric", "birth year"),
        QuestionCase("هل ذكرت الوثيقة سنة 1896؟", "C", "L-numeric", "unsupported year"),

        # M. Bibliography/reference contamination
        QuestionCase("ما هو رابط ويكيبيديا المذكور؟", "B", "M-reference", "reference section"),
        QuestionCase("هل تحتوي الإجابة على روابط فقط؟", "C", "M-reference", "should avoid link-only responses"),

        # Edge cases
        QuestionCase("", "C", "edge-empty", "empty"),
        QuestionCase("؟", "C", "edge-short", "short"),
        QuestionCase("URL؟ https://example.com", "C", "edge-url", "external url question"),
        QuestionCase("1940؟", "B", "edge-number", "single number query"),
    ]

    return cases


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    expected = record["expected_class"]
    behavior = record["observed_behavior"]

    if expected == "A":
        ok = behavior in {"answered", "partial"}
    elif expected == "B":
        ok = behavior in {"answered", "partial"}
    else:
        ok = behavior == "refusal"

    return {
        "meets_class_expectation": ok,
    }


def main() -> None:
    cases = build_cases()

    loader = PDFLoader(PDF_PATH)
    raw_text = loader.load()

    processor = ArabicTextProcessor()
    cleaned_text = processor.process(raw_text)

    chunker = TextChunker()
    chunk_records = chunker.split_with_metadata(
        text=cleaned_text,
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

    generator = AnswerGenerator()

    records: list[dict[str, Any]] = []

    invalid_json_count = 0
    fallback_count = 0
    refusal_count = 0
    partial_count = 0
    grounded_answer_count = 0
    unsupported_answer_count = 0

    reference_retrieval_count = 0
    irrelevant_chunk_count = 0

    mrr_total = 0.0
    top1_relevant_count = 0
    recall_hits = 0
    precision_hits = 0
    precision_total = 0
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0

    answerable_total = 0
    answerable_covered = 0
    unsupported_total = 0
    unsupported_refused = 0

    numeric_checks_total = 0
    numeric_checks_pass = 0

    for case in cases:
        results = retriever.retrieve(case.question, k=TOP_K)
        generation_k = min(retriever.suggest_k(case.question), len(results))

        retrieved = []
        relevant_ranks = []

        for rank, item in enumerate(results, start=1):
            chunk_text = item["chunk"]
            relevance = lexical_coverage(case.question, chunk_text)
            ref_chunk = is_reference_chunk(chunk_text)

            if ref_chunk:
                reference_retrieval_count += 1

            is_relevant = relevance >= 0.2
            if not is_relevant:
                irrelevant_chunk_count += 1
            else:
                precision_hits += 1
                relevant_ranks.append(rank)

            precision_total += 1

            retrieved.append(
                {
                    "rank": rank,
                    "chunk_id": item.get("chunk_id"),
                    "fused_score": item.get("fused_score", item["score"]),
                    "dense_score": item.get("dense_score", 0.0),
                    "sparse_score": item.get("sparse_score", 0.0),
                    "features": item.get("features", {}),
                    "is_reference_chunk": ref_chunk,
                    "relevance_overlap": round(relevance, 4),
                    "is_relevant_proxy": is_relevant,
                    "metadata": item.get("metadata", {}),
                    "text": chunk_text,
                }
            )

        if relevant_ranks:
            recall_hits += 1
            mrr_total += 1.0 / min(relevant_ranks)
            if min(relevant_ranks) == 1:
                top1_relevant_count += 1

        if any(item["is_relevant_proxy"] for item in retrieved[:1]):
            hit_at_1 += 1
        if any(item["is_relevant_proxy"] for item in retrieved[:3]):
            hit_at_3 += 1
        if any(item["is_relevant_proxy"] for item in retrieved[:5]):
            hit_at_5 += 1

        context = "\n\n".join(item["chunk"] for item in results[:generation_k])

        debug = generator.generate(
            question=case.question,
            context=context,
            retrieved_chunks=results,
            return_debug=True,
        )

        decision = debug["decision"]
        payload = debug.get("payload")
        answer = debug["final_answer"]

        behavior = classify_behavior(answer)
        if behavior == "refusal":
            refusal_count += 1
        if behavior == "partial":
            partial_count += 1

        if decision in {"validated_model_answer", "partial_reconstructed_answer", "extractive_reconstructed_fallback"}:
            grounded_answer_count += 1

        if decision not in {"validated_model_answer", "partial_reconstructed_answer", "extractive_reconstructed_fallback", "model_refusal", "safe_refusal", "empty_question_refusal"}:
            unsupported_answer_count += 1

        if payload is None:
            invalid_json_count += 1

        if decision not in {"validated_model_answer", "model_refusal"}:
            fallback_count += 1

        numeric_supported = extract_numbers(answer).issubset(extract_numbers(context))
        numeric_checks_total += 1
        if numeric_supported:
            numeric_checks_pass += 1

        if case.expected_class in {"A", "B"}:
            answerable_total += 1
            if behavior in {"answered", "partial"}:
                answerable_covered += 1
        elif case.expected_class == "C":
            unsupported_total += 1
            if behavior == "refusal":
                unsupported_refused += 1

        query_info = analyze_query(case.question)

        record = {
            "question": case.question,
            "expected_class": case.expected_class,
            "category": case.category,
            "note": case.note,
            "query_analysis": query_info,
            "observed_behavior": behavior,
            "decision": decision,
            "json_valid": payload is not None,
            "payload_supported": None if payload is None else payload.get("supported"),
            "final_answer": answer,
            "raw_model_output": debug.get("raw_model_output", ""),
            "retrieved": retrieved,
            "numeric_claims_supported": numeric_supported,
            "answer_artifacts": contains_bad_artifacts(answer),
        }

        record.update(score_record(record))
        records.append(record)

    total_questions = len(records)
    total_retrieved = total_questions * TOP_K

    chunk_lengths = [len(chunk) for chunk in chunks]
    avg_chunk_length = sum(chunk_lengths) / max(1, len(chunk_lengths))

    section_titles = [item.get("section", "") for item in chunk_records]
    heading_preservation_ratio = sum(
        1 for title in section_titles if title and title != "النص"
    ) / max(1, len(section_titles))

    sentence_boundary_ratio = sum(
        1
        for chunk in chunks
        if re.search(r"[\.؟!]$", chunk.strip())
    ) / max(1, len(chunks))

    class_success_rate = sum(
        1 for item in records if item["meets_class_expectation"]
    ) / max(1, total_questions)

    summary = {
        "question_count": total_questions,
        "chunk_count": len(chunks),
        "chunk_avg_length": round(avg_chunk_length, 2),
        "heading_preservation_ratio": round(heading_preservation_ratio, 4),
        "sentence_boundary_preservation_ratio": round(sentence_boundary_ratio, 4),
        "retrieval_hit_at_1": round(hit_at_1 / max(1, total_questions), 4),
        "retrieval_hit_at_3": round(hit_at_3 / max(1, total_questions), 4),
        "retrieval_hit_at_5": round(hit_at_5 / max(1, total_questions), 4),
        "invalid_json_rate": round(invalid_json_count / max(1, total_questions), 4),
        "structured_output_validity": round(1.0 - (invalid_json_count / max(1, total_questions)), 4),
        "fallback_rate": round(fallback_count / max(1, total_questions), 4),
        "refusal_rate": round(refusal_count / max(1, total_questions), 4),
        "refusal_accuracy": round(unsupported_refused / max(1, unsupported_total), 4),
        "partial_answer_rate": round(partial_count / max(1, total_questions), 4),
        "grounded_answer_rate": round(grounded_answer_count / max(1, total_questions), 4),
        "answerable_question_coverage": round(answerable_covered / max(1, answerable_total), 4),
        "grounded_answer_coverage": round(grounded_answer_count / max(1, answerable_total), 4),
        "unsafe_answer_rate": round(unsupported_answer_count / max(1, total_questions), 4),
        "numeric_grounding_accuracy": round(numeric_checks_pass / max(1, numeric_checks_total), 4),
        "class_expectation_success_rate": round(class_success_rate, 4),
        "reference_chunk_retrieval_rate": round(reference_retrieval_count / max(1, total_retrieved), 4),
        "irrelevant_chunk_rate": round(irrelevant_chunk_count / max(1, total_retrieved), 4),
        "precision_at_k_proxy": round(precision_hits / max(1, precision_total), 4),
        "recall_at_k_proxy": round(recall_hits / max(1, total_questions), 4),
        "mrr_proxy": round(mrr_total / max(1, total_questions), 4),
        "top1_relevant_rate": round(top1_relevant_count / max(1, total_questions), 4),
        "raw_text_artifacts": contains_bad_artifacts(raw_text),
        "cleaned_text_artifacts": contains_bad_artifacts(cleaned_text),
    }

    by_category: dict[str, dict[str, float]] = {}
    for item in records:
        category = item["category"]
        by_category.setdefault(category, {"count": 0, "success": 0, "refusal": 0})
        by_category[category]["count"] += 1
        if item["meets_class_expectation"]:
            by_category[category]["success"] += 1
        if item["observed_behavior"] == "refusal":
            by_category[category]["refusal"] += 1

    for category, values in by_category.items():
        count = max(1, values["count"])
        values["success_rate"] = round(values["success"] / count, 4)
        values["refusal_rate"] = round(values["refusal"] / count, 4)

    report = {
        "summary": summary,
        "by_category": by_category,
        "records": records,
    }

    out_path = ROOT / "audit_results.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Wrote", out_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
