"""RAG Bottleneck Experiment: Three-condition A/B/C ablation study.

EXPERIMENT DESIGN
-----------------
Goal: Isolate the true production bottleneck among:
  - Retrieval (wrong chunks retrieved)
  - Source-text fragmentation (right chunks, but noisy/fragmented)
  - Model/generator capability (right clean evidence, but model still fails)

Conditions:
  A = Current production pipeline (results loaded from existing audit_results.json)
  B = Oracle retrieval: correct raw corpus chunks injected, retrieval bypassed,
      grounding validation ACTIVE
  C = Oracle clean evidence: same correct chunks with conservative line-join repair,
      grounding validation ACTIVE

Conservative cleaning for condition C (clean_for_condition_c):
  - Only fixes: isolated 'ي' → 'في' (truncated preposition at line breaks)
  - Only fixes: 'الممل' → 'المملكة' (documented suffix truncation)
  - Only fixes: period-leading lines joined to previous sentence
  - Only joins: newlines within chunks
  - Does NOT complete truncated word stems (وز, التعل, والروا, etc.)
  - Does NOT add any words absent from the original text
  - Does NOT change any factual content

SAFETY REQUIREMENTS (met in all conditions):
  - Grounding validation is active in all conditions (B and C included)
  - Source chunk text is passed as-is for condition B
  - Clean text for condition C is derived algorithmically, documented and reproducible
  - Unsupported question safety controls included
  - unsafe_answer_rate must remain 0.0

DO NOT MODIFY PRODUCTION CODE. This is a diagnostic experiment only.

Usage:
  python rag_bottleneck_experiment.py

Outputs:
  bottleneck_experiment_results.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arabic_rag.generator import AnswerGenerator
from src.arabic_rag.loader import PDFLoader
from src.arabic_rag.text_processor import ArabicTextProcessor
from src.arabic_rag.chunker import TextChunker


# ── Corpus-derived evidence oracle ────────────────────────────────────────────

def load_corpus_chunks() -> list[str]:
    """Load corpus chunks exactly as the production pipeline does."""
    raw = PDFLoader("data/pdfs/sample.pdf").load()
    repaired = ArabicTextProcessor().process(raw)
    chunks_meta = TextChunker().split_with_metadata(repaired, "data/pdfs/sample.pdf")
    return [c["text"] for c in chunks_meta]


def clean_for_condition_c(text: str) -> str:
    """Conservative condition-C repair: fix line-fragmentation artifacts only.

    Exactly these transformations (no others):
    1. Line-terminal ي before next content → في (split preposition repair)
    2. Standalone ي surrounded by whitespace → في (isolated truncated preposition)
    3. Line-start ي → في (another split-preposition form)
    4. Period at line start → joined to previous line (PDF artifact)
    5. All remaining newlines within chunk joined to single spaces
    6. 'الممل' → 'المملكة' (documented suffix truncation of المملكة)
    7. Excess spaces collapsed

    NOT applied (would require fabricating missing characters):
    - Completing 'وز' → 'وزير'
    - Completing 'التعل' → 'التعليم'
    - Completing 'والروا' → 'والرواية'
    - Any word-stem completion
    - Adding any word not in the source
    """
    # Fix line-terminal ي before next content (e.g. "1940 ي\nالاحساء" → "1940 في الاحساء")
    text = re.sub(r"(\w) ي\n(\S)", r"\1 في \2", text)
    # Fix standalone ي surrounded by whitespace (isolated truncated في)
    text = re.sub(r"(?<=\s)ي(?=\s)", "في", text)
    # Fix ي at line start
    text = re.sub(r"(?m)^ي\s", "في ", text)
    # Fix period at line start (PDF extraction artifact)
    text = re.sub(r"\n\.", " .", text)
    # Join remaining newlines within chunk
    text = re.sub(r"\n+", " ", text)
    # Fix المملكة suffix truncation (الممل is never a standalone word)
    text = text.replace("الممل", "المملكة")
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ── Oracle evidence mapping ────────────────────────────────────────────────────
# Maps each question to its oracle chunk indices (0-based into the corpus).
# Evidence basis: RAG_FAILURE_ATTRIBUTION_REPORT.md + corpus chunk inspection.
# Questions with no recoverable oracle evidence are marked as UNANSWERABLE.

ORACLE_CHUNK_INDICES: dict[str, list[int] | None] = {
    # Passing answerable (positive controls)
    "من هو غازي القصيبي؟": [1, 2],
    "من هو غازي عبد الرحمن القصيبي؟": [1, 2, 7],
    "في أي عام وُلد غازي القصيبي؟": [1, 2],
    "أين ولد غازي القصيبي؟": [1, 2],
    "1940؟": [1, 2],

    # Failing answerable (primary experiment targets)
    "متى وأين ولد غازي القصيبي؟": [1, 2],           # birth year + location
    "هل تذكر الوثيقة تاريخ وفاة غازي القصيبي؟": [7, 8],  # تو2010
    "في أي مدينة تشير الوثيقة إلى مولده؟": [1, 2],   # الاحساء
    "ما المناصب الحكومية التي شغلها غازي القصيبي؟": [2, 3],  # مناصب + وز
    "اذكر الأعمال الأدبية المذكورة في الوثيقة.": [4, 5],   # literary works
    "ماذا تذكر المقدمة عن غازي القصيبي؟": [0, 1],    # intro chunks
    "ما الذي تذكره الخاتمة عن غازي القصيبي؟": [7, 8], # conclusion chunks
    "لخص نشأته وتعليمه ومناصبه الحكومية.": [1, 2, 3], # multi-chunk
    "أين ولد وماذا درس وما أبرز أعماله؟": [1, 2, 3, 4], # multi-chunk
    "هل تصفه الوثيقة بأنه أديب؟": [4, 7],            # works + conclusion
    "أين ولد ومتى توفي؟": [1, 7],                    # birth + death
    "What does the document say about Ghazi Al-Gosaibi?": [0, 1, 7],
    "ما هي Government roles المذكورة؟": [2, 3],       # positions chunks
    "اذكر السنة المذكورة لميلاده.": [1, 2],           # birth year
    "ما هو رابط ويكيبيديا المذكور؟": None,            # UNANSWERABLE: reference section stripped
}

# Unsupported questions used as refusal-accuracy safety controls in conditions B/C.
# Oracle chunks injected are general context chunks (not supporting the unsupported fact).
SAFETY_CONTROLS: dict[str, list[int]] = {
    "هل كان غازي القصيبي طبيبًا؟": [1, 2],       # C expected: should still refuse
    "ما اسم زوجته؟": [1, 2],                      # C expected: should still refuse
    "ما لون عيني غازي القصيبي؟": [1, 2],          # C expected: should still refuse
    "كم كان راتبه عندما كان وزيرًا؟": [2, 3],    # C expected: should still refuse
}


# ── Experiment logic ──────────────────────────────────────────────────────────

def classify_behavior(answer: str) -> str:
    if answer.strip() == AnswerGenerator.INSUFFICIENT_INFO_MESSAGE:
        return "refusal"
    if "لكن لا توفر الوثيقة" in answer:
        return "partial"
    return "answered"


def detect_hallucination(raw_output: str, oracle_chunks: list[str]) -> bool:
    """Heuristic hallucination detection: check if model fabricated key numbers."""
    all_corpus_nums = set(re.findall(r"\d+", " ".join(oracle_chunks)))
    model_nums = set(re.findall(r"\d{3,}", raw_output))  # 3+ digit numbers only
    fabricated = model_nums - all_corpus_nums
    return bool(fabricated)


def run_condition_b_or_c(
    question: str,
    oracle_chunk_texts: list[str],
    generator: AnswerGenerator,
    condition_label: str,
) -> dict[str, Any]:
    """Run the generator on oracle evidence. Grounding stays active."""
    if not oracle_chunk_texts:
        return {
            "condition": condition_label,
            "question": question,
            "oracle_chunks_used": [],
            "evidence_available": False,
            "json_valid": None,
            "grounding_decision": "unanswerable_no_corpus_evidence",
            "behavior": "refusal",
            "final_answer": AnswerGenerator.INSUFFICIENT_INFO_MESSAGE,
            "hallucination_detected": False,
            "numeric_correct": None,
            "raw_model_output": "",
            "payload_supported": None,
        }

    retrieved_chunks = [
        {"chunk": chunk_text, "score": 1.0}
        for chunk_text in oracle_chunk_texts
    ]
    context = "\n\n".join(oracle_chunk_texts)

    debug = generator.generate(
        question=question,
        context=context,
        retrieved_chunks=retrieved_chunks,
        return_debug=True,
    )

    answer = debug["final_answer"]
    decision = debug["decision"]
    raw_output = debug.get("raw_model_output", "")
    payload = debug.get("payload")
    behavior = classify_behavior(answer)

    hallucination = detect_hallucination(raw_output, oracle_chunk_texts)
    numeric_correct: bool | None = None
    if re.findall(r"\d+", answer):
        corpus_nums = set(re.findall(r"\d+", context))
        answer_nums = set(re.findall(r"\d+", answer))
        numeric_correct = answer_nums.issubset(corpus_nums)

    return {
        "condition": condition_label,
        "question": question,
        "oracle_chunks_used": oracle_chunk_texts,
        "evidence_available": True,
        "json_valid": payload is not None,
        "grounding_decision": decision,
        "behavior": behavior,
        "final_answer": answer,
        "hallucination_detected": hallucination,
        "numeric_correct": numeric_correct,
        "raw_model_output": raw_output,
        "payload_supported": payload.get("supported") if payload else None,
    }


def main() -> None:
    print("Loading corpus chunks...")
    corpus_chunks = load_corpus_chunks()
    print(f"Corpus: {len(corpus_chunks)} chunks")

    print("Loading existing condition-A results...")
    audit_data = json.loads(Path("audit_results.json").read_text(encoding="utf-8"))
    condition_a_by_question = {
        r["question"]: r for r in audit_data["records"]
    }

    print("Initializing generator...")
    generator = AnswerGenerator()
    print("Generator ready.\n")

    # All questions to run through B and C
    target_questions = list(ORACLE_CHUNK_INDICES.keys())
    safety_questions = list(SAFETY_CONTROLS.keys())

    results: list[dict[str, Any]] = []

    # ── Condition A: load from existing audit ─────────────────────────────────
    print("=== CONDITION A (existing pipeline) ===")
    for q in target_questions:
        if q not in condition_a_by_question:
            print(f"  WARNING: {q[:50]} not in audit results")
            continue
        r = condition_a_by_question[q]
        a_record = {
            "condition": "A",
            "question": q,
            "expected_class": r["expected_class"],
            "category": r["category"],
            "oracle_chunks_used": [],
            "evidence_available": True,
            "json_valid": r["json_valid"],
            "grounding_decision": r["decision"],
            "behavior": r["observed_behavior"],
            "final_answer": r["final_answer"],
            "hallucination_detected": False,
            "numeric_correct": r["numeric_claims_supported"],
            "raw_model_output": r.get("raw_model_output", ""),
            "payload_supported": r.get("payload_supported"),
            "meets_expectation": r["meets_class_expectation"],
        }
        results.append(a_record)
    print(f"  Loaded {len([r for r in results if r['condition'] == 'A'])} condition-A records from audit.\n")

    # ── Conditions B and C: fresh model runs ─────────────────────────────────
    for q in target_questions:
        idx_list = ORACLE_CHUNK_INDICES.get(q)
        audit_rec = condition_a_by_question.get(q, {})
        expected_class = audit_rec.get("expected_class", "?")
        category = audit_rec.get("category", "?")
        meets_a = audit_rec.get("meets_class_expectation", False)

        print(f"Q [{category}]: {q[:60]}")
        print(f"  A: {audit_rec.get('observed_behavior','?')} | oracle_chunks={idx_list}")

        # ── Condition B (oracle raw chunks) ───────────────────────────────────
        if idx_list is None:
            oracle_raw = []
        else:
            oracle_raw = [corpus_chunks[i] for i in idx_list if i < len(corpus_chunks)]

        rec_b = run_condition_b_or_c(q, oracle_raw, generator, "B")
        rec_b["expected_class"] = expected_class
        rec_b["category"] = category
        rec_b["meets_expectation"] = (
            rec_b["behavior"] in {"answered", "partial"}
            if expected_class in {"A", "B"}
            else rec_b["behavior"] == "refusal"
        )
        print(f"  B: {rec_b['behavior']} | grounding={rec_b['grounding_decision']}")
        results.append(rec_b)

        # ── Condition C (oracle clean chunks) ─────────────────────────────────
        oracle_clean = [clean_for_condition_c(t) for t in oracle_raw]

        rec_c = run_condition_b_or_c(q, oracle_clean, generator, "C")
        rec_c["expected_class"] = expected_class
        rec_c["category"] = category
        rec_c["meets_expectation"] = (
            rec_c["behavior"] in {"answered", "partial"}
            if expected_class in {"A", "B"}
            else rec_c["behavior"] == "refusal"
        )
        print(f"  C: {rec_c['behavior']} | grounding={rec_c['grounding_decision']}")
        results.append(rec_c)
        print()

    # ── Safety controls (unsupported questions, conditions B and C) ───────────
    print("=== SAFETY CONTROLS (unsupported questions) ===")
    for q, idx_list in SAFETY_CONTROLS.items():
        oracle_raw = [corpus_chunks[i] for i in idx_list if i < len(corpus_chunks)]
        oracle_clean = [clean_for_condition_c(t) for t in oracle_raw]

        print(f"Safety Q: {q[:55]}")

        rec_b = run_condition_b_or_c(q, oracle_raw, generator, "B_safety")
        rec_b["expected_class"] = "C"
        rec_b["category"] = "safety_control"
        rec_b["meets_expectation"] = (rec_b["behavior"] == "refusal")
        print(f"  B_safety: {rec_b['behavior']} (must be refusal)")
        results.append(rec_b)

        rec_c = run_condition_b_or_c(q, oracle_clean, generator, "C_safety")
        rec_c["expected_class"] = "C"
        rec_c["category"] = "safety_control"
        rec_c["meets_expectation"] = (rec_c["behavior"] == "refusal")
        print(f"  C_safety: {rec_c['behavior']} (must be refusal)")
        results.append(rec_c)
        print()

    # ── Compute summary statistics ────────────────────────────────────────────
    def compute_stats(condition_filter: str) -> dict[str, float | int]:
        recs = [r for r in results if r["condition"] == condition_filter]
        answerable = [r for r in recs if r.get("expected_class") in ("A", "B")]
        unsupported = [r for r in recs if r.get("expected_class") == "C"]

        if not recs:
            return {}

        answered_count = sum(1 for r in answerable if r["behavior"] in ("answered", "partial"))
        unsafe_count = sum(
            1 for r in recs
            if r["behavior"] == "answered" and r.get("expected_class") == "C"
        )
        return {
            "condition": condition_filter,
            "total_runs": len(recs),
            "answerable_count": len(answerable),
            "answerable_coverage": round(answered_count / max(1, len(answerable)), 4),
            "unsafe_answer_rate": round(unsafe_count / max(1, len(recs)), 4),
            "refusal_rate": round(
                sum(1 for r in answerable if r["behavior"] == "refusal") / max(1, len(answerable)),
                4,
            ),
            "json_valid_rate": round(
                sum(1 for r in recs if r.get("json_valid") is True) / max(1, len(recs)),
                4,
            ),
            "hallucination_rate": round(
                sum(1 for r in recs if r.get("hallucination_detected")) / max(1, len(recs)),
                4,
            ),
        }

    stats_a = compute_stats("A")
    stats_b = compute_stats("B")
    stats_c = compute_stats("C")

    # Per-category breakdown (failing answerable, conditions A/B/C)
    categories = sorted({r["category"] for r in results if r.get("expected_class") in ("A", "B")})
    cat_breakdown = {}
    for cat in categories:
        cat_breakdown[cat] = {}
        for cond in ("A", "B", "C"):
            cat_recs = [r for r in results if r["condition"] == cond and r["category"] == cat]
            if cat_recs:
                answered = sum(1 for r in cat_recs if r["behavior"] in ("answered", "partial"))
                cat_breakdown[cat][cond] = {
                    "count": len(cat_recs),
                    "answered": answered,
                    "coverage": round(answered / len(cat_recs), 4),
                }

    # Per-question B-vs-C comparison
    per_question = {}
    for q in target_questions:
        qa = next((r for r in results if r["condition"] == "A" and r["question"] == q), None)
        qb = next((r for r in results if r["condition"] == "B" and r["question"] == q), None)
        qc = next((r for r in results if r["condition"] == "C" and r["question"] == q), None)
        if qa and qb and qc:
            per_question[q] = {
                "A_behavior": qa["behavior"],
                "B_behavior": qb["behavior"],
                "C_behavior": qc["behavior"],
                "A_decision": qa["grounding_decision"],
                "B_decision": qb["grounding_decision"],
                "C_decision": qc["grounding_decision"],
                "B_json_valid": qb["json_valid"],
                "C_json_valid": qc["json_valid"],
                "B_hallucination": qb["hallucination_detected"],
                "C_hallucination": qc["hallucination_detected"],
                "A_to_B_change": qa["behavior"] != qb["behavior"],
                "B_to_C_change": qb["behavior"] != qc["behavior"],
                "evidence_available": qb["evidence_available"],
            }

    output = {
        "experiment_description": {
            "A": "Current production pipeline (from audit_results.json)",
            "B": "Oracle retrieval: correct raw corpus chunks, grounding active",
            "C": "Oracle clean evidence: line-join + في-repair only, grounding active",
            "clean_c_changes": [
                "Isolated 'ي' → 'في' (truncated preposition repair)",
                "Line-terminal 'ي' before content → 'في'",
                "Line-start 'ي' → 'في'",
                "Period at line-start joined to previous sentence",
                "All newlines within chunk joined to spaces",
                "'الممل' → 'المملكة' (documented suffix truncation)",
            ],
            "clean_c_NOT_changed": [
                "No word-stem completions (وز stays وز, not وزير)",
                "No content added beyond what the source clearly states",
                "No paraphrasing of facts",
            ],
        },
        "statistics": {
            "A": stats_a,
            "B": stats_b,
            "C": stats_c,
        },
        "category_breakdown": cat_breakdown,
        "per_question_comparison": per_question,
        "detailed_records": results,
    }

    out_path = ROOT / "bottleneck_experiment_results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== EXPERIMENT COMPLETE ===")
    print(f"Results written to: {out_path}")
    print()
    print("SUMMARY:")
    for cond, stats in [("A", stats_a), ("B", stats_b), ("C", stats_c)]:
        print(f"  Condition {cond}: answerable_coverage={stats.get('answerable_coverage','?')}  "
              f"json_valid={stats.get('json_valid_rate','?')}  "
              f"hallucination={stats.get('hallucination_rate','?')}  "
              f"unsafe={stats.get('unsafe_answer_rate','?')}")
    print()
    print("PER-QUESTION CHANGES (A→B, B→C):")
    for q, comp in per_question.items():
        a_b = "IMPROVED" if comp["A_to_B_change"] and comp["A_behavior"] != "answered" and comp["B_behavior"] in ("answered", "partial") else ("REGRESSED" if comp["A_to_B_change"] and comp["A_behavior"] in ("answered","partial") and comp["B_behavior"] == "refusal" else "same")
        b_c = "IMPROVED" if comp["B_to_C_change"] and comp["B_behavior"] != "answered" and comp["C_behavior"] in ("answered","partial") else ("REGRESSED" if comp["B_to_C_change"] and comp["B_behavior"] in ("answered","partial") and comp["C_behavior"] == "refusal" else "same")
        if a_b != "same" or b_c != "same":
            print(f"  A→B:{a_b} B→C:{b_c}  | {q[:60]}")


if __name__ == "__main__":
    main()
