"""Compute oracle ceilings and failure category breakdown."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arabic_rag.loader import PDFLoader
from src.arabic_rag.text_processor import ArabicTextProcessor
from src.arabic_rag.chunker import TextChunker
from src.arabic_rag.retriever import Retriever
from src.arabic_rag.query_analysis import analyze_query
from src.arabic_rag.generator import AnswerGenerator

# ── Build corpus ───────────────────────────────────────────────────────────────
raw = PDFLoader("data/pdfs/sample.pdf").load()
repaired = ArabicTextProcessor().process(raw)
chunks_meta = TextChunker().split_with_metadata(repaired, "data/pdfs/sample.pdf")
chunks = [c["text"] for c in chunks_meta]
ALL_CORPUS = "\n".join(chunks)

# Stub retriever just for BM25 and helper methods
r_stub = Retriever.__new__(Retriever)
r_stub.fusion_weights = {}
r_stub._bm25_k1 = 1.2
r_stub._bm25_b = 0.75
r_stub._build_sparse_index(chunks)

# Generator stub for grounding helpers
g_stub = AnswerGenerator.__new__(AnswerGenerator)

# ── Load audit ─────────────────────────────────────────────────────────────────
d = json.load(open("audit_results.json", encoding="utf-8"))
recs = d["records"]
fails = [r for r in recs if r["expected_class"] in ("A", "B") and not r["meets_class_expectation"]]
answerable = [r for r in recs if r["expected_class"] in ("A", "B")]
print(f"Answerable total: {len(answerable)}")
print(f"Answerable fails: {len(fails)}")

# ── Failure category definitions ───────────────────────────────────────────────
# A. EXTRACTION_NOISE  B. CHUNK_BOUNDARY  C. QUERY_FORMULATION
# D. DENSE_RETRIEVAL   E. SPARSE_RETRIEVAL F. FUSION_RANKING
# G. INTENT_CLASSIFICATION H. MULTI_CHUNK_REASONING I. GROUNDING_VALIDATION
# J. GENERATION_STRUCTURED_OUTPUT K. DATASET_EXPECTATION L. OTHER

def evidence_in_corpus(key_terms_str):
    toks = set(re.findall(r"[\u0621-\u064AA-Za-z0-9]+", re.sub(r"[\u064B-\u0652]", "", key_terms_str)))
    for chunk in chunks:
        ctoks = set(re.findall(r"[\u0621-\u064AA-Za-z0-9]+", re.sub(r"[\u064B-\u0652]", "", chunk)))
        if toks & ctoks:
            return True
    return False

# Oracle: can the correct chunk be found anywhere in corpus?
# If yes, the failure is a retrieval/ranking/grounding/generation problem, not missing-evidence.

oracle_hits = 0
results = []

for r in fails:
    q = r["question"]
    info = analyze_query(q)
    retrieved_top5 = r["retrieved"][:5]
    rel_in_top5 = sum(1 for x in retrieved_top5 if x["is_relevant_proxy"])
    rel_in_top1 = retrieved_top5[0]["is_relevant_proxy"] if retrieved_top5 else False
    rel_in_top3 = any(x["is_relevant_proxy"] for x in retrieved_top5[:3])

    # Check oracle: look for key_terms in any corpus chunk
    key_terms_flat = " ".join(info["key_terms"])
    oracle_in_corpus = evidence_in_corpus(key_terms_flat)
    if oracle_in_corpus:
        oracle_hits += 1

    # Grounding fallback analysis: if evidence were in top-5, would grounding pass it?
    source_map = {f"S{i+1}": c["text"] for i, c in enumerate(retrieved_top5)}
    support = g_stub._collect_support_sentences(info, source_map)
    grounding_would_pass = g_stub._can_return_partial_answer(info, support) if support else False

    # Structured output analysis
    json_valid = r["json_valid"]
    payload_supported_str = str(r.get("payload_supported", ""))
    model_out = r.get("raw_model_output", "")[:200]

    # Determine primary failure category
    if not oracle_in_corpus:
        if q.strip() and all(c.isascii() for c in q.replace(" ", "").replace("؟", "").replace(".", "")):
            primary = "C. QUERY_FORMULATION"
        elif not info["entities"] and not info["key_terms"]:
            primary = "C. QUERY_FORMULATION"
        else:
            primary = "A. EXTRACTION_NOISE"
    elif not rel_in_top5:
        # Evidence exists but not retrieved
        sparse_scores = []
        for i in range(len(chunks)):
            norm = r_stub._normalize_terms(r_stub._tokenize(key_terms_flat))
            sc = r_stub._bm25_score(norm, i)
            if sc > 0:
                sparse_scores.append((i + 1, sc))
        has_sparse_signal = bool(sparse_scores)
        if q.count("و") >= 2 or len(info["key_terms"]) > 5:
            primary = "H. MULTI_CHUNK_REASONING"
        elif not info["entities"]:
            primary = "C. QUERY_FORMULATION"
        elif not has_sparse_signal:
            primary = "E. SPARSE_RETRIEVAL"
        else:
            primary = "D. DENSE_RETRIEVAL"
    elif rel_in_top5 and not grounding_would_pass:
        primary = "I. GROUNDING_VALIDATION"
    elif rel_in_top5 and grounding_would_pass:
        if not json_valid or payload_supported_str == "None":
            primary = "J. GENERATION_STRUCTURED_OUTPUT"
        else:
            primary = "I. GROUNDING_VALIDATION"
    else:
        primary = "L. OTHER"

    results.append({
        "question": q,
        "category": r["category"],
        "decision": r["decision"],
        "oracle_in_corpus": oracle_in_corpus,
        "rel_in_top1": rel_in_top1,
        "rel_in_top3": rel_in_top3,
        "rel_in_top5": bool(rel_in_top5),
        "grounding_would_pass": grounding_would_pass,
        "json_valid": json_valid,
        "primary_failure": primary,
    })

print("\n=== PER-FAILURE ATTRIBUTION ===")
for item in results:
    print(f"\nQ: {item['question'][:70]}")
    print(f"  bench_cat={item['category']} decision={item['decision']}")
    print(f"  oracle_in_corpus={item['oracle_in_corpus']} rel@1={item['rel_in_top1']} rel@3={item['rel_in_top3']} rel@5={item['rel_in_top5']}")
    print(f"  grounding_would_pass={item['grounding_would_pass']} json_valid={item['json_valid']}")
    print(f"  PRIMARY_FAILURE: {item['primary_failure']}")

print("\n=== CATEGORY COUNTS ===")
from collections import Counter
cats = Counter(item["primary_failure"] for item in results)
total = len(results)
for cat, count in sorted(cats.items()):
    print(f"  {cat}: {count} ({100*count/total:.1f}%)")

print("\n=== LAYER ATTRIBUTION ===")
retrieval_fail = sum(1 for r in results if r["primary_failure"] in {
    "C. QUERY_FORMULATION", "D. DENSE_RETRIEVAL", "E. SPARSE_RETRIEVAL",
    "F. FUSION_RANKING", "G. INTENT_CLASSIFICATION", "H. MULTI_CHUNK_REASONING",
    "A. EXTRACTION_NOISE", "B. CHUNK_BOUNDARY"
})
grounding_fail = sum(1 for r in results if r["primary_failure"] in {"I. GROUNDING_VALIDATION"})
generation_fail = sum(1 for r in results if r["primary_failure"] in {"J. GENERATION_STRUCTURED_OUTPUT"})
dataset_fail = sum(1 for r in results if r["primary_failure"] in {"K. DATASET_EXPECTATION"})
other_fail = sum(1 for r in results if r["primary_failure"] in {"L. OTHER"})

print(f"  Retrieval-layer failures:  {retrieval_fail} ({100*retrieval_fail/total:.1f}%)")
print(f"  Grounding-layer failures:  {grounding_fail} ({100*grounding_fail/total:.1f}%)")
print(f"  Generation-layer failures: {generation_fail} ({100*generation_fail/total:.1f}%)")
print(f"  Dataset/benchmark issues:  {dataset_fail} ({100*dataset_fail/total:.1f}%)")
print(f"  Other:                     {other_fail} ({100*other_fail/total:.1f}%)")

print(f"\n=== ORACLE CEILINGS ===")
print(f"  Oracle@corpus (evidence exists anywhere): {oracle_hits}/{total} ({100*oracle_hits/total:.1f}%)")
total_answerable = len(answerable)
currently_correct = sum(1 for r in answerable if r["meets_class_expectation"])
print(f"  Current answerable coverage: {currently_correct}/{total_answerable} = {100*currently_correct/total_answerable:.1f}%")

# Retrieval ceiling: how many would pass if retrieval were perfect (oracle)?
retrieval_ceiling = currently_correct + oracle_hits
print(f"  Retrieval ceiling (perfect retrieval): {retrieval_ceiling}/{total_answerable} = {100*retrieval_ceiling/total_answerable:.1f}%")

# Grounding ceiling: evidence in top5 but grounding blocked it
grounding_blocked = sum(1 for r in results if r["rel_in_top5"] and not r["grounding_would_pass"])
grounding_passed = sum(1 for r in results if r["rel_in_top5"] and r["grounding_would_pass"])
print(f"  Evidence-in-top5 but grounding blocked: {grounding_blocked}")
print(f"  Evidence-in-top5 and grounding would pass: {grounding_passed}")

# Generation ceiling: after grounding passes, how many get lost in structured output
gen_blocked = sum(1 for r in results if r["primary_failure"] == "J. GENERATION_STRUCTURED_OUTPUT")
print(f"  Generation blocked after grounding pass: {gen_blocked}")
