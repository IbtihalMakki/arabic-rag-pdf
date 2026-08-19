import json
from src.arabic_rag.generator import AnswerGenerator
from src.arabic_rag.query_analysis import analyze_query

g = AnswerGenerator.__new__(AnswerGenerator)

d = json.load(open("audit_results.json", encoding="utf-8"))
recs = d["records"]
fails = [r for r in recs if r["expected_class"] in ("A", "B") and not r["meets_class_expectation"]]

print("=== GROUNDING FALLBACK ANALYSIS ===")
for r in fails:
    q = r["question"]
    info = analyze_query(q)
    chunks_top5 = r["retrieved"][:5]
    source_map = {f"S{i+1}": c["text"] for i, c in enumerate(chunks_top5)}

    support = g._collect_support_sentences(info, source_map)
    can_return = g._can_return_partial_answer(info, support) if support else False

    print("Q:", q[:70])
    print("  decision_was:", r["decision"])
    print("  key_terms:", info["key_terms"])
    print("  entities:", info["entities"])
    print("  support_sentences_found:", len(support))
    print("  can_return_partial:", can_return)
    if support:
        sq = support[0]["quote"]
        print("  first_support:", sq[:100])
    print()
