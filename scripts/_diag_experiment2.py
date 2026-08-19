import json
d = json.load(open("bottleneck_experiment_results.json", encoding="utf-8"))
recs = d["detailed_records"]

# Show detailed B outputs for 6 specific diagnostic cases
cases = [
    "في أي مدينة تشير الوثيقة إلى مولده؟",
    "متى وأين ولد غازي القصيبي؟",
    "ما الذي تذكره الخاتمة عن غازي القصيبي؟",
    "هل تصفه الوثيقة بأنه أديب؟",
    "ما المناصب الحكومية التي شغلها غازي القصيبي؟",
    "هل تذكر الوثيقة تاريخ وفاة غازي القصيبي؟",
]

for q in cases:
    for cond in ["B", "C"]:
        r = next((x for x in recs if x["question"] == q and x["condition"] == cond), None)
        if not r:
            continue
        print("=== Cond", cond, "|", q[:60], "===")
        print("  json_valid:", r["json_valid"], "| hallucination:", r["hallucination_detected"])
        print("  grounding:", r["grounding_decision"])
        print("  answer:", r["final_answer"][:120])
        print("  raw_out:", r["raw_model_output"][:200])
        print()

# Safety control check
print("=== SAFETY CONTROLS ===")
safety = [r for r in recs if r["condition"] in ("B_safety", "C_safety")]
for r in safety:
    print("  Cond", r["condition"], "|", r["question"][:50], "->", r["behavior"], "(unsafe=", r["behavior"] == "answered", ")")

# Count for report
b_ans = [r for r in recs if r["condition"] == "B" and r.get("expected_class") in ("A", "B")]
c_ans = [r for r in recs if r["condition"] == "C" and r.get("expected_class") in ("A", "B")]
print()
print("B answerable summary:", {d: sum(1 for r in b_ans if r["behavior"] == d) for d in ["answered", "partial", "refusal"]})
print("C answerable summary:", {d: sum(1 for r in c_ans if r["behavior"] == d) for d in ["answered", "partial", "refusal"]})
print("B hallucinations:", sum(1 for r in b_ans if r["hallucination_detected"]), "/", len(b_ans))
print("C hallucinations:", sum(1 for r in c_ans if r["hallucination_detected"]), "/", len(c_ans))
print("B json_valid:", sum(1 for r in b_ans if r["json_valid"]), "/", len(b_ans))
print("C json_valid:", sum(1 for r in c_ans if r["json_valid"]), "/", len(c_ans))
print("B grounding_decisions:", {k: sum(1 for r in b_ans if r["grounding_decision"] == k) for k in set(r["grounding_decision"] for r in b_ans)})
