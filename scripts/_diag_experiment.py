import json
d = json.load(open("bottleneck_experiment_results.json", encoding="utf-8"))

print("=== STATISTICS A/B/C ===")
for cond in ["A", "B", "C"]:
    s = d["statistics"][cond]
    print("  Cond", cond,
          "coverage=", s.get("answerable_coverage"),
          "json_valid=", s.get("json_valid_rate"),
          "hallucination=", s.get("hallucination_rate"),
          "unsafe=", s.get("unsafe_answer_rate"))
print()

print("=== ORACLE B/C DETAIL FOR FAILING QUESTIONS ===")
pq = d["per_question_comparison"]
failing_qs = [
    "متى وأين ولد غازي القصيبي؟",
    "هل تذكر الوثيقة تاريخ وفاة غازي القصيبي؟",
    "في أي مدينة تشير الوثيقة إلى مولده؟",
    "ما المناصب الحكومية التي شغلها غازي القصيبي؟",
    "اذكر الأعمال الأدبية المذكورة في الوثيقة.",
    "ماذا تذكر المقدمة عن غازي القصيبي؟",
    "ما الذي تذكره الخاتمة عن غازي القصيبي؟",
    "لخص نشأته وتعليمه ومناصبه الحكومية.",
    "أين ولد وماذا درس وما أبرز أعماله؟",
    "هل تصفه الوثيقة بأنه أديب؟",
    "أين ولد ومتى توفي؟",
    "What does the document say about Ghazi Al-Gosaibi?",
    "ما هي Government roles المذكورة؟",
    "اذكر السنة المذكورة لميلاده.",
    "ما هو رابط ويكيبيديا المذكور؟",
]
for q in failing_qs:
    if q in pq:
        p = pq[q]
        print("Q:", q[:60])
        print("  A:", p["A_behavior"], "B:", p["B_behavior"], "C:", p["C_behavior"])
        print("  B_grounding:", p["B_decision"])
        print("  B_json_valid:", p["B_json_valid"], "B_hallucination:", p["B_hallucination"])
        print("  C_json_valid:", p["C_json_valid"], "C_hallucination:", p["C_hallucination"])
        print()

# Check grounding rejection breakdown for B
print("=== GROUNDING REJECTION ANALYSIS FOR CONDITION B ===")
recs = d["detailed_records"]
b_recs = [r for r in recs if r["condition"] == "B" and r.get("expected_class") in ("A", "B")]
decisions = {}
for r in b_recs:
    dec = r["grounding_decision"]
    decisions[dec] = decisions.get(dec, 0) + 1
print("B grounding decisions:", decisions)

c_recs = [r for r in recs if r["condition"] == "C" and r.get("expected_class") in ("A", "B")]
decisions_c = {}
for r in c_recs:
    dec = r["grounding_decision"]
    decisions_c[dec] = decisions_c.get(dec, 0) + 1
print("C grounding decisions:", decisions_c)

# Raw model output sample for condition B failing questions
print()
print("=== SAMPLE RAW MODEL OUTPUTS (CONDITION B, FAILING) ===")
for r in b_recs:
    if r["behavior"] == "refusal" and r.get("json_valid") is not None:
        mo = r.get("raw_model_output", "")[:200]
        print("Q:", r["question"][:55])
        print("  json_valid:", r["json_valid"], "hallucination:", r["hallucination_detected"])
        print("  model_out:", mo[:150])
        print()
