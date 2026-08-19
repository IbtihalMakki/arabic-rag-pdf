"""Bridge activation diagnostic: verify bridge fires and measure BM25 impact
without running the full model."""
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
from src.arabic_rag.vocab_bridge import VocabularyBridge, ALIAS_MAP, _normalize

raw = PDFLoader("data/pdfs/sample.pdf").load()
repaired = ArabicTextProcessor().process(raw)
chunks_meta = TextChunker().split_with_metadata(repaired, "data/pdfs/sample.pdf")
chunks = [c["text"] for c in chunks_meta]

# Build a bare retriever stub (no model) just for BM25 + bridge
import math
from collections import defaultdict

class MockEmbedding:
    def encode(self, texts):
        import numpy as np
        return np.zeros((len(texts), 384))

class MockVS:
    def __init__(self, chunks_, metas_):
        self.chunks = chunks_
        self.metadatas = metas_
    def search(self, qe, k):
        return []

metas = [{"chunk_id": c["chunk_id"]} for c in chunks_meta]
vs = MockVS(chunks, metas)
r = Retriever.__new__(Retriever)
r.fusion_weights = {"dense_rrf": 0.0, "sparse_rrf": 1.0, "lexical": 0.0,
                    "entity": 0.0, "intent": 0.0, "reference_penalty": 0.35}
r.rrf_k = 60
r._bm25_k1 = 1.2
r._bm25_b = 0.75
r._build_sparse_index(chunks)
corpus_vocab = set()
for doc_tf in r._term_tf_by_doc:
    corpus_vocab |= doc_tf.keys()
r._vocab_bridge = VocabularyBridge(corpus_vocab)
r.vector_store = vs
r.embedding_model = MockEmbedding()

print("=== BRIDGE ACTIVATION AND BM25 IMPACT ===")
print(f"Corpus vocab size: {r._vocab_bridge.corpus_vocab_size()}")
print(f"Active bridges (corpus-filtered): {r._vocab_bridge.active_bridges()}")
print()

# Test each EXTRACTION_NOISE failing question
noise_questions = [
    "في أي مدينة تشير الوثيقة إلى مولده؟",
    "اذكر الأعمال الأدبية المذكورة في الوثيقة.",
    "لخص نشأته وتعليمه ومناصبه الحكومية.",
    "هل تصفه الوثيقة بأنه أديب؟",
    "ما هي Government roles المذكورة؟",
    "اذكر السنة المذكورة لميلاده.",
    "ما هو رابط ويكيبيديا المذكور؟",
]

for q in noise_questions:
    info = analyze_query(q)
    base_terms = r._normalize_terms(set(info.get("key_terms", [])))
    
    # Expand via query variants
    from src.arabic_rag.retriever import Retriever as R2
    variants = r._expand_query_variants(q, info)
    expanded = set(base_terms)
    for v in variants:
        expanded |= r._normalize_terms(r._tokenize(v))
    
    # Apply bridge
    bridged, activations = r._vocab_bridge.expand(expanded)
    
    # BM25 WITHOUT bridge
    scores_no_bridge = [(i+1, round(r._bm25_score(expanded, i), 4)) for i in range(len(chunks))]
    scores_no_bridge.sort(key=lambda x: x[1], reverse=True)
    
    # BM25 WITH bridge
    scores_bridge = [(i+1, round(r._bm25_score(bridged, i), 4)) for i in range(len(chunks))]
    scores_bridge.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Q: {q}")
    print(f"  base_terms: {sorted(base_terms)}")
    print(f"  bridge_activations: {activations}")
    print(f"  bridge_added_terms: {sorted(bridged - expanded)}")
    print(f"  BM25 top3 WITHOUT bridge: {scores_no_bridge[:3]}")
    print(f"  BM25 top3 WITH bridge:    {scores_bridge[:3]}")
    improved = any(s > 0 for _, s in scores_bridge[:3]) and not any(s > 0 for _, s in scores_no_bridge[:3])
    print(f"  IMPROVED (was zero, now positive): {improved}")
    print()
