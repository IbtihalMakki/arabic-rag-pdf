import math
import re
from collections import defaultdict
from typing import Any

import numpy as np

from src.arabic_rag.embeddings import EmbeddingModel
from src.arabic_rag.query_analysis import analyze_query
from src.arabic_rag.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        fusion_weights: dict[str, float] | None = None,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

        # Rank-fusion weights. These are calibrated to keep dense semantics
        # while allowing sparse lexical matches to surface noisy Arabic spans.
        self.fusion_weights = {
            "dense_rrf": 0.50,
            "sparse_rrf": 0.50,
            "lexical": 0.20,
            "entity": 0.15,
            "intent": 0.15,
            "reference_penalty": 0.35,
        }

        if fusion_weights:
            self.fusion_weights.update(fusion_weights)

        self.rrf_k = 60

        # Sparse BM25-like index over the same chunk corpus.
        self._build_sparse_index(self.vector_store.chunks)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        normalized = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", text)
        return set(re.findall(r"[\u0621-\u064AA-Za-z0-9]+", normalized))

    @staticmethod
    def _normalize_token(token: str) -> str:
        token = token.strip().lower()
        token = re.sub(r"[^\u0621-\u064Aa-z0-9]", "", token)
        token = token.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        token = token.replace("ى", "ي")
        token = token.replace("ة", "ه")
        token = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", token)
        token = re.sub(r"(.)\1+", r"\1", token)
        return token

    def _normalize_terms(self, terms: set[str]) -> set[str]:
        return {
            self._normalize_token(term)
            for term in terms
            if term.strip()
        }

    @staticmethod
    def _reference_noise(chunk_text: str) -> bool:
        lowered = chunk_text.lower()
        return (
            "المصادر" in chunk_text
            or "http://" in lowered
            or "https://" in lowered
            or "www." in lowered
        )

    def _build_sparse_index(self, chunks: list[str]) -> None:
        self._bm25_k1 = 1.2
        self._bm25_b = 0.75

        self._doc_tokens: list[list[str]] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_len = 0.0

        self._term_doc_freq: dict[str, int] = defaultdict(int)
        self._term_tf_by_doc: list[dict[str, int]] = []

        for chunk in chunks:
            norm_tokens = list(self._normalize_terms(self._tokenize(chunk)))
            self._doc_tokens.append(norm_tokens)
            self._doc_lengths.append(len(norm_tokens))

            tf: dict[str, int] = defaultdict(int)
            for token in norm_tokens:
                tf[token] += 1
            self._term_tf_by_doc.append(tf)

            for token in set(norm_tokens):
                self._term_doc_freq[token] += 1

        if self._doc_lengths:
            self._avg_doc_len = sum(self._doc_lengths) / len(self._doc_lengths)

    def _bm25_score(self, query_terms: set[str], doc_idx: int) -> float:
        if not query_terms:
            return 0.0

        n_docs = max(1, len(self._doc_tokens))
        tf_doc = self._term_tf_by_doc[doc_idx]
        doc_len = max(1, self._doc_lengths[doc_idx])

        score = 0.0
        for term in query_terms:
            tf = tf_doc.get(term, 0)
            if tf <= 0:
                continue

            df = self._term_doc_freq.get(term, 0)
            idf = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))

            numer = tf * (self._bm25_k1 + 1.0)
            denom = tf + self._bm25_k1 * (
                1.0 - self._bm25_b + self._bm25_b * (doc_len / max(1e-9, self._avg_doc_len))
            )
            score += idf * (numer / max(1e-9, denom))

        return float(score)

    def _sparse_search(self, query_terms: set[str], k: int) -> list[dict[str, Any]]:
        if not query_terms or not self.vector_store.chunks:
            return []

        scored: list[tuple[int, float]] = []
        for idx in range(len(self.vector_store.chunks)):
            score = self._bm25_score(query_terms, idx)
            if score > 0.0:
                scored.append((idx, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        scored = scored[:k]

        out = []
        for idx, score in scored:
            out.append(
                {
                    "index": idx,
                    "chunk_id": idx + 1,
                    "chunk": self.vector_store.chunks[idx],
                    "score": float(score),
                    "metadata": self.vector_store.metadatas[idx] if idx < len(self.vector_store.metadatas) else {},
                }
            )
        return out

    def _intent_score(
        self,
        intents: list[str],
        chunk_text: str,
        metadata: dict[str, Any],
    ) -> float:
        text = chunk_text
        score = 0.0

        if "temporal" in intents and re.search(r"\d{3,4}", text):
            score += 0.35

        if "location" in intents and re.search(r"\b(في|ب|مدينة|منطقة|المملكة|الأحساء|الاحساء)\b", text):
            score += 0.30

        if "list" in intents and ("•" in text or ":" in text):
            score += 0.35

        if "education" in intents and re.search(r"جامعة|تعليم|دراسة|الحقوق", text):
            score += 0.35

        if "works" in intents and re.search(r"روا|ديوان|كتب|قصائد|الشعر", text):
            score += 0.35

        if "identity" in intents and re.search(r"\b(هو|احد|أحد|ولد|وُلد)\b", text):
            score += 0.30

        section = str(metadata.get("section", ""))
        if section and any(intent in {"works", "education", "list"} for intent in intents):
            if ("نش" in section or "تعليم" in section or "حياه" in section or "حياة" in section):
                score += 0.10

        return min(score, 1.0)

    @staticmethod
    def _rrf_score(rank: int | None, k: int) -> float:
        if rank is None:
            return 0.0
        return 1.0 / (k + rank)

    def _expand_query_variants(self, query: str, query_info: dict[str, Any]) -> list[str]:
        intents = set(query_info.get("intents", []))
        variants = [query]

        if "temporal" in intents:
            variants.append(f"{query} تاريخ سنة عام")

        if "location" in intents:
            variants.append(f"{query} مكان مدينة الاحساء")

        if "list" in intents:
            variants.append(f"{query} مناصب اعمال نقاط")

        if "education" in intents:
            variants.append(f"{query} تعليم جامعة الحقوق")

        if "works" in intents:
            variants.append(f"{query} روايات شعر ديوان اعمال")

        return variants

    def suggest_k(self, query: str) -> int:
        info = analyze_query(query)
        intents = set(info.get("intents", []))
        tokens = info.get("tokens", [])

        if any(intent in intents for intent in {"list", "works", "education"}):
            return 5

        if "temporal" in intents and "location" in intents:
            return 5

        if len(tokens) >= 7 or query.count("و") >= 2:
            return 5

        if any(intent in intents for intent in {"location", "temporal"}):
            return 4

        return 3

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        candidate_multiplier: int = 4,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        if not self.vector_store.chunks:
            return []

        query_info = analyze_query(query)

        target_k = k if k is not None else self.suggest_k(query)
        target_k = max(1, min(target_k, len(self.vector_store.chunks)))

        candidate_k = max(target_k * candidate_multiplier, 8)
        candidate_k = min(candidate_k, len(self.vector_store.chunks))

        base_query_terms = self._normalize_terms(set(query_info.get("key_terms", [])))
        entity_terms = self._normalize_terms(set(query_info.get("entities", [])))
        intents = query_info.get("intents", [])

        query_variants = self._expand_query_variants(query, query_info)

        expanded_query_terms = set(base_query_terms)
        for variant in query_variants:
            expanded_query_terms |= self._normalize_terms(self._tokenize(variant))

        query_terms = expanded_query_terms

        dense_variant_embeddings = self.embedding_model.encode(query_variants)
        query_embedding = np.mean(dense_variant_embeddings, axis=0)

        dense_results = self.vector_store.search(query_embedding, k=candidate_k)
        sparse_results = self._sparse_search(query_terms, k=candidate_k)

        dense_rank = {item["index"]: rank for rank, item in enumerate(dense_results, start=1)}
        sparse_rank = {item["index"]: rank for rank, item in enumerate(sparse_results, start=1)}

        dense_score_by_idx = {item["index"]: float(item["score"]) for item in dense_results}
        sparse_score_by_idx = {item["index"]: float(item["score"]) for item in sparse_results}

        candidate_indices = set(dense_rank) | set(sparse_rank)

        fused: list[dict[str, Any]] = []
        for idx in candidate_indices:
            chunk_text = self.vector_store.chunks[idx]
            metadata = self.vector_store.metadatas[idx] if idx < len(self.vector_store.metadatas) else {}

            chunk_terms = self._normalize_terms(self._tokenize(chunk_text))

            lexical = 0.0
            if query_terms:
                lexical = len(query_terms & chunk_terms) / len(query_terms)

            entity = 0.0
            if entity_terms:
                entity = len(entity_terms & chunk_terms) / len(entity_terms)

            intent = self._intent_score(intents, chunk_text, metadata)
            reference_penalty = 1.0 if self._reference_noise(chunk_text) else 0.0

            dense_rrf = self._rrf_score(dense_rank.get(idx), self.rrf_k)
            sparse_rrf = self._rrf_score(sparse_rank.get(idx), self.rrf_k)

            fused_score = (
                self.fusion_weights["dense_rrf"] * dense_rrf
                + self.fusion_weights["sparse_rrf"] * sparse_rrf
                + self.fusion_weights["lexical"] * lexical
                + self.fusion_weights["entity"] * entity
                + self.fusion_weights["intent"] * intent
                - self.fusion_weights["reference_penalty"] * reference_penalty
            )

            fused.append(
                {
                    "index": idx,
                    "chunk": chunk_text,
                    "score": float(fused_score),
                    "fused_score": float(fused_score),
                    "dense_score": dense_score_by_idx.get(idx, 0.0),
                    "sparse_score": sparse_score_by_idx.get(idx, 0.0),
                    "chunk_id": idx + 1,
                    "metadata": metadata,
                    "features": {
                        "dense_rank": dense_rank.get(idx),
                        "sparse_rank": sparse_rank.get(idx),
                        "dense_rrf": round(dense_rrf, 6),
                        "sparse_rrf": round(sparse_rrf, 6),
                        "lexical": round(lexical, 4),
                        "entity": round(entity, 4),
                        "intent": round(intent, 4),
                        "reference_penalty": reference_penalty,
                    },
                }
            )

        fused.sort(key=lambda item: item["fused_score"], reverse=True)

        return fused[:target_k]
