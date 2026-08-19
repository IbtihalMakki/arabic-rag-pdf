"""Vocabulary bridge: maps natural-language query terms to corpus fragment forms.

RETRIEVAL ONLY.  This module expands BM25 sparse-retrieval query vocabulary to
bridge the gap between standard Arabic question vocabulary and fragmented/
truncated tokens produced by PDF text extraction.

Guarantees:
- Source chunk text is NEVER modified.
- Bridge aliases are NEVER passed to the grounding validator as evidence.
- Bridge aliases are NEVER used to fabricate factual content.
- Only operates at retrieval time (BM25 query-term expansion).

Root-cause documentation (see RAG_FAILURE_ATTRIBUTION_REPORT.md):
    Natural query term      →  Extracted corpus token(s) found
    ───────────────────────────────────────────────────────────
    ميلاد / مولد / ولاده   →  ولد / وُلد
    وزير / وزراء / وزاره   →  وز  (truncated during PDF extraction)
    أديب / أدبية           →  الادب / اعماله / ادب
    نشأة / نشأته           →  بيئه / عائله / كنف
    تعليم / دراسة          →  التعل / جامعه / الحقوق
    وفاة / توفي            →  تو2010
    government / roles     →  مناصب / حكوم / وز
    Gosaibi / Ghazi        →  القصييبي / غازي
"""

from __future__ import annotations

import re
from typing import Any


# ── Documented alias map ────────────────────────────────────────────────────────
# Keys are NORMALIZED natural-language query terms (diacritics stripped, alef
# variants unified, tatweel removed, same normalization as _normalize_token).
# Values are SETS of normalized corpus fragment tokens confirmed to be present
# in the indexed chunks for this document.
#
# IMPORTANT: Add entries here ONLY when supported by documented evidence in the
# failure attribution report.  This map is intentionally minimal.
_RAW_ALIAS_MAP: dict[str, set[str]] = {

    # ── Birth / date / year ───────────────────────────────────────────────────
    "ميلاد":     {"ولد"},
    "مولد":      {"ولد"},
    "ولاده":     {"ولد"},
    "مواليد":    {"ولد"},
    "لميلاده":   {"ولد"},
    "مولده":     {"ولد", "الاحساء"},

    # ── Death / end ──────────────────────────────────────────────────────────
    "وفاه":      {"تو2010", "الخاتمه"},
    "وفات":      {"تو2010"},

    # ── Government roles (truncated to "وز" in corpus) ───────────────────────
    "وزير":      {"وز", "مناصب"},
    "وزراء":     {"وز", "مناصب"},
    "وزاره":     {"وز", "مناصب"},
    "وزاري":     {"وز", "مناصب"},
    "roles":     {"مناصب", "حكوم", "وز"},
    "government": {"مناصب", "حكوم"},

    # ── Literary identity (truncated "ادب" form in corpus) ──────────────────
    "اديب":      {"الادب", "اعماله"},
    "اديبه":     {"الادب", "اعماله"},
    "ادبي":      {"الادب"},
    "ادبيه":     {"الادب", "اعماله"},

    # ── Education / study ────────────────────────────────────────────────────
    "تعليم":     {"التعل", "جامعه", "الحقوق"},
    "دراسه":     {"التعل", "جامعه", "الحقوق"},
    "تعلم":      {"التعل", "جامعه"},

    # ── Origin / upbringing ──────────────────────────────────────────────────
    "نشاه":      {"بيئه", "عائله", "كنف"},
    "نشات":      {"بيئه", "عائله", "كنف"},
    "نشاته":     {"بيئه", "عائله", "كنف"},

    # ── Works / literary output ───────────────────────────────────────────────
    "روايات":    {"روا", "الروا"},
    "قصص":       {"روا", "الروا"},
    "مؤلفات":    {"الادب", "اعماله", "روا"},

    # ── English entity bridge (for K-mixed questions) ─────────────────────────
    "ghazi":     {"غازي"},
    "gosaibi":   {"القصييبي", "القصيبي"},
    "gosabi":    {"القصييبي", "القصيبي"},
    "algosaibi": {"القصييبي", "غازي"},
}


def _normalize(token: str) -> str:
    """Same normalization as Retriever._normalize_token, duplicated here to
    keep VocabularyBridge self-contained and avoid circular imports."""
    token = token.strip().lower()
    token = re.sub(r"[^\u0621-\u064Aa-z0-9]", "", token)
    token = token.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    token = token.replace("ى", "ي")
    token = token.replace("ة", "ه")
    token = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", token)
    token = re.sub(r"(.)\1+", r"\1", token)
    return token


# Pre-normalize the alias map once at module load time.
ALIAS_MAP: dict[str, set[str]] = {
    _normalize(k): {_normalize(v) for v in vs}
    for k, vs in _RAW_ALIAS_MAP.items()
    if _normalize(k)
}


class VocabularyBridge:
    """Expands BM25 query vocabulary using documented corpus-gap aliases.

    Usage
    -----
    bridge = VocabularyBridge(corpus_vocab)
    expanded, activated = bridge.expand(normalized_query_terms)

    Parameters
    ----------
    corpus_vocab : set[str]
        Set of normalized tokens that actually appear in the indexed corpus.
        Used to filter out bridge aliases that would produce zero BM25 hits.
    """

    def __init__(self, corpus_vocab: set[str] | None = None) -> None:
        self._corpus_vocab = corpus_vocab if corpus_vocab is not None else set()
        # Restrict alias targets to tokens actually present in the corpus.
        # When corpus_vocab is None (not provided), keep all aliases.
        # When corpus_vocab is provided (even if empty), filter strictly.
        filter_by_corpus = corpus_vocab is not None
        self._effective_map: dict[str, set[str]] = {}
        for query_term, aliases in ALIAS_MAP.items():
            if filter_by_corpus:
                live_aliases = aliases & self._corpus_vocab
            else:
                live_aliases = aliases
            if live_aliases:
                self._effective_map[query_term] = live_aliases

    def expand(
        self,
        normalized_query_terms: set[str],
    ) -> tuple[set[str], list[dict[str, Any]]]:
        """Expand query terms with corpus-fragment aliases.

        Returns
        -------
        expanded_terms : set[str]
            Union of original terms plus activated bridge aliases.
        activated_bridges : list[dict]
            Diagnostics: one entry per fired bridge term, containing
            ``query_term``, ``aliases``, and ``was_bridged``.
        """
        expanded = set(normalized_query_terms)
        activated: list[dict[str, Any]] = []

        for qt in normalized_query_terms:
            aliases = self._effective_map.get(qt)
            if aliases:
                expanded |= aliases
                activated.append(
                    {
                        "query_term": qt,
                        "aliases": sorted(aliases),
                        "was_bridged": True,
                    }
                )

        return expanded, activated

    def corpus_vocab_size(self) -> int:
        return len(self._corpus_vocab)

    def active_bridges(self) -> int:
        return len(self._effective_map)
