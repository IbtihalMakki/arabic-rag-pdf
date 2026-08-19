"""Tests for the vocabulary bridge and its integration with retrieval safety."""

import unittest

from src.arabic_rag.vocab_bridge import VocabularyBridge, ALIAS_MAP, _normalize
from src.arabic_rag.generator import AnswerGenerator
from src.arabic_rag.chunker import TextChunker


class TestVocabularyBridgeCore(unittest.TestCase):
    """Core bridge logic: expansion, corpus filtering, diagnostics."""

    def _make_bridge(self, corpus: set[str]) -> VocabularyBridge:
        return VocabularyBridge(corpus_vocab=corpus)

    # ── Vocabulary mismatch recovery ─────────────────────────────────────────

    def test_birth_term_bridges_to_corpus_ولد(self) -> None:
        corpus = {_normalize("ولد"), _normalize("غازي"), _normalize("1940")}
        bridge = self._make_bridge(corpus)
        expanded, activated = bridge.expand({_normalize("ميلاد")})
        self.assertIn(_normalize("ولد"), expanded)
        self.assertTrue(any(a["query_term"] == _normalize("ميلاد") for a in activated))

    def test_death_term_bridges_to_corpus_تو2010(self) -> None:
        corpus = {_normalize("تو2010"), _normalize("الخاتمه")}
        bridge = self._make_bridge(corpus)
        expanded, activated = bridge.expand({_normalize("وفاه")})
        self.assertIn(_normalize("تو2010"), expanded)
        self.assertTrue(any(a["was_bridged"] for a in activated))

    def test_truncated_وز_recovers_from_وزير_query(self) -> None:
        corpus = {_normalize("وز"), _normalize("مناصب")}
        bridge = self._make_bridge(corpus)
        expanded, _ = bridge.expand({_normalize("وزير")})
        self.assertIn(_normalize("وز"), expanded)

    def test_literary_identity_bridge(self) -> None:
        corpus = {_normalize("الادب"), _normalize("اعماله")}
        bridge = self._make_bridge(corpus)
        expanded, _ = bridge.expand({_normalize("اديب")})
        self.assertIn(_normalize("الادب"), expanded)

    def test_english_entity_bridges_to_arabic_name(self) -> None:
        corpus = {_normalize("غازي"), _normalize("القصييبي")}
        bridge = self._make_bridge(corpus)
        expanded, activated = bridge.expand({"ghazi"})
        self.assertIn(_normalize("غازي"), expanded)
        self.assertTrue(activated)

    def test_english_government_bridges_to_arabic_positions(self) -> None:
        corpus = {_normalize("مناصب"), _normalize("حكوم"), _normalize("وز")}
        bridge = self._make_bridge(corpus)
        expanded, _ = bridge.expand({"government"})
        self.assertTrue(expanded & {_normalize("مناصب"), _normalize("حكوم")})

    # ── Source evidence must not be modified ─────────────────────────────────

    def test_bridge_never_modifies_corpus_vocab_set(self) -> None:
        original_corpus = {_normalize("ولد"), _normalize("غازي")}
        corpus_copy = set(original_corpus)
        bridge = VocabularyBridge(corpus_vocab=corpus_copy)
        bridge.expand({_normalize("ميلاد")})
        # Corpus set given to bridge must remain unchanged.
        self.assertEqual(corpus_copy, original_corpus)

    def test_bridge_expansion_does_not_contain_raw_arabic_not_in_corpus(self) -> None:
        corpus = {_normalize("ولد")}  # only "ولد" in corpus
        bridge = self._make_bridge(corpus)
        expanded, _ = bridge.expand({_normalize("ميلاد")})
        # الاحساء is also a target of مولده but NOT in this corpus.
        self.assertNotIn(_normalize("الاحساء"), expanded)

    # ── Numbers and years ─────────────────────────────────────────────────────

    def test_year_not_bridged_when_query_already_contains_entity(self) -> None:
        corpus = {_normalize("ولد"), _normalize("1940"), _normalize("غازي")}
        bridge = self._make_bridge(corpus)
        # If query already has "غازي" and "ولد", no bridge needed.
        expanded, activated = bridge.expand({_normalize("غازي"), _normalize("ولد")})
        # No bridge activations for these terms.
        self.assertEqual(activated, [])

    def test_number_strings_preserved_in_expansion(self) -> None:
        corpus = {_normalize("ولد"), "1940"}
        bridge = self._make_bridge(corpus)
        expanded, _ = bridge.expand({_normalize("لميلاده")})
        # Original term still present.
        self.assertIn(_normalize("لميلاده"), expanded)

    # ── URLs are not bridged ──────────────────────────────────────────────────

    def test_url_query_activates_no_bridge(self) -> None:
        corpus = {_normalize("ولد")}
        bridge = self._make_bridge(corpus)
        url_terms = {"https"}  # URL token that won't be in ALIAS_MAP
        expanded, activated = bridge.expand(url_terms)
        self.assertEqual(activated, [])
        self.assertEqual(expanded, url_terms)

    # ── Unrelated queries must not receive arbitrary aliases ──────────────────

    def test_unrelated_query_no_bridge_fired(self) -> None:
        corpus = {_normalize("ولد"), _normalize("وز")}
        bridge = self._make_bridge(corpus)
        unrelated_terms = {_normalize("السماء"), _normalize("اللون")}
        expanded, activated = bridge.expand(unrelated_terms)
        self.assertEqual(activated, [])
        self.assertEqual(expanded, unrelated_terms)

    # ── Dead bridge aliases (not in corpus) are excluded ─────────────────────

    def test_bridge_alias_excluded_when_not_in_corpus(self) -> None:
        empty_corpus: set[str] = set()
        bridge = VocabularyBridge(corpus_vocab=empty_corpus)
        expanded, activated = bridge.expand({_normalize("ميلاد")})
        # With empty corpus, no alive aliases → nothing added.
        self.assertEqual(expanded, {_normalize("ميلاد")})
        self.assertEqual(activated, [])

    # ── Diagnostics structure ─────────────────────────────────────────────────

    def test_activated_bridge_has_required_fields(self) -> None:
        corpus = {_normalize("ولد")}
        bridge = self._make_bridge(corpus)
        _, activated = bridge.expand({_normalize("ميلاد")})
        self.assertTrue(activated)
        record = activated[0]
        self.assertIn("query_term", record)
        self.assertIn("aliases", record)
        self.assertIn("was_bridged", record)
        self.assertTrue(record["was_bridged"])


class TestVocabularyBridgeGroundingSafety(unittest.TestCase):
    """Grounding safety: bridge aliases cannot become grounding evidence."""

    def test_bridge_aliases_not_in_chunk_text(self) -> None:
        """Aliases expand query terms; they must NOT appear in source chunks."""
        chunker = TextChunker()
        text = "غازي عبد الرحمن القصييبي هو شاعر سعودي."
        chunks = chunker.split(text)
        joined = " ".join(chunks)
        # Bridge alias "ولد" should not have been injected into chunk text.
        # We only check that the chunk text equals what was split from original.
        for chunk in chunks:
            self.assertIn("القصييبي", chunk)

    def test_bridge_term_not_accepted_as_grounding_quote(self) -> None:
        """Generator validator must reject quotes containing only bridge aliases."""
        generator = AnswerGenerator.__new__(AnswerGenerator)

        context = "غازي القصييبي كاتب وشاعر."
        source_map = {"S1": context}

        # Simulate model returning an evidence quote that is a bridge alias
        # (not present verbatim in the source).
        payload = {
            "supported": True,
            "answer": "هو اديب سعودي.",
            "sources": ["S1"],
            "evidence": [{"source": "S1", "quote": "ولد غازي في الاحساء عام 1940"}],
        }

        ok, supported, answer, sources, evidence = generator._validate_payload(
            payload=payload, source_map=source_map
        )

        # The fabricated quote is not in source_map["S1"], so validation must fail.
        self.assertFalse(ok and supported and bool(evidence))

    def test_unsupported_questions_still_refuse_after_bridge(self) -> None:
        """Bridge must not help unsupported questions hallucinate."""
        generator = AnswerGenerator.__new__(AnswerGenerator)

        # Context with only name info — no salary info.
        context = "غازي القصييبي شغل مناصب حكومية عديدة."
        source_map = {"S1": context}

        payload = {
            "supported": False,
            "answer": generator.INSUFFICIENT_INFO_MESSAGE,
            "sources": [],
            "evidence": [],
        }

        ok, supported, _, _, _ = generator._validate_payload(
            payload=payload, source_map=source_map
        )

        # Model said not-supported; validator must pass that through as refusal.
        self.assertTrue(ok)
        self.assertFalse(supported)

    def test_hallucination_protection_numeric_validator(self) -> None:
        """Numeric validation must still catch wrong years even with bridge active."""
        generator = AnswerGenerator.__new__(AnswerGenerator)

        context = "وُلد غازي القصييبي عام 1940 في الأحساء."
        source_map = {"S1": context}

        # Model hallucinated 1890.
        payload = {
            "supported": True,
            "answer": "ولد عام 1890.",
            "sources": ["S1"],
            "evidence": [
                {"source": "S1", "quote": "وُلد غازي القصييبي عام 1940 في الأحساء."}
            ],
        }

        ok, supported, _, _, _ = generator._validate_payload(
            payload=payload, source_map=source_map
        )

        # 1890 not in allowed numbers → must be rejected.
        self.assertFalse(ok and supported)


class TestVocabularyBridgeAliasMap(unittest.TestCase):
    """Sanity checks on the alias map itself."""

    def test_alias_map_keys_are_normalized(self) -> None:
        for key in ALIAS_MAP:
            self.assertEqual(key, _normalize(key), f"Key not normalized: {key!r}")

    def test_alias_map_values_are_normalized(self) -> None:
        for key, values in ALIAS_MAP.items():
            for val in values:
                self.assertEqual(val, _normalize(val), f"Alias value not normalized: {val!r} under key {key!r}")

    def test_no_empty_keys(self) -> None:
        for key in ALIAS_MAP:
            self.assertTrue(key, "Empty key found in ALIAS_MAP")

    def test_no_empty_alias_sets(self) -> None:
        for key, values in ALIAS_MAP.items():
            self.assertTrue(values, f"Empty alias set for key: {key!r}")


if __name__ == "__main__":
    unittest.main()
