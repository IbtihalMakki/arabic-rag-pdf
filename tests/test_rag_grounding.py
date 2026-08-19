import unittest

from src.arabic_rag.chunker import TextChunker
from src.arabic_rag.generator import AnswerGenerator


class TestRagGrounding(unittest.TestCase):

    def _make_generator_stub(self) -> AnswerGenerator:
        # Bypass heavy model loading; we only test grounding logic.
        return AnswerGenerator.__new__(AnswerGenerator)

    def test_supported_fact_preserved(self) -> None:
        generator = self._make_generator_stub()

        context = (
            "وُلد غازي عبد الرحمن القصييبي في الثاني من مارس عام 1940 في الأحساء، "
            "وهو أحد أبرز الشخصيات الأدبية والسياسية."
        )

        retrieved = [{"chunk": context, "score": 3.0}]

        generator._run_model = lambda messages: (
            '{"supported": true, "answer": "غازي القصييبي كاتب سعودي.", '
            '"sources": ["S1"], '
            '"evidence": ["وُلد غازي عبد الرحمن القصييبي في الثاني من مارس عام 1940 في الأحساء، وهو أحد أبرز الشخصيات الأدبية والسياسية."]}'
        )

        answer = generator.generate(
            question="من هو غازي القصيبي؟",
            context=context,
            retrieved_chunks=retrieved,
        )

        self.assertIn("1940", answer)
        self.assertTrue("الأحساء" in answer or "الاحساء" in answer)
        self.assertIn("المصادر: S1", answer)

    def test_unsupported_fact_refuses_instead_of_hallucinating(self) -> None:
        generator = self._make_generator_stub()

        context = "غازي القصييبي أديب وسياسي سعودي."
        retrieved = [{"chunk": context, "score": 2.0}]

        generator._run_model = lambda messages: (
            '{"supported": false, '
            '"answer": "لا توجد معلومات كافية في المستند للإجابة عن هذا السؤال.", '
            '"sources": [], "evidence": []}'
        )

        answer = generator.generate(
            question="ما اسم زوجة غازي القصيبي؟",
            context=context,
            retrieved_chunks=retrieved,
        )

        self.assertEqual(answer, AnswerGenerator.INSUFFICIENT_INFO_MESSAGE)

    def test_invalid_hallucinated_payload_falls_back_to_context(self) -> None:
        generator = self._make_generator_stub()

        context = "وُلد غازي القصييبي عام 1940 في الأحساء."
        retrieved = [{"chunk": context, "score": 2.5}]

        # Hallucinated year in answer; evidence does not support it.
        generator._run_model = lambda messages: (
            '{"supported": true, '
            '"answer": "ولد عام 1896 في الرياض.", '
            '"sources": ["S1"], '
            '"evidence": ["وُلد غازي القصييبي عام 1940 في الأحساء."]}'
        )

        answer = generator.generate(
            question="من هو غازي القصيبي؟",
            context=context,
            retrieved_chunks=retrieved,
        )

        self.assertIn("1940", answer)
        self.assertNotIn("1896", answer)
        self.assertNotIn("الرياض", answer)

    def test_reference_chunk_filtered_before_retrieval(self) -> None:
        chunker = TextChunker(chunk_size=240, chunk_overlap=0)

        text = (
            "غازي عبد الرحمن القصييبي هو أحد أبرز الشخصيات الأدبية والسياسية.\n\n"
            "المصادر\n"
            "1 - https://www.aldiwan.net/cat\n"
            "2 - https://ar.wikipedia.org/wiki/ghazi\n"
        )

        chunks = chunker.split(text)
        combined = "\n".join(chunks)

        self.assertIn("غازي عبد الرحمن القصييبي", combined)
        self.assertNotIn("https://www.aldiwan.net/cat", combined)


if __name__ == "__main__":
    unittest.main()
