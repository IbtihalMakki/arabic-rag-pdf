import unittest

from src.arabic_rag.chunker import TextChunker
from src.arabic_rag.generator import AnswerGenerator
from src.arabic_rag.loader import PDFLoader


class TestRagQuality(unittest.TestCase):
    def _make_generator_stub(self) -> AnswerGenerator:
        return AnswerGenerator.__new__(AnswerGenerator)

    def test_supported_question_returns_grounded_content(self) -> None:
        generator = self._make_generator_stub()

        context = (
            "غازي عبد الرحمن القصييبي هو أحد أبرز الشخصيات الأدبية. "
            "وُلد عام 1940 في الأحساء."
        )
        retrieved = [{"chunk": context, "score": 2.0}]

        # Force model output to be invalid so fallback path is exercised.
        generator._run_model = lambda messages: "{invalid"

        answer = generator.generate(
            question="من هو غازي القصيبي؟",
            context=context,
            retrieved_chunks=retrieved,
        )

        self.assertIn("غازي عبد الرحمن القصييبي", answer)
        self.assertIn("المصادر: S1", answer)

    def test_unsupported_attribute_question_refuses(self) -> None:
        generator = self._make_generator_stub()

        context = "غازي عبد الرحمن القصييبي هو أحد أبرز الشخصيات الأدبية."
        retrieved = [{"chunk": context, "score": 1.0}]

        generator._run_model = lambda messages: "{invalid"

        answer = generator.generate(
            question="ما لون عيني غازي القصيبي؟",
            context=context,
            retrieved_chunks=retrieved,
        )

        self.assertEqual(answer, AnswerGenerator.INSUFFICIENT_INFO_MESSAGE)

    def test_numeric_grounding_rejects_unsupported_numbers(self) -> None:
        generator = self._make_generator_stub()

        context = "وُلد غازي عبد الرحمن القصييبي عام 1940 في الأحساء."
        retrieved = [{"chunk": context, "score": 3.0}]

        generator._run_model = lambda messages: (
            '{"supported": true, "answer": "ولد عام 1896.", '
            '"sources": ["S1"], '
            '"evidence": ["وُلد غازي عبد الرحمن القصييبي عام 1940 في الأحساء."]}'
        )

        answer = generator.generate(
            question="متى ولد غازي القصيبي؟",
            context=context,
            retrieved_chunks=retrieved,
        )

        self.assertNotIn("1896", answer)
        self.assertTrue(
            "1940" in answer
            or answer == AnswerGenerator.INSUFFICIENT_INFO_MESSAGE
        )

    def test_source_validation_rejects_nonexistent_quotes(self) -> None:
        generator = self._make_generator_stub()

        context = "غازي القصييبي أديب وسياسي سعودي."
        retrieved = [{"chunk": context, "score": 2.0}]

        generator._run_model = lambda messages: (
            '{"supported": true, "answer": "إجابة", '
            '"sources": ["S1"], '
            '"evidence": ["اقتباس غير موجود في المصدر"]}'
        )

        answer = generator.generate(
            question="من هو غازي القصيبي؟",
            context=context,
            retrieved_chunks=retrieved,
        )

        self.assertIn("غازي القصييبي", answer)
        self.assertIn("المصادر: S1", answer)

    def test_arabic_output_quality_no_known_corruption(self) -> None:
        text = PDFLoader("data/pdfs/sample.pdf").load()

        self.assertNotIn("اغ يز ا يصقلبي", text)
        self.assertNotIn("0491", text)

    def test_reference_chunk_filtering(self) -> None:
        chunker = TextChunker(chunk_size=240, chunk_overlap=0)

        text = (
            "غازي عبد الرحمن القصييبي هو أحد أبرز الشخصيات الأدبية والسياسية.\n\n"
            "المصادر\n"
            "1 - https://www.aldiwan.net/cat\n"
            "2 - https://ar.wikipedia.org/wiki/ghazi\n"
        )

        chunks = chunker.split(text)
        joined = "\n".join(chunks)

        self.assertIn("غازي عبد الرحمن القصييبي", joined)
        self.assertNotIn("https://www.aldiwan.net/cat", joined)

    def test_mixed_rtl_ltr_not_globally_reversed(self) -> None:
        text = PDFLoader("data/pdfs/sample.pdf").load()

        self.assertIn("https://ar.wikipedia.org/wiki/", text)
        self.assertNotIn("sptth", text)


if __name__ == "__main__":
    unittest.main()
