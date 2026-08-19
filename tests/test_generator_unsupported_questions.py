import unittest

from src.arabic_rag.generator import AnswerGenerator


class TestGeneratorUnsupportedQuestions(unittest.TestCase):
    def _stub(self) -> AnswerGenerator:
        return AnswerGenerator.__new__(AnswerGenerator)

    def test_doctor_question_refuses_without_support(self) -> None:
        generator = self._stub()
        context = "غازي عبد الرحمن القصييبي أديب وسياسي سعودي."
        retrieved = [{"chunk": context, "score": 1.0}]

        generator._run_model = lambda messages: "{invalid"

        answer = generator.generate(
            question="هل كان غازي القصيبي طبيبًا؟",
            context=context,
            retrieved_chunks=retrieved,
        )

        self.assertEqual(answer, AnswerGenerator.INSUFFICIENT_INFO_MESSAGE)

    def test_salary_question_refuses_without_support(self) -> None:
        generator = self._stub()
        context = "شغل القصييبي مناصب حكومية عديدة."
        retrieved = [{"chunk": context, "score": 1.0}]

        generator._run_model = lambda messages: "{invalid"

        answer = generator.generate(
            question="كم كان راتبه عندما كان وزيرًا؟",
            context=context,
            retrieved_chunks=retrieved,
        )

        self.assertEqual(answer, AnswerGenerator.INSUFFICIENT_INFO_MESSAGE)


if __name__ == "__main__":
    unittest.main()
