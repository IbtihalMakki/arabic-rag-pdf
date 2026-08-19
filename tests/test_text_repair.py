import unittest

from src.arabic_rag.text_processor import ArabicTextProcessor


class TestTextRepair(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = ArabicTextProcessor()

    def test_obvious_broken_arabic_fragments_are_repaired(self) -> None:
        text = "كان هذا الشخص\nي\nمتميزا في عمله"
        diagnostics = self.processor.process_with_diagnostics(text)

        self.assertGreaterEqual(diagnostics["repair_count"], 1)
        self.assertIn("الشخص ي متميزا", diagnostics["repaired_text"])

    def test_mixed_rtl_ltr_preserved(self) -> None:
        text = "استخدم نظام API\nversion 2.0 للتجربة"
        repaired = self.processor.process(text)

        self.assertIn("API", repaired)
        self.assertIn("version 2.0", repaired)

    def test_numbers_and_years_preserved(self) -> None:
        text = "ولد عام\n1940 في الاحساء"
        repaired = self.processor.process(text)

        self.assertIn("1940", repaired)
        self.assertIn("ولد عام\n1940", repaired)

    def test_urls_are_preserved(self) -> None:
        text = "المصدر:\nhttps://example.com/path?q=1\nشرح اضافي"
        diagnostics = self.processor.process_with_diagnostics(text)

        repaired = diagnostics["repaired_text"]
        self.assertIn("https://example.com/path?q=1", repaired)
        self.assertIn("\nhttps://example.com/path?q=1\n", repaired)

    def test_english_terms_preserved(self) -> None:
        text = "Hybrid retrieval\nuses BM25 and FAISS"
        repaired = self.processor.process(text)

        self.assertIn("Hybrid retrieval", repaired)
        self.assertIn("uses BM25 and FAISS", repaired)

    def test_already_correct_text_not_forced_into_change(self) -> None:
        text = "غازي القصيبي هو اديب سعودي.\n\nولد عام 1940 في الاحساء."
        diagnostics = self.processor.process_with_diagnostics(text)

        self.assertEqual(diagnostics["repair_count"], 0)
        self.assertEqual(
            diagnostics["repaired_text"],
            "غازي القصيبي هو اديب سعودي.\n\nولد عام 1940 في الاحساء.",
        )

    def test_do_not_merge_heading_or_bullet_boundaries(self) -> None:
        text = "المقدمة\nكان النص هنا\n• بند اول\n• بند ثان"
        repaired = self.processor.process(text)

        self.assertIn("المقدمة\nكان النص هنا", repaired)
        self.assertIn("\n• بند اول\n• بند ثان", repaired)

    def test_non_short_fragments_are_not_force_merged(self) -> None:
        text = "كان هذا الشخص\nي متميزا\nفي عمله"
        repaired = self.processor.process(text)

        self.assertIn("الشخص\nي متميزا", repaired)


if __name__ == "__main__":
    unittest.main()
