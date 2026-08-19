import re


class ArabicTextProcessor:
    def _normalize(self, text: str) -> str:
        # Normalize common Alef variants without flattening all morphology.
        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ى", "ي", text)

        # Remove tatweel.
        text = text.replace("ـ", "")

        # Normalize inline spaces while preserving paragraph/newline structure.
        text = text.replace("\r\n", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _repair_fragmented_lines(self, normalized_text: str) -> tuple[str, list[dict[str, str]]]:
        """Conservatively stitch obvious short line fragments.

        This does not add characters or perform global reordering.
        It only removes hard breaks where isolated 1-2 letter Arabic lines
        split nearby content.
        """
        lines = normalized_text.split("\n")
        repaired: list[str] = []
        repairs: list[dict[str, str]] = []
        index = 0

        while index < len(lines):
            line = lines[index].strip()

            if not line:
                repaired.append("")
                index += 1
                continue

            has_next = index + 1 < len(lines)
            next_line = lines[index + 1].strip() if has_next else ""

            if re.fullmatch(r"[\u0621-\u064A]{1,2}", line) and has_next and next_line:
                if repaired and repaired[-1].strip():
                    left = repaired[-1].rstrip()
                    merged = f"{left} {line} {next_line}"
                    repaired[-1] = merged
                    repairs.append(
                        {
                            "reason": "short_fragment_wrap",
                            "confidence": "high",
                            "left": left,
                            "right": f"{line} {next_line}",
                            "merged": merged,
                        }
                    )
                else:
                    merged = f"{line} {next_line}"
                    repaired.append(merged)
                    repairs.append(
                        {
                            "reason": "short_fragment_wrap",
                            "confidence": "high",
                            "left": "",
                            "right": f"{line} {next_line}",
                            "merged": merged,
                        }
                    )
                index += 2
                continue

            repaired.append(line)
            index += 1

        return "\n".join(repaired), repairs

    def process_with_diagnostics(self, extracted_text: str) -> dict:
        normalized_text = self._normalize(extracted_text)
        repaired_text, repairs = self._repair_fragmented_lines(normalized_text)

        return {
            "original_extracted_text": extracted_text,
            "normalized_text": normalized_text,
            "repaired_text": repaired_text,
            "repair_count": len(repairs),
            "repairs": repairs,
        }

    def process(self, text: str) -> str:
        diagnostics = self.process_with_diagnostics(text)
        return diagnostics["repaired_text"]
