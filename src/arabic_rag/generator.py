import json
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.arabic_rag.query_analysis import analyze_query


class AnswerGenerator:
    INSUFFICIENT_INFO_MESSAGE = (
        "\u0644\u0627 \u062a\u0648\u062c\u062f \u0645\u0639\u0644\u0648\u0645\u0627\u062a \u0643\u0627\u0641\u064a\u0629 \u0641\u064a \u0627\u0644\u0645\u0633\u062a\u0646\u062f \u0644\u0644\u0625\u062c\u0627\u0628\u0629 \u0639\u0646 \u0647\u0630\u0627 \u0627\u0644\u0633\u0624\u0627\u0644."
    )

    SOURCE_PREFIX = "\u0627\u0644\u0645\u0635\u0627\u062f\u0631: "

    STOPWORDS = {
        "\u0645\u0646", "\u0645\u0627", "\u0645\u0627\u0630\u0627", "\u0645\u062a\u0649", "\u0627\u064a\u0646", "\u0623\u064a\u0646", "\u0647\u0644", "\u0647\u0648", "\u0647\u064a",
        "\u0639\u0646", "\u0641\u064a", "\u0639\u0644\u0649", "\u0627\u0644\u0649", "\u0625\u0644\u0649", "\u0643\u0645", "\u0627\u0644\u062a\u064a", "\u0627\u0644\u0630\u064a", "\u0647\u0630\u0647",
        "\u0630\u0644\u0643", "\u0643\u0627\u0646", "\u0643\u0627\u0646\u062a", "\u0627\u0630\u0627", "\u0625\u0630\u0627", "\u0627\u0648", "\u0623\u0648", "\u0642\u062f", "\u0644\u0642\u062f",
        "\u0627\u0644\u0648\u062b\u064a\u0642\u0629", "\u0627\u0644\u0645\u0633\u062a\u0646\u062f", "\u0627\u0639\u062a\u0645\u0627\u062f\u0627", "\u0627\u0639\u062a\u0645\u0627\u062f\u064b\u0627", "\u0633\u0624\u0627\u0644", "\u0625\u062c\u0627\u0628\u0629",
    }

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Generator device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _extract_numbers(text: str) -> set[str]:
        return set(re.findall(r"\d+", text))

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        normalized = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", text)
        return set(re.findall(r"[\u0621-\u064AA-Za-z0-9]+", normalized))

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        pieces = re.split(r"(?<=[\.\!\?\u061f\n])\s+", text)
        return [item.strip() for item in pieces if item.strip()]

    @staticmethod
    def _normalize_token(token: str) -> str:
        token = token.strip().lower()
        token = re.sub(r"[^\u0621-\u064Aa-z0-9]", "", token)
        token = token.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        token = token.replace("ى", "ي")
        token = token.replace("ة", "ه")
        token = re.sub(r"[\u064B-\u0652]", "", token)
        token = re.sub(r"(.)\1+", r"\1", token)
        return token

    def _normalized_terms(self, tokens: set[str]) -> set[str]:
        return {self._normalize_token(token) for token in tokens if token.strip()}

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"```$", "", cleaned)
        return cleaned.strip()

    def _build_source_map(
        self,
        context: str,
        retrieved_chunks: list[dict[str, Any]] | None,
    ) -> dict[str, str]:
        if retrieved_chunks:
            return {
                f"S{index}": item["chunk"]
                for index, item in enumerate(retrieved_chunks, start=1)
            }
        return {"S1": context}

    def _build_prompt_context(self, source_map: dict[str, str]) -> str:
        return "\n\n".join(f"[{source_id}] {chunk}" for source_id, chunk in source_map.items())

    def _build_messages(self, question: str, prompt_context: str) -> list[dict[str, str]]:
        system_content = (
            "\u0623\u0646\u062a \u0645\u0648\u0644\u062f \u0625\u062c\u0627\u0628\u0627\u062a \u0644\u0646\u0638\u0627\u0645 RAG \u0628\u0627\u0644\u0639\u0631\u0628\u064a\u0629.\n"
            "\u0627\u0633\u062a\u062e\u062f\u0645 \u0627\u0644\u0645\u0635\u0627\u062f\u0631 \u0641\u0642\u0637 \u0628\u062f\u0648\u0646 \u0623\u064a \u062d\u0642\u0627\u0626\u0642 \u062e\u0627\u0631\u062c\u064a\u0629.\n"
            "\u0623\u0639\u062f \u0627\u0644\u0646\u062a\u064a\u062c\u0629 JSON \u0641\u0642\u0637 \u0628\u0627\u0644\u0634\u0643\u0644:\n"
            "{\"supported\": true|false, \"answer\": \"...\", \"sources\": [\"S1\"], \"evidence\": [{\"source\": \"S1\", \"quote\": \"...\"}]}.\n"
            f"\u0627\u0630\u0627 \u0644\u0645 \u062a\u0648\u062c\u062f \u0627\u062c\u0627\u0628\u0629 \u0645\u062f\u0639\u0648\u0645\u0629 \u0623\u0639\u062f: {{\"supported\": false, \"answer\": \"{self.INSUFFICIENT_INFO_MESSAGE}\", \"sources\": [], \"evidence\": []}}."
        )

        user_content = (
            "\u0627\u0644\u0645\u0635\u0627\u062f\u0631:\n"
            "----------------\n"
            f"{prompt_context}\n"
            "----------------\n\n"
            "\u0627\u0644\u0633\u0624\u0627\u0644:\n"
            f"{question}\n"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _run_model(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=320,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.08,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def _normalize_json_like(self, text: str) -> str:
        normalized = self._strip_code_fences(text)
        normalized = normalized.replace("“", '"').replace("”", '"')
        normalized = normalized.replace("‘", "'").replace("’", "'")

        normalized = re.sub(
            r"([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:",
            r'\1"\2":',
            normalized,
        )

        normalized = re.sub(
            r"'([^'\\\n]*(?:\\.[^'\\\n]*)*)'",
            lambda match: '"' + match.group(1).replace('"', '\\"') + '"',
            normalized,
        )

        normalized = normalized.replace("None", "null")
        normalized = normalized.replace("True", "true")
        normalized = normalized.replace("False", "false")
        normalized = re.sub(r",\s*([\]}])", r"\1", normalized)

        return normalized

    def _regex_payload_recovery(self, raw_text: str) -> dict[str, Any] | None:
        text = self._strip_code_fences(raw_text)

        supported_match = re.search(r"supported\s*[:=]\s*(true|false)", text, re.IGNORECASE)
        answer_match = re.search(r"answer\s*[:=]\s*\"([\s\S]*?)\"", text, re.IGNORECASE)
        if not answer_match:
            answer_match = re.search(r"answer\s*[:=]\s*'([\s\S]*?)'", text, re.IGNORECASE)

        answer = answer_match.group(1).strip() if answer_match else ""
        sources = sorted(set(re.findall(r"S\d+", text)))

        evidence: list[dict[str, str]] = []
        for source, quote in re.findall(
            r"\{\s*\"source\"\s*:\s*\"(S\d+)\"\s*,\s*\"quote\"\s*:\s*\"([^\"]+)\"\s*\}",
            text,
        ):
            evidence.append({"source": source, "quote": quote.strip()})

        if not supported_match and not answer:
            return None

        return {
            "supported": (supported_match.group(1).lower() == "true") if supported_match else bool(answer),
            "answer": answer,
            "sources": sources,
            "evidence": evidence,
        }

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        stripped = self._strip_code_fences(text)

        candidates = [stripped]
        brace_match = re.search(r"\{[\s\S]*\}", stripped)
        if brace_match:
            candidates.insert(0, brace_match.group(0))

        for candidate in candidates:
            normalized = self._normalize_json_like(candidate)
            try:
                payload = json.loads(normalized)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload

        return self._regex_payload_recovery(stripped)

    def _normalize_evidence(self, evidence: Any, fallback_source: str = "") -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []

        if not isinstance(evidence, list):
            return normalized

        for item in evidence:
            source = fallback_source
            quote = ""

            if isinstance(item, dict):
                source = str(item.get("source", fallback_source)).strip()
                quote = str(item.get("quote", "")).strip()
            else:
                quote = str(item).strip()

            if not quote or len(quote) < 4:
                continue

            normalized.append({"source": source, "quote": quote})

        unique: list[dict[str, str]] = []
        seen = set()
        for item in normalized:
            key = (item["source"], item["quote"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)

        return unique

    def _answer_supported_by_evidence(self, answer: str, evidence_text: str) -> bool:
        answer_tokens = {
            token
            for token in self._tokenize(answer)
            if len(token) >= 3 and token not in self.STOPWORDS
        }

        evidence_tokens = {
            token
            for token in self._tokenize(evidence_text)
            if len(token) >= 3
        }

        if not answer_tokens:
            return True

        overlap = len(answer_tokens & evidence_tokens)
        return (overlap / len(answer_tokens)) >= 0.5

    def _validate_payload(
        self,
        payload: dict[str, Any] | None,
        source_map: dict[str, str],
    ) -> tuple[bool, bool, str, list[str], list[dict[str, str]]]:
        if not payload:
            return (False, False, "", [], [])

        supported = bool(payload.get("supported", False))
        answer = str(payload.get("answer", "")).strip()

        sources = payload.get("sources", [])
        if not isinstance(sources, list):
            sources = []
        sources = [str(item) for item in sources if str(item) in source_map]

        default_source = sources[0] if sources else ""
        evidence_items = self._normalize_evidence(payload.get("evidence", []), fallback_source=default_source)

        if not supported:
            return (True, False, answer, [], [])

        if not sources:
            return (False, True, answer, [], evidence_items)

        allowed_text = "\n".join(source_map[source_id] for source_id in sources)

        valid_quotes: list[dict[str, str]] = []
        for item in evidence_items:
            source = item["source"] if item["source"] in source_map else default_source
            quote = item["quote"]
            if quote and quote in source_map.get(source, ""):
                valid_quotes.append({"source": source, "quote": quote})

        if not valid_quotes:
            return (False, True, answer, sources, [])

        answer_numbers = self._extract_numbers(answer)
        allowed_numbers = self._extract_numbers(allowed_text)
        if answer_numbers and not answer_numbers.issubset(allowed_numbers):
            return (False, True, answer, sources, valid_quotes)

        evidence_text = " ".join(item["quote"] for item in valid_quotes)
        if answer and not self._answer_supported_by_evidence(answer, evidence_text):
            return (False, True, answer, sources, valid_quotes)

        return (True, True, answer, sources, valid_quotes)

    def _collect_support_sentences(
        self,
        question_info: dict[str, Any],
        source_map: dict[str, str],
        max_sentences: int = 3,
    ) -> list[dict[str, str]]:
        key_terms = set(question_info.get("key_terms", []))
        candidates: list[tuple[float, dict[str, str]]] = []

        for source_id, chunk in source_map.items():
            for sentence in self._split_sentences(chunk):
                if len(sentence) < 8:
                    continue

                sentence_terms = self._tokenize(sentence)
                overlap = len(key_terms & sentence_terms)
                if overlap <= 0:
                    continue

                score = float(overlap)
                if re.search(r"\d{3,4}", sentence):
                    score += 0.4

                candidates.append((score, {"source": source_id, "quote": sentence}))

        candidates.sort(key=lambda item: item[0], reverse=True)

        picked: list[dict[str, str]] = []
        seen_quotes = set()
        for _, item in candidates:
            quote = item["quote"]
            if quote in seen_quotes:
                continue
            seen_quotes.add(quote)
            picked.append(item)
            if len(picked) >= max_sentences:
                break

        return picked

    def _can_return_partial_answer(
        self,
        question_info: dict[str, Any],
        support_sentences: list[dict[str, str]],
    ) -> bool:
        if not support_sentences:
            return False

        key_terms = {
            token
            for token in question_info.get("key_terms", [])
            if len(token) >= 2 and token not in self.STOPWORDS
        }
        entity_terms = set(question_info.get("entities", []))

        norm_key_terms = self._normalized_terms(key_terms)
        norm_entity_terms = self._normalized_terms(entity_terms)
        non_entity_terms = norm_key_terms - norm_entity_terms

        support_tokens = self._normalized_terms(
            self._tokenize(" ".join(item["quote"] for item in support_sentences))
        )

        if not non_entity_terms:
            # For identity-style questions, entity overlap is enough.
            return bool(norm_entity_terms & support_tokens) or bool(support_tokens)

        overlap = len(non_entity_terms & support_tokens)
        coverage = overlap / max(1, len(non_entity_terms))

        if question_info.get("is_yes_no", False):
            return coverage >= 0.8

        return coverage >= 0.6

    def _build_partial_answer(
        self,
        question_info: dict[str, Any],
        support_sentences: list[dict[str, str]],
    ) -> str:
        if not support_sentences:
            return self.INSUFFICIENT_INFO_MESSAGE

        key_terms = {
            token
            for token in question_info.get("key_terms", [])
            if len(token) >= 2 and token not in self.STOPWORDS
        }
        entity_terms = set(question_info.get("entities", []))
        support_tokens = self._normalized_terms(
            self._tokenize(" ".join(item["quote"] for item in support_sentences))
        )

        core = " ".join(item["quote"] for item in support_sentences[:2])

        norm_key_terms = self._normalized_terms(key_terms)
        norm_entity_terms = self._normalized_terms(entity_terms)
        non_entity_missing = sorted(
            token for token in (norm_key_terms - norm_entity_terms) if token not in support_tokens
        )

        if non_entity_missing and len(non_entity_missing) <= 3:
            missing_text = "\u060c ".join(non_entity_missing)
            return (
                f"{core} "
                "\u0644\u0643\u0646 \u0644\u0627 \u062a\u0648\u0641\u0631 \u0627\u0644\u0648\u062b\u064a\u0642\u0629 "
                f"\u0645\u0639\u0644\u0648\u0645\u0627\u062a \u0648\u0627\u0636\u062d\u0629 \u0639\u0646: {missing_text}."
            )

        return core

    def _clean_answer(self, answer: str) -> str:
        cleaned = answer.strip()
        cleaned = re.sub(r"^#+\s*", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _format_final_answer(self, answer: str, sources: list[str]) -> str:
        answer = self._clean_answer(answer)
        if not sources:
            return answer

        source_text = ", ".join(sorted(set(sources)))
        return f"{answer}\n{self.SOURCE_PREFIX}{source_text}"

    def generate(
        self,
        question: str,
        context: str,
        retrieved_chunks: list[dict[str, Any]] | None = None,
        return_debug: bool = False,
    ) -> str | dict[str, Any]:
        if not question.strip():
            final_answer = self.INSUFFICIENT_INFO_MESSAGE
            if return_debug:
                return {
                    "final_answer": final_answer,
                    "decision": "empty_question_refusal",
                    "raw_model_output": "",
                    "payload": None,
                    "payload_answer": "",
                    "sources": [],
                    "evidence": [],
                    "source_map": {},
                    "prompt_context": "",
                }
            return final_answer

        source_map = self._build_source_map(context=context, retrieved_chunks=retrieved_chunks)
        prompt_context = self._build_prompt_context(source_map)
        messages = self._build_messages(question, prompt_context)

        raw_output = self._run_model(messages)
        payload = self._extract_json_object(raw_output)

        payload_ok, supported, payload_answer, sources, evidence = self._validate_payload(
            payload=payload,
            source_map=source_map,
        )

        question_info = analyze_query(question)

        if payload_ok and supported:
            evidence_text = " ".join(item["quote"] for item in evidence)
            answer_text = evidence_text if evidence_text else payload_answer
            final_answer = self._format_final_answer(answer_text, sources)
            decision = "validated_model_answer"
        elif payload_ok and not supported:
            final_answer = self.INSUFFICIENT_INFO_MESSAGE
            decision = "model_refusal"
        else:
            support_sentences = self._collect_support_sentences(question_info, source_map)

            if self._can_return_partial_answer(question_info, support_sentences):
                answer_text = self._build_partial_answer(question_info, support_sentences)
                source_ids = [item["source"] for item in support_sentences]

                if answer_text == self.INSUFFICIENT_INFO_MESSAGE:
                    final_answer = answer_text
                    decision = "safe_refusal"
                else:
                    final_answer = self._format_final_answer(answer_text, source_ids)
                    decision = "extractive_reconstructed_fallback"
            else:
                final_answer = self.INSUFFICIENT_INFO_MESSAGE
                decision = "safe_refusal"

        if return_debug:
            return {
                "final_answer": final_answer,
                "decision": decision,
                "raw_model_output": raw_output,
                "payload": payload,
                "payload_answer": payload_answer,
                "sources": sources,
                "evidence": evidence,
                "source_map": source_map,
                "prompt_context": prompt_context,
            }

        return final_answer
