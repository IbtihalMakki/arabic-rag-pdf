import re
from typing import Any


QUESTION_STOPWORDS = {
    "من", "ما", "ماذا", "متى", "اين", "أين", "هل", "هو", "هي",
    "عن", "في", "على", "الى", "إلى", "كم", "التي", "الذي", "هذه",
    "ذلك", "مع", "او", "أو", "اذا", "إذا", "كان", "كانت", "يكون",
    "الوثيقة", "المستند", "اعتمادا", "اعتمادًا", "يرجى", "رجاء",
}

ANSWER_TYPE_HINTS = {
    "identity": {"من", "هو", "هي", "نبذة", "تعريف"},
    "temporal": {"متى", "تاريخ", "عام", "سنة", "ولد", "وفاة", "توفي"},
    "location": {"اين", "أين", "مكان", "مدينة", "بلد", "ولد"},
    "list": {"ابرز", "أبرز", "اذكر", "عدد", "ماهي", "ما هي", "المناصب", "اعمال", "أعمال"},
    "education": {"درس", "تعليم", "جامعة", "دراسة"},
    "works": {"رواية", "روايات", "ديوان", "كتاب", "اعمال", "أعمال", "ادبية", "أدبية"},
    "verification": {"هل", "صحيح", "كان", "كانت"},
}

ENTITY_HINTS = {
    "name_tokens": {"غازي", "القصيبي", "قصييبي", "عبد", "الرحمن"},
}


def _tokenize(text: str) -> list[str]:
    normalized = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", text)
    return re.findall(r"[\u0621-\u064AA-Za-z0-9]+", normalized)


def analyze_query(question: str) -> dict[str, Any]:
    stripped = question.strip()
    tokens = _tokenize(stripped)
    lowered_tokens = [token.lower() for token in tokens]

    token_set = set(tokens)
    lowered_set = set(lowered_tokens)

    key_terms = {
        token
        for token in tokens
        if len(token) >= 2 and token not in QUESTION_STOPWORDS
    }

    entities = {
        token
        for token in tokens
        if token in ENTITY_HINTS["name_tokens"]
    }

    intents: set[str] = set()
    for intent_name, hints in ANSWER_TYPE_HINTS.items():
        if token_set & hints or lowered_set & {hint.lower() for hint in hints}:
            intents.add(intent_name)

    if not stripped:
        intents.add("empty")

    if not intents:
        intents.add("generic")

    is_yes_no = stripped.startswith("هل")

    return {
        "question": question,
        "tokens": tokens,
        "key_terms": sorted(key_terms),
        "entities": sorted(entities),
        "intents": sorted(intents),
        "is_yes_no": is_yes_no,
    }
