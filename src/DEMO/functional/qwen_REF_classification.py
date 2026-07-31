# Классификация справочных полей через Qwen: вместо NLI-подхода KERYX
# (N forward pass на N кандидатов) - один generative-запрос на поле,
# модель отвечает строго JSON с номером выбранного варианта.
#
# Использование:
#   from src.DEMO.functional.qwen_REF_classification import classify_fields_by_qwen
#   predictions = classify_fields_by_qwen(text) # все поля
#   predictions = classify_fields_by_qwen(text, ["AppealKind"]) # только часть

import re
import json
import logging
from typing import Dict, List, Optional

from src.DEMO.loading.qwen_loader import get_qwen
from src.DEMO.loading.REF_fields_loader import get_field_candidates

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Человекочитаемые названия полей для промпта (см. list.docx).
# Если поля нет в словаре - в промпт уйдёт сырое имя field_name.
FIELD_LABELS = {
    "AppealKind": "Вид обращения",
    "StatusId": "Тип обращения",
    "ItemID": "Форма обращения",
    "DeliveryTypeId": "Источник поступления",
    "RegistrationPlaceId": "Место события",
    "ConsiderationType": "Обращение (первичное / повторное / неоднократное)",
    "PetitionerCategory": "Категория заявителя",
    "PetitionerDistrict": "Район проживания заявителя",
}

CONSIDERATION_TYPE_MAPPING = {
    "первичное": "0",
    "повторное": "1",
    "неоднократное": "2",
}

FIELD_PROMPT = """Определи значение поля "{field_label}" для этого обращения гражданина.
Выбери РОВНО ОДИН вариант из списка ниже. Если ни один вариант не подходит (совсем не понятно по тексту), то
ставь null.

Варианты:
{options_block}

ОБРАЩЕНИЕ:
{text}

Ответь СТРОГО в формате JSON, без пояснений и без markdown-разметки:
{{"index": <номер варианта>}}"""


def _build_options_block(candidates: List[str]) -> str:
    return "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))


def _build_prompt(text: str, field_label: str, candidates: List[str]) -> str:
    return FIELD_PROMPT.format(
        field_label=field_label,
        options_block=_build_options_block(candidates),
        text=text[:3000],
    )


def _parse_choice_index(response: str, n_candidates: int) -> Optional[int]:
    if not response:
        return None

    # 1) пробуем распарсить как чистый JSON
    idx = _try_parse_json(response.strip(), n_candidates)
    if idx is not None:
        return idx

    # 2) ищем JSON-объект внутри ответа (на случай лишнего текста вокруг)
    match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if match:
        idx = _try_parse_json(match.group(), n_candidates)
        if idx is not None:
            return idx

    # 3) крайний фолбэк - просто первое число в ответе
    match = re.search(r"\d+", response)
    if match:
        raw_idx = int(match.group())
        if 1 <= raw_idx <= n_candidates:
            return raw_idx - 1

    return None


def _try_parse_json(candidate_str: str, n_candidates: int) -> Optional[int]:
    try:
        data = json.loads(candidate_str)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    raw_idx = data.get("index")
    if raw_idx is None:
        return None

    try:
        raw_idx = int(raw_idx)
    except (TypeError, ValueError):
        return None

    if 1 <= raw_idx <= n_candidates:
        return raw_idx - 1
    return None


def _fallback_by_substring(response: str, candidates: List[str]) -> Optional[int]:
    if not response:
        return None
    response_lower = response.strip().lower()
    for i, c in enumerate(candidates):
        if c.strip().lower() in response_lower or response_lower in c.strip().lower():
            return i
    return None


def classify_field_by_qwen(
    text: str,
    field_name: str,
    candidates: List[str],
    field_label: Optional[str] = None,
) -> Optional[str]:

    if not candidates:
        logger.warning(f"[{field_name}] No candidates provided, skip")
        return None

    label = field_label or FIELD_LABELS.get(field_name, field_name)
    prompt = _build_prompt(text, label, candidates)

    qwen = get_qwen()
    response = qwen.chat(prompt, temperature=0.0, max_tokens=20)

    idx = _parse_choice_index(response, len(candidates))
    if idx is None:
        idx = _fallback_by_substring(response, candidates)

    if idx is None:
        logger.warning(f"[{field_name}] Could not parse Qwen response: '{response}'")
        return None

    value = candidates[idx]
    return value


def classify_fields_by_qwen(text: str, field_names: Optional[List[str]] = None) -> Dict[str, str]:

    all_candidates = get_field_candidates()
    field_names = field_names or list(all_candidates.keys())

    predictions = {}
    for field_name in field_names:
        candidates = all_candidates.get(field_name)
        if not candidates:
            logger.warning(f"[{field_name}] No candidates found in REF_fields_loader, skip")
            continue

        value = classify_field_by_qwen(text, field_name, candidates)
        if value is None:
            continue

        if field_name == "ConsiderationType":
            value = CONSIDERATION_TYPE_MAPPING.get(value.strip().lower(), value)

        predictions[field_name] = value

    logger.info(f"Qwen field predictions: {predictions}")
    return predictions


logger.info("Qwen REF-field classifier ready")
