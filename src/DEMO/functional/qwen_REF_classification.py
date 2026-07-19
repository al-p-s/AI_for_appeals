import json
import logging
import re
from typing import Dict, List, Optional

from src.DEMO.loading.REF_fields_loader import get_field_candidates
from src.DEMO.loading.qwen_loader import get_qwen

logger = logging.getLogger(__name__)


def _build_fields_description(field_candidates: Dict[str, List[str]]) -> str:
    description = []
    for field, candidates in field_candidates.items():
        if len(candidates) > 30:
            short_list = candidates[:5] + ["... (всего " + str(len(candidates)) + ")"]
            description.append(f"  {field}: {', '.join(short_list)}")
        else:
            description.append(f"  {field}: {', '.join(candidates)}")
    return "\n".join(description)


def _get_prompt_template() -> str:
    return """Ты — система классификации полей обращения.

Проанализируй текст обращения и определи значения для следующих полей.

Доступные поля и их возможные значения:
{fields_description}

ВАЖНЫЕ ПРАВИЛА:
1. Выбирай ТОЛЬКО из предложенного списка значений для каждого поля.
2. Если значение невозможно определить - ставь null.
3. ConsiderationType — это статус рассмотрения обращения. Его значение может быть только 0, 1 или 2. (0 - обращение новое/не рассмотрено,
1 - повторное обращение, 2 - неоднократное обращение)
4. PetitionerDistrict — район проживания заявителя. Определяй ТОЛЬКО если явно указан в тексте.
5. RegistrationPlaceId — место события (где произошло нарушение/проблема). Определяй ТОЛЬКО если явно указан в тексте.
6. DeliveryTypeId — источник поступления обращения (откуда пришло).
Если заявитель писал по email — значит "Заявитель (электронная почта)".

Ответ должен быть ТОЛЬКО JSON без пояснений.

Текст:
{text}"""


def parse_json_from_response(content: str) -> Optional[Dict]:
    try:
        return json.loads(content)
    except:
        pass

    json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass

    json_match = re.search(r'\{[^{}]*(\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass

    logger.debug(f"Can't find JSON in: {content[:200]}...")
    return None


def validate_and_fix_result(result: Dict, field_candidates: Dict[str, List[str]]) -> Dict:
    validated = {}
    for field, candidates in field_candidates.items():
        value = result.get(field)

        if not value or not isinstance(value, str):
            if field == "PetitionerCategory":
                validated[field] = "категория не установлена"
            elif field == "RegistrationPlaceId":
                validated[field] = "Иное"
            else:
                validated[field] = None
            continue

        if value in candidates:
            validated[field] = value
            continue

        value_lower = value.lower().strip()
        found = False
        for candidate in candidates:
            if value_lower in candidate.lower() or candidate.lower() in value_lower:
                validated[field] = candidate
                found = True
                break

        if not found:
            validated[field] = None

    return validated


def classify_all_fields_qwen(text: str) -> Dict:
    logger.info("Start REF fields classification by Qwen")

    try:
        field_candidates = get_field_candidates()
        if not field_candidates:
            logger.error("Can't load candidates for fields")
            return {}

        if len(text) > 8000:
            text = text[:8000]
            logger.info(f"Text shrink up to 8000 symbols")

        fields_desc = _build_fields_description(field_candidates)

        prompt_template = _get_prompt_template()
        prompt = prompt_template.format(
            fields_description=fields_desc,
            text=text
        )

        qwen = get_qwen()
        response = qwen.chat(prompt, max_tokens=1500)

        if not response:
            logger.warning("Qwen returned empty answer")
            return {field: None for field in field_candidates}

        result = parse_json_from_response(response)

        if not result:
            logger.warning("Can't extract JSON from LLM answer")
            logger.debug(f"Answer: {response[:500]}...")
            return {field: None for field in field_candidates}

        validated = validate_and_fix_result(result, field_candidates)
        logger.info(f"Fields classification complete: {validated}")
        return validated

    except Exception as e:
        logger.error(f"Fields classification error: {e}", exc_info=True)
        return {}
