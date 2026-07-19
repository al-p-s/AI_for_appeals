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
2. Если значение невозможно определить — ставь "не определяется".
3. ConsiderationType может быть только "0", "1" или "2".
4. PetitionDistrict — район проживания заявителя.
5. RegistrationPlaceId — место события/административная единица.
6. StatusId — это тип/подтип обращения (Заявление, Жалоба и т.д.).

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
            validated[field] = "не определяется"
            continue

        if value in candidates:
            validated[field] = value
            continue

        value_lower = value.lower().strip()
        found = False
        for candidate in candidates:
            if candidate == "не определяется":
                continue
            if value_lower in candidate.lower() or candidate.lower() in value_lower:
                validated[field] = candidate
                found = True
                break

        if not found:
            validated[field] = "не определяется"

    return validated


def classify_all_fields_qwen(text: str) -> Dict:
    logger.info("Start fields classification by Qwen")

    try:
        field_candidates = get_field_candidates()
        if not field_candidates:
            logger.error("Can't load candidates for fields")
            return {}

        max_len = 6000
        if len(text) > max_len:
            text = text[:max_len]
            logger.info(f"Text shrink up to {max_len} symbols")

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
            return {field: "can't be defined" for field in field_candidates}

        result = parse_json_from_response(response)

        if not result:
            logger.warning("Can't extract JSON from LLM answer")
            logger.debug(f"Answer: {response[:500]}...")
            return {field: "can't be defined" for field in field_candidates}

        validated = validate_and_fix_result(result, field_candidates)
        logger.info(f"Fields classification complete: {validated}")
        return validated

    except Exception as e:
        logger.error(f"Fields classification error: {e}", exc_info=True)
        return {}
