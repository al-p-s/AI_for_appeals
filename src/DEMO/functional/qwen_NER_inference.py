import json
import logging
import re
import os
from typing import Dict, List, Optional
from src.DEMO.loading.qwen_loader import get_qwen

logger = logging.getLogger(__name__)


NER_PROMPT = """Ты — система извлечения именованных сущностей (NER).

Извлеки из текста персональные данные и верни ТОЛЬКО JSON без пояснений.

Схема JSON:
{{"FIRST_NAME": null, "LAST_NAME": null, "MIDDLE_NAME": null, "PHONE_NUMBER": [], "PERSONAL_EMAIL": [], "GOV_EMAIL": [], "POSTAL_CODE": null, "REGION": null, "CITY": null, "STREET": null, "HOUSE": null, "ROOM": null}}

Правила:
- FIRST_NAME — имя
- LAST_NAME — фамилия
- MIDDLE_NAME — отчество
- PHONE_NUMBER — список телефонов
- PERSONAL_EMAIL — личные email (gmail, yandex, mail.ru и т.п.)
- GOV_EMAIL — гос/корпоративные email
- POSTAL_CODE — почтовый индекс
- REGION — субъект РФ (область, край, республика и т.д.)
- CITY — населённый пункт
- STREET — улица, проспект, переулок и т.д.
- HOUSE — номер дома
- ROOM — квартира, офис, кабинет

Текст:
{text}"""


def extract_entities(text: str) -> Dict:
    logger.info("NER extraction started")

    entities = _extract_with_llm(text)
    return _convert_llm_to_pipeline_format(entities)


def _extract_with_llm(text: str) -> Dict:

    try:
        if len(text) > 6000:
            text = text[:6000]
            logger.info(f"Text shrink up to 6000 symbols")

        prompt = NER_PROMPT.format(text=text)

        qwen = get_qwen()
        response = qwen.chat(prompt)

        if not response:
            logger.warning("LLM returned empty answer")
            return _empty_result()

        entities = _parse_json_from_response(response)

        if not entities:
            logger.warning("Can't extract JSON from LLM answer")
            return _empty_result()

        logger.info(f"LLM extracted entites: {list(entities.keys())}")
        return entities

    except Exception as e:
        logger.error(f"LLM NER error: {e}")
        return _empty_result()


def _parse_json_from_response(content: str) -> Optional[Dict]:
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


def _empty_result() -> Dict:
    return {
        "FIRST_NAME": None,
        "LAST_NAME": None,
        "MIDDLE_NAME": None,
        "PHONE_NUMBER": [],
        "PERSONAL_EMAIL": [],
        "GOV_EMAIL": [],
        "POSTAL_CODE": None,
        "REGION": None,
        "CITY": None,
        "STREET": None,
        "HOUSE": None,
        "ROOM": None
    }


def _convert_llm_to_pipeline_format(entities: Dict) -> Dict:
    result = {}

    for key in ["FIRST_NAME", "LAST_NAME", "MIDDLE_NAME"]:
        if entities.get(key):
            result[key] = [entities[key]]

    for key in ["PERSONAL_EMAIL", "GOV_EMAIL", "PHONE_NUMBER"]:
        if entities.get(key):
            if isinstance(entities[key], list):
                result[key] = entities[key]
            else:
                result[key] = [entities[key]]

    address_parts = []
    if entities.get("POSTAL_CODE"):
        address_parts.append(entities["POSTAL_CODE"])
    if entities.get("REGION"):
        address_parts.append(entities["REGION"])
    if entities.get("CITY"):
        address_parts.append(entities["CITY"])
    if entities.get("STREET"):
        address_parts.append(entities["STREET"])
    if entities.get("HOUSE"):
        address_parts.append(f"д. {entities['HOUSE']}")
    if entities.get("ROOM"):
        address_parts.append(f"кв. {entities['ROOM']}")

    if address_parts:
        result["ADDRESS"] = [", ".join(address_parts)]

    for key in ["POSTAL_CODE", "REGION", "CITY", "STREET", "HOUSE", "ROOM"]:
        if entities.get(key):
            result[key] = [entities[key]]

    logger.info(f"Converted entities: {result}")
    return result


def format_ner_entities(entities: Dict) -> str:
    if not entities:
        return "Key fields not found"

    result = []
    SKIP_LABELS = {"GOV_EMAIL"}

    for label, values in entities.items():
        if label in SKIP_LABELS:
            continue
        if not values:
            continue

        if isinstance(values, list):
            uniq_values = list(dict.fromkeys([v for v in values if v]))
        else:
            uniq_values = [str(values)] if values else []

        if uniq_values:
            result.append(f"{label}: {', '.join(uniq_values)}")

    return "\n".join(result)
