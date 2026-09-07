import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional
from src.DEMO.loading.qwen_loader import get_qwen
from src.DEMO.paths_config import ORGS_XML_PATH

logger = logging.getLogger(__name__)

NER_PROMPT_PERSONAL = """Ты — система извлечения именованных сущностей (NER).
Извлеки персональные данные и верни ТОЛЬКО JSON.
Схема: {{"FIRST_NAME": null, "LAST_NAME": null, "MIDDLE_NAME": null, "PHONE_NUMBER": [], "PERSONAL_EMAIL": [], "GOV_EMAIL": [], "DATE": null}}

- FIRST_NAME, LAST_NAME, MIDDLE_NAME — имя, фамилия и отчество исходного автора обращения (заявителя). ИЗВЛЕКАЙ ИХ В ЛЮБОМ СЛУЧАЕ, даже если автор выступает от лица компании (указан как директор, руководитель, представитель ООО/ИП и т.д.). Если сообщение переслано, ищи ФИО изначального автора.
- PHONE_NUMBER — телефон заявителя. Их может быть несколько.
- PERSONAL_EMAIL — email заявителя. Сюда относятся любые email-адреса автора обращения, даже если они выглядят как корпоративные или содержат название организации.
- GOV_EMAIL — email ведомства, администрации или госслужащего, куда направлено обращение или откуда оно переслано.
- DATE — дата создания или подписания самого обращения заявителем в формате ISO 8601: YYYY-MM-DDTHH:MM:SS.000. Если время не указано, ставь полночь (T00:00:00.000).
Дата может быть указана в обращении в разных форматах:
- "21.02.2025" -> "2025-02-21T00:00:00.000"
- "21 февраля 2025" -> "2025-02-21T00:00:00.000"

Текст:
{text}"""

NER_PROMPT_ADDRESS = """Ты — система извлечения именованных сущностей (NER).
Извлеки географические данные (адрес) и верни ТОЛЬКО JSON.
Схема: {{"POSTAL_CODE": null, "REGION": null, "CITY": null, "STREET": null, "HOUSE": null, "ROOM": null}}

Текст:
{text}"""

NER_PROMPT_SENDER = """Ты — система извлечения именованных сущностей (NER).
Извлеки данные об организации-отправителе обращения и её исходящих реквизитах. Верни ТОЛЬКО JSON.
Схема: {{"SENDER_ORG": null, "EXTERNAL_NUMBER": null, "EXTERNAL_DATE": null}}

Список известных организаций из справочника:
{orgs_list}

Инструкция для SENDER_ORG:
1. Если организация в тексте соответствует (или является синонимом/сокращением) организации из СПИСКА ВЫШЕ — возвращай её ЭТАЛОННОЕ наименование из этого списка.
2. Если в тексте упоминается организация, которой НЕТ в списке — возвращай её наименование из текста как есть. ВАЖНО - не придумывай название. Выводи в таком случае именно то, что указано в тексте.
3. Если организация-отправитель не упоминается — возвращай null.

- EXTERNAL_NUMBER — исходящий регистрационный номер документа от организации-отправителя (например: "01-12/345", "№ 123-А").
- EXTERNAL_DATE — исходящая дата документа от организации-отправителя (НЕ от заявителя/просителя/гражданина) в формате ISO 8601: YYYY-MM-DDTHH:MM:SS.000. Если время не указано, ставь T00:00:00.000.

Текст:
{text}"""

_org_names_cache = None


def _get_org_names_prompt() -> str:
    global _org_names_cache
    if _org_names_cache is not None:
        return _org_names_cache

    orgs_path = ORGS_XML_PATH
    names = []
    if orgs_path.exists():
        try:
            import xml.etree.ElementTree as ET
            root = ET.parse(str(orgs_path)).getroot()
            for comp in root.iter("CompaniesRow"):
                name = comp.get("Name", "").strip()
                if name and name not in names:
                    names.append(name)
        except Exception as e:
            logger.warning(f"Failed to load orgs for prompt: {e}")

    if names:
        _org_names_cache = "\n".join(f"- {n}" for n in names)
    else:
        _org_names_cache = "(список организаций недоступен)"
    return _org_names_cache


def extract_entities(text: str) -> Dict:
    logger.info("Start NER by Qwen (split prompts)")
    entities = _extract_with_llm(text)
    return _convert_llm_to_pipeline_format(entities)


def _extract_with_llm(text: str) -> Dict:
    qwen = get_qwen()

    try:
        resp_pers = qwen.chat(NER_PROMPT_PERSONAL.format(text=text))
        resp_addr = qwen.chat(NER_PROMPT_ADDRESS.format(text=text))
        orgs_list = _get_org_names_prompt()
        resp_sender = qwen.chat(NER_PROMPT_SENDER.format(orgs_list=orgs_list, text=text))

        dict_pers = _parse_json_from_response(resp_pers) if resp_pers else {}
        dict_addr = _parse_json_from_response(resp_addr) if resp_addr else {}
        dict_sender = _parse_json_from_response(resp_sender) if resp_sender else {}

        # Берем пустой шаблон и накатываем сверху то, что нашла сеть
        result = _empty_result()
        if isinstance(dict_pers, dict): result.update({k: v for k, v in dict_pers.items() if k in result})
        if isinstance(dict_addr, dict): result.update({k: v for k, v in dict_addr.items() if k in result})
        if isinstance(dict_sender, dict): result.update({k: v for k, v in dict_sender.items() if k in result})

        logger.info(f"LLM extracted entites (merged): {list(k for k, v in result.items() if v)}")
        return result

    except Exception as e:
        logger.error(f"LLM NER error: {e}")
        return _empty_result()


def _parse_json_from_response(content: str) -> Optional[Dict]:
    try: return json.loads(content)
    except: pass
    match = re.search(r'\{[^{}]*\}', content, re.DOTALL) or re.search(r'\{[^{}]*(\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
    if match:
        try: return json.loads(match.group())
        except: pass
    return None


def _empty_result() -> Dict:
    return {
        "FIRST_NAME": None, "LAST_NAME": None, "MIDDLE_NAME": None,
        "PHONE_NUMBER": [], "PERSONAL_EMAIL": [], "GOV_EMAIL": [],
        "POSTAL_CODE": None, "REGION": None, "CITY": None,
        "STREET": None, "HOUSE": None, "ROOM": None, "DATE": None,
        "SENDER_ORG": None, "EXTERNAL_NUMBER": None, "EXTERNAL_DATE": None
    }


def _convert_llm_to_pipeline_format(entities: Dict) -> Dict:
    result = {}

    for key in ["FIRST_NAME", "LAST_NAME", "MIDDLE_NAME", "DATE", "SENDER_ORG", "EXTERNAL_NUMBER", "EXTERNAL_DATE"]:
        if entities.get(key):
            if isinstance(entities[key], list):
                result[key] = [str(v) for v in entities[key] if v]
            else:
                result[key] = [str(entities[key])]

    for key in ["PERSONAL_EMAIL", "GOV_EMAIL", "PHONE_NUMBER"]:
        if entities.get(key):
            if isinstance(entities[key], list):
                flat_list = []
                for item in entities[key]:
                    if isinstance(item, list):
                        flat_list.extend([str(v) for v in item if v])
                    elif item:
                        flat_list.append(str(item))
                result[key] = flat_list
            else:
                result[key] = [str(entities[key])]

    for key in ["POSTAL_CODE", "REGION", "CITY", "STREET", "HOUSE", "ROOM"]:
        if entities.get(key):
            if isinstance(entities[key], list):
                result[key] = [str(v) for v in entities[key] if v]
            else:
                result[key] = [str(entities[key])]

    return result


def format_ner_entities(entities: Dict) -> str:
    if not entities:
        return "Key fields not found"

    result = []

    for label, values in entities.items():
        if not values:
            continue

        if isinstance(values, list):
            uniq_values = list(dict.fromkeys([v for v in values if v]))
        else:
            uniq_values = [str(values)] if values else []

        if uniq_values:
            result.append(f"{label}: {', '.join(uniq_values)}")

    return "\n".join(result)
