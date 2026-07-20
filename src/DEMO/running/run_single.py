# single run of full appeal-processing pipeline
# summarization -> hierarchical classification L2/L3/L4 -> reference fields classification

import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("../logs/run_single.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

from src.DEMO.functional.make_qwen_summary import summarize
from src.DEMO.functional.keryx_classifier import classify_hierarchy, format_preds
from src.DEMO.functional.keryx_REF_classification import classify_all_fields
from src.DEMO.functional.qwen_NER_inference import extract_entities

logger = logging.getLogger(__name__)

HARDCODED_TEXT = """
Эбращение № 7320927 a\n\nДата дедлайна по исполнению: 17.03.2\n\n(©) Информация о гражданине Автоопределение @)\nФИО
гражданина: ©. гитоево_ Татьяна Сергеевна s :\nЭлектронная почта: muvaveva— GU@ mail. ru\n\nНомер телефона:
“s7sasov2795 = Ot -\n\n+7(952)501-27-95\n\nИНН гражданина:\n\nФИО обратившегося:\n\nАдрес обратившегося: |\n\nСекретно
О да @ Her\n\n@ География обращения Автоопределение iif)\n\nАдрес источника\n\nФедеральный Округ\n\nРегион\n\n
Муниципальное\nобразование\n\nАдрес\n\nМ источник обращения\n\nКанал\nПоток\n\nСобытие\n\nДАН И ОРГАНИЗАЦИЙ\nВходящий
№2 8\n\n8 Информация об исполнителе 19.02 25\n\nОрганизация\n\n9 информация о волонтёре Обработка волонтёром: —\n\nВ
Ход действий\n\nВолонтёры:\n\nBE текст обращения\nКоличество спама и обсценной лексики: 0 %\n\nМеня зовут Рассчитается
Татьяна Сергеевна. Я звоню из города Челябинска. Я хотела бы попросить п\nомощи у Владимира Владимировича вопросе
основе домов улицы Ярославская. Дом четырнадцать.\nДома давным давно в аварийном состоянии, уже практически нет подачи
нормальной воды, отоплен\nие. Никто не занимается обслуживанием дома два РА, в том числе только управляющая компания
бе\nрет деньги. Им каждый год обещают, что их расселят, но уже очень много лет их никто не расселяет д\nом, но уже
стоит на ладан дышит. Вот я хотела бы попросить Владимира Владимировича как-то помо\nчь в решении этого вопроса и уже
наконец то, чтобы их расстелили. Спасибо большое.\n\nСохранить\n\n\nсуо. организация | | |\n\nРешение исполнителя\n\n
Ответ исполнителя : Развернуть\n\nТематика обращения\n\nТип категории\n\nКатегория обращения\n\nПодкатегория\n
обращения\n\n[$] География гражданина Автоопределение ©)\n\nАдрес источника\nФедеральный Округ\n\nРегион\n\n
Муниципальное\nобразование\n\nАдрес\n(®} Дополнительная информация\n\nСрочно: ©) Да @) Нет Обращались ранее: ©) Да
©) Нет\n\nВозраст на момент\nсоздания обращения\n\nОсобые метки сообщения: ap\n\nСистема-источник\n\n
"""


def _clean_email(value: str) -> str:
    value = re.sub(r"\s+", "", value)
    value = value.strip(" ()[]{}:;,.")
    return value


def _clean_name(value: str) -> str:
    value = value.strip(" .")
    if not value:
        return value
    if " " in value:
        value = value.split()[0]
    value = re.split(r"(?<=[а-яёa-z])(?=[А-ЯЁA-Z])", value)[0]
    return value.capitalize()


def _clean_addr_part(value: str) -> str:
    return value.strip(" .,")


NER_CLEANERS = {
    "GOV_EMAIL": _clean_email,
    "PERSONAL_EMAIL": _clean_email,
    "LAST_NAME": _clean_name,
    "FIRST_NAME": _clean_name,
    "MIDDLE_NAME": _clean_name,
    "POSTAL_CODE": _clean_addr_part,
    "REGION": _clean_addr_part,
    "CITY": _clean_addr_part,
    "STREET": _clean_addr_part,
    "HOUSE": _clean_addr_part,
    "ROOM": _clean_addr_part,
}


def postprocess_entities(entities: dict) -> dict:
    for label, cleaner in NER_CLEANERS.items():
        if label in entities and entities[label]:
            if isinstance(entities[label], list):
                cleaned = [cleaner(v) for v in entities[label] if v]
                entities[label] = list(dict.fromkeys(cleaned))
            else:
                entities[label] = cleaner(entities[label])

    address_parts = []
    address_order = ["POSTAL_CODE", "REGION", "CITY", "STREET", "HOUSE", "ROOM"]

    for key in address_order:
        if key in entities and entities[key]:
            value = entities[key]
            if isinstance(value, list) and value:
                address_parts.append(value[0])
            elif isinstance(value, str):
                address_parts.append(value)

    if address_parts:
        entities["FULL_ADDRESS"] = [", ".join(address_parts)]

    logger.info(f"Postprocessed entities: {entities}")
    return entities


def classify_text(text: str):

    summary = summarize(text)

    pred_l2, pred_l3, pred_l4 = classify_hierarchy(summary)

    logger.info(f"Summary: {summary}")
    logger.info(f"L2 predictions: {[(c['code'], c['name'], round(s, 3)) for c, s in pred_l2]}")
    logger.info(f"L3 predictions: {[(c['code'], c['name'], round(s, 3)) for c, s in pred_l3]}")
    logger.info(f"L4 predictions: {[(c['code'], c['name'], round(s, 3)) for c, s in pred_l4]}")

    field_predictions = classify_all_fields(text)

    entities = extract_entities(text)
    entities = postprocess_entities(entities)

    return (
        summary,
        format_preds(pred_l2),
        format_preds(pred_l3),
        format_preds(pred_l4),
        field_predictions,
        entities,
    )


def main():
    logger.info("=" * 55)
    logger.info("Test inference on hard-coded text...")
    summary, l2, l3, l4, fields, entities = classify_text(HARDCODED_TEXT)
    logger.info("Done")

if __name__ == "__main__":
    main()
