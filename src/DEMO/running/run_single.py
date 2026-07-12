# single run of full appeal-processing pipeline
# summarization -> hierarchical classification L2/L3/L4 -> reference fields classification

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("../logs/run_single.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

from src.DEMO.loading.gigachat_loader import summarize
from src.DEMO.functional.keryx_classifier import classify_hierarchy, classify_all_fields, format_preds
from src.DEMO.functional.ner_inference import extract_entities

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

def classify_text(text: str):
    summary = summarize(text)

    pred_l2, pred_l3, pred_l4 = classify_hierarchy(summary)

    logger.info(f"Summary: {summary}")
    logger.info(f"L2 predictions: {[(c['code'], c['name'], round(s, 3)) for c, s in pred_l2]}")
    logger.info(f"L3 predictions: {[(c['code'], c['name'], round(s, 3)) for c, s in pred_l3]}")
    logger.info(f"L4 predictions: {[(c['code'], c['name'], round(s, 3)) for c, s in pred_l4]}")

    field_predictions = classify_all_fields(text)
    entities = extract_entities(text)
    logger.info(f"NER: {entities}")

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
