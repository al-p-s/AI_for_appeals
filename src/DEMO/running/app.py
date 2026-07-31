import gradio as gr
import logging

from src.DEMO.text_extraction.text_extraction_paddle import pdf_extract
from src.DEMO.running.run_single import classify_text
from src.DEMO.functional.xml_export import build_xml_from_results, parse_l4_codes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("../logs/full_pipeline.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Справочные поля: KERYX-модели (обучены отдельно на каждое поле)
FIELD_LABELS = {
    "AppealKind": "Вид обращения",
    "StatusId": "Тип обращения",
    "ItemID": "Форма обращения",
    "DeliveryTypeId": "Источник поступления",
    "RegistrationPlaceId": "Место события",
    "ConsiderationType": "Первичное/повторное/неоднократное",
    "PetitionerCategory": "Категория заявителя",
    "PetitionerDistrict": "Район проживания заявителя",
}

CONSIDERATION_TYPE_LABELS = {
    "0": "Первичное",
    "1": "Повторное",
    "2": "Неоднократное",
}

# NER-поля: остаются на Qwen (см. qwen_NER_inference.py)
NER_LABELS = {
    "LAST_NAME": "Фамилия",
    "FIRST_NAME": "Имя",
    "MIDDLE_NAME": "Отчество",
    "PERSONAL_EMAIL": "Email",
    "PHONE_NUMBER": "Телефон",
    "DATE": "Дата",
    "FULL_ADDRESS": "Адрес",
}


def build_fields_table(field_predictions: dict) -> list[list[str]]:
    rows = []
    for field_key, label in FIELD_LABELS.items():
        value = field_predictions.get(field_key, "")
        if field_key == "ConsiderationType":
            value = CONSIDERATION_TYPE_LABELS.get(str(value).strip(), value)
        value = value if value not in (None, "") else "не определено"
        rows.append([label, value])
    return rows


def build_ner_table(entities: dict) -> list[list[str]]:
    rows = []
    for ner_key, label in NER_LABELS.items():
        values = entities.get(ner_key, [])
        uniq = list(dict.fromkeys(v for v in values if v))
        value_str = ", ".join(uniq) if uniq else "не определено"
        rows.append([label, value_str])
    return rows


def process_pdf(pdf_file):
    logger.info("=" * 50)
    logger.info("PDF processing started")

    if pdf_file is None:
        return "Файл не загружен", "", [], []

    logger.info(f"Uploaded file: {pdf_file.name}")

    try:
        logger.info(f"Text extraction started | file={pdf_file.name}")
        text = pdf_extract(pdf_file.name)
        logger.info(f"Text: {text}")
        logger.info(f"Text extraction finished | chars={len(text)}")
    except Exception as e:
        logger.exception("Text extraction failed")
        return f"Ошибка при извлечении текста: {e}", "", [], []

    try:
        logger.info("Classification started")
        # classify_text уже включает: Qwen-суммаризацию, каскадную KERYX-классификацию (L2/L3/L4),
        # KERYX-классификацию справочных полей и Qwen-NER
        summary, l2_text, l3_text, l4_text, field_predictions, entities = classify_text(text)
        logger.info("Classification finished")
        try:
            l4_codes = parse_l4_codes(l4_text)
            xml_path = build_xml_from_results(summary, l4_codes, field_predictions, entities, pdf_file.name)
            logger.info(f"XML exported: {xml_path}")
        except Exception as e:
            logger.exception("XML export failed")
    except Exception as e:
        logger.exception("Classification failed")
        return text, f"Ошибка классификации: {e}", [], []

    fields_table = build_fields_table(field_predictions)
    ner_table = build_ner_table(entities)

    logger.info("PDF processing finished")

    return summary, l4_text, fields_table, ner_table


with gr.Blocks(title="Классификатор обращений") as demo:
    gr.Markdown("# Классификация обращений граждан")
    gr.Markdown("Загрузите PDF с обращением – получите суммаризацию, рубрику и ключевые поля.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="PDF-файл", file_types=[".pdf"])
            btn = gr.Button("Обработать", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Результат")
            summary_text = gr.Textbox(label="Суммаризация", lines=4)
            l4_output = gr.Textbox(label="Категория(-и) обращения", lines=4)

            gr.Markdown("#### Справочные поля")
            fields_output = gr.Dataframe(
                headers=["Поле", "Значение"],
                datatype=["str", "str"],
                col_count=(2, "fixed"),
                interactive=False,
                wrap=True,
            )

            gr.Markdown("#### Личные данные заявителя")
            ner_output = gr.Dataframe(
                headers=["Поле", "Значение"],
                datatype=["str", "str"],
                col_count=(2, "fixed"),
                interactive=False,
                wrap=True,
            )

    btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=[summary_text, l4_output, fields_output, ner_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
