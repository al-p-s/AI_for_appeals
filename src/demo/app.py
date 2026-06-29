import gradio as gr
import logging

from src.features.OCR.pdf_extractor_test import pdf_extract
from single_inference import classify_text
from ner_inference import extract_entities

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/full_pipeline.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Человекочитаемые названия полей
FIELD_LABELS = {
    "AppealKind": "Вид обращения",
    "ConsiderationType": "Тип обращения (первичное/повторное)",
    "ItemID": "Форма обращения",
    "PetitionerDistrict": "Район проживания заявителя",
    "RegistrationPlaceId": "Место события",
    "StatusId": "Тип обращения",
    "DeliveryTypeId": "Источник поступления",
    "PetitionerCategory": "Категория заявителя",
}

NER_LABELS = {
    "LAST_NAME": "Фамилия",
    "FIRST_NAME": "Имя",
    "MIDDLE_NAME": "Отчество",
    "PERSONAL_EMAIL": "Email",
    "PHONE_NUMBER": "Телефон",
    "ADDRESS": "Адрес",
}


def format_field_predictions(field_predictions: dict) -> str:
    lines = []
    for field_name, value in field_predictions.items():
        label = FIELD_LABELS.get(field_name, field_name)
        lines.append(f"{label}: {value}")
    return "\n".join(lines) if lines else "Поля не определены"


def format_ner_readable(entities: dict) -> str:
    lines = []
    for ner_key, label in NER_LABELS.items():
        values = entities.get(ner_key, [])
        uniq = list(dict.fromkeys(values))
        value_str = ", ".join(uniq) if uniq else "не определено"
        lines.append(f"{label}: {value_str}")
    return "\n".join(lines)


def process_pdf(pdf_file):
    logger.info("=" * 50)
    logger.info("PDF processing started")
    logger.info(f"Uploaded file: {pdf_file.name}")

    if pdf_file is None:
        return "Файл не загружен", "", "", ""

    try:
        logger.info(f"Text extraction started | file={pdf_file.name}")
        text = pdf_extract(pdf_file.name)
        logger.info(f"Text: {text}")
        logger.info(f"Text extraction finished | chars={len(text)}")
    except Exception as e:
        logger.exception("Text extraction failed")
        return f"Ошибка при извлечении текста: {e}", "", "", ""

    try:
        logger.info("Classification started")
        summary, l2_text, l3_text, l4_text, field_predictions = classify_text(text)
        logger.info("Classification finished")
    except Exception as e:
        logger.exception("Classification failed")
        return text, f"Ошибка классификации: {e}", "", ""

    extracted_entities = extract_entities(text)
    ner_output = format_ner_readable(extracted_entities)
    fields_output = format_field_predictions(field_predictions)

    logger.info("PDF processing finished")

    return (
        summary,
        l4_text,
        ner_output,
        fields_output,
    )


with gr.Blocks(title="Классификатор обращений") as demo:
    gr.Markdown("#Классификация обращений граждан")
    gr.Markdown("Загрузите PDF с обращением – получите суммаризацию, рубрику и ключевые поля.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="PDF-файл", file_types=[".pdf"])
            btn = gr.Button("Обработать", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Результат")
            summary_text = gr.Textbox(label="Суммаризация", lines=4)
            l4_output = gr.Textbox(label="Категория(-и) обращения")
            ner_output = gr.Textbox(label="Личные данные заявителя", lines=7)
            fields_output = gr.Textbox(label="Классифицированные поля", lines=10)

    btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=[summary_text, l4_output, ner_output, fields_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
