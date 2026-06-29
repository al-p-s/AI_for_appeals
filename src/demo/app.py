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

NER_LABELS = {
    "LAST_NAME": "Фамилия",
    "FIRST_NAME": "Имя",
    "MIDDLE_NAME": "Отчество",
    "PERSONAL_EMAIL": "Email",
    "PHONE_NUMBER": "Телефон",
    "ADDRESS": "Адрес",
}

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

    logger.info("PDF processing finished")

    return (
        summary,
        l4_text,
        ner_output,
        field_predictions.get("AppealKind", ""),
        field_predictions.get("ItemID", ""),
        field_predictions.get("DeliveryTypeId", ""),
        field_predictions.get("RegistrationPlaceId", ""),
        field_predictions.get("PetitionerDistrict", ""),
        field_predictions.get("PetitionerCategory", ""),
        field_predictions.get("StatusId", ""),
        field_predictions.get("ConsiderationType", ""),
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
            appeal_kind = gr.Textbox(label="Вид обращения")
            item_id = gr.Textbox(label="Форма обращения")
            delivery_type = gr.Textbox(label="Источник поступления")
            registration_place = gr.Textbox(label="Место события")
            petitioner_district = gr.Textbox(label="Район проживания заявителя")
            petitioner_category = gr.Textbox(label="Категория заявителя")
            status_id = gr.Textbox(label="Тип обращения")
            consideration_type = gr.Textbox(label="Первичное/повторное")

    btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=[summary_text, l4_output, ner_output,
                 appeal_kind, item_id, delivery_type,
                 registration_place, petitioner_district,
                 petitioner_category, status_id, consideration_type]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
