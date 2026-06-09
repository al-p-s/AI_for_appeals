import gradio as gr
import logging

from extract_text import pdf_extract
from single_inference import classify_text
from ner_inference import extract_entities, format_ner_entities

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("full_pipeline.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def process_pdf(pdf_file):

    logger.info("=" * 50)
    logger.info("PDF processing started")
    logger.info(f"Uploaded file: {pdf_file.name}")

    if pdf_file is None:
        return "Файл не загружен"

    try:
        logger.info(f"Text extraction started | file={pdf_file.name}")
        text = pdf_extract(pdf_file.name)
        logger.info(f"Text: {text}")
        logger.info(f"Text extraction finished | chars={len(text)}")
    except Exception as e:
        logger.exception("Text extraction failed")
        return f"Ошибка при извлечении текста: {e}"

    try:
        logger.info("Classification started")
        summary, l2_text, l3_text, l4_text = classify_text(text)
        logger.info("Classification finished")
    except Exception as e:
        logger.exception("Classification failed")
        return text, f"Ошибка классификации: {e}"

    extracted_entities = extract_entities(text)
    entities = format_ner_entities(extracted_entities)

    logger.info("PDF processing finished")

    return (
        summary,
        # l2_text,
        # l3_text,
        l4_text,
        entities,
    )

with gr.Blocks(title="Классификатор обращений") as demo:
    gr.Markdown("# 📄 Классификация обращений граждан")
    gr.Markdown("Загрузите PDF с обращением – получите суммаризацию и рубрику по трём уровням.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="PDF-файл", file_types=[".pdf"])
            btn = gr.Button("Обработать", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Результат")
            summary_text = gr.Textbox(label="Суммаризация", lines=4)
            # l2_output = gr.Textbox(label="Категория L2")
            # l3_output = gr.Textbox(label="Категория L3")
            l4_output = gr.Textbox(label="Категория(-и) обращения:")
            ner_output = gr.Textbox(label="Ключевые поля", lines=10)

    btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=[summary_text, l4_output, ner_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
