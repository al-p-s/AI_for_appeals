import os
import gradio as gr

from extract_text import pdf_extract
from single_inference import classify_text

def process_pdf(pdf_file):
    if pdf_file is None:
        return "Файл не загружен", "", "", ""

    try:
        text = pdf_extract(pdf_file.name)
    except Exception as e:
        return f"Ошибка при извлечении текста: {e}", "", "", ""

    try:
        summary, l2_code, l2_name, l3_code, l3_name, l4_code, l4_name = classify_text(text)
    except Exception as e:
        return text, f"Ошибка классификации: {e}", "", "", ""

    return (
        summary,
        f"{l2_code} — {l2_name}",
        f"{l3_code} — {l3_name}",
        f"{l4_code} — {l4_name}",
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
            l2_output = gr.Textbox(label="Категория L2")
            l3_output = gr.Textbox(label="Категория L3")
            l4_output = gr.Textbox(label="Категория L4")

    btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=[summary_text, l2_output, l3_output, l4_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
