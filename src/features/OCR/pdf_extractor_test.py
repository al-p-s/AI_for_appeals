"""
Тестовый скрипт: PDF -> poppler (pdf2image) -> GLM-OCR (LM Studio) OCR.
Никакого paddle/easyocr/гомоглифов — чистый VLM-пайплайн через локальный LM Studio.
LM Studio должен быть запущен с загруженной моделью glm-ocr.
"""
import os
import base64
from io import BytesIO

from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
from openai import OpenAI

load_dotenv()

POPPLER_PATH = os.getenv('POPPLER_PATH')
LMSTUDIO_BASE_URL = os.getenv('LMSTUDIO_BASE_URL', 'http://127.0.0.1:1234/v1')
LMSTUDIO_MODEL = os.getenv('LMSTUDIO_MODEL', 'glm-ocr')

DPI = 300

DEFAULT_PROMPT = (
    "Извлеки весь текст с изображения документа."
    "Сохрани исходную структуру (абзацы, таблицы как markdown)."
    "Ничего не добавляй от себя, не переводи, не исправляй орфографию."
    "Верни только распознанный текст, без комментариев."
)

client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")


def pdf_to_images(pdf_path: str, dpi: int = DPI) -> list[Image.Image]:
    """Рендерит PDF в список PIL-изображений через poppler."""
    return convert_from_path(
        pdf_path,
        dpi=dpi,
        poppler_path=POPPLER_PATH,
        fmt='png',
    )


def image_to_text(image: Image.Image, prompt: str = DEFAULT_PROMPT) -> str:
    """Прогоняет одно изображение через GLM-OCR, возвращает распознанный текст."""
    buf = BytesIO()
    image.save(buf, format='PNG')
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    response = client.chat.completions.create(
        model=LMSTUDIO_MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


def pdf_extract(pdf_path: str, dpi: int = DPI) -> str:
    """Извлекает текст из всех страниц PDF через GLM-OCR."""
    images = pdf_to_images(pdf_path, dpi=dpi)
    pages_text = []
    for i, image in enumerate(images):
        print(f"[INFO] страница {i + 1}/{len(images)}...")
        text = image_to_text(image)
        pages_text.append(text)
    return '\n\n'.join(pages_text)
