# Извлечение текстового слоя из PDF через GLM-OCR (LM Studio, VLM-пайплайн).

import os
import base64
import logging
from io import BytesIO
from typing import List, Optional

from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

load_dotenv()

POPPLER_PATH = os.getenv("POPPLER_PATH")
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "glm-ocr")

DPI = 300
PAGE_SEPARATOR = "\n\n"

DEFAULT_PROMPT = (
    "Извлеки весь текст с изображения документа."
    "Сохрани исходную структуру (абзацы, таблицы как markdown)."
    "Ничего не добавляй от себя, не переводи, не исправляй орфографию."
    "Верни только распознанный текст, без комментариев."
)


class GlmOcrLoader:

    def __init__(self, base_url: str = LMSTUDIO_BASE_URL, model: str = LMSTUDIO_MODEL):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self._check_connection(base_url)
        logger.info(f"GLM-OCR loaded: {model}")

    def _check_connection(self, base_url: str):
        try:
            self.client.models.list()
            logger.info("LM Studio API available")
        except Exception as e:
            logger.error(f"LM Studio connection error: {e}")

    def image_to_text(self, image: Image.Image, prompt: str = DEFAULT_PROMPT) -> Optional[str]:
        buf = BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
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
            content = response.choices[0].message.content
            if not content:
                logger.warning("GLM-OCR returned empty answer")
            return content
        except Exception as e:
            logger.error(f"GLM-OCR request error: {e}")
            return None


_glm_ocr_instance: Optional[GlmOcrLoader] = None


def get_glm_ocr() -> GlmOcrLoader:
    global _glm_ocr_instance
    if _glm_ocr_instance is None:
        _glm_ocr_instance = GlmOcrLoader()
    return _glm_ocr_instance


def pdf_to_images(pdf_path: str, dpi: int = DPI) -> List[Image.Image]:
    return convert_from_path(
        pdf_path,
        dpi=dpi,
        poppler_path=POPPLER_PATH,
        fmt="png",
    )


# text_extraction_glm_ocr.py
def extract_text_from_pdf(pdf_path: str, dpi: int = 160, prompt: str = DEFAULT_PROMPT) -> str:

    logger.info(f"OCR started: {pdf_path}")

    ocr = get_glm_ocr()
    images = pdf_to_images(pdf_path, dpi=dpi)

    pages_text = []
    for i, image in enumerate(images, start=1):
        logger.info(f"OCR page {i}/{len(images)}: {pdf_path}")
        text = ocr.image_to_text(image, prompt=prompt)
        pages_text.append(text or "")

    full_text = PAGE_SEPARATOR.join(pages_text).strip()
    logger.info(f"OCR finished: {pdf_path} | {len(full_text)} symbols, {len(images)} pages")
    return full_text


logger.info("GLM-OCR extractor is ready")