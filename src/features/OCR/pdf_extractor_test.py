"""
Тестовый скрипт: PDF -> poppler (pdf2image) -> Qwen2.5-VL-7B-Instruct OCR.

Никакого paddle/easyocr/гомоглифов — чистый VLM-пайплайн.
Модель должна быть уже скачана локально (huggingface-cli download ...).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

load_dotenv()

POPPLER_PATH = os.getenv('POPPLER_PATH')

MODEL_PATH = os.getenv('QWEN_MODEL_PATH')

DPI = 200

DEFAULT_PROMPT = (
    "Извлеки весь текст с изображения документа. "
    "Сохрани исходную структуру (абзацы, таблицы как markdown). "
    "Ничего не добавляй от себя, не переводи, не исправляй орфографию. "
    "Верни только распознанный текст, без комментариев."
)

_model = None
_processor = None

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

def get_model():
    global _model, _processor
    if _model is None:
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=bnb_config,
        )
        _processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return _model, _processor


def pdf_to_images(pdf_path: str, dpi: int = DPI) -> list[Image.Image]:
    """Рендерит PDF в список PIL-изображений через poppler."""
    return convert_from_path(
        pdf_path,
        dpi=dpi,
        poppler_path=POPPLER_PATH,
        fmt='png',
    )


def image_to_text(image: Image.Image, prompt: str = DEFAULT_PROMPT) -> str:
    """Прогоняет одно изображение через Qwen2.5-VL, возвращает распознанный текст."""
    model, processor = get_model()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def pdf_extract(pdf_path: str, dpi: int = DPI) -> str:
    """Извлекает текст из всех страниц PDF через Qwen2.5-VL."""
    images = pdf_to_images(pdf_path, dpi=dpi)
    pages_text = []
    for i, image in enumerate(images):
        print(f"[INFO] страница {i + 1}/{len(images)}...")
        text = image_to_text(image)
        pages_text.append(text)
    return '\n\n'.join(pages_text)
