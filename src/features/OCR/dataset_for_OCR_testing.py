import os
import json
import base64
import time
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path

load_dotenv()

client = OpenAI()

PDF_DIRECTORY = r'E:/работа/новые выбранные'
OUTPUT_PATH = "../../../data/OCR_improving/OCR_test.json"
POPPLER_PATH = os.getenv('POPPLER_PATH')

MODEL = "gpt-4o"
DPI = 150

PROMPT = """Ты — точный OCR-движок. Перед тобой скан документа на русском языке
(возможны короткие вставки на английском - email, домены, технические термины).

Распознай ВЕСЬ текст на изображении максимально точно, сохраняя:
- оригинальные переносы строк и абзацы
- порядок чтения (сверху вниз, слева направо)
- email-адреса и латинские вставки символ в символ, без "исправлений"
- числа, даты, знаки препинания как есть

Если какой-то фрагмент текста неразборчив или повреждён - пропускай,
НЕ придумывай и не угадывай содержание. Не добавляй никаких комментариев от себя,
никаких "вот распознанный текст" - выведи ТОЛЬКО сам распознанный текст документа."""


def pdf_to_base64_images(pdf_path, dpi=DPI):
    """Конвертирует все страницы PDF в список base64-encoded PNG."""
    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=POPPLER_PATH, fmt='png')
    encoded = []
    for img in images:
        buf = BytesIO()
        img.save(buf, format='PNG')
        encoded.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
    return encoded


def recognize_with_vision(pdf_path, max_retries=3):
    """Распознаёт текст всех страниц PDF через GPT-4o Vision, склеивает по страницам."""
    images_b64 = pdf_to_base64_images(pdf_path)
    page_texts = []

    for page_num, img_b64 in enumerate(images_b64):
        content = [
            {"type": "text", "text": PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"},
            },
        ]

        last_error = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": content}],
                    temperature=0,
                )
                page_text = response.choices[0].message.content.strip()
                page_texts.append(page_text)
                break
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                print(f"  Страница {page_num}: ошибка ({e}), повтор через {wait}с...")
                time.sleep(wait)
        else:
            print(f"  Страница {page_num}: не удалось распознать после {max_retries} попыток ({last_error})")
            page_texts.append(f"[ОШИБКА РАСПОЗНАВАНИЯ: {last_error}]")

    return '\n'.join(page_texts)


def main():
    if not os.path.isdir(PDF_DIRECTORY):
        raise SystemExit(f"Директория не найдена: {PDF_DIRECTORY}")

    folders = sorted(
        f for f in os.listdir(PDF_DIRECTORY)
        if os.path.isdir(os.path.join(PDF_DIRECTORY, f))
    )

    existing = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            existing = {item['file_name']: item for item in data.get('appeals', [])}
        print(f"Найден существующий черновик: {len(existing)} документов уже обработано, пропускаем их")

    results = list(existing.values())

    i = 0

    for folder in folders:
        if i == 50:
            break

        if folder in existing:
            continue

        folder_path = os.path.join(PDF_DIRECTORY, folder)
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        if not pdf_files:
            print(f"PDF не найден в {folder}, пропуск")
            continue

        pdf_path = os.path.join(folder_path, pdf_files[0])
        print(f"Обработка: {folder}")

        text = recognize_with_vision(pdf_path)

        results.append({
            "file_name": folder,
            "text": text,
        })

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"appeals": results}, f, ensure_ascii=False, indent=4)

        i = i+1

    print(f"\nГотово. Черновик сохранён в {OUTPUT_PATH}")
    print(f"Всего документов: {len(results)}")


if __name__ == "__main__":
    main()
