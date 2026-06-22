import os
import tempfile
from dotenv import load_dotenv
import re
import easyocr

import pdfplumber
import PyPDF2
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTFigure
from pdf2image import convert_from_path
from PIL import Image
import numpy as np

from paddleocr import PaddleOCR

load_dotenv()

POPPLER_PATH = os.getenv('POPPLER_PATH')

_ocr_engine = None
_easyocr_engine = None


CYRILLIC_TO_LATIN = str.maketrans({
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'х': 'x', 'у': 'y',
    'і': 'i', 'п': 'n', 'г': 'r', 'к': 'k', 'м': 'm', 'н': 'h', 'б': 'b',
    'и': 'u', 'д': 'd', 'т': 'т',
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O',
    'Р': 'P', 'С': 'C', 'Т': 'T', 'Х': 'X', 'У': 'Y', 'Г': 'G', 'П': 'P',
})

# Email-подстрока внутри произвольного текста (не вся строка целиком!)
EMAIL_SEARCH_RE = re.compile(
    r'[A-Za-zА-Яа-я0-9._%+\-]+@[A-Za-zА-Яа-я0-9.\-]+\.[A-Za-zА-Яа-я]{2,}'
)
EMAIL_STRICT_RE = re.compile(r'^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$')

KNOWN_TLDS = ['ru', 'com', 'net', 'org']
KNOWN_DOMAIN_FRAGMENTS = ['mail', 'gmail', 'yandex', 'cheladmin', 'inbox', 'icloud', 'mvd', 'urfo']


PHONE_RE = re.compile(
    r'(?<!\d)'          # нет цифры слева  — не часть длинного числа
    r'(\+?[78])'        # код страны: +7, 7, 8 (+ опциональный)
    r'[\s\-]?'          # разделитель после кода
    r'[\(\[]?'          # открывающая скобка (необязательно)
    r'(\d{3})'          # код оператора  (3 цифры)
    r'[\)\]]?'          # закрывающая скобка
    r'[\s\-]?'          # разделитель
    r'(\d{3})'          # первые 3 цифры
    r'[\s\-]?'          # разделитель
    r'(\d{2})'          # цифры 4–5
    r'[\s\-]?'          # разделитель
    r'(\d{2})'          # цифры 6–7
    r'(?!\d)',          # нет цифры справа — не часть длинного числа
)


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(
            lang='ru',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,  # выключено: баг oneDNN/PIR на CPU в paddlepaddle 3.x (см. https://github.com/PaddlePaddle/Paddle/issues, "ConvertPirAttribute2RuntimeAttribute")
            # device='gpu',  # раскомментируй, если есть CUDA и paddlepaddle-gpu установлен
        )
    return _ocr_engine


def get_easyocr_engine():
    global _easyocr_engine
    if _easyocr_engine is None:
        # lang_list=['en'] достаточно: нас интересуют только латинские email-кандидаты
        # gpu=False - если нет CUDA; поставь True, если есть
        _easyocr_engine = easyocr.Reader(['en'], gpu=False)
    return _easyocr_engine


def text_extraction(element):
    line_text = element.get_text()
    line_formats = []
    for text_line in element:
        if isinstance(text_line, LTTextContainer):
            for character in text_line:
                if isinstance(character, LTChar):
                    line_formats.append(character.fontname)
                    line_formats.append(character.size)
    return line_text, list(set(line_formats))


def extract_table(pdf_path, page_num, table_num):
    with pdfplumber.open(pdf_path) as pdf:
        return pdf.pages[page_num].extract_tables()[table_num]


def table_converter(table):
    rows = []
    for row in table:
        cleaned = [
            item.replace('\n', ' ') if item and '\n' in item
            else '' if item is None
            else item
            for item in row
        ]
        rows.append('|' + '|'.join(cleaned) + '|')
    return '\n'.join(rows)


def is_element_inside_any_table(element, page, tables):
    x0, y0up, x1, y1up = element.bbox
    y0 = page.bbox[3] - y1up
    y1 = page.bbox[3] - y0up
    for table in tables:
        tx0, ty0, tx1, ty1 = table.bbox
        if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
            return True
    return False


def find_table_for_element(element, page, tables):
    x0, y0up, x1, y1up = element.bbox
    y0 = page.bbox[3] - y1up
    y1 = page.bbox[3] - y0up
    for i, table in enumerate(tables):
        tx0, ty0, tx1, ty1 = table.bbox
        if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
            return i
    return None


def find_email_candidate(text):
    """Ищет email-подстроку ВНУТРИ строки, возвращает Match или None."""
    return EMAIL_SEARCH_RE.search(text)


def is_valid_email(text):
    return bool(EMAIL_STRICT_RE.match(text.strip()))


def fix_homoglyphs(text):
    return text.translate(CYRILLIC_TO_LATIN)


def estimate_substring_bbox(full_text, match, line_box):
    """Грубая оценка x-координат email-подстроки внутри bbox всей строки,
    исходя из доли символов (предполагаем примерно равноширокий шрифт)."""
    x1, y1, x2, y2 = line_box
    total_len = max(len(full_text), 1)
    char_w = (x2 - x1) / total_len
    sub_x1 = x1 + match.start() * char_w
    sub_x2 = x1 + match.end() * char_w
    return [sub_x1, y1, sub_x2, y2]


def crop_region(img_array, box, padding=6, upscale=3, extra_left_padding=15):
    h, w = img_array.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1 - padding - extra_left_padding)  # больше запаса слева
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    crop = img_array[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_img = Image.fromarray(crop)
    new_size = (crop_img.width * upscale, crop_img.height * upscale)
    crop_img = crop_img.resize(new_size, Image.LANCZOS)
    return np.array(crop_img)


def reocr_email_with_easyocr(img_array, box):
    crop_array = crop_region(img_array, box, padding=8, upscale=3)
    if crop_array is None:
        return None
    reader = get_easyocr_engine()
    results = reader.readtext(crop_array, detail=0, paragraph=False)
    if not results:
        return None
    candidate = ''.join(results).strip().replace(' ', '')
    return candidate


def normalize_easyocr_email(text):
    """Чинит типичные баги EasyOCR: лишний мусор по краям, потерянные точки, частые опечатки символов."""
    # Убираем мусор вроде '<', '>', '(', ')', ведущие "a:" и т.п.
    text = re.sub(r'^[^A-Za-z0-9]*', '', text)
    text = re.sub(r'[^A-Za-z0-9]*$', '', text)

    # Частые символьные путаницы EasyOCR (НЕ кириллица, а просто плохое распознавание латиницы)
    text = text.replace('O', '0') if text.count('O') == 1 and any(c.isdigit() for c in text) else text
    # ^ осторожно с таким глобальным replace, лучше применять только в доменной части после @, разберём ниже

    if '@' not in text:
        return text

    local, _, domain = text.partition('@')

    # Если в домене нет точки вообще - пробуем вставить перед известным TLD
    if '.' not in domain:
        for tld in KNOWN_TLDS:
            if domain.endswith(tld) and len(domain) > len(tld):
                domain = domain[:-len(tld)] + '.' + tld
                break

    return f'{local}@{domain}'


def correct_email_in_line(text, box, img_array):
    """Находит email-подстроку в строке, чинит её, возвращает строку
    с заменённым (если получилось) email-фрагментом."""
    match = find_email_candidate(text)
    if not match:
        return text  # email в строке вообще не найден - нечего чинить

    raw_email = match.group(0)
    print(f"[DEBUG] candidate found: '{raw_email}'")

    # Шаг 1: дешёвая гомоглиф-коррекция
    fixed = fix_homoglyphs(raw_email)
    if is_valid_email(fixed):
        print(f"[DEBUG] homoglyph fix succeeded: '{raw_email}' -> '{fixed}'")
        return text[:match.start()] + fixed + text[match.end():]

    print(f"[DEBUG] homoglyph fix failed, calling easyocr on box={box}")

    # Шаг 2: гомоглифы не помогли - зовём EasyOCR на узкий кроп ИМЕННО email-подстроки
    sub_box = estimate_substring_bbox(text, match, box)
    easyocr_text = reocr_email_with_easyocr(img_array, sub_box)

    if easyocr_text:
        easyocr_text = normalize_easyocr_email(easyocr_text)
        easy_match = find_email_candidate(easyocr_text)
        candidate = easy_match.group(0) if easy_match else easyocr_text

        if is_valid_email(candidate):
            return text[:match.start()] + candidate + text[match.end():]

        candidate_fixed = fix_homoglyphs(candidate)
        if is_valid_email(candidate_fixed):
            return text[:match.start()] + candidate_fixed + text[match.end():]

    print(f"[email OCR] не удалось восстановить: paddle='{raw_email}' easyocr='{easyocr_text}'")
    return text  # не получилось - оставляем оригинал


def normalize_phone(match: re.Match) -> str:
    """Принимает Match, возвращает строку '7XXXXXXXXXX' (11 цифр)."""
    groups = match.groups()           # (+7/7/8, ddd, ddd, dd, dd)
    digits = ''.join(groups).replace('+', '')
    if digits.startswith('8'):
        digits = '7' + digits[1:]
    print(f"[DEBUG phone] '{match.group(0).strip()}' -> '{digits}'")
    return digits


def normalize_phones_in_text(text: str) -> str:
    """
    Находит все телефонные номера в тексте и заменяет их нормализованной
    формой '7XXXXXXXXXX'.  Остальной текст не трогается.
    """
    return PHONE_RE.sub(normalize_phone, text)


def image_to_text(image_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    engine = get_ocr_engine()
    results = engine.predict(img_array)
    if not results:
        return ''

    res = results[0]
    texts = res.get('rec_texts') if hasattr(res, 'get') else res['rec_texts']
    boxes = res.get('rec_boxes') if hasattr(res, 'get') else res['rec_boxes']
    if not texts:
        return ''

    lines = []
    for text, box in zip(texts, boxes):
        top_y = float(box[1])
        left_x = float(box[0])

        final_text = text
        if find_email_candidate(text):
            final_text = correct_email_in_line(text, box, img_array)

        lines.append((top_y, left_x, final_text))

    lines.sort(key=lambda x: (round(x[0] / 10), x[1]))
    raw_text = '\n'.join(text for _, _, text in lines)
    return normalize_phones_in_text(raw_text)


def pdf_extract(pdf_path, dpi=100):
    all_content = []

    with tempfile.TemporaryDirectory() as tmpdir:
        cropped_path = os.path.join(tmpdir, 'cropped.pdf')
        image_path = os.path.join(tmpdir, 'page.png')

        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            with pdfplumber.open(pdf_path) as plumber_pdf:

                for pagenum, page in enumerate(extract_pages(pdf_path)):
                    page_content = []
                    pageObj = pdf_reader.pages[pagenum]

                    tables = plumber_pdf.pages[pagenum].find_tables()
                    text_from_tables = [
                        table_converter(
                            plumber_pdf.pages[pagenum].extract_tables()[i]
                        )
                        for i in range(len(tables))
                    ]

                    added_tables = set()

                    page_elements = sorted(
                        [(el.y1, el) for el in page._objs],
                        key=lambda a: a[0],
                        reverse=True
                    )

                    for _, element in page_elements:

                        if tables and is_element_inside_any_table(element, page, tables):
                            table_idx = find_table_for_element(element, page, tables)
                            if table_idx is not None and table_idx not in added_tables:
                                page_content.append(text_from_tables[table_idx])
                                added_tables.add(table_idx)
                            continue

                        if isinstance(element, LTTextContainer):
                            line_text, _ = text_extraction(element)
                            page_content.append(line_text)

                        elif isinstance(element, LTFigure):
                            try:
                                # Вырезаем область изображения (без deepcopy!)
                                writer = PyPDF2.PdfWriter()
                                pageObj.mediabox.lower_left = (element.x0, element.y0)
                                pageObj.mediabox.upper_right = (element.x1, element.y1)
                                writer.add_page(pageObj)

                                with open(cropped_path, 'wb') as f:
                                    writer.write(f)

                                images = convert_from_path(
                                    cropped_path,
                                    dpi=dpi,
                                    poppler_path=POPPLER_PATH,
                                    fmt='png',
                                    grayscale=True
                                )
                                images[0].save(image_path, 'PNG')
                                ocr_text = image_to_text(image_path)
                                if ocr_text.strip():
                                    page_content.append(ocr_text)
                            except Exception as e:
                                print(f"Ошибка при обработке изображения на стр. {pagenum}: {e}")

                    all_content.append(''.join(page_content))

    result = '\n'.join(all_content)
    return normalize_phones_in_text(result)