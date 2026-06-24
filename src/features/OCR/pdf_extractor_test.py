import os
import re
import tempfile
from dotenv import load_dotenv

load_dotenv()

cudnn_path = os.getenv('CUDNN_PATH')
if cudnn_path:
    os.environ['PATH'] = cudnn_path + os.pathsep + os.environ.get('PATH', '')

import easyocr
import numpy as np
import pdfplumber
import PyPDF2
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LTFigure, LTTextContainer
from PIL import Image
from pathlib import Path


load_dotenv()

POPPLER_PATH = os.getenv('POPPLER_PATH')
EASYOCR_MODEL_DIR = os.getenv('EASYOCR_MODEL_DIR', str(Path.home() / '.EasyOCR' / 'model'))

_ocr_engine = None
_easyocr_engine = None


# Кириллические гомоглифы → латинские эквиваленты (для коррекции email)
CYRILLIC_TO_LATIN = str.maketrans({
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'х': 'x', 'у': 'y',
    'і': 'i', 'п': 'n', 'г': 'r', 'к': 'k', 'м': 'm', 'н': 'h', 'б': 'b',
    'и': 'u', 'д': 'd', 'т': 't',
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O',
    'Р': 'P', 'С': 'C', 'Т': 'T', 'Х': 'X', 'У': 'Y', 'Г': 'G', 'П': 'P',
})

# Email-подстрока внутри произвольного текста (не вся строка целиком)
EMAIL_SEARCH_RE = re.compile(
    r'[A-Za-zА-Яа-я0-9._%+\-]+@[A-Za-zА-Яа-я0-9.\-]+\.[A-Za-zА-Яа-я]{2,}'
)
# Валидный email целиком (только латиница)
EMAIL_STRICT_RE = re.compile(
    r'^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$'
)

KNOWN_TLDS = ['ru', 'com', 'net', 'org']

EASYOCR_DOMAIN_FIXES = str.maketrans({
    'т': 'm',   # mail.ru → тail.ru → mailru (EasyOCR: m → т)
    'п': 'n',   # yandex → яndex и подобное (EasyOCR: n → п)
})

CYRILLIC_TO_LATIN_DOMAIN = str.maketrans({
    **{ord(k): ord(v) for k, v in {
        'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'х': 'x', 'у': 'y',
        'і': 'i', 'п': 'n', 'г': 'r', 'к': 'k', 'м': 'm', 'н': 'h', 'б': 'b',
        'и': 'u', 'д': 'd',
        'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O',
        'Р': 'P', 'С': 'C', 'Т': 'T', 'Х': 'X', 'У': 'Y', 'Г': 'G', 'П': 'P',
    }.items()},
    # Переопределяем т→m (приоритет над т→t из CYRILLIC_TO_LATIN)
    ord('т'): ord('m'),
})


# Форматы: +7/7/8, скобки, дефисы, пробелы — строго 11 цифр (код страны + 10)
PHONE_RE = re.compile(
    r'(?<!\d)'      # нет цифры слева — не часть длинного числа
    r'(\+?[78])'    # код страны: +7, 7, 8
    r'[\s\-]?'      # разделитель после кода
    r'[\(\[]?'      # открывающая скобка (необязательно)
    r'(\d{3})'      # код оператора (3 цифры)
    r'[\)\]]?'      # закрывающая скобка
    r'[\s\-]?'      # разделитель
    r'(\d{3})'      # первые 3 цифры номера
    r'[\s\-]?'      # разделитель
    r'(\d{2})'      # цифры 4–5
    r'[\s\-]?'      # разделитель
    r'(\d{2})'      # цифры 6–7
    r'(?!\d)',      # нет цифры справа — не часть длинного числа
)

# =============================================================================
# OCR-движки
# =============================================================================

def get_ocr_engine() -> PaddleOCR:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(
            lang='ru',
            text_detection_model_name='PP-OCRv5_server_det',
            text_recognition_model_name='eslav_PP-OCRv5_mobile_rec',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _ocr_engine


def get_easyocr_engine() -> easyocr.Reader:
    global _easyocr_engine
    if _easyocr_engine is None:
        _easyocr_engine = easyocr.Reader(
            ['en'],
            gpu=False,
            model_storage_directory=EASYOCR_MODEL_DIR,
        )
    return _easyocr_engine

# =============================================================================
# PDF — вспомогательные функции
# =============================================================================

def text_extraction(element) -> tuple[str, list]:
    """Извлекает текст и список шрифтов/размеров из LTTextContainer."""
    line_text = element.get_text()
    line_formats = []
    for text_line in element:
        if isinstance(text_line, LTTextContainer):
            for character in text_line:
                if isinstance(character, LTChar):
                    line_formats.append(character.fontname)
                    line_formats.append(character.size)
    return line_text, list(set(line_formats))


def extract_table(pdf_path: str, page_num: int, table_num: int) -> list:
    """Возвращает сырую таблицу с указанной страницы."""
    with pdfplumber.open(pdf_path) as pdf:
        return pdf.pages[page_num].extract_tables()[table_num]


def table_converter(table: list) -> str:
    """Конвертирует таблицу pdfplumber в markdown-подобный текст (|col|col|)."""
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


def is_element_inside_any_table(element, page, tables) -> bool:
    """Проверяет, попадает ли элемент внутрь любой из таблиц на странице."""
    x0, y0up, x1, y1up = element.bbox
    y0 = page.bbox[3] - y1up
    y1 = page.bbox[3] - y0up
    for table in tables:
        tx0, ty0, tx1, ty1 = table.bbox
        if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
            return True
    return False


def find_table_for_element(element, page, tables) -> int | None:
    """Возвращает индекс таблицы, в которую попадает элемент, или None."""
    x0, y0up, x1, y1up = element.bbox
    y0 = page.bbox[3] - y1up
    y1 = page.bbox[3] - y0up
    for i, table in enumerate(tables):
        tx0, ty0, tx1, ty1 = table.bbox
        if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
            return i
    return None

# =============================================================================
# Email — коррекция
# =============================================================================

def find_email_candidate(text: str) -> re.Match | None:
    """Ищет email-подстроку внутри строки, возвращает Match или None."""
    return EMAIL_SEARCH_RE.search(text)


def is_valid_email(text: str) -> bool:
    """Проверяет, является ли строка валидным email (только латиница)."""
    return bool(EMAIL_STRICT_RE.match(text.strip()))


def fix_homoglyphs(text: str) -> str:
    """Заменяет кириллические гомоглифы латинскими эквивалентами.

    Local-part и domain обрабатываются разными таблицами:
    - local: CYRILLIC_TO_LATIN (т→t)
    - domain: CYRILLIC_TO_LATIN_DOMAIN (т→m, т.к. m чаще встречается в доменах)
    """
    if '@' not in text:
        return text.translate(CYRILLIC_TO_LATIN)

    local, _, domain = text.partition('@')
    return local.translate(CYRILLIC_TO_LATIN) + '@' + domain.translate(CYRILLIC_TO_LATIN_DOMAIN)


def estimate_substring_bbox(full_text: str, match: re.Match, line_box: list) -> list:
    """Грубая оценка x-координат подстроки внутри bbox строки.

    Предполагает равноширокий шрифт — используется для кропа email-кандидата.
    """
    x1, y1, x2, y2 = line_box
    char_w = (x2 - x1) / max(len(full_text), 1)
    sub_x1 = x1 + match.start() * char_w
    sub_x2 = x1 + match.end() * char_w
    return [sub_x1, y1, sub_x2, y2]


def crop_region(
    img_array: np.ndarray,
    box: list,
    padding: int = 6,
    upscale: int = 3,
    extra_left_padding: int = 15,
) -> np.ndarray | None:
    """Вырезает и масштабирует регион изображения для повторного OCR."""
    h, w = img_array.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1 - padding - extra_left_padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    crop = img_array[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_img = Image.fromarray(crop).resize(
        (crop.shape[1] * upscale, crop.shape[0] * upscale),
        Image.LANCZOS,
    )
    return np.array(crop_img)


def reocr_email_with_easyocr(img_array: np.ndarray, box: list) -> str | None:
    """Запускает EasyOCR на кропе региона, возвращает склеенный текст или None."""
    crop_array = crop_region(img_array, box, padding=8, upscale=3)
    if crop_array is None:
        return None
    results = get_easyocr_engine().readtext(crop_array, detail=0, paragraph=False)
    if not results:
        return None
    return ''.join(results).strip().replace(' ', '')


# Regex для обрезки local-part и domain вокруг @
_LOCAL_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._%-+]*$')
_DOMAIN_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9.-]*')
# Fallback: ищем нижнерегистровый email целиком в замусоренной строке EasyOCR
_EMAIL_IN_JUNK_RE = re.compile(r'[a-z0-9][a-z0-9._%-+]{2,}@[a-z0-9][a-z0-9.-]+')


def _extract_around_at(text: str) -> str:
    """Вырезает email-подстроку вокруг @ из замусоренного вывода EasyOCR.

    EasyOCR часто добавляет мусор перед email ('4Ta:evendi@mailmu',
    'LOLWldLlU):CKatransport@cheladmintu'). Простой strip не помогает,
    потому что мусор содержит буквы и цифры.

    Стратегия:
    1. Берём максимальный local-part перед @ и domain после @.
    2. Если local начинается с блока заглавных — обрезаем до первой строчной.
    3. Если заглавные остались — ищем email целиком в нижнем регистре (fallback).
    """
    at_pos = text.find('@')
    if at_pos == -1:
        return text

    local_match = _LOCAL_RE.search(text[:at_pos])
    domain_match = _DOMAIN_RE.match(text[at_pos + 1:])
    local = local_match.group(0) if local_match else ''
    domain = domain_match.group(0) if domain_match else ''

    if not local or not domain:
        return text

    # Эвристика: блок из 2+ заглавных в начале local — скорее всего мусор
    if re.match(r'^[A-Z]{2,}', local):
        transition = re.search(r'(?<=[A-Z])([a-z])', local)
        if transition:
            local = local[transition.start():]

    # Fallback: если заглавные в начале остались — ищем email в нижнем регистре
    if re.match(r'^[A-Z]', local):
        m = _EMAIL_IN_JUNK_RE.search(text.lower())
        if m:
            return text[m.start():m.end()]

    return f'{local}@{domain}'


def normalize_easyocr_email(text: str) -> str:
    """Чинит типичные артефакты EasyOCR.

    Порядок шагов:
    1. _extract_around_at — вырезаем email-подстроку, убирая мусор-префиксы.
    2. EASYOCR_DOMAIN_FIXES — кириллица т→m, п→n в доменной части.
    3. mu/nu/tu на конце домена → .ru  (inboxnu, mailmu, cheladmintu).
    4. tuz на конце → .ru  (mailtuz — ещё один артефакт EasyOCR).
    5. Нет точки в домене — вставляем перед известным TLD.
    """
    if '@' not in text:
        return text

    text = _extract_around_at(text)

    if '@' not in text:
        return text

    local, _, domain = text.partition('@')

    # Шаг 2: кириллические гомоглифы специфичные для EasyOCR (только домен)
    domain = domain.translate(EASYOCR_DOMAIN_FIXES)

    # Шаг 3–4: артефакты окончания .ru
    domain = re.sub(r'(?<=[a-z])(mu|nu|tu)$', '.ru', domain)
    domain = re.sub(r'(?<=[a-z])tuz$', '.ru', domain)

    # Шаг 5: нет точки в домене — вставляем перед известным TLD
    if '.' not in domain:
        for tld in KNOWN_TLDS:
            if domain.endswith(tld) and len(domain) > len(tld):
                domain = domain[:-len(tld)] + '.' + tld
                break

    return f'{local}@{domain}'


def correct_email_in_line(text: str, box: list, img_array: np.ndarray) -> str:
    """Находит email-кандидата в строке и пытается его исправить.

    Шаг 1: дешёвая замена гомоглифов.
    Шаг 2: если не помогло — повторный OCR через EasyOCR на кропе подстроки.
    Возвращает строку с заменённым email или оригинал, если восстановить не удалось.
    """
    match = find_email_candidate(text)
    if not match:
        return text

    raw_email = match.group(0)
    print(f"[DEBUG email] candidate: '{raw_email}'")

    # Шаг 1 — гомоглифы
    fixed = fix_homoglyphs(raw_email)
    if is_valid_email(fixed):
        print(f"[DEBUG email] homoglyph fix: '{raw_email}' -> '{fixed}'")
        return text[:match.start()] + fixed + text[match.end():]

    print(f"[DEBUG email] homoglyph fix failed, calling EasyOCR on box={box}")

    # Шаг 2 — EasyOCR на кропе
    sub_box = estimate_substring_bbox(text, match, box)
    easyocr_raw = reocr_email_with_easyocr(img_array, sub_box)

    if easyocr_raw:
        easyocr_text = normalize_easyocr_email(easyocr_raw)
        easy_match = find_email_candidate(easyocr_text)
        candidate = easy_match.group(0) if easy_match else easyocr_text

        if is_valid_email(candidate):
            return text[:match.start()] + candidate + text[match.end():]

        candidate_fixed = fix_homoglyphs(candidate)
        if is_valid_email(candidate_fixed):
            return text[:match.start()] + candidate_fixed + text[match.end():]

    print(f"[DEBUG email] failed: paddle='{raw_email}' easyocr='{easyocr_raw}'")
    return text

# =============================================================================
# Телефон — нормализация
# =============================================================================

def normalize_phone(match: re.Match) -> str:
    """Принимает Match телефона, возвращает '7XXXXXXXXXX' (11 цифр)."""
    digits = ''.join(match.groups()).replace('+', '')
    if digits.startswith('8'):
        digits = '7' + digits[1:]
    print(f"[DEBUG phone] '{match.group(0).strip()}' -> '{digits}'")
    return digits


def normalize_phones_in_text(text: str) -> str:
    """Заменяет все телефонные номера в тексте нормализованной формой '7XXXXXXXXXX'."""
    return PHONE_RE.sub(normalize_phone, text)

# =============================================================================
# Основные функции извлечения текста
# =============================================================================

def image_to_text(image_path: str) -> str:
    """Извлекает текст из изображения через PaddleOCR.

    Для каждой строки применяет коррекцию email.
    В конце нормализует телефонные номера по всему тексту.
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    results = get_ocr_engine().predict(img_array)
    if not results:
        return ''

    res = results[0]
    texts = res.get('rec_texts') if hasattr(res, 'get') else res['rec_texts']
    boxes = res.get('rec_boxes') if hasattr(res, 'get') else res['rec_boxes']
    if not texts:
        return ''

    lines = []
    for text, box in zip(texts, boxes):
        final_text = correct_email_in_line(text, box, img_array) if find_email_candidate(text) else text
        lines.append((float(box[1]), float(box[0]), final_text))

    lines.sort(key=lambda x: (round(x[0] / 10), x[1]))
    raw_text = '\n'.join(t for _, _, t in lines)
    return normalize_phones_in_text(raw_text)


def pdf_extract(pdf_path: str, dpi: int = 100) -> str:
    """Извлекает текст из PDF: текстовый слой через pdfminer, изображения через OCR.

    Таблицы обрабатываются отдельно через pdfplumber и форматируются в markdown.
    В конце нормализует телефонные номера по всему документу.
    """
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
                        table_converter(plumber_pdf.pages[pagenum].extract_tables()[i])
                        for i in range(len(tables))
                    ]
                    added_tables = set()

                    page_elements = sorted(
                        [(el.y1, el) for el in page._objs],
                        key=lambda a: a[0],
                        reverse=True,
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
                                # Вырезаем область изображения (без deepcopy)
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
                                    grayscale=True,
                                )
                                images[0].save(image_path, 'PNG')
                                ocr_text = image_to_text(image_path)
                                if ocr_text.strip():
                                    page_content.append(ocr_text)
                            except Exception as e:
                                print(f"[ERROR] стр. {pagenum}: {e}")

                    all_content.append(''.join(page_content))

    return normalize_phones_in_text('\n'.join(all_content))
