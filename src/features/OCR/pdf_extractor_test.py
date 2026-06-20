import os
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_enable_pir_in_executor'] = '0'
import tempfile
from dotenv import load_dotenv

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

# Инициализация PaddleOCR один раз на уровне модуля (модель тяжёлая, загружать каждый раз дорого)
# lang='cyrillic' покрывает русский + латиницу (английские вставки, email, домены и т.д.)
# use_angle_cls=True — автоматически определяет и поворачивает текст, если он перевёрнут
# show_log=False — отключаем подробный лог Paddle, чтобы не засорять вывод
_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        # API PaddleOCR 3.x: класс теперь строится поверх paddlex-пайплайнов.
        # use_textline_orientation - аналог старого use_angle_cls (поворот строк текста).
        # use_doc_orientation_classify / use_doc_unwarping выключаем - не нужны для уже
        # вырезанных изображений из PDF (это для фото целых документов "с перекосом").
        _ocr_engine = PaddleOCR(
            lang='ru',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,  # выключено: баг oneDNN/PIR на CPU в paddlepaddle 3.x (см. https://github.com/PaddlePaddle/Paddle/issues, "ConvertPirAttribute2RuntimeAttribute")
            # device='gpu',  # раскомментируй, если есть CUDA и paddlepaddle-gpu установлен
        )
    return _ocr_engine


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


def image_to_text(image_path):
    """OCR через PaddleOCR (API 3.x). Возвращает текст, склеенный по строкам,
    отсортированным сверху вниз (порядок чтения не гарантирован движком)."""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    engine = get_ocr_engine()
    results = engine.predict(img_array)

    if not results:
        return ''

    res = results[0]
    # В новом API результат - объект с доступом как у словаря (res['rec_texts'] и т.д.)
    texts = res.get('rec_texts') if hasattr(res, 'get') else res['rec_texts']
    # rec_boxes - список [x1, y1, x2, y2] на строку, используем для сортировки по порядку чтения
    boxes = res.get('rec_boxes') if hasattr(res, 'get') else res['rec_boxes']

    if not texts:
        return ''

    lines = []
    for text, box in zip(texts, boxes):
        top_y = float(box[1])
        left_x = float(box[0])
        lines.append((top_y, left_x, text))

    # Сортировка: сверху вниз, при близком top_y - слева направо
    lines.sort(key=lambda x: (round(x[0] / 10), x[1]))

    return '\n'.join(text for _, _, text in lines)


def pdf_extract(pdf_path, dpi=150):
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

    return '\n'.join(all_content)
