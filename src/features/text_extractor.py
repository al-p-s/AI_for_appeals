import PyPDF2
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTFigure
import pdfplumber
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD')
POPPLER_PATH = os.getenv('POPPLER_PATH')


def text_extraction(element):
    line_text = element.get_text()
    line_formats = []
    for text_line in element:
        if isinstance(text_line, LTTextContainer):
            for character in text_line:
                if isinstance(character, LTChar):
                    line_formats.append(character.fontname)
                    line_formats.append(character.size)
    return (line_text, list(set(line_formats)))


def extract_table(pdf_path, page_num, table_num):
    with pdfplumber.open(pdf_path) as pdf:
        table = pdf.pages[page_num].extract_tables()[table_num]
    return table


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
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, lang='rus+eng')


def pdf_extract(pdf_path):
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

                    current_table_idx = 0
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
                                # Вырезаем изображение
                                writer = PyPDF2.PdfWriter()
                                pageObj.mediabox.lower_left = (element.x0, element.y0)
                                pageObj.mediabox.upper_right = (element.x1, element.y1)
                                writer.add_page(pageObj)
                                with open(cropped_path, 'wb') as f:
                                    writer.write(f)

                                images = convert_from_path(
                                    cropped_path,
                                    poppler_path=POPPLER_PATH
                                )
                                images[0].save(image_path, 'PNG')
                                page_content.append(image_to_text(image_path))
                            except Exception as e:
                                print(f"Ошибка при обработке изображения на стр. {pagenum}: {e}")

                    all_content.append(''.join(page_content))

    return '\n'.join(all_content)
