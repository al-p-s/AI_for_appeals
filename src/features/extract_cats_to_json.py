import fitz
import re
import json


## НЕ ЗАПУСКАТЬ ПО ПРИКОЛУ!! ТАМ НАДО ПОСЛЕ ЗАПУСКА ВРУЧНУЮ ПРАВИТЬ НЕСКОЛЬКО КАТЕГОРИЙ!!


def parse_classifier(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    parent_pattern = r'(\d{4}\.\d{4}\.\d{4})\.0000\s+(.+?)(?=\d{4}\.\d{4}|\Z)'
    parents = {}
    for m in re.finditer(parent_pattern, full_text, re.DOTALL):
        pcode = m.group(1)
        pname = re.sub(r'\s+', ' ', m.group(2).strip())
        pname = re.split(r'\d{4}\.\d{4}', pname)[0].strip()
        parents[pcode] = pname

    pattern = r'(\d{4}(?:\.\d{4})+)\s+(.+?)(?=\d{4}\.\d{4}|\Z)'

    results = []
    seen_codes = set()

    for match in re.finditer(pattern, full_text, re.DOTALL):
        code = match.group(1)
        segments = code.split('.')

        if len(segments) != 4:
            continue
        if code.endswith('.0000'):
            continue

        if code in seen_codes:
            continue
        seen_codes.add(code)

        name = match.group(2).strip()
        name = re.sub(r'\s+', ' ', name)
        name = re.sub(r'\s*\d+При.+$', '', name).strip()
        name = re.split(r'\d{4}\.\d{4}', name)[0].strip()

        parent_key = '.'.join(code.split('.')[:3])
        parent_name = parents.get(parent_key, "")

        results.append({
            "code": code,
            "parent_code": parent_key + ".0000",
            "parent_name": parent_name,
            "name": name
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"categories": results}, f, ensure_ascii=False, indent=2)

    print(f"Готово. Найдено категорий: {len(results)}")
    print(f"Сохранено в: {output_path}")

if __name__ == "__main__":
    PDF_PATH = r"../../data/classifier/classifier_2024.pdf"
    OUTPUT_PATH = r"../../data/classifier/cats.json"
    parse_classifier(PDF_PATH, OUTPUT_PATH)
