import pymupdf as fitz
import re
import json

def parse_classifier_l2(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    pattern = r'(\d{4}\.\d{4})\.0000\.0000\s+(.+?)(?=\d{4}\.\d{4}|\Z)'
    results = []
    seen = set()

    for m in re.finditer(pattern, full_text, re.DOTALL):
        code_prefix = m.group(1)
        parts = code_prefix.split('.')
        if parts[1] == '0000':
            continue
        full_code = code_prefix + '.0000.0000'
        if full_code in seen:
            continue
        seen.add(full_code)
        name = re.sub(r'\s+', ' ', m.group(2).strip())
        name = re.split(r'\d{4}\.\d{4}', name)[0].strip()
        results.append({"code": full_code, "name": name})

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"categories": results}, f, ensure_ascii=False, indent=2)
    print(f"Готово. L2 категорий: {len(results)}")

if __name__ == "__main__":
    parse_classifier_l2(
        r"../../data/classifier/classifier_2024.pdf",
        r"../../data/classifier/cats2.json"
    )
