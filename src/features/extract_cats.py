import os
import re
import json

def decode_xml(path):
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        return raw.decode('utf-16')
    except Exception:
        return raw.decode('utf-8', errors='ignore')


def get_card_id_by_filename(appeals_dir, pdf_name):
    for folder in os.listdir(appeals_dir):
        folder_path = os.path.join(appeals_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for f in os.listdir(folder_path):
            if f.lower() == pdf_name.lower():
                return folder.strip('{}')
    return None


def find_question_xml_by_contract_ref(questions_dir, card_id):
    for fname in os.listdir(questions_dir):
        if not fname.lower().endswith('.xml'):
            continue
        fpath = os.path.join(questions_dir, fname)
        try:
            text = decode_xml(fpath)
        except Exception:
            continue
        if card_id.upper() in text.upper():
            if re.search(rf'ContractRef="{re.escape(card_id)}"', text, re.IGNORECASE):
                return text

    return None


def extract_category(xml_text):
    pattern = r'Name="(\d{4}\.\d{4}\.\d{4}\.\d{4}[^"]*)"'
    m = re.search(pattern, xml_text)
    if m:
        return m.group(1).strip()
    return None


def build_dataset(cropped_appeals_dir, appeals_dir, questions_dir, output_path):
    results = []
    no_folder = []
    no_question = []
    no_category = []

    pdf_files = [f for f in os.listdir(cropped_appeals_dir) if f.lower().endswith('.pdf')]
    print(f"Найдено PDF: {len(pdf_files)}")

    for pdf_name in pdf_files:
        card_id = get_card_id_by_filename(appeals_dir, pdf_name)
        if not card_id:
            print(f"  [!] Папка не найдена для: {pdf_name}")
            no_folder.append(pdf_name)
            continue

        xml_text = find_question_xml_by_contract_ref(questions_dir, card_id)
        if not xml_text:
            print(f"  [!] XML ответа не найден для CardID: {card_id} ({pdf_name})")
            no_question.append(pdf_name)
            continue

        category = extract_category(xml_text)
        if not category:
            print(f"  [!] Категория не найдена в XML ответа для: {pdf_name}")
            no_category.append(pdf_name)
            continue

        results.append({
            "file_name": pdf_name,
            "card_id": card_id,
            "category": category
        })
        # print(f"  [+] {pdf_name} -> {category}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"appeals": results}, f, ensure_ascii=False, indent=4)

    print(f"\n=== Готово ===")
    print(f"Успешно: {len(results)}")
    print(f"Не найдена папка: {len(no_folder)}")
    print(f"Не найден XML ответа: {len(no_question)}")
    print(f"Не найдена категория: {len(no_category)}")
    print(f"Результат сохранён в: {output_path}")


if __name__ == "__main__":
    CROPPED_APPEALS_DIR = r"../../data/cropped_appeals"
    APPEALS_DIR = r"../../data/Обращения январь 2025"
    QUESTIONS_DIR = r"../../data/Обращения январь 2025/Questions"
    OUTPUT_PATH = r"../../data/appeals_with_cats.json"

    build_dataset(CROPPED_APPEALS_DIR, APPEALS_DIR, QUESTIONS_DIR, OUTPUT_PATH)
