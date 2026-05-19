import os
import json

from src.features.text_extractor import pdf_extract

directory = r'C:/appeals/appeals_350_sasha/appeals_350_sasha'
files = os.listdir(directory)
json_path = "../../data/appeals_350_SASHKA.json"

all_appeals = []

for folder in files:
    folder_path = os.path.join(directory, folder)
    if not os.path.isdir(folder_path):
        continue

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"PDF не найден в {folder}")
        continue

    pdf_path = os.path.join(folder_path, pdf_files[0])
    text = pdf_extract(pdf_path)
    print(folder)

    appeal_data = {
        "file_name": folder,
        "text": text
    }
    all_appeals.append(appeal_data)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump({"appeals": all_appeals}, f, ensure_ascii=False, indent=4)
