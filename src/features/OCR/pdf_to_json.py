import os
import json
import time

from src.features.OCR.pdf_extractor_test import pdf_extract
from src.DEMO.config import PDF_DIRECTORY, BASE_DATA_DIR

directory = PDF_DIRECTORY
files = os.listdir(directory)
json_path = BASE_DATA_DIR / "data" / "OCR_improving" / "OCR_test.json"

all_appeals = []

start_time = time.time()
i = 0
for folder in files:

    if i > 50:
        break

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
    i = i+1

elapsed_time = time.time() - start_time
print(elapsed_time)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump({"appeals": all_appeals}, f, ensure_ascii=False, indent=4)
