import os
import json

from src.features.text_extractor import pdf_extract

directory = r'../../data/cropped_appeals'
files = os.listdir(directory)
json_path = "../../data/appeals.json"

all_appeals = []

for file in files:
    text = pdf_extract(os.path.join(directory, file))

    appeal_data = {
        "file_name": file,
        "text": text
    }

    all_appeals.append(appeal_data)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump({"appeals": all_appeals}, f, ensure_ascii=False, indent=4)
