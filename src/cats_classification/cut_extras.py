import json
import re

INPUT_PATH = "../../data/sets_to_learn/dataset_1068_v2.json"
OUTPUT_PATH = "../../data/sets_to_learn/dataset_1068_v2_cleaned.json"

# 1. Удаление префикса "Эксперт по обработке обращений граждан в органы власти."
prefix_pattern = re.compile(r'^\s*Эксперт по обработке обращений граждан в органы власти\.\s*', re.IGNORECASE)

# 2. Удаление конструкций "проживающая по адресу: [адрес не указан]" и подобных
# с опциональной запятой и пробелом перед ними
address_pattern = re.compile(
    r'(?:,\s*)?' # опциональная запятая перед
    r'(?:\b(?:проживающ(?:ий|ая))\s+)?' # опционально "проживающий/проживающая"
    r'по\s+адресу\s*:?\s*' # "по адресу" с опциональным двоеточием
    r'\[(?:адрес\s*)?(?:не\s*указан|скрыт|данные\s*не\s*указаны|адрес)?\]'  # [адрес], [адрес не указан] и т.д.
    r'\s*[.,]?\s*',
    re.IGNORECASE
)

with open(INPUT_PATH, encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if "summary" in item:
        s = item["summary"]
        s = prefix_pattern.sub("", s) # убираем префикс
        s = address_pattern.sub("", s) # убираем "проживающ... по адресу: ..."
        # подчищаем двойные пробелы и пробелы перед знаками препинания
        s = re.sub(r'\s{2,}', ' ', s)
        s = re.sub(r'\s([.,])', r'\1', s)
        item["summary"] = s.strip()

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Очищено записей: {len(data)} → {OUTPUT_PATH}")
