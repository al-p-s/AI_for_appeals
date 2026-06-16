import json
import re

INPUT_PATH = "../../data/sets_to_learn/dataset_157_test_prompt.json"
OUTPUT_PATH = "../../data/sets_to_learn/dataset_157_cleaned.json"

# 1. Удаление префикса Эксперт по обработке обращений граждан в органы власти.
prefix_pattern = re.compile(r'^\s*Эксперт по обработке обращений граждан в органы власти\.\s*', re.IGNORECASE)

# 2. Удаление конструкций "проживающая по адресу: [адрес не указан]" и подобных
address_pattern = re.compile(
    r'(?:,\s*)?'
    r'(?:\b(?:проживающ(?:ий|ая))\s+)?'
    r'по\s+адресу\s*:?\s*'
    r'\[(?:адрес\s*)?(?:не\s*указан|скрыт|данные\s*не\s*указаны|адрес)?\]'
    r'\s*[.,]?\s*',
    re.IGNORECASE
)

annotation_prefix_pattern = re.compile(
    r'^\s*Аннотация[^\n.:]*[.:\n]\s*',
    re.IGNORECASE
)

with open(INPUT_PATH, encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if "summary" in item:
        s = item["summary"]
        s = prefix_pattern.sub("", s)
        s = address_pattern.sub("", s)
        s = annotation_prefix_pattern.sub("", s)

        s = re.sub(r'\s{2,}', ' ', s)
        s = re.sub(r'\s([.,])', r'\1', s)
        item["summary"] = s.strip()

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Очищено записей: {len(data)} → {OUTPUT_PATH}")
