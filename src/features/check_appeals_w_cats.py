import json
import re
from typing import List, Dict, Any


def find_problematic_appeals(data: Dict[str, Any]) -> List[str]:
    problematic = []
    pattern = re.compile(r'^(\d+)\.(\d+)\.(\d+)\.(\d+)(?:\s|$)')

    for appeal in data.get("appeals", []):
        file_name = appeal.get("file_name")
        categories = appeal.get("categories", [])

        if not isinstance(categories, list):
            continue

        # Проверка на дубликаты
        if len(categories) != len(set(categories)):
            problematic.append(file_name)
            continue

        # Проверка формата каждой категории
        has_format_issue = False
        for cat in categories:
            match = pattern.match(cat)
            if not pattern.match(cat):
                has_format_issue = True
                break
            if match.group(4) == "0000":
                has_format_issue = True
                break

        if has_format_issue:
            problematic.append(file_name)

    return problematic


if __name__ == "__main__":
    file_path = "../../data/sets_to_learn/appeals_w_cats/chel_appeals_with_cats99_test.json"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Файл не найден: {file_path}")
        exit(1)

    bad_files = find_problematic_appeals(data)

    if bad_files:
        print("Обращения с проблемами в категориях:")
        for fname in bad_files:
            print(fname)
    else:
        print("Проблемных обращений не найдено.")
