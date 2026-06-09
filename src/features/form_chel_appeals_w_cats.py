import json
import os
import glob
from pathlib import Path

APPEALS_JSON = r"../../data/appeals_99_test_sasha_NEW_OCR.json"
CATS_DIR = r"D:\Обращения\chel_cats"
OUTPUT_JSON = r"../../data/sets_to_learn/appeals_w_cats/chel_appeals_with_cats99_test_NEW_OCR.json"

def load_appeals(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {a["file_name"]: a["text"] for a in data["appeals"]}

def load_categories(cats_dir: str) -> dict[str, list[str]]:
    result = {}
    for txt_path in glob.glob(os.path.join(cats_dir, "*.txt")):
        stem = Path(txt_path).stem
        with open(txt_path, encoding="utf-16") as f:
            raw = f.read().strip()
        cats = [c.strip() for c in raw.split("//") if c.strip()]
        result[stem] = cats
    return result

def build_dataset(appeals: dict, categories: dict) -> list[dict]:
    dataset = []
    matched = 0
    missing_cats = []
    missing_text = []

    all_keys = set(appeals) | set(categories)

    for key in sorted(all_keys):
        text = appeals.get(key)
        cats = categories.get(key)

        if text is None:
            missing_text.append(key)
            continue
        if cats is None:
            missing_cats.append(key)
            continue

        dataset.append({
            "file_name": key,
            "text": text,
            "categories": cats,
        })
        matched += 1

    print(f"Собрано записей: {matched}")
    print(f"Нет категорий (TXT): {len(missing_cats)}")
    if missing_cats:
        print("  Без категорий:", missing_cats[:10], "...")

    return dataset

def main():
    print("Загружаю тексты...")
    appeals = load_appeals(APPEALS_JSON)
    print(f"  Загружено обращений: {len(appeals)}")

    print("Загружаю категории...")
    categories = load_categories(CATS_DIR)
    print(f"  Загружено TXT-файлов: {len(categories)}")

    print("Строю датасет...")
    dataset = build_dataset(appeals, categories)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"appeals": dataset}, f, ensure_ascii=False, indent=2)

    print(f"\nГотово → {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
