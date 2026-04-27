import json
import os
import glob
from pathlib import Path
from collections import defaultdict

APPEALS_JSON = r"../../data/appeals_chelyabinsk_MISHA.json"
CATS_DIR = r"D:\Обращения\chel_cats"
PDFS_DIR     = r"D:\Обращения\part_Chelyabinsk_appeals\misha_250"
OUTPUT_JSON = r"../../data/chel_appeals_with_cats2.json"


def scan_folders(pdfs_dir: str):
    pdf_to_folder = defaultdict(list)
    empty_folders = []

    for entry in os.scandir(pdfs_dir):
        if not entry.is_dir():
            continue
        pdfs = [f.name for f in os.scandir(entry.path) if f.name.lower().endswith(".pdf")]
        if not pdfs:
            empty_folders.append(entry.name)
        for pdf in pdfs:
            pdf_to_folder[pdf].append(entry.name)

    return pdf_to_folder, empty_folders


def report_issues(pdf_to_folder: dict, empty_folders: list):
    print("\n── Диагностика ──────────────────────────────────────────────────")

    if empty_folders:
        print(f"Пустые папки (нет PDF): {len(empty_folders)}")
        for f in empty_folders:
            print(f"  {f}")
    else:
        print("Пустых папок: 0")

    duplicates = {k: v for k, v in pdf_to_folder.items() if len(v) > 1}
    if duplicates:
        print(f"\nДублирующиеся PDF ({len(duplicates)} имён в нескольких папках):")
        for pdf_name, folders in sorted(duplicates.items()):
            print(f"  {pdf_name}")
            for folder in folders:
                print(f"    └─ {folder}")
    else:
        print("Дублирующихся PDF: 0")

    print("─────────────────────────────────────────────────────────────────\n")


def load_appeals(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["appeals"]


def load_categories(cats_dir: str, folder_id: str) -> list[str] | None:
    txt_path = os.path.join(cats_dir, folder_id + ".txt")
    if not os.path.exists(txt_path):
        return None
    with open(txt_path, encoding="utf-16") as f:
        raw = f.read().strip()
    return [c.strip() for c in raw.split("//") if c.strip()]


def build_dataset(appeals: list, pdf_to_folder: dict, cats_dir: str) -> list[dict]:
    dataset = []
    missing_folder = []
    missing_cats   = []
    duplicate_skip = []

    for appeal in appeals:
        pdf_name  = appeal["file_name"]
        text      = appeal["text"]
        folders   = pdf_to_folder.get(pdf_name)

        if not folders:
            missing_folder.append(pdf_name)
            continue

        if len(folders) > 1:
            duplicate_skip.append(pdf_name)
            continue

        folder_id = folders[0]
        cats = load_categories(cats_dir, folder_id)

        if cats is None:
            missing_cats.append((pdf_name, folder_id))
            continue

        dataset.append({
            "file_name":  folder_id,
            "text":       text,
            "categories": cats,
        })

    print(f"Собрано записей:              {len(dataset)}")
    print(f"Не найдена папка (PDF нет):   {len(missing_folder)}")
    print(f"Пропущено из-за дублей:       {len(duplicate_skip)}")
    print(f"Нет TXT категорий:            {len(missing_cats)}")

    if missing_folder:
        print("\n  Нет папки для:", missing_folder[:10])
    if missing_cats:
        print("\n  Нет TXT для:")
        for pdf, fid in missing_cats[:10]:
            print(f"    {pdf} → {fid}")

    return dataset


def main():
    print("Сканирую папки с PDF...")
    pdf_to_folder, empty_folders = scan_folders(PDFS_DIR)
    print(f"  Уникальных PDF-имён: {len(pdf_to_folder)}")

    report_issues(pdf_to_folder, empty_folders)

    print("Загружаю тексты...")
    appeals = load_appeals(APPEALS_JSON)
    print(f"  Загружено обращений: {len(appeals)}")

    print("Строю датасет...")
    dataset = build_dataset(appeals, pdf_to_folder, CATS_DIR)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"appeals": dataset}, f, ensure_ascii=False, indent=2)

    print(f"\nГотово → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
