import argparse
import json
import os
import xml.etree.ElementTree as ET

CONSIDERATION_MAP = {
    "0": 0,
    "1": 1,
    "2": 2,
}


def read_xml(path: str) -> ET.Element:
    for enc in ("utf-16", "utf-16-le", "utf-8"):
        try:
            with open(path, encoding=enc) as f:
                content = f.read()
            # убираем BOM если есть
            content = content.lstrip("\ufeff")
            return ET.fromstring(content)
        except (UnicodeDecodeError, ET.ParseError):
            continue
    raise ValueError(f"Не удалось прочитать XML: {path}")


def build_id_to_name(root: ET.Element) -> dict:
    id_map = {}
    for elem in root.iter():
        row_id = elem.get("RowID")
        name = elem.get("Name")
        if row_id and name:
            id_map[row_id] = name
    return id_map


ZERO_UUID = "00000000-0000-0000-0000-000000000000"


def resolve(value: str | None, id_map: dict) -> str:
    if not value or value == ZERO_UUID:
        return ""
    return id_map.get(value, value)


def extract_record(root: ET.Element, file_name: str, text: str = "") -> dict:
    id_map = build_id_to_name(root)

    card = root.find("CardDocument")
    if card is None:
        card = root  # на случай если root и есть CardDocument

    main = card.find("MainInfo")
    pet = card.find("PetitionerInfo")
    sender = card.find("SenderPartner")

    def m(attr):
        return (main.get(attr) or "") if main is not None else ""

    def p(attr):
        return (pet.get(attr) or "") if pet is not None else ""

    def s(attr):
        return (sender.get(attr) or "") if sender is not None else ""

    consideration_raw = m("ConsiderationType")
    consideration = CONSIDERATION_MAP.get(consideration_raw)

    record = {
        "file_name": file_name,
        "text": text,
        # Информация
        "AppealKind": resolve(m("AppealKind"), id_map),
        "StatusId": resolve(m("StatusId"), id_map),
        "ItemID": resolve(m("ItemID"), id_map),
        "DeliveryTypeId": resolve(m("DeliveryTypeId"), id_map),
        "RegistrationPlaceId": resolve(m("RegistrationPlaceId"), id_map),
        "AppealDate": m("AppealDate"),
        "ConsiderationType": consideration,
        "Content": m("Content"),
        # Информация о заявителе
        "PetitionerSurname": p("PetitionerSurname"),
        "PetitionerName": p("PetitionerName"),
        "PetitionerPatronymic": p("PetitionerPatronymic"),
        "PetitionerAddress": p("PetitionerAddress"),
        "PetitionerEmail": p("PetitionerEmail"),
        "PetitionerPhone": p("PetitionerPhone"),
        "PetitionerCategory": resolve(p("PetitionerCategory"), id_map),
        "PetitionerDistrict": resolve(p("PetitionerDistrict"), id_map),
        "PetitionerComments": p("PetitionerComments"),
        "PetitionerOrganization": p("PetitionerOrganization"),
        # Доп. адресаты
        "AdditionalRecipients": m("AdditionalRecipients"),
        # Информация об отправителе
        "SenderOrg": resolve(s("SenderOrg"), id_map),
        # Исходящий номер / дата (из отправителя)
        "ExternalNumber": m("ExternalNumber"),
        "ExternalDate": m("ExternalDate"),
    }
    return record


def main():
    parser = argparse.ArgumentParser(description="Извлечение полей из XML-обращений в JSON")
    parser.add_argument("--input", required=True, help="Папка с XML-файлами")
    parser.add_argument("--output", default="appeals_extracted.json", help="Путь к выходному JSON")
    args = parser.parse_args()

    input_dir = Path(args.input)
    xml_files = sorted(input_dir.glob("*.xml"))

    if not xml_files:
        print(f"XML-файлы не найдены в {input_dir}")
        return

    results = []
    errors = []

    for xml_path in xml_files:
        try:
            root = read_xml(str(xml_path))
            record = extract_record(root, xml_path.stem)
            results.append(record)
            print(f"  OK  {xml_path.name}")
        except Exception as e:
            errors.append({"file": xml_path.name, "error": str(e)})
            print(f"  ERR {xml_path.name}: {e}")

    output = {
        "appeals": results,
        "errors": errors,
        "total": len(results),
        "failed": len(errors),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nГотово. Обработано: {len(results)}, ошибок: {len(errors)}")
    print(f"Результат: {args.output}")


if __name__ == "__main__":
    INPUT_DIR = r"D:\Обращения\xml_w_fields"
    OUTPUT_FILE = "../../../data/sets_to_learn/fields/target_fields_test_100.json"
    DATASET_PATH = "../../../data/sets_to_learn/appeals_w_cats/100_for_test_G.json"

    from pathlib import Path

    # берём file_name из датасета
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    dataset = data["appeals"]
    target_names = {r["file_name"].lower() for r in dataset}
    file_to_text = {r["file_name"].lower(): r.get("text", "") for r in dataset}

    input_dir = Path(INPUT_DIR)

    xml_files = sorted(
        f for f in input_dir.glob("*.xml")
        if f.stem.lower() in target_names
    )

    print(f"Найдено XML под фильтр: {len(xml_files)} из {len(list(input_dir.glob('*.xml')))}")

    found_stems = {f.stem.lower() for f in xml_files}
    missing = [r["file_name"] for r in dataset if r["file_name"].lower() not in found_stems]

    results = []
    errors = []

    for xml_path in xml_files:
        try:
            root = read_xml(str(xml_path))
            text = file_to_text.get(xml_path.stem.lower(), "")
            record = extract_record(root, xml_path.stem, text)
            results.append(record)
            print(f"  OK  {xml_path.stem}")
        except Exception as e:
            errors.append({"file": xml_path.stem, "error": str(e)})
            print(f"  ERR {xml_path.stem}: {e}")

    output = {
        "appeals": results,
        "errors": errors,
        "total": len(results),
        "failed": len(errors),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nГотово. Обработано: {len(results)}, ошибок: {len(errors)}")
    print(f"Результат: {OUTPUT_FILE}")
    if missing:
        print(f"\nНе найдены XML для {len(missing)} записей:")
        for name in missing:
            print(f"  MISS  {name}")
