import json


def load_json(path: str) -> any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def build_true_l4_map(first_data: list) -> dict[str, list[str]]:
    mapping = {}
    for item in first_data:
        file_name = item.get("file_name")
        true_l4 = item.get("true_l4")
        if file_name and true_l4 is not None:
            mapping[file_name] = true_l4
    return mapping


def patch_appeals(second_data: dict | list, true_l4_map: dict) -> tuple[int, int]:
    if isinstance(second_data, dict):
        appeals = second_data.get("appeals", [])
    elif isinstance(second_data, list):
        appeals = second_data
    else:
        raise ValueError("Второй JSON должен быть списком или объектом с ключом 'appeals'")

    total = len(appeals)
    patched = 0

    actually_changed = 0

    for appeal in appeals:
        file_name = appeal.get("file_name")
        if file_name in true_l4_map:
            new_categories = true_l4_map[file_name]
            old_categories = appeal.get("categories")
            appeal["categories"] = new_categories
            patched += 1
            if old_categories != new_categories:
                actually_changed += 1

    return total, patched, actually_changed


def main():
    first_path  = "../dataset_1340_G.json"
    second_path = "appeals_w_cats_1340.json"
    output_path = "appeals_w_cats1340_G.json"

    print(f"Читаю первый JSON:  {first_path}")
    first_data = load_json(first_path)

    print(f"Читаю второй JSON:  {second_path}")
    second_data = load_json(second_path)

    # Первый JSON может быть списком или объектом с ключом, содержащим список
    if isinstance(first_data, list):
        items = first_data
    elif isinstance(first_data, dict):
        # Берём первое значение, которое является списком
        items = next((v for v in first_data.values() if isinstance(v, list)), [])
    else:
        print("Ошибка: неожиданный формат первого JSON")
        sys.exit(1)

    true_l4_map = build_true_l4_map(items)
    print(f"Найдено записей с true_l4: {len(true_l4_map)}")

    total, patched, actually_changed = patch_appeals(second_data, true_l4_map)
    print(f"Обращений в файле:       {total}")
    print(f"Найдено совпадений:      {patched}")
    print(f"Реально изменено значений: {actually_changed}")

    save_json(second_data, output_path)
    print(f"Результат сохранён в: {output_path}")


if __name__ == "__main__":
    main()
