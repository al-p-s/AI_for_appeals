import json
import os


def load_dataset(file_path):
    """Загружает датасет из JSON файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_dataset(data, file_path):
    """Сохраняет датасет в JSON файл"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_file_list(txt_file_path):
    """Загружает список файлов из txt файла"""
    file_list = []
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Убираем пробелы и переносы строк
            clean_line = line.strip()
            if clean_line:  # Пропускаем пустые строки
                file_list.append(clean_line)
    return file_list


def transfer_appeals(source_dataset_path, target_dataset_path, file_list_path, output_source_path, output_target_path):
    """
    Переносит записи из исходного датасета в целевой на основе списка файлов

    Args:
        source_dataset_path: путь к исходному датасету (откуда переносим)
        target_dataset_path: путь к целевому датасету (куда переносим)
        file_list_path: путь к txt файлу со списком file_name для переноса
        output_source_path: путь для сохранения обновленного исходного датасета
        output_target_path: путь для сохранения обновленного целевого датасета
    """

    # Загружаем датасеты
    print("Загрузка датасетов...")
    source_dataset = load_dataset(source_dataset_path)
    target_dataset = load_dataset(target_dataset_path)

    # Загружаем список файлов для переноса
    print("Загрузка списка файлов...")
    files_to_transfer = load_file_list(file_list_path)
    print(f"Найдено {len(files_to_transfer)} файлов для переноса")

    # Создаем множество для быстрого поиска
    files_to_transfer_set = set(files_to_transfer)

    # Разделяем записи
    appeals_to_keep = []
    appeals_to_transfer = []
    transferred_count = 0

    print("Обработка записей...")
    for appeal in source_dataset.get("appeals", []):
        file_name = appeal.get("file_name", "")
        if file_name in files_to_transfer_set:
            appeals_to_transfer.append(appeal)
            transferred_count += 1
        else:
            appeals_to_keep.append(appeal)

    # Обновляем исходный датасет
    source_dataset["appeals"] = appeals_to_keep

    # Добавляем записи в целевой датасет
    if "appeals" not in target_dataset:
        target_dataset["appeals"] = []
    target_dataset["appeals"].extend(appeals_to_transfer)

    # Сохраняем обновленные датасеты
    print("Сохранение обновленных датасетов...")
    save_dataset(source_dataset, output_source_path)
    save_dataset(target_dataset, output_target_path)

    print(f"\nГотово!")
    print(f"Перенесено записей: {transferred_count}")
    print(f"Осталось в исходном датасете: {len(appeals_to_keep)}")
    print(f"Стало в целевом датасете: {len(target_dataset['appeals'])}")

    # Проверка на ненайденные файлы
    not_found = files_to_transfer_set - {appeal['file_name'] for appeal in
                                         source_dataset.get('appeals', []) + appeals_to_transfer}
    if not_found:
        print(f"\nВнимание! Не найдено в исходном датасете {len(not_found)} файлов из списка:")
        for file_name in list(not_found)[:10]:  # Показываем первые 10
            print(f"  - {file_name}")
        if len(not_found) > 10:
            print(f"  ... и еще {len(not_found) - 10}")


# Настройка путей к файлам
if __name__ == "__main__":
    # Укажите свои пути к файлам
    SOURCE_DATASET = "../../../data/sets_to_learn/appeals_w_cats/appeals_w_cats_test_98_NEW.json"  # Исходный датасет
    TARGET_DATASET = "../../../data/sets_to_learn/appeals_w_cats/BRAND_NEW_w_cats.json"  # Целевой датасет
    FILE_LIST = "../../../data/sets_to_learn/appeals_w_cats/to_transfer"  # Список файлов для переноса

    # Пути для сохранения результатов (можно те же файлы)
    OUTPUT_SOURCE = "../../../data/sets_to_learn/appeals_w_cats/appeals_w_cats_test_98_NEW.json"
    OUTPUT_TARGET = "../../../data/sets_to_learn/appeals_w_cats/BRAND_NEW_w_cats.json"

    # Проверка существования файлов
    for file_path in [SOURCE_DATASET, TARGET_DATASET, FILE_LIST]:
        if not os.path.exists(file_path):
            print(f"Ошибка: файл {file_path} не найден!")
            exit(1)

    # Запуск переноса
    transfer_appeals(
        SOURCE_DATASET,
        TARGET_DATASET,
        FILE_LIST,
        OUTPUT_SOURCE,
        OUTPUT_TARGET
    )