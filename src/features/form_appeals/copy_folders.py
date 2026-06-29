import json
import os
import shutil

# ЗАДАЙТЕ ПУТИ ЗДЕСЬ
json_path = "../../../data/sets_to_learn/appeals_w_cats/100_for_test_G.json" # Путь к JSON файлу
source_dir = r"D:\Обращения\ALL" # Папка, где лежат папки для копирования
dest_dir = r"D:\Обращения\100_test_appeals" # Папка, куда копировать

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Извлекаем все file_name из обращений
file_names = set()
if 'appeals' in data:
    for appeal in data['appeals']:
        if 'file_name' in appeal:
            file_names.add(appeal['file_name'])

print(f"Найдено {len(file_names)} уникальных имен папок в JSON")

# Создаем целевую папку, если она не существует
os.makedirs(dest_dir, exist_ok=True)

# Копируем совпадающие папки
copied_count = 0
not_found_count = 0

for folder_name in file_names:
    source_folder_path = os.path.join(source_dir, folder_name)
    dest_folder_path = os.path.join(dest_dir, folder_name)

    if os.path.exists(source_folder_path) and os.path.isdir(source_folder_path):
        if os.path.exists(dest_folder_path):
            print(f"Предупреждение: Папка '{folder_name}' уже существует в целевой директории, пропускаем")
            continue

        try:
            shutil.copytree(source_folder_path, dest_folder_path)
            print(f"✓ Скопирована папка: {folder_name}")
            copied_count += 1
        except Exception as e:
            print(f"✗ Ошибка при копировании папки '{folder_name}': {e}")
    else:
        print(f"✗ Папка '{folder_name}' не найдена в исходной директории")
        not_found_count += 1

print(f"\nГотово! Скопировано: {copied_count}, не найдено: {not_found_count}")
