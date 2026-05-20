# import json
#
# input_file = "../../data/sets_to_learn/dataset_hier721_prompt2.json"
# output_file = "../../data/sets_to_learn/dataset_hier720_v2.json"
#
# with open(input_file, 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
# for item in data:
#     if "candidates_l2" in item:
#         del item["candidates_l2"]
#
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)
#
# print(f"Готово! Результат сохранён в {output_file}")

# import json
#
# DATASET_PATH = "../../data/sets_to_learn/dataset_hier720_v2.json"
# CATS2_PATH = "../../data/classifier/cats2.json"
# CATS3_PATH = "../../data/classifier/cats3.json"
# CATS4_PATH = "../../data/classifier/cats4.json"
# OUTPUT_PATH = "../../data/sets_to_learn/dataset_hier720_v2.json"
#
# with open(DATASET_PATH, encoding="utf-8") as f:
#     dataset = json.load(f)
#
# code2name = {}
# for path in (CATS2_PATH, CATS3_PATH, CATS4_PATH):
#     with open(path, encoding="utf-8") as f:
#         for item in json.load(f)["categories"]:
#             code2name[item["code"]] = item["name"]
#
# for record in dataset:
#     for level in ("l2", "l3", "l4"):
#         key = f"true_{level}"
#         code = record.get(key)
#         if code and code in code2name:
#             record[key] = f"{code} {code2name[code]}"
#
# with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
#     json.dump(dataset, f, ensure_ascii=False, indent=2)
#
# print(f"Готово! Обогащённый датасет сохранён в {OUTPUT_PATH}")

import json


def find_missing_files(file1_path, file2_path):
    # Загружаем данные из обоих файлов
    with open(file1_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)

    with open(file2_path, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)

    # Извлекаем все file_name из первого файла
    file_names1 = {appeal['file_name'] for appeal in data1['appeals']}

    # Извлекаем все file_name из второго файла
    file_names2 = {item['file_name'] for item in data2}

    # Находим file_name, которые есть в первом, но нет во втором
    missing_in_second = file_names1 - file_names2

    return missing_in_second


# Использование
file1 = '../../data/sets_to_learn/appeals_w_cats/appeals_w_cats1068.json'  # путь к первому файлу
file2 = '../../data/sets_to_learn/dataset_hier1068_multi.json'  # путь ко второму файлу

missing = find_missing_files(file1, file2)

print(f"Найдено {len(missing)} файлов, которые есть в первом, но нет во втором:")
for file_name in missing:
    print(f"  - {file_name}")