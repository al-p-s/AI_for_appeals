import json

PATH_1 = "../../data/sets_to_learn/dataset_1068_short.json"
PATH_2 = "../../data/sets_to_learn/dataset_155_short.json"
OUTPUT_PATH = "../../data/sets_to_learn/dataset_1223.json"

with open(PATH_1, encoding="utf-8") as f:
    data1 = json.load(f)

with open(PATH_2, encoding="utf-8") as f:
    data2 = json.load(f)

# проверка на дубли по file_name
names1 = {item["file_name"] for item in data1}
duplicates = [item["file_name"] for item in data2 if item["file_name"] in names1]

if duplicates:
    print(f"Найдено дублей: {len(duplicates)}")
    for d in duplicates:
        print(f"  {d}")
    data2 = [item for item in data2 if item["file_name"] not in names1]
    print(f"Дубли исключены. Добавляется: {len(data2)} записей")
else:
    print("Дублей нет")

merged = data1 + data2
print(f"Итого записей: {len(merged)}")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"Сохранено → {OUTPUT_PATH}")
