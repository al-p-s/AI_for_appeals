import json

input_file = "../../data/sets_to_learn/dataset_hier721.json"
output_file = "../../data/sets_to_learn/dataset_hier721_new.json"

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    if "candidates_l2" in item:
        del item["candidates_l2"]

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Готово! Результат сохранён в {output_file}")