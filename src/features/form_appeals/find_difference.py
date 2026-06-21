import json

FILE_WITHOUT_CATEGORIES = "../../data/appeals_test_99_NEW.json"
FILE_WITH_CATEGORIES = "../../data/sets_to_learn/appeals_w_cats/appeals_w_cats_test_99_NEW.json"

def load_file_names(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item["file_name"] for item in data["appeals"]}

names1 = load_file_names(FILE_WITHOUT_CATEGORIES)
names2 = load_file_names(FILE_WITH_CATEGORIES)

only_in_first = names1 - names2

print(f"Первый файл ({FILE_WITHOUT_CATEGORIES}): {len(names1)} обращений")
print(f"Второй файл ({FILE_WITH_CATEGORIES}): {len(names2)} обращений")

if not only_in_first:
    print("Все обращения из первого файла присутствуют во втором.")
else:
    print(f"\nТолько в первом файле ({len(only_in_first)} шт.):")
    for name in sorted(only_in_first):
        print(name)

