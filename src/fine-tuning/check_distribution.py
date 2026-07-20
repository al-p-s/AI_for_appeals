import json
from collections import Counter

DATASET_PATH = "../../data/sets_to_learn/fields/target_fields_v3.json"
FIELD_NAME = "PetitionerDistrict"

with open(DATASET_PATH, encoding="utf-8") as f:
    records = json.load(f)["appeals"]

dist = Counter()
for r in records:
    value = r.get(FIELD_NAME)
    if value is None:
        value = "не определяется"
    elif isinstance(value, int):
        value = str(value)
    elif isinstance(value, str):
        value = value.strip() or "не определяется"
    else:
        value = "не определяется"
    dist[value] += 1

print(f"\nDistribution for '{FIELD_NAME}' ({len(records)} records):")
for val, count in dist.most_common():
    print(f"  {count:5d} | {val}")
