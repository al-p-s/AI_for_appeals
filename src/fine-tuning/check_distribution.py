import json
from collections import Counter

DATASET_PATH = "../../data/sets_to_learn/fields/target_fields_v3.json"
FIELD_NAME = "PetitionerCategory"

with open(DATASET_PATH, encoding="utf-8") as f:
    records = json.load(f)["appeals"]

dist = Counter(
    r.get(FIELD_NAME, "").strip() or "не определяется"
    for r in records
)

print(f"\nDistribution for '{FIELD_NAME}' ({len(records)} records):")
for val, count in dist.most_common():
    print(f"  {count:5d} | {val}")
