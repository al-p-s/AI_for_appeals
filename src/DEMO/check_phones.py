import json
from collections import Counter

with open("../../data/sets_to_learn/fields/target_fields_test_100.json", encoding="utf-8") as f:
    records = json.load(f)["appeals"]

phones = [r.get("PetitionerPhone", "").strip() for r in records]
print(Counter(bool(p) for p in phones))
print("Примеры:", [p for p in phones if p][:10])
