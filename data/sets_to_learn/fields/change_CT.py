import json

DATASET_PATH = "target_fields_v3.json"
OUTPUT_PATH = "target_fields_v3_new.json"

# Маппинг: число -> слово
CONSIDERATION_TYPE_MAPPING = {
    0: "первичное",
    1: "повторное",
    2: "неоднократное",
}

with open(DATASET_PATH, encoding="utf-8") as f:
    data = json.load(f)

for appeal in data["appeals"]:
    val = appeal.get("ConsiderationType")
    if val in CONSIDERATION_TYPE_MAPPING:
        appeal["ConsiderationType"] = CONSIDERATION_TYPE_MAPPING[val]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Done. Saved to {OUTPUT_PATH}")
print(f"Total appeals: {len(data['appeals'])}")
