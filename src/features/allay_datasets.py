import json

with open("../../data/training_data_bio.json", "r", encoding="utf-8") as f:
    data1 = json.load(f)

with open("../../data/training_data_bio_MISHA.json", "r", encoding="utf-8") as f:
    data2 = json.load(f)

data = data1 + data2

with open("../../data/dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)