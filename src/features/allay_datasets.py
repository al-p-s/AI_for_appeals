import json

with open("../../data/NER_relearn2_0/Sasha_300_NEW.json", "r", encoding="utf-8") as f:
    data1 = json.load(f)

with open("../../data/NER_relearn2_0/Misha_300_NEW.json", "r", encoding="utf-8") as f:
    data2 = json.load(f)

data = data1 + data2

with open("../../data/NER_relearn2_0/raw_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)