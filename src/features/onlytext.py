import json

with open("../../data/appeals_chelyabinsk_MISHA.json", "r", encoding="utf-8") as f:
    data = json.load(f)

newData = []

for i in range(len(data["appeals"])):
    newData.append({"text": data["appeals"][i]["text"]})

with open("../../data/text_Chelyabinsk_MISHA.json", "w", encoding="utf-8") as f:
    json.dump(newData, f, ensure_ascii=False)
