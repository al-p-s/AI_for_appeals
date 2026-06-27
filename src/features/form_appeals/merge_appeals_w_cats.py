import json

with open('../../../data/sets_to_learn/appeals_w_cats/appeals_w_cats_119_G.json', 'r', encoding='utf-8') as f:
    data1 = json.load(f)

with open('../../../data/sets_to_learn/appeals_w_cats/appeals_w_cats1222_upd.json', 'r', encoding='utf-8') as f:
    data2 = json.load(f)

merged = {"appeals": data1["appeals"] + data2["appeals"]}

with open('../../../data/sets_to_learn/appeals_w_cats/appeals_w_cats_1341.json', 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
