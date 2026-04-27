import json

A = r"../../data/chel_appeals_with_cats.json"
B = r"../../data/chel_appeals_with_cats2.json"
OUT = r"../../data/chel_appeals_w_cats.json"

with open(A, encoding="utf-8") as f:
    a = json.load(f)["appeals"]

with open(B, encoding="utf-8") as f:
    b = json.load(f)["appeals"]

merged = a + b
print(f"Часть 1: {len(a)}  |  Часть 2: {len(b)}  |  Итого: {len(merged)}")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"appeals": merged}, f, ensure_ascii=False, indent=2)

print(f"Готово → {OUT}")
