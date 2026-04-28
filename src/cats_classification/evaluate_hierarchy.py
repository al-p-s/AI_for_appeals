import json
import re
from pathlib import Path
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from transformers import AutoModelForSequenceClassification
import torch
from geracl import GeraclHF, ZeroShotClassificationPipeline

APPEALS_PATH = "../../data/appeals.json"
APPEALS_CATS = "../../data/appeals_with_cats.json"
# CATS_L1 = "../../data/classifier/cats1.json"
CATS_L2 = "../../data/classifier/cats2.json"
CATS_L3 = "../../data/classifier/cats3.json"
CATS_L4 = "../../data/classifier/cats.json"

GIGACHAT_PATH = "../../models/gigaChat_lite"
# GERACL_PATH = "../../models/GeRaCl-USER2-base"
GERACL_PATH_L2 = "../../models/GeRaCl-finetuned/L2"
GERACL_PATH_L3 = "../../models/GeRaCl-finetuned/L3"
GERACL_PATH_L4 = "../../models/GeRaCl-finetuned/L4"

# MAX_NEW_TOKENS = 80
EVAL_LIMIT = None
ERRORS_OUT = "../../data/classifier/eval_errors_hierarchy255.json"

SUMM_PROMPT = """Ты — эксперт по суммаризации обращений граждан. Напиши выжимку в 1-2 предложения, строго по правилам:

1. Не повторяй: ФИО (пиши «житель», «жительница» или «заявитель»), номер документа, телефон, email, социальное положение, входящие номера, приветствия и подписи.
2. Отрази суть: КТО (житель такого-то района/улицы) → ЧТО ПРОСИТ или НА ЧТО ЖАЛУЕТСЯ → ПОЧЕМУ (одна-две главные причины).
3. Говори коротко, без канцелярита («просит согласовать», «требует уборки», «выражает негодование» вместо «прошу обеспечить проведение мероприятий»).
4. Не используй терминологию Классификатора обращений — она нужна для классификации, а не для выжимки.

ОБРАЩЕНИЕ:
{text}"""

def get_prefix(code: str, level: int) -> str:
    parts = code.split(".")
    return ".".join(parts[:level] + ["0000"] * (4 - level))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_gigachat() -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(GIGACHAT_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        GIGACHAT_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    generation_config = GenerationConfig.from_pretrained(GIGACHAT_PATH, trust_remote_code=True)
    # generation_config.max_new_tokens = MAX_NEW_TOKENS
    generation_config.do_sample = False
    return tokenizer, model, generation_config

def load_geracl(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to("cuda").eval()
    return tokenizer, model

def summarize(text: str, tokenizer, model, generation_config) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": SUMM_PROMPT.format(text=text[:3000])}],
        tokenize=False, add_generation_prompt=True
    )
    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    data = {k: v.to(model.device) for k, v in data.items()}
    data.pop("token_type_ids", None)
    output_ids = model.generate(**data, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

def classify_level(summary, candidates, tokenizer_model):
    tokenizer, model = tokenizer_model
    labels = [c["name"] for c in candidates]
    scores = []
    for label in labels:
        enc = tokenizer(summary, label, return_tensors="pt",
                        truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits
        scores.append(logits[0][0].item())  # entailment score
    best_idx = scores.index(max(scores))
    return candidates[best_idx]["code"]


def main():
    appeals_text = {a["file_name"]: a["text"]
                    for a in load_json(APPEALS_PATH)["appeals"]}
    appeals_cats = {a["file_name"]: a["categories"]
                    for a in load_json(APPEALS_CATS)["appeals"]}

    # cats_l1 = load_json(CATS_L1)["categories"]
    cats_l2 = load_json(CATS_L2)["categories"]
    cats_l3 = load_json(CATS_L3)["categories"]
    cats_l4 = load_json(CATS_L4)["categories"]

    name_by_code = {c["code"]: c["name"] for c in cats_l2 + cats_l3 + cats_l4}

    def children(cats, parent_code, level):
        prefix = ".".join(parent_code.split(".")[:level])
        return [c for c in cats if c["code"].startswith(prefix + ".")]

    print("Загружаем GigaChat-Lite...")
    gigachat_tok, gigachat_model, gigachat_gen = load_gigachat()

    print("Загружаем GeRaCl...")
    geracl_l2 = load_geracl(GERACL_PATH_L2)
    geracl_l3 = load_geracl(GERACL_PATH_L3)
    geracl_l4 = load_geracl(GERACL_PATH_L4)

    total = 0
    correct = {2: 0, 3: 0, 4: 0}
    errors = []

    file_names = [fn for fn in appeals_cats if fn in appeals_text]
    file_names = file_names[:EVAL_LIMIT] if EVAL_LIMIT else file_names
    print(f"Обращений для оценки: {len(file_names)}\n")

    for i, file_name in enumerate(file_names):
        text = appeals_text[file_name]
        true_cat_str = appeals_cats[file_name][0]
        true_code    = true_cat_str.split(" ")[0]

        # true_l1 = get_prefix(true_code, 1)
        true_l2 = get_prefix(true_code, 2)
        true_l3 = get_prefix(true_code, 3)
        true_l4 = true_code

        summary = summarize(text, gigachat_tok, gigachat_model, gigachat_gen)

        # pred_l1 = classify_level(summary, cats_l1, geracl_pipe)
        # ok_l1   = pred_l1 == true_l1

        cands_l2 = cats_l2
        pred_l2 = classify_level(summary, cands_l2, geracl_l2)
        ok_l2 = pred_l2 == true_l2

        cands_l3 = children(cats_l3, pred_l2, 2)
        pred_l3 = classify_level(summary, cands_l3, geracl_l3) if cands_l3 else pred_l2
        ok_l3 = pred_l3 == true_l3

        cands_l4 = children(cats_l4, pred_l3, 3)
        pred_l4 = classify_level(summary, cands_l4, geracl_l4) if cands_l4 else pred_l3
        ok_l4 = pred_l4 == true_l4

        total += 1
        for lvl, ok in [(2, ok_l2), (3, ok_l3), (4, ok_l4)]:
            if ok:
                correct[lvl] += 1

        status = "✓" if ok_l4 else "✗"
        print(f"[{i+1}/{len(file_names)}] {status} {file_name}")
        print(f"  summary : {summary}")
        print(f"  true    : {true_l2} → {true_l3} → {true_l4}")
        print(f"  pred    : {pred_l2} → {pred_l3} → {pred_l4}")
        print(f"  levels  : L2={'✓' if ok_l2 else '✗'} "
              f"L3={'✓' if ok_l3 else '✗'} L4={'✓' if ok_l4 else '✗'}\n")

        if not ok_l4:
            errors.append({
                "file_name": file_name,
                "summary": summary,
                "true": {
                    "L2": f"{true_l2} {name_by_code.get(true_l2, '')}",
                    "L3": f"{true_l3} {name_by_code.get(true_l3, '')}",
                    "L4": f"{true_l4} {name_by_code.get(true_l4, '')}",
                },
                "pred": {
                    "L2": f"{pred_l2} {name_by_code.get(pred_l2, '')}",
                    "L3": f"{pred_l3} {name_by_code.get(pred_l3, '')}",
                    "L4": f"{pred_l4} {name_by_code.get(pred_l4, '')}",
                },
                "ok": {"L2": ok_l2, "L3": ok_l3, "L4": ok_l4},
            })

    print("=" * 50)
    print(f"Всего обращений: {total}")
    for lvl in [2, 3, 4]:
        acc = correct[lvl] / total * 100
        print(f"  L{lvl} accuracy: {correct[lvl]}/{total} = {acc:.1f}%")
    print("=" * 50)

    with open(ERRORS_OUT, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    print(f"Ошибки → {ERRORS_OUT}")


if __name__ == "__main__":
    main()
