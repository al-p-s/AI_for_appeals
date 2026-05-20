import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

APPEALS_CATS_PATH = "../../data/sets_to_learn/appeals_w_cats/appeals_w_cats1068.json"
CATS_L2_PATH = "../../data/classifier/cats2.json"
CATS_L3_PATH = "../../data/classifier/cats3.json"
CATS_L4_PATH = "../../data/classifier/cats4.json"
OUTPUT_PATH = "../../data/sets_to_learn/dataset_hier1068_multi.json"
GIGACHAT_PATH = "../../models/gigaChat_lite"
EVAL_LIMIT = None

SUMM_PROMPT = """Ты — эксперт по суммаризации обращений граждан. Напиши выжимку в 1-2 предложения, строго по правилам:

1. Не повторяй: ФИО (пиши «житель», «жительница» или «заявитель»), номер документа, телефон, email, социальное положение, входящие номера, приветствия и подписи.
2. Отрази суть: КТО (житель такого-то района/улицы) → ЧТО ПРОСИТ или НА ЧТО ЖАЛУЕТСЯ → ПОЧЕМУ (одна-две главные причины).
3. Говори коротко, без канцелярита («просит согласовать», «требует уборки», «выражает негодование» вместо «прошу обеспечить проведение мероприятий»).

ОБРАЩЕНИЕ:
{text}"""


def get_prefix(code: str, level: int) -> str:
    parts = code.split(".")
    return ".".join(parts[:level] + ["0000"] * (4 - level))


def load_gigachat() -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(GIGACHAT_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        GIGACHAT_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    gen_config = GenerationConfig.from_pretrained(GIGACHAT_PATH, trust_remote_code=True)
    gen_config.do_sample = False
    return tokenizer, model, gen_config


def summarize(text, tokenizer, model, gen_config) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": SUMM_PROMPT.format(text=text[:3000])}],
        tokenize=False, add_generation_prompt=True
    )
    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    data = {k: v.to(model.device) for k, v in data.items()}
    data.pop("token_type_ids", None)
    output_ids = model.generate(**data, generation_config=gen_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def children(cats, parent_code, level):
    prefix = ".".join(parent_code.split(".")[:level])
    return [c for c in cats if c["code"].startswith(prefix + ".")]


def main():
    with open(APPEALS_CATS_PATH, encoding="utf-8") as f:
        appeals = json.load(f)["appeals"]
    text_by_file = {a["file_name"]: a["text"] for a in appeals}

    cats_l3 = json.load(open(CATS_L3_PATH, encoding="utf-8"))["categories"]
    cats_l4 = json.load(open(CATS_L4_PATH, encoding="utf-8"))["categories"]

    name_l2 = {}
    name_l3 = {c["code"]: c["name"] for c in cats_l3}
    name_l4 = {c["code"]: c["name"] for c in cats_l4}

    for c in cats_l3:
        l2_code = get_prefix(c["code"], 2)
        if l2_code not in name_l2:
            name_l2[l2_code] = ""

    try:
        cats_l2_list = json.load(open(CATS_L2_PATH, encoding="utf-8"))["categories"]
        name_l2 = {c["code"]: c["name"] for c in cats_l2_list}
    except FileNotFoundError:
        pass

    if EVAL_LIMIT:
        appeals = appeals[:EVAL_LIMIT]

    tokenizer, model, gen_config = load_gigachat()

    dataset = []
    for i, appeal in enumerate(appeals):
        file_name = appeal["file_name"]
        text = text_by_file.get(file_name, "").strip()
        if not text:
            print(f"[{i+1}] {file_name} — нет текста, пропуск")
            continue

        true_codes = [cat.split(" ")[0] for cat in appeal["categories"]]
        true_l2_codes = list({get_prefix(c, 2) for c in true_codes})
        true_l3_codes = list({get_prefix(c, 3) for c in true_codes})
        true_l4_codes = true_codes  # уже листы

        # true_l2/l3/l4 — массивы строк "КОД НАЗВАНИЕ"
        true_l2 = [f"{c} {name_l2.get(c, '')}".strip() for c in true_l2_codes]
        true_l3 = [f"{c} {name_l3.get(c, '')}".strip() for c in true_l3_codes]
        true_l4 = [f"{c} {name_l4.get(c, '')}".strip() for c in true_l4_codes]

        candidates_l3 = {l2c: children(cats_l3, l2c, 2) for l2c in true_l2_codes}
        candidates_l4 = {l3c: children(cats_l4, l3c, 3) for l3c in true_l3_codes}

        print(f"[{i+1}/{len(appeals)}] {file_name} → суммаризация...")
        summary = summarize(text, tokenizer, model, gen_config)
        print(f"  {summary}")

        dataset.append({
            "file_name":      file_name,
            "summary":        summary,
            "true_l2":        true_l2,
            "true_l3":        true_l3,
            "true_l4":        true_l4,
            "candidates_l3":  candidates_l3,
            "candidates_l4":  candidates_l4,
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\nГотово. Записей: {len(dataset)} → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
