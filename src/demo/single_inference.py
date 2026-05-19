import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import AutoModelForSequenceClassification
import torch

APPEALS_PATH = "../../data/sets_to_learn/appeals_w_cats/appeals_w_cats720.json"
CATS_L2 = "../../data/classifier/cats2.json"
CATS_L3 = "../../data/classifier/cats3.json"
CATS_L4 = "../../data/classifier/cats4.json"

GIGACHAT_PATH = "../../models/gigaChat_lite"
KERYX_PATH_L2 = "../../models/KERYX_720p2/L2"
KERYX_PATH_L3 = "../../models/KERYX_720p2/L3"
KERYX_PATH_L4 = "../../models/KERYX_720p2/L4"

HARDCODED_TEXT = """
Жительница г. Москва, ул. Ленина, д. 15, кв. 42, Иванова Мария Ивановна, тел. 8-999-123-45-67,
просит разобраться с регулярными перебоями горячего водоснабжения в её доме.
Управляющая компания не реагирует на заявки, последний раз воды не было трое суток.
Прошу принять меры."""

SUMM_PROMPT = """Ты — эксперт по суммаризации обращений граждан. Напиши выжимку в 1-2 предложения, строго по правилам:

1. Не повторяй: ФИО (пиши «житель», «жительница» или «заявитель»), номер документа, телефон, email, социальное положение, входящие номера, приветствия и подписи.
2. Отрази суть: КТО (житель такого-то района/улицы) → ЧТО ПРОСИТ или НА ЧТО ЖАЛУЕТСЯ → ПОЧЕМУ (одна-две главные причины).
3. Говори коротко, без канцелярита («просит согласовать», «требует уборки», «выражает негодование» вместо «прошу обеспечить проведение мероприятий»).

ОБРАЩЕНИЕ:
{text}"""


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
    generation_config.do_sample = False
    # generation_config.max_new_tokens -- можно задать при необходимости
    return tokenizer, model, generation_config


def load_keryx(path):
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
                        truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits
        scores.append(logits[0][0].item())
    best_idx = scores.index(max(scores))
    return candidates[best_idx]["code"]


def get_children(cats, parent_code, level):
    prefix = ".".join(parent_code.split(".")[:level])
    return [c for c in cats if c["code"].startswith(prefix + ".")]


print("Инициализация классификатора (загрузка категорий и моделей)...")

cats_l2 = load_json(CATS_L2)["categories"]
cats_l3 = load_json(CATS_L3)["categories"]
cats_l4 = load_json(CATS_L4)["categories"]
name_by_code = {c["code"]: c["name"] for c in cats_l2 + cats_l3 + cats_l4}

gigachat_tok, gigachat_model, gigachat_gen = load_gigachat()
keryx_l2 = load_keryx(KERYX_PATH_L2)
keryx_l3 = load_keryx(KERYX_PATH_L3)
keryx_l4 = load_keryx(KERYX_PATH_L4)

print("Модели готовы к использованию.")


def classify_text(text: str):
    summary = summarize(text, gigachat_tok, gigachat_model, gigachat_gen)

    pred_l2 = classify_level(summary, cats_l2, keryx_l2)
    cands_l3 = get_children(cats_l3, pred_l2, 2)
    pred_l3 = classify_level(summary, cands_l3, keryx_l3) if cands_l3 else pred_l2
    cands_l4 = get_children(cats_l4, pred_l3, 3)
    pred_l4 = classify_level(summary, cands_l4, keryx_l4) if cands_l4 else pred_l3

    return (
        summary,
        pred_l2, name_by_code.get(pred_l2, ''),
        pred_l3, name_by_code.get(pred_l3, ''),
        pred_l4, name_by_code.get(pred_l4, '')
    )

def main():
    text = HARDCODED_TEXT
    print("Тестовый прогон на захардкоженном тексте...")
    summary, l2c, l2n, l3c, l3n, l4c, l4n = classify_text(text)
    print(f"\nСуммаризация:\n{summary}")
    print(f"L2: {l2c} {l2n}")
    print(f"L3: {l3c} {l3n}")
    print(f"L4: {l4c} {l4n}")
    print("\nГотово.")


if __name__ == "__main__":
    main()
