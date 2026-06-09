import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import AutoModelForSequenceClassification
import torch
import logging
from datetime import datetime

CATS_L2 = "../../data/classifier/cats2.json"
CATS_L3 = "../../data/classifier/cats3.json"
CATS_L4 = "../../data/classifier/cats4.json"

GIGACHAT_PATH = "../../models/gigaChat_lite"
KERYX_PATH_L2 = "../../models/KERYX_1068_multi/L2"
KERYX_PATH_L3 = "../../models/KERYX_1068_multi/L3"
KERYX_PATH_L4 = "../../models/KERYX_1068_multi/L4"

THRESHOLD_L2 = 0.9
THRESHOLD_L3 = 0.85
THRESHOLD_L4 = 0.8

HARDCODED_TEXT = """
Эбращение № 7320927 a\n\nДата дедлайна по исполнению: 17.03.2\n\n(©) Информация о гражданине Автоопределение @)\nФИО гражданина: ©. гитоево_ Татьяна Сергеевна s :\nЭлектронная почта: muvaveva— GU@ mail. ru\n\nНомер телефона: “s7sasov2795 = Ot -\n\n+7(952)501-27-95\n\nИНН гражданина:\n\nФИО обратившегося:\n\nАдрес обратившегося: |\n\nСекретно О да @ Her\n\n@ География обращения Автоопределение iif)\n\nАдрес источника\n\nФедеральный Округ\n\nРегион\n\nМуниципальное\nобразование\n\nАдрес\n\nМ источник обращения\n\nКанал\nПоток\n\nСобытие\n\nДАН И ОРГАНИЗАЦИЙ\nВходящий №2 8\n\n8 Информация об исполнителе 19.02 25\n\nОрганизация\n\n9 информация о волонтёре Обработка волонтёром: —\n\nВ Ход действий\n\nВолонтёры:\n\nBE текст обращения\nКоличество спама и обсценной лексики: 0 %\n\nМеня зовут Рассчитается Татьяна Сергеевна. Я звоню из города Челябинска. Я хотела бы попросить п\nомощи у Владимира Владимировича вопросе основе домов улицы Ярославская. Дом четырнадцать.\nДома давным давно в аварийном состоянии, уже практически нет подачи нормальной воды, отоплен\nие. Никто не занимается обслуживанием дома два РА, в том числе только управляющая компания бе\nрет деньги. Им каждый год обещают, что их расселят, но уже очень много лет их никто не расселяет д\nом, но уже стоит на ладан дышит. Вот я хотела бы попросить Владимира Владимировича как-то помо\nчь в решении этого вопроса и уже наконец то, чтобы их расстелили. Спасибо большое.\n\nСохранить\n\n\nсуо. организация | | |\n\nРешение исполнителя\n\nОтвет исполнителя : Развернуть\n\nТематика обращения\n\nТип категории\n\nКатегория обращения\n\nПодкатегория\nобращения\n\n[$] География гражданина Автоопределение ©)\n\nАдрес источника\nФедеральный Округ\n\nРегион\n\nМуниципальное\nобразование\n\nАдрес\n(®} Дополнительная информация\n\nСрочно: ©) Да @) Нет Обращались ранее: ©) Да ©) Нет\n\nВозраст на момент\nсоздания обращения\n\nОсобые метки сообщения: ap\n\nСистема-источник\n\n
"""

SUMM_PROMPT = """Ты — эксперт по суммаризации обращений граждан. Напиши выжимку в 1-2 предложения, строго по правилам:

1. Не повторяй: ФИО (пиши «житель», «жительница» или «заявитель»), номер документа, телефон, email, социальное положение, входящие номера, приветствия и подписи.
2. Отрази суть: КТО (житель такого-то района/улицы) → ЧТО ПРОСИТ или НА ЧТО ЖАЛУЕТСЯ → ПОЧЕМУ (одна-две главные причины).
3. Говори коротко, без канцелярита («просит согласовать», «требует уборки», «выражает негодование» вместо «прошу обеспечить проведение мероприятий»).

ОБРАЩЕНИЕ:
{text}"""

# LOG_FILE = "full_pipeline.log"
#
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
#     handlers=[
#         logging.FileHandler(LOG_FILE, encoding="utf-8"),
#         logging.StreamHandler()
#     ]
# )

logger = logging.getLogger(__name__)

def format_preds(preds):
    return "\n".join(
        f"{c['code']} — {c['name']}"
        for c, s in preds
    )

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


def classify_level_multi(summary, candidates, tokenizer_model, threshold, prefix=""):
    tokenizer, model = tokenizer_model
    scores = []
    logger.info(f"Classification started | candidates={len(candidates)} | threshold={threshold}")
    for c in candidates:
        enc = tokenizer(summary, prefix + c["name"], return_tensors="pt",
                        truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
        scores.append((c, probs[0][0].item()))

    # result = [(c, s) for c, s in scores if s >= threshold]
    max_score = max(s for _, s in scores)
    result = [(c, s) for c, s in scores if s >= max_score * threshold]
    logger.info(f"Classification finished | selected={len(result)}")
    if not result:
        result = [max(scores, key=lambda x: x[1])]
    return result


def get_children(cats, parent_code, level):
    prefix = ".".join(parent_code.split(".")[:level])
    return [c for c in cats if c["code"].startswith(prefix + ".")]


def classify_text(text: str):
    summary = summarize(text, gigachat_tok, gigachat_model, gigachat_gen)

    pred_l2 = classify_level_multi(summary, cats_l2, keryx_l2, THRESHOLD_L2)

    pred_l3 = []
    for l2, l2_score in pred_l2:
        cands_l3 = get_children(cats_l3, l2["code"], 2)
        l2_name = name_by_code.get(l2["code"], "")
        if cands_l3:
            for l3, l3_score in classify_level_multi(summary, cands_l3, keryx_l3, THRESHOLD_L3, prefix=f"{l2_name} → "):
                pred_l3.append((l3, l3_score))

    pred_l4 = []
    for l3, l3_score in pred_l3:
        cands_l4 = get_children(cats_l4, l3["code"], 3)
        l3_name = name_by_code.get(l3["code"], "")
        l2_code = ".".join(l3["code"].split(".")[:2]) + ".0000.0000"
        l2_name = name_by_code.get(l2_code, "")
        if cands_l4:
            for l4, l4_score in classify_level_multi(summary, cands_l4, keryx_l4, THRESHOLD_L4,
                                                     prefix=f"{l2_name} → {l3_name} → "):
                pred_l4.append((l4, l4_score))

    logger.info(f"Summary: {summary}")

    logger.info(f"L2 predictions: {[(c['code'], c['name'], round(s, 3)) for c, s in pred_l2]}")
    logger.info(f"L3 predictions: {[(c['code'], c['name'], round(s, 3)) for c, s in pred_l3]}")
    logger.info(f"L4 predictions: {[(c['code'], c['name'], round(s, 3)) for c, s in pred_l4]}")
    return (
        summary,
        format_preds(pred_l2),
        format_preds(pred_l3),
        format_preds(pred_l4),
    )

logger.info("Classificator initialization (cats and models loading)...")

cats_l2 = load_json(CATS_L2)["categories"]
cats_l3 = load_json(CATS_L3)["categories"]
cats_l4 = load_json(CATS_L4)["categories"]
name_by_code = {c["code"]: c["name"] for c in cats_l2 + cats_l3 + cats_l4}

gigachat_tok, gigachat_model, gigachat_gen = load_gigachat()
keryx_l2 = load_keryx(KERYX_PATH_L2)
keryx_l3 = load_keryx(KERYX_PATH_L3)
keryx_l4 = load_keryx(KERYX_PATH_L4)

logger.info("Models ready for inference.")

def main():
    text = HARDCODED_TEXT
    logger.info("Test inference on hard-code text...")
    summary, l2, l3, l4 = classify_text(text)

    logger.info("Done")

if __name__ == "__main__":
    main()
