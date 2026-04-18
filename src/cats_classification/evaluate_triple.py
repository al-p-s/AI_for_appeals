import json
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
import torch
from geracl import GeraclHF, ZeroShotClassificationPipeline
from src.embeddings.USER2Embeddings import USER2Embeddings

CHROMA_PATH = "../../data/chroma"
COLLECTION_NAME = "appeals_cats"
GERACL_MODEL = "../../models/GeRaCl-USER2-base"
GIGACHAT_MODEL = "../../models/gigaChat_lite"
APPEALS_WITH_CATS = "../../data/appeals_with_cats.json"
APPEALS_WITH_TEXT = "../../data/appeals.json"
TOP_K = 10
EVAL_LIMIT = 1

MARKERS = [
    "Текст обращения:",
    "Текст обращения :",
    "Содержание обращения:",
    "Содержание обращения :",
    "Вопрос:"
]

MARKERS = []


def load_chroma() -> Chroma:
    embeddings = USER2Embeddings()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def load_geracl() -> ZeroShotClassificationPipeline:
    model = GeraclHF.from_pretrained(GERACL_MODEL).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(GERACL_MODEL)
    return ZeroShotClassificationPipeline(model, tokenizer, device="cuda")


def load_gigachat() -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(GIGACHAT_MODEL, trust_remote_code=True)

    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        GIGACHAT_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    generation_config = GenerationConfig.from_pretrained(GIGACHAT_MODEL, trust_remote_code=True)

    return tokenizer, model, generation_config


def summarize(text: str, tokenizer, model, generation_config) -> str:
    query = (
        "Выдели основную суть проблемы из обращения гражданина. "
        "Одно предложение, официальный стиль, без вводных слов.\n\n"
        f"ОБРАЩЕНИЕ:\n{text}"
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tokenize=False, add_generation_prompt=True
    )
    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    data = {k: v.to(model.device) for k, v in data.items()}
    data.pop("token_type_ids", None)

    output_ids = model.generate(**data, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def extract_content(text: str) -> tuple[str, str]:
    for marker in MARKERS:
        idx = text.find(marker)
        if idx != -1:
            extracted = text[idx + len(marker):].strip()
            return extracted, marker
    return text, ""


def get_true_codes(categories: list[str]) -> tuple[set[str], list[dict]]:
    codes = set()
    labeled = []
    for cat in categories:
        parts = cat.split(" ", 1)
        code = parts[0].strip()
        name = parts[1].strip() if len(parts) > 1 else ""
        codes.add(code)
        labeled.append({"code": code, "name": name})
    return codes, labeled


def classify(text: str, chroma_db: Chroma, geracl_pipe: ZeroShotClassificationPipeline) -> tuple[str, list[str], list[dict], int]:
    results = chroma_db.similarity_search_with_score(text, k=TOP_K)

    candidates = []
    for doc, score in results:
        candidates.append({
            "code": doc.metadata["code"],
            "name": doc.metadata["name"],
            "label": doc.metadata["name"],
        })

    labels = [c["label"] for c in candidates]
    best_idx = geracl_pipe([text], [labels], batch_size=1)[0]
    predicted_code = candidates[best_idx]["code"]
    top_k_codes = [c["code"] for c in candidates]

    return predicted_code, top_k_codes, candidates, best_idx


def evaluate():
    with open(APPEALS_WITH_CATS, "r", encoding="utf-8") as f:
        cats_data = json.load(f)
    with open(APPEALS_WITH_TEXT, "r", encoding="utf-8") as f:
        text_data = json.load(f)

    text_by_file = {a["file_name"]: a["text"] for a in text_data["appeals"]}
    appeals = cats_data["appeals"]

    if EVAL_LIMIT:
        appeals = appeals[:EVAL_LIMIT]
        print(f"[!] Режим теста: первые {EVAL_LIMIT} обращений")

    print(f"Всего обращений: {len(appeals)}")
    print("Загружаем модели...")
    chroma_db = load_chroma()
    geracl_pipe = load_geracl()
    gigachat_tokenizer, gigachat_model, gigachat_gen_config = load_gigachat()
    print("Готово.\n")

    marker_found = 0
    marker_stats = {m: 0 for m in MARKERS}

    full_strict = 0;   full_topk = 0
    cut_strict = 0;    cut_topk = 0
    summ_strict = 0;   summ_topk = 0

    total = 0
    skipped = 0
    errors_summ = []

    for appeal in tqdm(appeals, desc="Evaluating"):
        file_name = appeal["file_name"]
        text = text_by_file.get(file_name, "").strip()

        if not text:
            skipped += 1
            continue

        true_codes, true_labeled = get_true_codes(appeal["categories"])
        extracted, found_marker = extract_content(text)

        if found_marker:
            marker_found += 1
            marker_stats[found_marker] += 1

        # суммаризируем обрезанный текст (если маркер найден) или полный
        text_for_summary = extracted if found_marker else text
        summary = summarize(text_for_summary, gigachat_tokenizer, gigachat_model, gigachat_gen_config)

        try:
            pred_full,  topk_full,  _,              _            = classify(text,     chroma_db, geracl_pipe)
            pred_cut,   topk_cut,   candidates_cut, best_idx_cut = classify(extracted, chroma_db, geracl_pipe)
            pred_summ,  topk_summ,  candidates_summ, best_idx_summ = classify(summary, chroma_db, geracl_pipe)
        except Exception as e:
            print(f"Ошибка на {file_name}: {e}")
            skipped += 1
            continue

        total += 1

        if pred_full in true_codes:    full_strict += 1
        if true_codes & set(topk_full): full_topk += 1

        if pred_cut in true_codes:     cut_strict += 1
        if true_codes & set(topk_cut):  cut_topk += 1

        if pred_summ in true_codes:    summ_strict += 1
        if true_codes & set(topk_summ): summ_topk += 1
        else:
            errors_summ.append({
                "file": file_name,
                "summary": summary,
                "true": true_labeled,
                "predicted_code": pred_summ,
                "predicted_label": candidates_summ[best_idx_summ]["label"],
                "top_k": [{"code": c["code"], "label": c["label"]} for c in candidates_summ],
            })

    print(f"\n{'='*62}")
    print(f"Обработано: {total}  |  Пропущено: {skipped}")
    print(f"Маркер найден: {marker_found}/{total} ({marker_found/total*100:.1f}%)")
    print(f"\nСтатистика по маркерам:")
    for m, cnt in marker_stats.items():
        if cnt > 0:
            print(f"  '{m}': {cnt}")
    print(f"\n{'='*62}")
    print(f"{'':30} {'Полный':>9} {'Обрезанный':>11} {'Суммаризация':>13}")
    print(f"{'Strict accuracy':30} {full_strict/total*100:>8.1f}% {cut_strict/total*100:>10.1f}% {summ_strict/total*100:>12.1f}%")
    print(f"{'Top-{} accuracy'.format(TOP_K):30} {full_topk/total*100:>8.1f}% {cut_topk/total*100:>10.1f}% {summ_topk/total*100:>12.1f}%")
    print(f"{'='*62}")

    if errors_summ:
        with open("../../data/eval_errors_summ.json", "w", encoding="utf-8") as f:
            json.dump(errors_summ, f, ensure_ascii=False, indent=2)
        print(f"\nКейсы где топ-{TOP_K} не попал (суммаризация): {len(errors_summ)} шт. → data/eval_errors_summ.json")


if __name__ == "__main__":
    evaluate()
