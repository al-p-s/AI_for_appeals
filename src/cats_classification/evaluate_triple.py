import json
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
import torch
from geracl import GeraclHF, ZeroShotClassificationPipeline
from src.embeddings.USER2Embeddings import USER2Embeddings

CHROMA_PATH = "../../data/chroma_backup"
COLLECTION_NAME = "appeals_cats"
GERACL_MODEL = "../../models/GeRaCl-USER2-base"
GIGACHAT_MODEL = "../../models/gigaChat_lite"
APPEALS_WITH_CATS = "../../data/appeals_with_cats.json"
APPEALS_WITH_TEXT = "../../data/appeals.json"
TOP_K = 60
EVAL_LIMIT = 10


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
    model = AutoModelForCausalLM.from_pretrained(
        GIGACHAT_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    generation_config = GenerationConfig.from_pretrained(GIGACHAT_MODEL, trust_remote_code=True)
    generation_config.max_new_tokens = 80
    generation_config.do_sample = False
    return tokenizer, model, generation_config


def summarize(text: str, tokenizer, model, generation_config) -> str:
    query = (
        '''Ты — эксперт по классификации обращений граждан РФ.
    Прочитай обращение и сформулируй его суть в виде 2-3 коротких номинальных фраз (существительное + прилагательное / уточнение), как
    будто это название раздела официального классификатора.
    Без глаголов, без предложений, без имён и адресов.
    ОБРАЩЕНИЕ: {text}'''
        # f"""Ты — эксперт по классификации обращений граждан.
        # Прочитай обращение и выпиши ключевые слова и короткие фразы, которые наиболее точно описывают суть проблемы.
        # Игнорируй эмоции и медицинские детали. Выдели только суть административного запроса к органу власти.
        # Первым словом укажи тематику: Земля / Транспорт / ЖКХ / Соцзащита / Строительство / Здравоохранение / и т.д.
        # Затем — ключевые слова сути проблемы, аварии или происшествия. Игнорируй имена, даты, адреса, реквизиты.
        # Ответ — только список слов и коротких фраз через запятую, без предложений и пояснений.\n\n
        # ОБРАЩЕНИЕ:\n{text}"""
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
            "score": float(score),
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

    full_strict = 0;  full_topk = 0
    summ_strict = 0;  summ_topk = 0

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
        print(f"Суммаризация {file_name}...")
        summary = summarize(text, gigachat_tokenizer, gigachat_model, gigachat_gen_config)

        try:
            print(f"Classify full...")
            pred_full, topk_full, _, _                            = classify(text,    chroma_db, geracl_pipe)
            print(f"Classify summ...")
            pred_summ, topk_summ, candidates_summ, best_idx_summ = classify(summary, chroma_db, geracl_pipe)
        except Exception as e:
            print(f"Ошибка на {file_name}: {e}")
            skipped += 1
            continue

        total += 1

        if pred_full in true_codes:     full_strict += 1
        if true_codes & set(topk_full): full_topk += 1

        if pred_summ in true_codes:     summ_strict += 1
        if true_codes & set(topk_summ): summ_topk += 1
        else:
            errors_summ.append({
                "file": file_name,
                "summary": summary,
                "true": true_labeled,
                "predicted_code": pred_summ,
                "predicted_label": candidates_summ[best_idx_summ]["label"],
                "top_k": [
                    {"code": c["code"], "label": c["label"], "score": c["score"]}
                    for c in candidates_summ
                ],
            })

    print(f"\n{'='*50}")
    print(f"Обработано: {total}  |  Пропущено: {skipped}")
    print(f"\n{'='*50}")
    print(f"{'':30} {'Полный':>9} {'Суммаризация':>13}")
    print(f"{'Strict accuracy':30} {full_strict/total*100:>8.1f}% {summ_strict/total*100:>12.1f}%")
    print(f"{'Top-{} accuracy'.format(TOP_K):30} {full_topk/total*100:>8.1f}% {summ_topk/total*100:>12.1f}%")
    print(f"{'='*50}")

    if errors_summ:
        with open("../../data/classifier/eval_errors_summ10.json", "w", encoding="utf-8") as f:
            json.dump(errors_summ, f, ensure_ascii=False, indent=2)
        print(f"\nПромахи top-{TOP_K} (суммаризация): {len(errors_summ)} шт.")


if __name__ == "__main__":
    evaluate()
