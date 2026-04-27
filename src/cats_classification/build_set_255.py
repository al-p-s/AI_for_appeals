import json
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from src.embeddings.USER2Embeddings import USER2Embeddings

CHROMA_PATH = "../../data/chroma"
COLLECTION_NAME = "appeals_cats"
GIGACHAT_MODEL = "../../models/gigaChat_lite"
APPEALS_WITH_CATS = "../../data/appeals_with_cats.json"
APPEALS_WITH_TEXT = "../../data/appeals.json"
OUTPUT_PATH = "../../data/sets_to_learn/dataset_255.json"
TOP_K = 60

MARKERS = [
    "Текст обращения:",
    "Текст обращения :",
    "Содержание обращения:",
    "Содержание обращения :",
    "Вопрос:"
]


def load_chroma() -> Chroma:
    embeddings = USER2Embeddings()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def load_gigachat() -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(GIGACHAT_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        GIGACHAT_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    generation_config = GenerationConfig.from_pretrained(GIGACHAT_MODEL, trust_remote_code=True)
    # generation_config.max_new_tokens = 30
    # generation_config.temperature = 0.1
    # generation_config.top_p = 0.9
    # generation_config.repetition_penalty = 1.2
    # generation_config.do_sample = False
    return tokenizer, model, generation_config


def summarize(text: str, tokenizer, model, generation_config) -> str:
    # query = (
    #     f"""Ты — классификатор обращений. Прочитай обращение и выпиши через запятую основные ключевые слова и фразы,
    #     которые отражают суть проблемы гражданина. Включи адреса, номера, названия организаций, тип проблемы (отопление, вода, дороги и т.д.).
    #     Затем на основе этих ключевых слов составь одно официальное предложение-резюме.\n\n
    #     ОБРАЩЕНИЕ:\n{text}"""
    # )
    query = (
        f"""Ты — эксперт по классификации обращений граждан.
        Извлеки из текста обращения ключевую информацию, необходимую для точного определения категории по классификатору.
        Можешь ориентироваться на Официальный Классификатор Обращений Граждан РФ.
        Сформулируй результат как одно компактное предложение (не более 40 слов) в официальном стиле, без вводных конструкций.\n\n
        ОБРАЩЕНИЕ:\n{text}"""
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


def extract_content(text: str) -> str:
    for marker in MARKERS:
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
    return text


def get_true_categories(categories: list[str]) -> list[dict]:
    result = []
    for cat in categories:
        parts = cat.split(" ", 1)
        result.append({
            "code": parts[0].strip(),
            "name": parts[1].strip() if len(parts) > 1 else ""
        })
    return result


def get_chroma_candidates(text: str, chroma_db: Chroma) -> list[dict]:
    results = chroma_db.similarity_search_with_score(text, k=TOP_K)
    return [
        {
            "code": doc.metadata["code"],
            "name": doc.metadata["name"],
            "score": float(score)
        }
        for doc, score in results
    ]


def build():
    with open(APPEALS_WITH_CATS, "r", encoding="utf-8") as f:
        cats_data = json.load(f)
    with open(APPEALS_WITH_TEXT, "r", encoding="utf-8") as f:
        text_data = json.load(f)

    text_by_file = {a["file_name"]: a["text"] for a in text_data["appeals"]}
    appeals = cats_data["appeals"]

    print(f"Всего обращений: {len(appeals)}")
    print("Загружаем модели...")
    chroma_db = load_chroma()
    gigachat_tokenizer, gigachat_model, gigachat_gen_config = load_gigachat()
    print("Готово.\n")

    dataset = []
    skipped = 0
    true_not_in_topk = 0

    for appeal in tqdm(appeals, desc="Building dataset"):
        file_name = appeal["file_name"]
        text = text_by_file.get(file_name, "").strip()

        if not text:
            skipped += 1
            continue

        extracted = extract_content(text)
        summary = summarize(extracted, gigachat_tokenizer, gigachat_model, gigachat_gen_config)
        true_cats = get_true_categories(appeal["categories"])
        candidates = get_chroma_candidates(summary, chroma_db)

        true_codes = {c["code"] for c in true_cats}
        candidate_codes = {c["code"] for c in candidates}
        hit = bool(true_codes & candidate_codes)

        if not hit:
            true_not_in_topk += 1

        dataset.append({
            "file_name": file_name,
            "summary": summary,
            "true_categories": true_cats,
            "chroma_candidates": candidates,
            "true_in_topk": hit
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"appeals": dataset}, f, ensure_ascii=False, indent=2)

    total = len(dataset)
    print(f"\nГотово. Обработано: {total} | Пропущено: {skipped}")
    print(f"Истинная категория в top-{TOP_K}: {total - true_not_in_topk}/{total} ({(total - true_not_in_topk)/total*100:.1f}%)")
    print(f"Промахи (не в top-{TOP_K}): {true_not_in_topk} → {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
