import json
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from src.embeddings.USER2Embeddings import USER2Embeddings
import torch
from tqdm import tqdm

CHROMA_PATH = "../../data/chroma"
COLLECTION_NAME = "appeals_cats"
GIGACHAT_MODEL = "../../models/gigaChat_lite"
CATEGORIES_PATH = "../../data/classifier/cats.json"


def load_chroma() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=USER2Embeddings(),
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
    generation_config.max_new_tokens = 120
    generation_config.do_sample = False
    return tokenizer, model, generation_config


def generate_examples(category_name: str, tokenizer, model, generation_config) -> list[str]:
    query = (
        f"""Ты — эксперт по обращениям граждан.
Категория классификатора: «{category_name}»

Напиши 5 коротких ключевых фраз (каждая до 10 слов), которые типично используют граждане в обращениях по этой теме.
Фразы должны отражать реальные жалобы, просьбы или вопросы граждан.
Ответ — только пронумерованный список, без пояснений."""
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
    raw = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    examples = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # убираем нумерацию "1. ", "1) " и т.д.
        for sep in [". ", ") ", ": "]:
            if line[1:3] == sep or (len(line) > 2 and line[0].isdigit() and line[1] == sep[0]):
                line = line.split(sep, 1)[-1].strip()
                break
        if line:
            examples.append(line)

    return examples[:5]


def enrich():
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        categories = json.load(f)["categories"]

    chroma_db = load_chroma()
    tokenizer, model, generation_config = load_gigachat()

    print(f"Категорий: {len(categories)}")
    print(f"Документов в Chroma до: {chroma_db._collection.count()}\n")

    docs = []
    doc_ids = []
    for cat in tqdm(categories, desc="Generating"):
        examples = generate_examples(cat["name"], tokenizer, model, generation_config)
        for i, ex in enumerate(examples):
            docs.append(Document(
                page_content=ex,
                metadata={
                    "code": cat["code"],
                    "name": cat["name"],
                }
            ))
            doc_ids.append(f"{cat['code']}_s{i}")

        if len(docs) >= 100:
            chroma_db.add_documents(docs, ids=doc_ids)
            docs = []
            doc_ids = []

    if docs:
        chroma_db.add_documents(docs, ids=doc_ids)

    print(f"\nДокументов в Chroma после: {chroma_db._collection.count()}")


if __name__ == "__main__":
    enrich()
