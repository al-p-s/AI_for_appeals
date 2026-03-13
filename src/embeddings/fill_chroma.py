import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

ROSBERTA_PATH = "../../models/ru-en-RoSBERTa"
CHROMA_PATH = "../../data/chroma"
CATEGORIES_JSON = "../../data/classifier/cats.json"
COLLECTION_NAME = "appeals_cats"


def load_categories(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["categories"]


def build_chroma(categories: list[dict]) -> Chroma:
    texts = []
    ids = []
    metadatas = []

    for cat in categories:
        parent = cat.get("parent_name", "").strip()
        name = cat.get("name", "").strip()
        code = cat["code"]

        text = f"{parent} / {name}" if parent else name

        texts.append(text)
        ids.append(code)
        metadatas.append({
            "code": code,
            "parent_code": cat.get("parent_code", ""),
            "parent_name": parent,
            "name": name,
        })

    print(f"Загружено категорий: {len(texts)}")
    print("Инициализируем эмбеддинг модель...")

    embeddings = HuggingFaceEmbeddings(
        model_name=ROSBERTA_PATH,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Строим Chroma базу...")
    chroma_db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        ids=ids,
        metadatas=metadatas,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )

    print(f"Готово. База сохранена в: {CHROMA_PATH}")
    return chroma_db


if __name__ == "__main__":
    categories = load_categories(CATEGORIES_JSON)
    build_chroma(categories)
