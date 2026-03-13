from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer
from geracl import GeraclHF, ZeroShotClassificationPipeline

from src.embeddings.USER2Embeddings import USER2Embeddings

CHROMA_PATH = "../../data/chroma"
COLLECTION_NAME = "appeals_cats"
GERACL_MODEL = "../../models/GeRaCl-USER2-base"

TOP_K = 15

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


def classify(text: str, chroma_db: Chroma, geracl_pipe: ZeroShotClassificationPipeline) -> dict:
    # шаг 1: retrieval — топ-K кандидатов из Chroma
    results = chroma_db.similarity_search_with_score(text, k=TOP_K)

    candidates = []
    for doc, score in results:
        candidates.append({
            "code": doc.metadata["code"],
            "name": doc.metadata["name"],
            "parent_name": doc.metadata["parent_name"],
            "label": f"{doc.metadata['parent_name']} / {doc.metadata['name']}" if doc.metadata["parent_name"] else doc.metadata["name"],
            "retrieval_score": float(score),
        })

    # шаг 2: reranking — GeRaCl выбирает лучшую из кандидатов
    labels = [c["label"] for c in candidates]
    best_idx = geracl_pipe([text], [labels], batch_size=1)[0]
    best = candidates[best_idx]

    return {
        "text": text,
        "result": {
            "code": best["code"],
            "name": best["name"],
            "parent_name": best["parent_name"],
        },
        "candidates": candidates,
    }


if __name__ == "__main__":
    print("Загружаем модели...")
    chroma_db = load_chroma()
    geracl_pipe = load_geracl()
    print("Готово.\n")

    test_texts = [
        "Здравствуйте жена вчера шла на работу упала на спину и ударилась локтем. Получила вывих. Кто-нибудь будет чистить тротуары? Там всё во льду и буграх. Ходить не возможно!",
        "Не работает светофор на перекрёстке улиц Ленина и Кирова уже две недели",
        "Прошу рассмотреть вопрос о выделении земельного участка под строительство",
    ]

    for text in test_texts:
        print(f"Текст: {text}")
        result = classify(text, chroma_db, geracl_pipe)
        r = result["result"]
        print(f"Категория: [{r['code']}] {r['parent_name']} / {r['name']}")
        print(f"Топ-3 кандидата из Chroma:")
        for c in result["candidates"][:3]:
            print(f"  [{c['code']}] {c['label']} (score: {c['retrieval_score']:.4f})")
        print()
