from langchain_community.vectorstores import Chroma
from src.embeddings.USER2Embeddings import USER2Embeddings

db = Chroma(
    persist_directory="../../data/chroma",
    embedding_function=USER2Embeddings(),
    collection_name="appeals_cats"
)

targets = [
    "Государственный земельный надзор",
    "Устранение аварийных ситуаций на магистральных коммуникациях. Работа аварийных коммунальных служб",
    "Городской, сельский и междугородний пассажирский транспорт",
]

for name in targets:
    print(f"\n=== {name} ===")
    results = db.similarity_search_with_score(name, k=10)
    for doc, score in results:
        print(f"  {score:.4f}  {doc.metadata['name']}")
