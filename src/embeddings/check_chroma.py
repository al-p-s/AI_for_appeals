from langchain_community.vectorstores import Chroma
from src.embeddings.USER2Embeddings import USER2Embeddings
import random

db = Chroma(
    persist_directory="../../data/chroma",
    embedding_function=USER2Embeddings(),
    collection_name="appeals_cats"
)

col = db._collection.get(include=["documents", "metadatas"])
docs = list(zip(col["documents"], col["metadatas"]))

synthetic = [(d, m) for d, m in docs if "_s" in col["ids"][docs.index((d, m))] if False]

all_ids = col["ids"]
synthetic = [(col["documents"][i], col["metadatas"][i])
             for i, id_ in enumerate(all_ids) if "_s" in id_]

for doc, meta in random.sample(synthetic, 30):
    print(f"[{meta['code']}] {meta['name']}")
    print(f"  → {doc}\n")
