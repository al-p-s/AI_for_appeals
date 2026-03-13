from typing import List
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

USER2_PATH = "../../models/USER2-base"


class USER2Embeddings(Embeddings):
    def __init__(self, model_path: str = USER2_PATH, device: str = "cuda"):
        self.model = SentenceTransformer(model_path, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            prompt_name="search_document",
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            text,
            prompt_name="search_query",
            normalize_embeddings=True,
        )
        return embedding.tolist()
