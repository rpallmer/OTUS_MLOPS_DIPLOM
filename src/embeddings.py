from collections.abc import Iterable

import torch
from sentence_transformers import SentenceTransformer

from config.config import settings


class Embedder:
    def __init__(self) -> None:
        # Use GPU when available; otherwise fallback to CPU.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dense = SentenceTransformer(settings.dense_model, device=device)

    def dense_embed(self, texts: Iterable[str]) -> list[list[float]]:
        vectors = self.dense.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()
