import torch
from sentence_transformers import CrossEncoder

from config.config import settings


class ReRanker:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(settings.reranker_model, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        if not candidates:
            return []
        pairs = [
            (query, f"{c.get('question', '')}\n{c.get('answer', '')}".strip())
            for c in candidates
        ]
        try:
            scores = self.model.predict(pairs, batch_size=4)
        except torch.OutOfMemoryError:
            # If vLLM occupies most VRAM, continue evaluation on CPU instead of failing.
            if self.device == "cuda":
                self.device = "cpu"
                self.model = CrossEncoder(settings.reranker_model, device="cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                scores = self.model.predict(pairs, batch_size=4)
            else:
                raise
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        return [
            {**cand, "rerank_score": float(score)}
            for cand, score in ranked[:top_k]
        ]
