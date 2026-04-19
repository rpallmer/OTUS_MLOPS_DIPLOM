from dataclasses import dataclass

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from config.config import settings
from src.embeddings import Embedder
from src.io_utils import read_jsonl
from src.reranker import ReRanker
from models.schemas import RetrievedDoc  

@dataclass

class Retriever_QA:
    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.qdrant_url)
        self.embedder = Embedder()
        self.reranker: ReRanker | None = None
        self._corpus_rows = read_jsonl(settings.data_corpus_QA_path)
        self._bm25_docs = [r["question"] for r in self._corpus_rows]
        self._bm25_tokens = [self._tokenize(x) for x in self._bm25_docs]
        self._bm25 = BM25Okapi(self._bm25_tokens) if self._bm25_tokens else None
        self._id_by_doc_idx = [r["id"] for r in self._corpus_rows]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def _dense_top_ids(self, col_name: str,query: str, limit: int) -> dict[int | str, int]:
        dense = self.embedder.dense_embed([query])[0]
        result = self.client.query_points(
            collection_name=col_name,
            query=dense,
            using="dense",
            limit=limit,
        )
        return {p.id: i + 1 for i, p in enumerate(result.points)}

    def _sparse_top_ids(self, query: str, limit: int) -> dict[int | str, int]:
        if not self._bm25:
            return {}
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
        return {self._id_by_doc_idx[i]: rank + 1 for rank, i in enumerate(ranked_idx)}

    def _fetch_points_by_ids(self, col_name: str, ids: list[int | str]) -> dict[int | str, dict]:
        points = self.client.retrieve(
            collection_name=col_name,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
        return {p.id: p.payload or {} for p in points}

    @staticmethod
    def _rrf_fuse(
        dense_ranks: dict[int | str, int],
        sparse_ranks: dict[int | str, int],
        limit: int,
    ) -> list[tuple[int | str, float]]:
        candidates = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        fused: list[tuple[int | str, float]] = []
        for pid in candidates:
            d = dense_ranks.get(pid)
            s = sparse_ranks.get(pid)
            score = 0.0
            if d is not None:
                score += 1.0 / (60 + d)
            if s is not None:
                score += 1.0 / (60 + s)
            fused.append((pid, score))
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:limit]

    def search(self, query: str,col_name: str, mode: str = "hybrid", top_k: int = 10,) -> list[RetrievedDoc]:
        dense = self.embedder.dense_embed([query])[0]

        if mode not in {"semantic", "hybrid", "semantic_rerank", "hybrid_rerank"}:
            raise ValueError(f"Unsupported mode: {mode}")

        base_limit = max(100, settings.rerank_candidates)
        dense_ranks = self._dense_top_ids(col_name,query, limit=base_limit)
        sparse_ranks = self._sparse_top_ids(query, limit=base_limit)

        if mode in {"semantic", "semantic_rerank"}:
            response = self.client.query_points(
                collection_name=col_name,
                query=dense,
                using="dense",
                limit=base_limit if mode == "semantic_rerank" else top_k,
                with_payload=True,
            )
            ranked_ids = [p.id for p in response.points]
            payloads = {p.id: (p.payload or {}) for p in response.points}
            scores = {p.id: float(p.score) for p in response.points}
        else:
            fused = self._rrf_fuse(dense_ranks, sparse_ranks, limit=base_limit)
            ranked_ids = [pid for pid, _ in fused]
            payloads = self._fetch_points_by_ids(col_name,ranked_ids)
            scores = dict(fused)

        if mode in {"semantic_rerank", "hybrid_rerank"}:
            if self.reranker is None:
                self.reranker = ReRanker()
            rerank_input = [
                {
                    "id": pid,
                    "question": str(payloads.get(pid, {}).get("question", "")),
                    "answer": str(payloads.get(pid, {}).get("answer", "")),
                    "source": str(payloads.get(pid, {}).get("source", "unknown")),
                    "category": str(payloads.get(pid, {}).get("category", "general")),
                }
                for pid in ranked_ids[: settings.rerank_candidates]
            ]
            reranked = self.reranker.rerank(query=query, candidates=rerank_input, top_k=top_k)
            ranked_ids = [r["id"] for r in reranked]
            scores = {r["id"]: float(r["rerank_score"]) for r in reranked}
        else:
            ranked_ids = ranked_ids[:top_k]

        docs: list[RetrievedDoc] = []
        for i, pid in enumerate(ranked_ids):
            payload = payloads.get(pid, {})
            docs.append(
                RetrievedDoc(
                    point_id=pid,
                    score=float(scores.get(pid, 0.0)),
                    question=str(payload.get("question", "")),
                    answer=str(payload.get("answer", "")),
                    source=str(payload.get("source", "unknown")),
                    category=str(payload.get("category", "general")),
                    dense_rank=dense_ranks.get(pid),
                    sparse_rank=sparse_ranks.get(pid),
                    final_rank=i + 1,
                )
            )
        return docs


def run_hybrid_search_QA(
    query: str,
    col_name: str,
    top_k: int = 5,
    mode: str = "hybrid"  # "semantic" | "hybrid" | "semantic_rerank" | "hybrid_rerank"
) -> list[RetrievedDoc]:

    retriever = Retriever_QA()
    results = retriever.search(query=query,col_name=col_name, mode=mode, top_k=top_k)
      
    return results