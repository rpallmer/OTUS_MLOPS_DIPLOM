import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name_QA: str = os.getenv("COLLECTION_NAME_QA", "QA_ES")  
    collection_name_KB: str = os.getenv("COLLECTION_NAME_KB", "KB_ES")
    data_corpus_QA_path: str = os.getenv("DATA_CORPUS_QA_PATH", "data/processed/corpus_QA.jsonl")
    data_corpus_KB_path: str = os.getenv("DATA_CORPUS_KB_PATH", "data/processed/corpus_KB.jsonl")
    data_eval_path: str = os.getenv("DATA_EVAL_PATH", "data/processed/eval.jsonl")
    dense_model: str = os.getenv("DENSE_MODEL", "models/BAAI--bge-m3")
    reranker_model: str = os.getenv("RERANKER_MODEL", "models/bge-reranker-v2-m3")
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "30"))
    llm_model: str = os.getenv("LLM_MODEL", "mlx-community/Qwen3.5-9B-MLX-4bit")
   
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))


settings = Settings()
