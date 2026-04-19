from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from config.config import settings
from embeddings import Embedder
from io_utils import read_jsonl


def recreate_collection(col_name: str,client: QdrantClient, dense_dim: int) -> None:
    client.recreate_collection(
        collection_name=col_name,
        vectors_config={"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)},
    )

def index_corpus(col_name: str,corpus_path: str | None = None, batch_size: int = 128) -> dict[str, int]:
    path = corpus_path #or settings.data_corpus_path
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Corpus is empty or missing: {path}")

    embedder = Embedder()
    client = QdrantClient(url=settings.qdrant_url)

    first_vec = embedder.dense_embed([rows[0]["question"]])[0]
    recreate_collection(col_name,client, dense_dim=len(first_vec))

    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [x["question"] for x in batch]
        dense_vectors = embedder.dense_embed(texts)
        points: list[PointStruct] = []
        for row, dvec in zip(batch, dense_vectors, strict=True):
            points.append(
                PointStruct(
                    id=row["id"],
                    vector={"dense": dvec},
                    payload={
                        "question": row["question"],
                        "answer": row["answer"],
                        "source": row.get("source", "unknown"),
                        "category": row.get("category", "general"),
                    },
                )
            )

        client.upsert(collection_name=col_name, points=points, wait=True)
        total += len(points)

    return {"indexed": total}

if __name__ == "__main__":
    result = index_corpus(settings.collection_name_QA,corpus_path=settings.data_corpus_QA_path, batch_size=128)
    print(f"✅ Проиндексировано QA документов: {result['indexed']}")    
    #result = index_corpus(settings.collection_name_KB,corpus_path=settings.data_corpus_KB_path, batch_size=128)
    #print(f"✅ Проиндексировано KB документов: {result['indexed']}")    