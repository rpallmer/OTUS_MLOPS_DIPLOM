import argparse
from statistics import mean

from src.io_utils import read_jsonl
from src.retrieval_QA import Retriever_QA
from tqdm import tqdm


def recall_at_k(relevant: set, retrieved: list, k: int) -> float:
    top = set(retrieved[:k])
    if not relevant:
        return 0.0
    return 1.0 if relevant.intersection(top) else 0.0


def mrr_at_10(relevant: set, retrieved: list) -> float:
    for idx, point_id in enumerate(retrieved[:10], start=1):
        if point_id in relevant:
            return 1.0 / idx
    return 0.0


def evaluate(mode: str, eval_rows: list[dict], top_k: int) -> dict[str, float]:
    retriever = Retriever_QA()
    r5, r10, mrr = [], [], []
    for row in tqdm(eval_rows, desc=f"Evaluating {mode}", unit="query"):
        docs=retriever.search(row["query"],col_name="QA_ES", mode=mode, top_k=top_k)
        retrieved_ids = [d.point_id for d in docs]
        relevant = set(row["relevant_ids"])
        r5.append(recall_at_k(relevant, retrieved_ids, 5))
        r10.append(recall_at_k(relevant, retrieved_ids, 10))
        mrr.append(mrr_at_10(relevant, retrieved_ids))

    return {
        "Recall@5": round(mean(r5), 4),
        "Recall@10": round(mean(r10), 4),
        "MRR@10": round(mean(mrr), 4),
    }


def main() -> None:

    eval_rows = read_jsonl("data/processed/eval.jsonl")
    if not eval_rows:
        raise ValueError("Eval file is empty. Run prepare_data.py first.")

    modes = ("semantic", "hybrid", "semantic_rerank", "hybrid_rerank")
    results: list[tuple[str, dict[str, float]]] = []

    for mode in modes:
        metrics = evaluate(mode, eval_rows, top_k=10)
        results.append((mode, metrics))

    print("\nRetrieval metrics")
    headers = ["Mode", "Recall@5", "Recall@10", "MRR@10"]
    rows = [
        [
            mode,
            f"{metrics['Recall@5']:.4f}",
            f"{metrics['Recall@10']:.4f}",
            f"{metrics['MRR@10']:.4f}",
        ]
        for mode, metrics in results
    ]
    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt_row(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    print(fmt_row(headers))
    print("|-" + "-|-".join("-" * w for w in widths) + "-|")
    for row in rows:
        print(fmt_row(row))


if __name__ == "__main__":
    main()
