"""Compare chunk sizes on the golden set without touching the database.

Re-chunks and re-embeds every stored article in memory for each config, then scores retrieval.
Usage: python -m travelrag.chunk_experiment
"""

import numpy as np

from .chunking import chunk_text, count_tokens
from .db import connect
from .golden import Ranked, load_golden, summarize
from .indexing import embed

CONFIGS = [(150, 20), (250, 40), (400, 50), (600, 80), (900, 100)]  # (size, overlap) in tokens
TOP_K = 8


def main() -> None:
    golden = load_golden()
    query_vectors = np.array(embed([g.query for g in golden]), dtype=np.float32)
    with connect(with_vectors=False) as conn:
        articles = conn.execute("select title, body from articles where source_type = 'feed'").fetchall()

    print(f"{'size/overlap':>12} {'chunks':>6} {'avg tok':>7} | {'recall@1':>8} {'recall@4':>8} | "
          f"{'detail@1':>8} {'detail@4':>8} {'detail@8':>8} {'d.mrr':>6} | {'ooi max':>7} {'ans min':>7} | ctx tok (top4)")
    for size, overlap in CONFIGS:
        rows: list[tuple[str, str]] = []  # (title, chunk text)
        for title, body in articles:
            rows.extend((title, part) for part in chunk_text(body, size, overlap))
        vectors = np.array(embed([f"{t}\n\n{c}" for t, c in rows]), dtype=np.float32)
        similarities = query_vectors @ vectors.T  # embeddings are unit length, so dot = cosine

        results: dict[str, Ranked] = {}
        for g, sims in zip(golden, similarities):
            top = np.argsort(-sims)[:TOP_K]
            results[g.id] = [(rows[i][0], float(sims[i]), rows[i][1]) for i in top]

        m = summarize(golden, results)
        avg_tokens = sum(count_tokens(c) for _, c in rows) / len(rows)
        print(f"{f'{size}/{overlap}':>12} {len(rows):>6} {avg_tokens:>7.0f} | {m['recall@1']:>8.2f} {m['recall@4']:>8.2f} | "
              f"{m['detail_recall@1']:>8.2f} {m['detail_recall@4']:>8.2f} {m['detail_recall@8']:>8.2f} {m['detail_mrr']:>6.2f} | "
              f"{m['out_of_index_top1_max']:>7.3f} {m['answerable_top1_min']:>7.3f} | {4 * avg_tokens:>6.0f}", flush=True)


if __name__ == "__main__":
    main()
