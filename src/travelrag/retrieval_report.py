"""Run the golden queries against the curated index (web-fallback pages excluded, so calibration is stable).

Usage: python -m travelrag.retrieval_report [top_k]
"""

import sys

from pgvector import Vector

from .config import get_settings
from .db import connect
from .golden import Ranked, first_relevant_rank, load_golden, summarize
from .indexing import embed


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else get_settings().retrieval_top_k
    golden = load_golden()
    vectors = embed([g.query for g in golden])

    results: dict[str, Ranked] = {}
    with connect() as conn:
        for g, vector in zip(golden, vectors):
            rows = conn.execute(
                "select title, similarity, content from match_chunks(%s, %s, false)", (Vector(vector), k)
            ).fetchall()
            results[g.id] = [(title, float(sim), content) for title, sim, content in rows]

    for g in golden:
        ranked = results[g.id]
        rank = first_relevant_rank(g, ranked) if g.category in ("answerable", "detail") else None
        verdict = (f"rank {rank}" if rank else "MISS") if g.category in ("answerable", "detail") else g.category
        print(f"{g.id} [{verdict:>12}] top1={ranked[0][1]:.3f}  {g.query}")
        if g.category != "answerable" or not rank:
            for title, sim, _ in ranked[:3]:
                print(f"        {sim:.3f}  {title[:80]}")

    print()
    for name, value in summarize(golden, results).items():
        print(f"{name:26} {value:.3f}")


if __name__ == "__main__":
    main()
