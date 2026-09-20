"""Corpus statistics used to choose the chunk size. Usage: python -m travelrag.stats"""

import statistics

from .chunking import count_tokens
from .db import connect


def pct(values: list[int], q: float) -> int:
    return sorted(values)[min(len(values) - 1, int(len(values) * q))]


def main() -> None:
    with connect(with_vectors=False) as conn:
        articles = conn.execute("select source, body from articles").fetchall()
        chunk_rows = conn.execute(
            "select a.source, count(*), avg(c.token_count)::int, max(c.token_count)"
            " from chunks c join articles a on a.id = c.article_id group by a.source"
        ).fetchall()
        generation = conn.execute("select value from app_state where key = 'ingest_generation'").fetchone()[0]

    by_source: dict[str, list[int]] = {}
    for source, body in articles:
        by_source.setdefault(source, []).append(count_tokens(body))
    chunks = {r[0]: r[1:] for r in chunk_rows}

    print(f"{'source':14} {'articles':>8} {'tok p50':>8} {'tok p90':>8} {'tok max':>8} {'chunks':>7} {'avg chunk':>9}")
    for source, tokens in sorted(by_source.items()):
        n_chunks, avg_chunk, _ = chunks.get(source, (0, 0, 0))
        print(f"{source:14} {len(tokens):>8} {int(statistics.median(tokens)):>8} {pct(tokens, 0.9):>8} "
              f"{max(tokens):>8} {n_chunks:>7} {avg_chunk:>9}")
    everything = [t for tokens in by_source.values() for t in tokens]
    if everything:
        print(f"\nall: {len(everything)} articles, median {int(statistics.median(everything))} tokens, "
              f"{sum(c[0] for c in chunks.values())} chunks, ingest_generation={generation}")


if __name__ == "__main__":
    main()
