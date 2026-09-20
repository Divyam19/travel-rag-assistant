"""Choose what actually goes in the prompt.

Raw vector search returns the top-k chunks, which in practice are often several chunks of one
page (a Kanpur flights question returned four chunks of the same trip.com page). Three steps fix
that: cap how many chunks any one article may contribute, pull in the chunks either side of each
hit so a table or list is not cut in half, and merge contiguous chunks into one citation.
"""

from dataclasses import replace

from .config import get_settings
from .db import connect
from .rag import Source

JOIN = "\n\n"


def drop_weak(sources: list[Source], floor_ratio: float) -> list[Source]:
    """Drop hits far below the best one. A 0.37 match beside a 0.72 match is noise, and expanding
    its neighbours would spend hundreds of tokens on an unrelated page. The top hit always stays."""
    if not sources:
        return []
    floor = sources[0].similarity * floor_ratio
    return [sources[0]] + [s for s in sources[1:] if s.similarity >= floor]


def cap_per_article(sources: list[Source], per_article: int, limit: int) -> list[Source]:
    """Keep rank order, but let no article contribute more than `per_article` chunks."""
    seen: dict[str, int] = {}
    kept: list[Source] = []
    for source in sources:
        if seen.get(source.url, 0) >= per_article:
            continue
        seen[source.url] = seen.get(source.url, 0) + 1
        kept.append(source)
        if len(kept) >= limit:
            break
    return kept


def _neighbour_text(chunk_ids: list[int], neighbours: int) -> dict[int, list[tuple[int, str]]]:
    """For each selected chunk, its article's chunks within `neighbours` positions, as (index, text)."""
    if not chunk_ids or neighbours <= 0:
        return {}
    with connect() as conn:
        rows = conn.execute(
            "select target.id, near.chunk_index, near.content"
            " from chunks target join chunks near on near.article_id = target.article_id"
            "  and abs(near.chunk_index - target.chunk_index) <= %s"
            " where target.id = any(%s) order by target.id, near.chunk_index",
            (neighbours, chunk_ids),
        ).fetchall()
    out: dict[int, list[tuple[int, str]]] = {}
    for chunk_id, index, content in rows:
        out.setdefault(chunk_id, []).append((index, content))
    return out


def expand_and_merge(sources: list[Source], neighbours: int) -> list[Source]:
    """Widen each hit to its neighbouring chunks, then merge same-article runs into one source."""
    expanded = _neighbour_text([s.chunk_id for s in sources if s.chunk_id], neighbours)

    by_article: dict[str, dict[int, str]] = {}
    best: dict[str, Source] = {}
    order: list[str] = []
    for source in sources:
        if source.url not in by_article:
            by_article[source.url] = {}
            best[source.url] = source  # first occurrence is the highest-scoring one
            order.append(source.url)
        pieces = expanded.get(source.chunk_id) or [(0, source.content)]
        by_article[source.url].update(dict(pieces))

    merged = []
    for n, url in enumerate(order, start=1):
        text = JOIN.join(content for _, content in sorted(by_article[url].items()))
        merged.append(replace(best[url], n=n, content=text))
    return merged


def select_context(sources: list[Source], limit: int | None = None) -> list[Source]:
    """Full selection: cap, then expand and merge. `limit` counts chunks before merging."""
    s = get_settings()
    strong = drop_weak(sources, s.context_relative_floor)
    capped = cap_per_article(strong, s.max_chunks_per_article, limit or s.context_top_n)
    return expand_and_merge(capped, s.neighbour_chunks)


# ---------------------------------------------------------------------------
# Web answers. These are built from pages held in memory, not from the index, so there is no
# chunk_id to expand from. Short pages go in whole; long ones contribute their most relevant
# window, chosen with a single batched embedding call across every page.
# ---------------------------------------------------------------------------

def _best_window(chunks: list[str], scores: list[float], budget: int) -> str:
    """Grow outwards from the best-scoring chunk until the token budget is spent."""
    from .chunking import count_tokens

    best = max(range(len(chunks)), key=lambda i: scores[i])
    low = high = best
    used = count_tokens(chunks[best])
    while used < budget and (low > 0 or high < len(chunks) - 1):
        before = scores[low - 1] if low > 0 else float("-inf")
        after = scores[high + 1] if high < len(chunks) - 1 else float("-inf")
        if before >= after and low > 0:
            low -= 1
            used += count_tokens(chunks[low])
        elif high < len(chunks) - 1:
            high += 1
            used += count_tokens(chunks[high])
        else:
            break
    return JOIN.join(chunks[low : high + 1])


def build_web_context(query_vector: list[float], pages: list, run_label: str = "adhoc",
                      query_id: str | None = None) -> list["Source"]:
    """Turn fetched pages into numbered sources, newest and most relevant first."""
    import numpy as np

    from .chunking import chunk_text, count_tokens
    from .llm import embed_texts

    s = get_settings()
    long_pages = [p for p in pages if count_tokens(p.text) > s.web_page_max_tokens]
    windows: dict[str, str] = {}
    if long_pages:
        per_page = {p.url: chunk_text(p.text, s.chunk_size_tokens, s.chunk_overlap_tokens) for p in long_pages}
        flat = [(url, c) for url, chunks in per_page.items() for c in chunks]
        vectors = np.array(embed_texts([c for _, c in flat], "embed_web", run_label, query_id), dtype=np.float32)
        similarity = vectors @ np.array(query_vector, dtype=np.float32)
        by_url: dict[str, list[float]] = {}
        for (url, _), score in zip(flat, similarity):
            by_url.setdefault(url, []).append(float(score))
        for url, chunks in per_page.items():
            windows[url] = _best_window(chunks, by_url[url], s.web_page_max_tokens)

    sources: list[Source] = []
    spent = 0
    for page in sorted(pages, key=lambda p: -p.score):
        text = windows.get(page.url, page.text)
        cost = count_tokens(text)
        if spent + cost > s.web_context_tokens and sources:
            break
        spent += cost
        sources.append(Source(len(sources) + 1, page.title, page.url, page.site, page.published_at,
                              page.score, text, "web", 0))
    return sources
