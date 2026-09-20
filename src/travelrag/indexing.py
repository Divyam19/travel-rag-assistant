"""Turn an article into embedded chunks and store them."""

from dataclasses import dataclass
from datetime import datetime

import psycopg
from openai import OpenAI

from .chunking import chunk_text, count_tokens
from .config import get_settings

EMBED_BATCH = 100


@dataclass
class PreparedChunk:
    content: str
    tokens: int
    embedding: list[float]


def embed(texts: list[str]) -> list[list[float]]:
    s = get_settings()
    client = OpenAI(api_key=s.openai_api_key)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        response = client.embeddings.create(model=s.embedding_model, input=texts[i : i + EMBED_BATCH])
        vectors.extend(item.embedding for item in response.data)
    return vectors


def prepare_chunks(title: str, body: str, size: int | None = None, overlap: int | None = None) -> list[PreparedChunk]:
    """Chunk and embed. Each chunk is embedded with its article title prepended, so a passage
    about "Level 2 advisory" still carries the country name; the stored text stays title-free."""
    s = get_settings()
    parts = chunk_text(body, size or s.chunk_size_tokens, s.chunk_overlap_tokens if overlap is None else overlap)
    vectors = embed([f"{title}\n\n{p}" for p in parts])
    return [PreparedChunk(p, count_tokens(p), v) for p, v in zip(parts, vectors)]


def store_chunks(conn: psycopg.Connection, article_id: int, chunks: list[PreparedChunk]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            "insert into chunks (article_id, chunk_index, content, token_count, embedding)"
            " values (%s, %s, %s, %s, %s)",
            [(article_id, i, c.content, c.tokens, c.embedding) for i, c in enumerate(chunks)],
        )


def save_article(conn: psycopg.Connection, *, url: str, url_hash: str, content_hash: str, source: str,
                 title: str, body: str, published_at: datetime | None, chunks: list[PreparedChunk],
                 source_type: str = "feed", expires_at: datetime | None = None,
                 replace: bool = False) -> int | None:
    """Insert an article and its chunks atomically; returns the new id, or None if an identical
    article already exists. With replace=True the old version of this URL is deleted first.
    The connection must be in autocommit mode so conn.transaction() is a real transaction."""
    with conn.transaction():
        if replace:
            conn.execute("delete from articles where url_hash = %s", (url_hash,))
        row = conn.execute(
            "insert into articles (url, url_hash, content_hash, source, source_type, title, body, published_at,"
            " expires_at) values (%s, %s, %s, %s, %s, %s, %s, %s, %s) on conflict do nothing returning id",
            (url, url_hash, content_hash, source, source_type, title, body, published_at, expires_at),
        ).fetchone()
        if row is None:
            return None
        store_chunks(conn, row[0], chunks)
        return row[0]
