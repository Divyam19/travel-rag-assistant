"""Baseline retrieve-then-answer chain: no agent loop, no cache, no routing.

Deliberately naive. Every retrieved chunk goes into the prompt and the large model answers,
which gives the "before" numbers the efficiency pass is measured against.
"""

from dataclasses import dataclass
from datetime import datetime

from pgvector import Vector

from .config import get_settings
from .db import connect
from .llm import chat, embed_query

SYSTEM_PROMPT = (
    "You are a travel assistant. Answer using ONLY the numbered sources provided. "
    "Cite sources inline like [1] or [2][3]. If the sources do not contain the answer, "
    "say so plainly instead of guessing. Do not give legal or visa advice."
)


@dataclass
class Source:
    n: int
    title: str
    url: str
    source: str
    published_at: datetime | None
    similarity: float
    content: str
    source_type: str  # 'feed' (curated index) or 'web' (fallback page written back)
    chunk_id: int = 0

    def to_dict(self, include_text: bool = True) -> dict:
        """The one JSON-ready form of a source, used by the API, the run logs and the evaluator.

        Every field keeps its Python name. The API used to rename `source` to `site` and
        `source_type` to `origin`, and the run log carried a third hand-written copy, so adding a
        field meant editing four places and the copies drifted. include_text=False drops the
        chunk text and id, which the browser does not need.
        """
        record = {"n": self.n, "title": self.title, "url": self.url, "source": self.source,
                  "published_at": self.published_at.isoformat() if self.published_at else None,
                  "similarity": round(self.similarity, 3), "source_type": self.source_type}
        if include_text:
            record.update(content=self.content, chunk_id=self.chunk_id)
        return record

    @classmethod
    def from_dict(cls, record: dict) -> "Source":
        published = record.get("published_at")
        return cls(record["n"], record["title"], record["url"], record["source"],
                   datetime.fromisoformat(published) if published else None, record["similarity"],
                   record.get("content", ""), record["source_type"], record.get("chunk_id", 0))


@dataclass
class Answer:
    text: str
    sources: list[Source]
    prompt_tokens: int
    completion_tokens: int


def search_index(vector: list[float], k: int) -> list[Source]:
    with connect() as conn:
        rows = conn.execute(
            "select title, url, source, published_at, similarity, content, source_type, chunk_id"
            " from match_chunks(%s, %s)",
            (Vector(vector), k),
        ).fetchall()
    return [Source(i + 1, *row) for i, row in enumerate(rows)]


def retrieve(question: str, k: int, run_label: str, query_id: str | None) -> list[Source]:
    return search_index(embed_query(question, run_label, query_id), k)


def format_context(sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        date = s.published_at.date().isoformat() if s.published_at else "undated"
        blocks.append(f"[{s.n}] {s.title} ({s.source}, {date})\n{s.content}")
    return "\n\n".join(blocks)


def answer_question(question: str, run_label: str = "adhoc", query_id: str | None = None) -> Answer:
    s = get_settings()
    sources = retrieve(question, s.retrieval_top_k, run_label, query_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n{format_context(sources)}\n\nQuestion: {question}"},
    ]
    text, prompt_tokens, completion_tokens = chat(messages, s.large_model, "answer", run_label, query_id)
    return Answer(text, sources, prompt_tokens, completion_tokens)
