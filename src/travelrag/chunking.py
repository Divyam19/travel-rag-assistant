"""Token-aware chunking that prefers paragraph, then sentence, boundaries."""

import re
from functools import lru_cache

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")  # tokenizer used by text-embedding-3-*
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=4096)
def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _split_long(paragraph: str, size: int) -> list[str]:
    """Break an oversized paragraph into pieces of at most `size` tokens."""
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(paragraph):
        if count_tokens(sentence) <= size:
            pieces.append(sentence)
            continue
        tokens = _enc.encode(sentence)  # a single huge sentence: hard split on tokens
        pieces.extend(_enc.decode(tokens[i : i + size]) for i in range(0, len(tokens), size))
    return pieces


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Pack paragraphs into chunks of about `size` tokens, repeating up to `overlap` tokens
    of trailing context at the start of the next chunk."""
    pieces: list[str] = []
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not paragraph:
            continue
        pieces.extend([paragraph] if count_tokens(paragraph) <= size else _split_long(paragraph, size))

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for piece in pieces:
        tokens = count_tokens(piece)
        if current and current_tokens + tokens > size:
            chunks.append("\n\n".join(current))
            carried: list[str] = []
            carried_tokens = 0
            for previous in reversed(current):
                previous_tokens = count_tokens(previous)
                if carried_tokens + previous_tokens > overlap:
                    break
                carried.insert(0, previous)
                carried_tokens += previous_tokens
            current, current_tokens = carried, carried_tokens
        current.append(piece)
        current_tokens += tokens
    if current:
        chunks.append("\n\n".join(current))
    return chunks
