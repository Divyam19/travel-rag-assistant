"""OpenAI calls that record their token usage in llm_calls."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from openai import OpenAI

from .config import get_settings
from .db import connect


@lru_cache
def client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key, timeout=30.0, max_retries=4)


# Usage rows are written from one background thread so logging never delays an answer. The
# executor's single worker keeps them in order, and pending writes finish before the process exits.
_usage_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="usage-log")


def _write_usage(row: tuple) -> None:
    try:
        with connect() as conn:
            conn.execute(
                "insert into llm_calls (run_label, query_id, purpose, model, prompt_tokens, completion_tokens)"
                " values (%s, %s, %s, %s, %s, %s)", row)
    except Exception:  # noqa: BLE001 - losing a usage row must never break a chat turn
        logging.getLogger(__name__).exception("could not record LLM usage")


def record_usage(run_label: str, query_id: str | None, purpose: str, model: str,
                 prompt_tokens: int, completion_tokens: int = 0) -> None:
    _usage_writer.submit(_write_usage, (run_label, query_id, purpose, model, prompt_tokens, completion_tokens))


def flush_usage() -> None:
    """Block until every queued usage row is written. Call before reading llm_calls."""
    _usage_writer.submit(lambda: None).result()


def embed_query(text: str, run_label: str = "adhoc", query_id: str | None = None) -> list[float]:
    model = get_settings().embedding_model
    response = client().embeddings.create(model=model, input=text)
    record_usage(run_label, query_id, "embed_query", model, response.usage.total_tokens)
    return response.data[0].embedding


def embed_texts(texts: list[str], purpose: str, run_label: str = "adhoc", query_id: str | None = None) -> list[list[float]]:
    model = get_settings().embedding_model
    response = client().embeddings.create(model=model, input=texts)
    record_usage(run_label, query_id, purpose, model, response.usage.total_tokens)
    return [item.embedding for item in response.data]


def parse_json(text: str) -> dict:
    """Parse a model's JSON reply; anything unparseable or not an object becomes {}."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _options(model: str, json_mode: bool, temperature: float | None, effort: str | None) -> dict:
    """gpt-5* models take reasoning_effort and reject temperature; older models are the reverse."""
    extra: dict = {"response_format": {"type": "json_object"}} if json_mode else {}
    if model.startswith("gpt-5"):
        if effort:
            extra["reasoning_effort"] = effort
    elif temperature is not None:
        extra["temperature"] = temperature
    return extra


def chat(messages: list[dict], model: str, purpose: str, run_label: str = "adhoc",
         query_id: str | None = None, json_mode: bool = False, temperature: float | None = None,
         effort: str | None = None) -> tuple[str, int, int]:
    """Return (text, prompt_tokens, completion_tokens). json_mode needs the word JSON in the prompt."""
    response = client().chat.completions.create(
        model=model, messages=messages, **_options(model, json_mode, temperature, effort))
    usage = response.usage
    record_usage(run_label, query_id, purpose, model, usage.prompt_tokens, usage.completion_tokens)
    return response.choices[0].message.content or "", usage.prompt_tokens, usage.completion_tokens


def chat_stream(messages: list[dict], model: str, purpose: str, run_label: str = "adhoc",
                query_id: str | None = None, temperature: float | None = None,
                effort: str | None = None):
    """Yield answer text as it arrives, then a final ("__done__", text, prompt, completion) tuple."""
    stream = client().chat.completions.create(
        model=model, messages=messages, stream=True, stream_options={"include_usage": True},
        **_options(model, False, temperature, effort))
    parts: list[str] = []
    usage = None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and (piece := chunk.choices[0].delta.content):
            parts.append(piece)
            yield "token", piece
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    record_usage(run_label, query_id, purpose, model, prompt_tokens, completion_tokens)
    yield "__done__", ("".join(parts), prompt_tokens, completion_tokens)
