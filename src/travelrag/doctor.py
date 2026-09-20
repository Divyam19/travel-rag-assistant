"""Verify config, database, and OpenAI connectivity. Usage: python -m travelrag.doctor

Deliberately does NOT call Tavily, to avoid spending credits on a health check.
"""

import sys

from openai import OpenAI

from .config import get_settings
from .db import connect

EXPECTED_TABLES = {
    "articles", "chunks", "app_state", "ingest_runs",
    "answer_cache", "web_search_cache", "tavily_usage",
}


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'ok' if ok else 'FAIL'}] {label}{': ' + detail if detail else ''}")
    return ok


def main() -> int:
    s = get_settings()  # exits with a readable message if .env is wrong
    results = [check("config loaded", True)]
    results.append(check("tavily key format", s.tavily_api_key.startswith("tvly-"),
                         "" if s.tavily_api_key.startswith("tvly-") else "expected a key starting with 'tvly-'"))

    try:
        with connect(with_vectors=False) as conn:
            results.append(check("database reachable", True))
            ext = conn.execute("select 1 from pg_extension where extname = 'vector'").fetchone()
            results.append(check("pgvector extension", ext is not None,
                                 "" if ext else "run: python -m travelrag.migrate"))
            rows = conn.execute(
                "select table_name from information_schema.tables where table_schema = 'public'"
            ).fetchall()
            missing = EXPECTED_TABLES - {r[0] for r in rows}
            results.append(check("schema tables", not missing,
                                 f"missing {sorted(missing)}; run: python -m travelrag.migrate" if missing else ""))
    except Exception as e:  # noqa: BLE001 - report any connection problem readably
        results.append(check("database reachable", False, str(e).splitlines()[0]))

    try:
        emb = OpenAI(api_key=s.openai_api_key).embeddings.create(model=s.embedding_model, input="health check")
        dim = len(emb.data[0].embedding)
        results.append(check("openai embeddings", dim == s.embedding_dim,
                             f"dimension {dim}" + ("" if dim == s.embedding_dim else f", expected {s.embedding_dim}")))
    except Exception as e:  # noqa: BLE001
        results.append(check("openai embeddings", False, str(e).splitlines()[0]))

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
