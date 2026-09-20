"""Run every golden query through the agent and check it took the expected path.

Usage: python -m travelrag.agent_check [run_label]
Set TAVILY_USE_FIXTURES=true to replay recorded Tavily responses for free. Re-running with the
same label replaces that label's earlier rows in llm_calls.

Pass criteria:
  answerable/detail  path 'index' and the top cited source is the expected one
  out_of_index       'web' or 'not_found', or 'index' when it cites an earlier web fallback page
  off-topic (o09/o10) 'off_topic'
  ambiguous          'clarify'
"""

import json
import sys
from collections import defaultdict

from .agent import run_agent
from .config import ROOT
from .db import connect
from .llm import flush_usage
from .golden import GoldenQuery, load_golden
from .pricing import cost_usd

OFF_TOPIC_IDS = {"o09", "o10"}


def passed(g: GoldenQuery, result) -> bool:
    if g.category in ("answerable", "detail"):
        return result.path == "index" and any(g.is_relevant(s.title, s.content) for s in result.sources[:1])
    if g.category == "ambiguous":
        return result.path == "clarify"
    if g.id in OFF_TOPIC_IDS:
        return result.path == "off_topic"
    if result.path in ("web", "not_found"):
        return True
    return result.path == "index" and any(s.source_type == "web" for s in result.sources)  # answered from an earlier fallback page


def usage_summary(label: str) -> tuple[int, int, float]:
    """(queries, chat tokens, estimated dollars) for one run label, embeddings excluded from tokens."""
    flush_usage()
    with connect(with_vectors=False) as conn:
        rows = conn.execute(
            "select model, purpose, sum(prompt_tokens), sum(completion_tokens) from llm_calls"
            " where run_label = %s group by model, purpose", (label,)).fetchall()
        queries = conn.execute("select count(distinct query_id) from llm_calls where run_label = %s", (label,)).fetchone()[0]
    tokens = sum(int(p) + int(c) for _, purpose, p, c in rows if not purpose.startswith("embed"))
    priced = [cost_usd(m, int(p), int(c)) for m, _, p, c in rows]
    dollars = sum(d for d in priced if d is not None)
    unpriced = sorted({m for (m, _, _, _), d in zip(rows, priced) if d is None})
    return queries, tokens, dollars, unpriced


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "agent"
    golden = load_golden()
    with connect(with_vectors=False) as conn:
        conn.execute("delete from llm_calls where run_label = %s", (label,))

    outcomes: dict[str, list[bool]] = defaultdict(list)
    paths: dict[str, int] = defaultdict(int)
    live_calls = 0
    out_path = ROOT / "logs" / f"{label}_answers.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as out:
        for g in golden:
            r = run_agent(g.query, run_label=label, query_id=g.id)
            ok = passed(g, r)
            outcomes[g.category].append(ok)
            paths[r.path] += 1
            live_calls += r.web_calls
            # Store the context verbatim: eval_check must grade an answer against the sources it
            # was actually written from, not against a fresh retrieval that may differ.
            out.write(json.dumps({
                "id": g.id, "query": g.query, "path": r.path, "answer": r.answer, "trace": r.trace,
                "sources": [s.title for s in r.sources],
                "context": [{"n": s.n, "title": s.title, "url": s.url, "source": s.source,
                             "published_at": s.published_at.isoformat() if s.published_at else None,
                             "similarity": s.similarity, "content": s.content,
                             "source_type": s.source_type, "chunk_id": s.chunk_id} for s in r.sources],
            }) + "\n")
            print(f"{'ok  ' if ok else 'FAIL'} {g.id} [{r.path:9}] tokens={r.prompt_tokens + r.completion_tokens:>5}  {g.query}", flush=True)

    print("\n== path checks")
    for category, results in outcomes.items():
        print(f"{category:14} {sum(results)}/{len(results)}")
    print(f"paths taken: {dict(paths)} | live Tavily calls this run: {live_calls}")

    print("\n== chat tokens per query (analysis, routing and answer calls; embeddings excluded)")
    for name in ("baseline", label):
        queries, tokens, dollars, unpriced = usage_summary(name)
        if queries:
            note = f"  (no verified price for {', '.join(unpriced)})" if unpriced else ""
            print(f"{name:10} {queries} queries  {tokens:>8} tokens  {tokens // queries:>6}/query  ~${dollars:.4f}{note}")


if __name__ == "__main__":
    main()
