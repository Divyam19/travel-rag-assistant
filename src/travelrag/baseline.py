"""Run every golden query through the baseline chain and report measured token usage.

Usage: python -m travelrag.baseline [run_label]
Re-running with the same label replaces that label's earlier rows in llm_calls.
"""

import json
import sys

from .config import ROOT, get_settings
from .db import connect
from .llm import flush_usage
from .golden import load_golden
from .pricing import cost_usd
from .rag import answer_question


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    golden = load_golden()
    with connect(with_vectors=False) as conn:
        conn.execute("delete from llm_calls where run_label = %s", (label,))

    out_path = ROOT / "logs" / f"{label}_answers.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as out:
        for g in golden:
            result = answer_question(g.query, run_label=label, query_id=g.id)
            out.write(json.dumps({
                "id": g.id, "category": g.category, "query": g.query, "answer": result.text,
                "sources": [{"title": s.title, "similarity": round(s.similarity, 3)} for s in result.sources],
            }) + "\n")
            print(f"{g.id} {g.category:12} prompt={result.prompt_tokens:>5} completion={result.completion_tokens:>4}  "
                  f"{result.text[:90]!r}", flush=True)

    flush_usage()
    with connect(with_vectors=False) as conn:
        rows = conn.execute(
            "select purpose, model, count(*), sum(prompt_tokens), sum(completion_tokens)"
            " from llm_calls where run_label = %s group by purpose, model order by purpose", (label,),
        ).fetchall()
    print(f"\n== {label}: {len(golden)} queries")
    total = 0.0
    for purpose, model, calls, prompt, completion in rows:
        cost = cost_usd(model, prompt, completion)
        total += cost or 0
        print(f"{purpose:12} {model:24} calls={calls:>3} prompt={prompt:>7} completion={completion:>6}"
              f"  ~${cost:.4f}" if cost is not None else f"{purpose:12} {model:24} (no price known)")
    print(f"per query: ~{total / len(golden) * 1000:.3f} milli-dollars; total ~${total:.4f}; "
          f"answers saved to {out_path.relative_to(ROOT)}")
    print(f"models: large={get_settings().large_model} small={get_settings().small_model}")


if __name__ == "__main__":
    main()
