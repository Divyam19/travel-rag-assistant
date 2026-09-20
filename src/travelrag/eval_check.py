"""Test the evaluation layer against deliberately corrupted answers and against clean ones.

Usage: python -m travelrag.eval_check [--tier1-only] [--from RUN_LABEL]

--from picks which agent_check run to grade (default "agent"), i.e. logs/<label>_answers.jsonl.

Adversarial cases (data/adversarial_cases.json) take a real answer from the last agent_check run and
alter a fact, append an invented claim, or add a legal claim. Controls are every real answer
from that run, which must not be flagged. Both are graded against the context stored with the
answer, so the evaluator sees exactly what the model saw.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from .config import ROOT
from .db import connect
from .llm import flush_usage
from .evaluation import EvalResult, evaluate_answer
from .golden import load_golden
from .pricing import cost_usd
from .rag import Source

LABEL = "eval_check"


def tamper(case: dict, answer: str) -> str | None:
    """Corrupt a real answer as the case describes, or return None if the wording is not there.

    A case lists alternative phrasings because the model rewords answers on every regeneration
    ("4:3" becomes "4 to 3"); the first that appears is used. A case that cannot be applied is
    reported as skipped rather than aborting the whole run.
    """
    if "replace" in case:
        candidates = case["replace"] if isinstance(case["replace"][0], list) else [case["replace"]]
        for old, new in candidates:
            if old in answer:
                return answer.replace(old, new)
        return None
    return answer + case.get("append", "")


def main() -> None:
    use_judge = "--tier1-only" not in sys.argv
    source_label = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else "agent"
    golden = {g.id: g for g in load_golden()}
    answers_path = ROOT / "logs" / f"{source_label}_answers.jsonl"
    if not answers_path.exists():
        sys.exit(f"No answers at {answers_path}. Run: python -m travelrag.agent_check {source_label}")
    answers = {r["id"]: r for r in map(json.loads, answers_path.read_text().splitlines())}
    cases = json.loads((ROOT / "data" / "adversarial_cases.json").read_text())
    with connect(with_vectors=False) as conn:
        conn.execute("delete from llm_calls where run_label = %s", (LABEL,))

    def sources_for(qid: str) -> list[Source]:
        rows = answers[qid].get("context")
        if not rows:
            sys.exit(f"{qid} has no stored context. Re-run: python -m travelrag.agent_check")
        return [Source(r["n"], r["title"], r["url"], r["source"],
                       datetime.fromisoformat(r["published_at"]) if r["published_at"] else None,
                       r["similarity"], r["content"], r["source_type"], r["chunk_id"]) for r in rows]

    def grade(qid: str, answer: str) -> EvalResult:
        sources = sources_for(qid)
        return evaluate_answer(golden[qid].query, answer, sources, LABEL, qid, use_judge)

    controls = [(qid, row["answer"]) for qid, row in answers.items() if row["path"] in ("index", "web")]
    tampered = {c["id"]: tamper(c, answers[c["base"]]["answer"]) for c in cases}
    runnable = [c for c in cases if tampered[c["id"]] is not None]
    skipped = [c["id"] for c in cases if tampered[c["id"]] is None]
    with ThreadPoolExecutor(4) as pool:  # each grade is mostly network wait, so threads help
        case_results = list(pool.map(lambda c: grade(c["base"], tampered[c["id"]]), runnable))
        control_results = list(pool.map(lambda c: grade(*c), controls))

    print("== adversarial cases")
    caught = 0
    for case, result in zip(runnable, case_results):
        ok = result.verdict == "flagged" if case["expect"] == "flagged" else result.verdict != "pass"
        caught += ok
        print(f"{'caught ' if ok else 'MISSED '} {case['id']} {case['kind']:13} verdict={result.verdict:10} tier={result.tier} "
              f"min_sim={result.min_sim:.2f}  {result.problems[:2]}")
    print(f"caught {caught}/{len(runnable)}" + (f"   SKIPPED {len(skipped)}: {', '.join(skipped)} (base answer reworded; add a phrasing)" if skipped else ""))

    print("\n== controls: real answers that should not be flagged")
    verdicts: dict[str, int] = {}
    tiers = {1: 0, 2: 0}
    false_flags = []
    sims = []
    for (qid, _), result in zip(controls, control_results):
        verdicts[result.verdict] = verdicts.get(result.verdict, 0) + 1
        tiers[result.tier] += 1
        sims.append(result.min_sim)
        if result.verdict == "flagged":
            false_flags.append((qid, result.min_sim, result.problems[:2]))
    total = sum(verdicts.values())
    print(f"{total} answers: {verdicts} | reached tier 2: {tiers[2]}/{total} ({tiers[2] / total:.0%})")
    print(f"min sentence similarity across controls: lowest {min(sims):.2f}, median {sorted(sims)[len(sims) // 2]:.2f}")
    for qid, sim, problems in false_flags:
        print(f"  false flag {qid} min_sim={sim:.2f} {problems}")

    print("\n== pass-threshold sweep: share of answers sent past Tier 1 (to the judge)")
    print(f"{'threshold':>9} {'controls':>9} {'adversarial':>12}")
    for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
        def routed(r):
            return r.min_sim < t or r.facts_bad or r.legal_topic
        print(f"{t:>9.2f} {sum(map(routed, control_results)) / len(control_results):>9.0%} "
              f"{sum(map(routed, case_results)) / len(case_results):>12.0%}")

    flush_usage()
    with connect(with_vectors=False) as conn:
        rows = conn.execute("select model, purpose, sum(prompt_tokens), sum(completion_tokens), count(*) from llm_calls"
                            " where run_label = %s and purpose in ('embed_eval', 'eval_judge') group by 1, 2", (LABEL,)).fetchall()
    for model, purpose, p, c, n in rows:
        print(f"{purpose:11} calls={n:>3} prompt={int(p):>6} completion={int(c):>5} ~${cost_usd(model, int(p), int(c)):.4f}")


if __name__ == "__main__":
    main()
