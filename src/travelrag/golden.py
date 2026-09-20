"""Golden query set and the retrieval metrics computed over it."""

import json
import statistics
from dataclasses import dataclass

from .config import ROOT

GOLDEN_PATH = ROOT / "data" / "golden_queries.json"


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    category: str  # answerable | detail | out_of_index | ambiguous
    query: str
    expect_titles: list[str]
    expect_text: str = ""  # detail queries: the retrieved chunk itself must contain this

    def is_relevant(self, title: str, content: str) -> bool:
        title_ok = any(t.lower() in title.lower() for t in self.expect_titles)
        text_ok = not self.expect_text or self.expect_text.lower() in content.lower()
        return title_ok and text_ok


def load_golden() -> list[GoldenQuery]:
    return [
        GoldenQuery(g["id"], g["category"], g["query"], g["expect_titles"], g.get("expect_text", ""))
        for g in json.loads(GOLDEN_PATH.read_text())
    ]


# A ranked result is (title, similarity, chunk content), best first.
Ranked = list[tuple[str, float, str]]


def first_relevant_rank(q: GoldenQuery, ranked: Ranked) -> int | None:
    return next((i + 1 for i, (title, _, content) in enumerate(ranked) if q.is_relevant(title, content)), None)


def best_threshold(answerable_top1: list[float], other_top1: list[float]) -> tuple[float, float]:
    """Cut-off on top-1 similarity that best separates answerable from out-of-index queries.
    Returns (threshold, accuracy). Confident means top-1 >= threshold."""
    candidates = sorted(set(answerable_top1 + other_top1))
    total = len(answerable_top1) + len(other_top1)
    scored = [
        (t, (sum(s >= t for s in answerable_top1) + sum(s < t for s in other_top1)) / total)
        for t in candidates
    ]
    return max(scored, key=lambda pair: (pair[1], -pair[0]))


def rank_metrics(golden: list[GoldenQuery], results: dict[str, Ranked], category: str, prefix: str) -> dict:
    ranks = [first_relevant_rank(g, results[g.id]) for g in golden if g.category == category]
    return {
        f"{prefix}recall@1": sum(r == 1 for r in ranks) / len(ranks),
        f"{prefix}recall@4": sum(r is not None and r <= 4 for r in ranks) / len(ranks),
        f"{prefix}recall@8": sum(r is not None for r in ranks) / len(ranks),
        f"{prefix}mrr": sum(1 / r for r in ranks if r) / len(ranks),
    }


def summarize(golden: list[GoldenQuery], results: dict[str, Ranked]) -> dict:
    a_top1 = [results[g.id][0][1] for g in golden if g.category == "answerable"]
    o_top1 = [results[g.id][0][1] for g in golden if g.category == "out_of_index"]
    threshold, accuracy = best_threshold(a_top1, o_top1)
    return {
        **rank_metrics(golden, results, "answerable", ""),
        **rank_metrics(golden, results, "detail", "detail_"),
        "answerable_top1_median": statistics.median(a_top1),
        "answerable_top1_min": min(a_top1),
        "out_of_index_top1_median": statistics.median(o_top1),
        "out_of_index_top1_max": max(o_top1),
        "threshold": threshold,
        "threshold_accuracy": accuracy,
    }
