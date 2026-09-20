"""context.py chooses what goes in the prompt. Neighbour expansion alone took detail-query recall
from 1/8 to 8/8, so its selection rules are pinned here."""

import numpy as np
import pytest

from travelrag import context
from travelrag.context import (build_web_context, cap_per_article, drop_weak, expand_and_merge,
                               interleave)
from travelrag.rag import Source
from travelrag.web import FetchedPage


def src(n, url, sim, text="text", chunk_id=0, source_type="feed"):
    return Source(n, f"Title {url}", url, "site", None, sim, text, source_type, chunk_id)


def test_interleave_alternates_and_keeps_every_group_represented():
    assert interleave([["a1", "a2", "a3", "a4"], ["b1", "b2"]]) == ["a1", "b1", "a2", "b2", "a3", "a4"]
    assert interleave([[], ["x"]]) == ["x"]
    assert interleave([]) == []


def test_a_page_cap_no_longer_starves_the_second_part():
    """Regression: concatenating then truncating gave part one all six slots' worth of pages."""
    part_one = [f"one-{i}" for i in range(4)]
    part_two = [f"two-{i}" for i in range(4)]
    kept = interleave([part_one, part_two])[:6]
    assert sum(k.startswith("two") for k in kept) == 3


def test_drop_weak_removes_far_below_best_but_always_keeps_the_top_hit():
    hits = [src(1, "a", 0.80), src(2, "b", 0.60), src(3, "c", 0.40)]
    assert [h.url for h in drop_weak(hits, 0.65)] == ["a", "b"]  # 0.40 < 0.65 * 0.80
    assert [h.url for h in drop_weak(hits, 1.0)] == ["a"]         # top hit survives any floor
    assert drop_weak([], 0.65) == []


def test_cap_per_article_keeps_rank_order_and_respects_the_limit():
    hits = [src(1, "a", .9), src(2, "a", .8), src(3, "b", .7), src(4, "a", .6), src(5, "c", .5)]
    assert [h.n for h in cap_per_article(hits, per_article=1, limit=10)] == [1, 3, 5]
    assert [h.n for h in cap_per_article(hits, per_article=2, limit=3)] == [1, 2, 3]


def test_expansion_merges_neighbouring_chunks_of_one_article_into_one_citation(monkeypatch):
    monkeypatch.setattr(context, "_neighbour_text", lambda ids, k: {
        10: [(1, "before"), (2, "hit"), (3, "after")],
        11: [(3, "after"), (4, "later")],  # overlaps the first hit's window
    })
    merged = expand_and_merge([src(1, "a", .9, chunk_id=10), src(2, "a", .8, chunk_id=11)], neighbours=1)
    assert len(merged) == 1 and merged[0].n == 1
    # ordered by position in the article, and the shared chunk appears once
    assert merged[0].content.split("\n\n") == ["before", "hit", "after", "later"]


def test_expansion_falls_back_to_the_hit_itself_when_neighbours_are_unavailable(monkeypatch):
    monkeypatch.setattr(context, "_neighbour_text", lambda ids, k: {})
    merged = expand_and_merge([src(1, "a", .9, text="only me", chunk_id=5)], neighbours=1)
    assert merged[0].content == "only me"


def test_distinct_articles_are_numbered_in_rank_order(monkeypatch):
    monkeypatch.setattr(context, "_neighbour_text", lambda ids, k: {})
    merged = expand_and_merge([src(1, "a", .9, chunk_id=1), src(2, "b", .8, chunk_id=2)], 1)
    assert [(m.n, m.url) for m in merged] == [(1, "a"), (2, "b")]


def page(url, text, score=1.0):
    return FetchedPage(f"T {url}", url, "site", text, None, score, True)


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Make chunk i of a page score by a fixed table, so window selection is predictable."""
    table = {"needle": 0.9, "filler": 0.1}

    def embed(texts, purpose, run_label="adhoc", query_id=None):
        return [[1.0 if "needle" in t else 0.0, 0.0] for t in texts]

    monkeypatch.setattr("travelrag.llm.embed_texts", embed)


def test_short_pages_go_in_whole_and_long_ones_contribute_their_best_window(fake_embeddings, monkeypatch):
    monkeypatch.setattr(context.get_settings(), "web_page_max_tokens", 300)  # make the long page exceed it
    long_text = "\n\n".join(["filler words " * 60] * 6 + ["the needle is here " * 30] + ["filler words " * 60] * 6)
    pages = [page("https://a.test/long", long_text), page("https://b.test/short", "a short page " * 5)]
    sources = build_web_context([1.0, 0.0], pages)
    long_source = next(s for s in sources if s.url.endswith("/long"))
    assert "needle" in long_source.content
    assert len(long_source.content) < len(long_text)  # a window, not the whole page
    assert next(s for s in sources if s.url.endswith("/short")).content == "a short page " * 5


def test_the_web_context_respects_its_token_budget_but_never_returns_nothing(fake_embeddings, monkeypatch):
    monkeypatch.setattr(context.get_settings(), "web_context_tokens", 500)
    pages = [page(f"https://p{i}.test", "word " * 300, score=1.0 - i / 10) for i in range(5)]
    sources = build_web_context([1.0, 0.0], pages)
    assert 1 <= len(sources) < 5
    assert [s.url for s in sources] == sorted((s.url for s in sources))  # rank order preserved
    assert [s.n for s in sources] == list(range(1, len(sources) + 1))
