"""Live web fallback: credit-guarded Tavily search, our own page fetching, corpus write-back.

Credit rules, in the order they apply to one fallback:
  1. a cached result for the same query is reused (0 credits);
  2. in fixtures mode a recorded response is replayed (0 credits, never calls Tavily);
  3. a hard daily cap on calls, enforced atomically in Postgres, refuses further searches;
  4. otherwise exactly one basic-depth Tavily search (1 credit), whose response is recorded
     as a fixture so it can be replayed for free later.
Tavily only finds URLs; the top pages are then fetched with our own scraper at no credit cost.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlparse

import psycopg
from psycopg.types.json import Jsonb
from tavily import TavilyClient

from .config import ROOT, get_settings
from .db import connect
from .fetch import Fetcher, FetchError
from .indexing import prepare_chunks, save_article
from .ingest import content_hash, extract_article, normalize_url, sha

FIXTURE_DIR = ROOT / "data" / "fixtures" / "tavily"
MIN_SNIPPET_WORDS = 30
WEB_FETCH_DELAY_SECONDS = 0.5  # politeness delay for one-off page fetches, shorter than the bulk scraper's

# One worker: indexing is disk/network bound and must not compete with the live turn.
_indexer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="web-index")


class WebSearchUnavailable(Exception):
    """Fallback could not run (cap reached, fixture missing, Tavily error); message is safe to show."""


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str


def parse_results(raw: list[dict]) -> list[WebResult]:
    seen: set[str] = set()
    results = []
    for r in raw:
        url = r.get("url", "")
        if urlparse(url).scheme not in ("http", "https") or not (r.get("content") or "").strip():
            continue
        key = normalize_url(url)
        if key in seen:
            continue
        seen.add(key)
        results.append(WebResult((r.get("title") or "").strip() or url, url, r["content"].strip()))
    return results


def spend_credit(conn: psycopg.Connection, cap: int, day: date | None = None) -> bool:
    """Count one Tavily call against today's cap. Atomic, so concurrent requests cannot overshoot."""
    if cap <= 0:
        return False
    row = conn.execute(
        "insert into tavily_usage (day, calls) values (%s, 1)"
        " on conflict (day) do update set calls = tavily_usage.calls + 1 where tavily_usage.calls < %s"
        " returning calls",
        (day or datetime.now(timezone.utc).date(), cap),
    ).fetchone()
    return row is not None


def search(query: str, time_range: str | None = None) -> tuple[list[WebResult], str]:
    """Return (results, origin) where origin is 'cache', 'fixture' or 'live (N credit)'.
    time_range ('day', 'week', 'month') limits results to recent pages for time-sensitive questions."""
    s = get_settings()
    key = sha(" ".join(query.lower().split()) + (f"|{time_range}" if time_range else ""))
    fixture = FIXTURE_DIR / f"{key[:16]}.json"
    with connect(with_vectors=False) as conn:
        conn.autocommit = True
        cached = conn.execute(
            "select results from web_search_cache where query_hash = %s"
            " and created_at > now() - make_interval(hours => %s)",
            (key, s.web_result_cache_hours),
        ).fetchone()
        if cached:
            return parse_results(cached[0]), "cache"

        if s.tavily_use_fixtures:
            if not fixture.exists():
                raise WebSearchUnavailable("fixtures mode is on and no recorded response exists for this query")
            raw, origin = json.loads(fixture.read_text()), "fixture"
        else:
            if not spend_credit(conn, s.tavily_daily_call_cap):
                raise WebSearchUnavailable(f"daily Tavily cap of {s.tavily_daily_call_cap} calls reached")
            try:
                raw = TavilyClient(api_key=s.tavily_api_key).search(
                    query=query[:400], search_depth=s.tavily_search_depth,
                    max_results=s.tavily_max_results, include_usage=True, timeout=15,
                    **({"time_range": time_range} if time_range else {}),
                )
            except Exception as e:  # noqa: BLE001 - SDK raises several error types
                raise WebSearchUnavailable(f"Tavily error: {type(e).__name__}") from e
            FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
            fixture.write_text(json.dumps(raw, indent=1))
            origin = f"live ({(raw.get('usage') or {}).get('credits', '?')} credit)"

        conn.execute(
            "insert into web_search_cache (query_hash, query_text, results) values (%s, %s, %s)"
            " on conflict (query_hash) do update set results = excluded.results, created_at = now()",
            (key, query, Jsonb(raw["results"])),
        )
    return parse_results(raw["results"]), origin


def store_results(results: list[WebResult], ttl_days: int | None = None) -> int:
    """Write result pages into the corpus as web articles that expire. Returns how many were new.

    The top web_fetch_top_n pages are fetched with our own scraper; others (and any page that
    blocks us) fall back to Tavily's snippet. Web pages never bump ingest_generation, so a
    fallback does not invalidate cached answers. ttl_days overrides how long they are kept: pages
    fetched for time-sensitive questions should expire quickly so they are not served stale later.
    """
    s = get_settings()
    fetcher = Fetcher(delay=WEB_FETCH_DELAY_SECONDS)
    expires = datetime.now(timezone.utc) + timedelta(days=ttl_days or s.web_article_ttl_days)
    stored = 0
    with connect() as conn:
        conn.autocommit = True
        for i, result in enumerate(results):
            url = normalize_url(result.url)
            url_hash = sha(url)
            if conn.execute("select 1 from articles where url_hash = %s", (url_hash,)).fetchone():
                continue
            title, body, published = result.title, result.snippet, None
            if i < s.web_fetch_top_n:
                try:
                    page_title, body, published = extract_article(fetcher, url)
                    title = page_title or title
                except FetchError:
                    body = result.snippet
            if len(body.split()) < MIN_SNIPPET_WORDS:
                continue
            c_hash = content_hash(body)
            if conn.execute("select 1 from articles where content_hash = %s", (c_hash,)).fetchone():
                continue
            article_id = save_article(
                conn, url=url, url_hash=url_hash, content_hash=c_hash, source=urlparse(url).netloc,
                title=title, body=body, published_at=published, chunks=prepare_chunks(title, body),
                source_type="web", expires_at=expires,
            )
            stored += article_id is not None
    return stored


# ---------------------------------------------------------------------------
# Interactive path: fetch pages, answer from them immediately, index in the background.
# The old flow wrote every page to Postgres and then re-queried it, which measured at 9.4s of a
# 27.8s turn. We now answer from the text already in memory and index afterwards.
# ---------------------------------------------------------------------------

@dataclass
class FetchedPage:
    title: str
    url: str
    site: str
    text: str
    published_at: datetime | None
    score: float          # Tavily's own relevance score
    full_text: bool       # True when we fetched the page, False when it is only a search snippet


@lru_cache(maxsize=1)
def _interactive_fetcher() -> Fetcher:
    """Shared across turns so robots.txt is fetched once per host, not once per search."""
    return Fetcher(delay=WEB_FETCH_DELAY_SECONDS, timeout=get_settings().web_fetch_timeout_seconds)


def fetch_pages(results: list[WebResult], scores: list[float] | None = None) -> list[FetchedPage]:
    """Fetch the top results concurrently. Anything that blocks or times out degrades to its
    snippet rather than failing the turn."""
    s = get_settings()
    fetcher = _interactive_fetcher()
    scores = scores or [1.0 - 0.01 * i for i in range(len(results))]

    def one(index_result: tuple[int, WebResult]) -> FetchedPage:
        i, result = index_result
        url = normalize_url(result.url)
        site = urlparse(url).netloc
        if i < s.web_fetch_top_n:
            try:
                title, body, published = extract_article(fetcher, url, s.web_fetch_timeout_seconds)
                return FetchedPage(title or result.title, url, site, body, published, scores[i], True)
            except (FetchError, Exception):  # noqa: BLE001 - a bad page must not break the turn
                pass
        return FetchedPage(result.title, url, site, result.snippet, None, scores[i], False)

    if not results:
        return []
    with ThreadPoolExecutor(min(len(results), 4)) as pool:
        pages = list(pool.map(one, enumerate(results)))
    return [p for p in pages if len(p.text.split()) >= MIN_SNIPPET_WORDS]


def index_pages_later(pages: list[FetchedPage], ttl_days: int | None = None) -> None:
    """Chunk, embed and store the pages off the critical path, so the user is not kept waiting."""
    _indexer.submit(_index_pages, pages, ttl_days)


def _index_pages(pages: list[FetchedPage], ttl_days: int | None) -> None:
    s = get_settings()
    expires = datetime.now(timezone.utc) + timedelta(days=ttl_days or s.web_article_ttl_days)
    try:
        with connect() as conn:
            conn.autocommit = True
            for page in pages:
                url_hash, c_hash = sha(page.url), content_hash(page.text)
                if conn.execute("select 1 from articles where url_hash = %s or content_hash = %s",
                                (url_hash, c_hash)).fetchone():
                    continue
                save_article(conn, url=page.url, url_hash=url_hash, content_hash=c_hash, source=page.site,
                             title=page.title, body=page.text, published_at=page.published_at,
                             chunks=prepare_chunks(page.title, page.text), source_type="web",
                             expires_at=expires)
    except Exception:  # noqa: BLE001 - background work must never surface into a chat turn
        logging.getLogger(__name__).exception("background indexing of web pages failed")


def flush_indexing() -> None:
    """Block until queued background indexing finishes. For tests and scripts."""
    _indexer.submit(lambda: None).result()
