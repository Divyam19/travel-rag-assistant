"""Discover, fetch, extract, chunk, embed and store articles.

Usage: python -m travelrag.ingest [--source NAME] [--limit N]

Safe to run repeatedly (cron): known URLs are skipped before any fetch, and identical
content is never embedded twice. One failing article or source never stops the run.
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import lxml.html
import psycopg
import trafilatura

from .config import get_settings
from .db import connect
from .fetch import Fetcher, FetchError
from .indexing import prepare_chunks, save_article
from .sources import SOURCES, Source

MIN_WORDS_FETCHED = 120  # shorter extractions are usually paywall stubs or navigation pages
MIN_WORDS_FEED = 30
TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "mc_", "ref")


class Stats(dict):
    def __init__(self) -> None:
        super().__init__(discovered=0, fetched=0, extracted=0, inserted=0, skipped_dupe=0, failed=0, chunks=0)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith(TRACKING_PARAMS)]
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path.rstrip("/") or "/", urlencode(query), ""))


def content_hash(text: str) -> str:
    return sha(" ".join(text.lower().split()))


def feed_entry_text(entry: feedparser.FeedParserDict) -> str:
    raw = (entry.get("content") or [{}])[0].get("value") or entry.get("summary", "")
    if not raw.strip():
        return ""
    raw = re.sub(r"(?i)</(p|div|li|h\d)>|<br\s*/?>", "\n\n", raw)
    text = lxml.html.fromstring(f"<div>{raw}</div>").text_content().replace("\xa0", " ")
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", text)).strip()


def entry_date(entry: feedparser.FeedParserDict) -> datetime | None:
    stamp = entry.get("published_parsed") or entry.get("updated_parsed")
    return datetime(*stamp[:6], tzinfo=timezone.utc) if stamp else None


def parse_date(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc) if value else None
    except ValueError:
        return None


def extract_article(fetcher: Fetcher, url: str, timeout: float | None = None) -> tuple[str, str, datetime | None]:
    """Return (title, body, published_at) for a page, or raise FetchError."""
    page = fetcher.get(url, timeout)
    out = trafilatura.extract(page.text, output_format="json", with_metadata=True)
    doc = json.loads(out) if out else {}
    body = (doc.get("text") or "").strip()
    if len(body.split()) < MIN_WORDS_FETCHED:
        raise FetchError("extracted text too short")
    return (doc.get("title") or "").strip(), body, parse_date(doc.get("date"))


def ingest_source(conn: psycopg.Connection, fetcher: Fetcher, src: Source, limit: int | None) -> tuple[Stats, list[str]]:
    s = get_settings()
    stats, failures = Stats(), []
    feed = feedparser.parse(fetcher.get(src.feed_url).content)
    entries = feed.entries[: limit or src.max_articles or s.scraper_max_articles_per_source]
    stats["discovered"] = len(entries)

    known_urls = dict(conn.execute("select url_hash, content_hash from articles where source = %s", (src.name,)))
    known_content = {r[0] for r in conn.execute("select content_hash from articles")}

    for entry in entries:
        link = entry.get("link", "")
        try:
            if not link:
                raise FetchError("entry without link")
            url = normalize_url(link)
            url_hash = sha(url)
            title, published = entry.get("title", "").strip(), entry_date(entry)

            if url_hash in known_urls and not src.use_feed_text:
                stats["skipped_dupe"] += 1  # known article: skip before spending a request
                continue
            if src.use_feed_text:
                body = feed_entry_text(entry)
                if len(body.split()) < MIN_WORDS_FEED:
                    raise FetchError("feed text too short")
            else:
                stats["fetched"] += 1
                page_title, body, page_date = extract_article(fetcher, url)
                title, published = title or page_title, published or page_date
            stats["extracted"] += 1

            c_hash = content_hash(body)
            if c_hash == known_urls.get(url_hash) or c_hash in known_content:
                stats["skipped_dupe"] += 1
                continue

            chunks = prepare_chunks(title, body)  # embedding happens outside the DB transaction
            article_id = save_article(
                conn, url=url, url_hash=url_hash, content_hash=c_hash, source=src.name, title=title,
                body=body, published_at=published, chunks=chunks, replace=url_hash in known_urls,
            )
            if article_id is None:
                stats["skipped_dupe"] += 1
                continue
            known_urls[url_hash] = c_hash
            known_content.add(c_hash)
            stats["inserted"] += 1
            stats["chunks"] += len(chunks)
        except FetchError as e:
            stats["failed"] += 1
            failures.append(f"{link or '?'}: {e}")
        except Exception as e:  # noqa: BLE001 - one bad article must not stop the run
            stats["failed"] += 1
            failures.append(f"{link or '?'}: {type(e).__name__}: {e}")
    return stats, failures


def run_source(fetcher: Fetcher, src: Source, limit: int | None) -> bool:
    with connect() as conn:
        conn.autocommit = True  # so each article's conn.transaction() is a real commit, not a savepoint
        run_id = conn.execute("insert into ingest_runs (source) values (%s) returning id", (src.name,)).fetchone()[0]
        stats, failures, error = Stats(), [], None
        try:
            stats, failures = ingest_source(conn, fetcher, src, limit)
        except Exception as e:  # noqa: BLE001 - e.g. the feed itself is unreachable
            error = f"{type(e).__name__}: {e}"
        if failures and not error:
            error = "; ".join(failures[:5])
        conn.execute(
            "update ingest_runs set finished_at = now(), discovered = %s, fetched = %s, extracted = %s,"
            " inserted = %s, skipped_dupe = %s, failed = %s, error = %s where id = %s",
            (stats["discovered"], stats["fetched"], stats["extracted"], stats["inserted"],
             stats["skipped_dupe"], stats["failed"], error, run_id),
        )
    summary = " ".join(f"{k}={v}" for k, v in stats.items())
    print(f"{src.name:14} {summary}" + (f"  !! {error}" if error else ""), flush=True)
    for f in failures[:5]:
        print(f"    failed: {f}", flush=True)
    return error is None or stats["inserted"] > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="only ingest this source name")
    parser.add_argument("--limit", type=int, help="max articles per source")
    args = parser.parse_args()
    sources = [s for s in SOURCES if not args.source or s.name == args.source]
    if not sources:
        sys.exit(f"Unknown source {args.source!r}. Known: {', '.join(s.name for s in SOURCES)}")
    fetcher = Fetcher()
    results = [run_source(fetcher, src, args.limit) for src in sources]
    with connect(with_vectors=False) as conn:  # web-fallback pages past their TTL
        purged = conn.execute("delete from articles where expires_at is not null and expires_at < now()").rowcount
    if purged:
        print(f"purged {purged} expired web articles")
    return 0 if any(results) else 1


if __name__ == "__main__":
    sys.exit(main())
