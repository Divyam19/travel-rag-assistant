"""Check which sources yield usable article text. Usage: python -m travelrag.probe [articles_per_source]

Makes no LLM or Tavily calls. Built on the same Fetcher as the ingester, so it obeys robots.txt and
throttling identically, and it applies the ingester's own acceptance threshold, so "usable" here
means "the ingester would store it".
"""

import json
import sys

import feedparser
import trafilatura

from .fetch import Fetcher, FetchError
from .ingest import MIN_WORDS_FETCHED, parse_date
from .sources import SOURCES


def probe_article(fetcher: Fetcher, url: str) -> tuple[int, bool]:
    """(words extracted, whether a publication date was found). Raises FetchError if unreachable."""
    page = fetcher.get(url)
    doc = json.loads(trafilatura.extract(page.text, output_format="json", with_metadata=True) or "{}")
    return len((doc.get("text") or "").split()), parse_date(doc.get("date")) is not None


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    fetcher = Fetcher()
    print(f"{'source':16} {'entries':>7} {'feedtext':>8} {'fetched':>7} {'usable':>6} {'dated':>5} {'avgwords':>8}  notes")
    for src in SOURCES:
        try:
            feed = feedparser.parse(fetcher.get(src.feed_url).content)
        except FetchError as e:
            print(f"{src.name:16} {'-':>7} {'-':>8} {'-':>7} {'-':>6} {'-':>5} {'-':>8}  feed: {e}")
            continue
        entries = feed.entries[:n]
        feed_words = [len((e.get("content", [{}])[0].get("value") or e.get("summary", "")).split()) for e in entries]
        fetched = usable = dated = 0
        words: list[int] = []
        notes: set[str] = set() if feed.entries else {"not a parseable feed"}
        for entry in entries:
            try:
                count, has_date = probe_article(fetcher, entry.get("link", ""))
            except FetchError as e:
                notes.add(str(e))
                continue
            fetched += 1
            dated += has_date
            if count >= MIN_WORDS_FETCHED:
                usable += 1
                words.append(count)
        print(f"{src.name:16} {len(feed.entries):>7} {sum(feed_words) // max(len(feed_words), 1):>8} "
              f"{fetched:>4}/{len(entries):<2} {usable:>3}/{len(entries):<2} {dated:>5} "
              f"{sum(words) // max(len(words), 1):>8}  {', '.join(sorted(notes))}")
    print(f"\nusable = extracted body of at least {MIN_WORDS_FETCHED} words (the ingester's threshold); "
          "feedtext = average words already in the feed entry")


if __name__ == "__main__":
    main()
