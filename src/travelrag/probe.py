"""Check which sources yield usable article text. Usage: python -m travelrag.probe [articles_per_source]

Makes no LLM or Tavily calls. Respects robots.txt and throttles requests.
"""

import json
import sys
import time
from urllib import robotparser
from urllib.parse import urlparse

import feedparser
import httpx
import trafilatura

from .config import get_settings
from .sources import SOURCES

MIN_WORDS = 150  # below this we treat an extraction as a teaser or a failure


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    s = get_settings()
    client = httpx.Client(
        headers={"User-Agent": s.scraper_user_agent}, timeout=s.scraper_timeout_seconds, follow_redirects=True
    )
    robots: dict[str, robotparser.RobotFileParser | None] = {}

    def allowed(url: str) -> bool:
        host = urlparse(url).netloc
        if host not in robots:
            rp = robotparser.RobotFileParser()
            try:
                r = client.get(f"https://{host}/robots.txt")
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except httpx.HTTPError:
                rp.parse([])
            robots[host] = rp
            time.sleep(s.scraper_request_delay_seconds)
        return robots[host].can_fetch(s.scraper_user_agent, url)

    print(f"{'source':14} {'feed':>5} {'entries':>7} {'feedtext':>8} {'fetched':>7} {'usable':>6} {'dated':>5} {'avgwords':>8}  notes")
    for src in SOURCES:
        notes: list[str] = []
        try:
            r = client.get(src.feed_url)
            feed = feedparser.parse(r.content)
            status = r.status_code
        except httpx.HTTPError as e:
            print(f"{src.name:14} {'ERR':>5} {'-':>7} {'-':>8} {'-':>7} {'-':>6} {'-':>5} {'-':>8}  {type(e).__name__}")
            continue
        entries = feed.entries[:n]
        if not feed.entries:
            notes.append("not a parseable feed")
        feed_words = [
            len((e.get("content", [{}])[0].get("value") or e.get("summary", "")).split()) for e in entries
        ]
        fetched = usable = dated = 0
        words: list[int] = []
        for e in entries:
            url = e.get("link", "")
            time.sleep(s.scraper_request_delay_seconds)
            if not url or not allowed(url):
                notes.append("robots.txt disallows" if url else "entry without link")
                continue
            try:
                page = client.get(url)
            except httpx.HTTPError as err:
                notes.append(type(err).__name__)
                continue
            if page.status_code != 200:
                notes.append(f"HTTP {page.status_code}")
                continue
            fetched += 1
            out = trafilatura.extract(page.text, output_format="json", with_metadata=True)
            doc = json.loads(out) if out else {}
            wc = len((doc.get("text") or "").split())
            if wc >= MIN_WORDS:
                usable += 1
                words.append(wc)
            if doc.get("date"):
                dated += 1
        avg_feed = sum(feed_words) // len(feed_words) if feed_words else 0
        avg_words = sum(words) // len(words) if words else 0
        print(
            f"{src.name:14} {status:>5} {len(feed.entries):>7} {avg_feed:>8} {fetched:>4}/{len(entries):<2} "
            f"{usable:>3}/{len(entries):<2} {dated:>5} {avg_words:>8}  {', '.join(sorted(set(notes)))}"
        )
        time.sleep(s.scraper_request_delay_seconds)
    print(f"\nusable = extracted body of at least {MIN_WORDS} words; feedtext = avg words already in the feed itself")


if __name__ == "__main__":
    main()
