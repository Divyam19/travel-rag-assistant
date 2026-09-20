"""Curated news sources. Add or remove entries here; the probe and ingester both read this list."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    feed_url: str
    topic: str  # what this source is good for, used to pick demo queries
    # True when article pages block us (or add nothing) but the feed entry carries the full text.
    use_feed_text: bool = False
    max_articles: int | None = None  # overrides SCRAPER_MAX_ARTICLES_PER_SOURCE


SOURCES = [
    Source("skift", "https://skift.com/feed/", "travel industry news"),
    Source("thepointsguy", "https://thepointsguy.com/feed/", "flight deals, points, airlines"),
    Source("simpleflying", "https://simpleflying.com/feed/", "aviation and airline news"),
    Source("matador", "https://www.matadornetwork.com/feed/", "destination guides"),
    Source("fcdo", "https://www.gov.uk/foreign-travel-advice.atom", "UK entry rules and safety advice"),
    # Article pages return 403, but each feed entry holds the advisory summary (one per country).
    Source("statedept", "https://travel.state.gov/_res/rss/TAsTWs.xml", "US travel advisories",
           use_feed_text=True, max_articles=300),
    # Indian coverage: the corpus was US/UK-centric, so questions about Indian routes, fares and
    # destinations always fell through to web search.
    Source("thehindu", "https://www.thehindu.com/life-and-style/travel/feeder/default.rss",
           "Indian destinations and travel features"),
    Source("et_aviation", "https://economictimes.indiatimes.com/industry/transportation/airlines-/-aviation/rssfeeds/13354027.cms",
           "Indian airlines, airports and aviation policy"),
    Source("traveltourworld", "https://www.travelandtourworld.com/feed/",
           "global travel news, often covering Indian travellers"),
]
# Dropped after probing: Travel + Leisure (HTTP 402), Travel Weekly (HTTP 403),
# Lonely Planet (no RSS at that URL, returns an HTML page); Times of India (travel feed IDs
# 2276337/2886704/3908999 are dead or point at Health); Conde Nast Traveller India (HTTP 400);
# Outlook Traveller (no response); Business Standard (HTTP 403).
