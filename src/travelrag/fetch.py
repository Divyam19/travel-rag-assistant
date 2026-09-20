"""Polite HTTP fetching: identifies itself, obeys robots.txt, throttles per host."""

import threading
import time
from urllib import robotparser
from urllib.parse import urlparse

import httpx

from .config import get_settings


class FetchError(Exception):
    """A page could not be fetched; the message is safe to log."""


class Fetcher:
    def __init__(self, delay: float | None = None, timeout: float | None = None) -> None:
        s = get_settings()
        self._agent = s.scraper_user_agent
        self._delay = s.scraper_request_delay_seconds if delay is None else delay
        # robots.txt is fetched on the critical path of a chat turn, so it needs the caller's
        # timeout too: one booking site took 20s to serve it and stalled the whole answer.
        self._timeout = timeout
        self._client = httpx.Client(
            headers={"User-Agent": self._agent}, timeout=s.scraper_timeout_seconds, follow_redirects=True
        )
        self._robots: dict[str, robotparser.RobotFileParser] = {}
        self._last_request: dict[str, float] = {}
        # Pages are fetched concurrently during a web fallback; guard the shared caches.
        self._lock = threading.Lock()

    def _throttle(self, host: str) -> None:
        with self._lock:  # only serialises requests to the same host
            wait = self._last_request.get(host, 0) + self._delay - time.monotonic()
            self._last_request[host] = time.monotonic() + max(wait, 0)
        if wait > 0:
            time.sleep(wait)

    def _allowed(self, url: str) -> bool:
        parts = urlparse(url)
        with self._lock:
            known = parts.netloc in self._robots
        if not known:
            rp = robotparser.RobotFileParser()
            self._throttle(parts.netloc)
            try:
                r = self._client.get(f"{parts.scheme}://{parts.netloc}/robots.txt", timeout=self._timeout)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except httpx.HTTPError:
                rp.parse([])  # robots.txt unreachable: treat as no restrictions
            with self._lock:
                self._robots[parts.netloc] = rp
        return self._robots[parts.netloc].can_fetch(self._agent, url)

    def get(self, url: str, timeout: float | None = None) -> httpx.Response:
        if not self._allowed(url):
            raise FetchError("robots.txt disallows")
        self._throttle(urlparse(url).netloc)
        try:
            wait = timeout or self._timeout
            r = self._client.get(url, timeout=wait) if wait else self._client.get(url)
        except httpx.HTTPError as e:
            raise FetchError(type(e).__name__) from e
        if r.status_code != 200:
            raise FetchError(f"HTTP {r.status_code}")
        return r
