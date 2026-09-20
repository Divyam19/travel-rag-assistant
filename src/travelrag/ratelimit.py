"""Protect the paid APIs behind /api/chat: a per-client sliding window and a daily total.

Each chat turn spends OpenAI tokens and up to three Tavily searches. Tavily already has a hard
daily cap, but nothing bounded OpenAI spend or one client hammering the endpoint. Limits are held
in memory: they reset when the server restarts, which is acceptable for a single-process demo and
NOT for a multi-worker deployment, where they would need to live in Postgres or Redis.
"""

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable


class RateLimiter:
    def __init__(self, per_minute: int, per_day: int,
                 clock: Callable[[], float] = time.monotonic,
                 today: Callable[[], object] = lambda: datetime.now(timezone.utc).date()) -> None:
        self.per_minute, self.per_day = per_minute, per_day
        self._clock, self._today = clock, today
        self._hits: dict[str, deque[float]] = {}
        self._day = today()
        self._day_count = 0
        self._lock = threading.Lock()

    def check(self, client: str) -> tuple[bool, int, str]:
        """Record a request if allowed. Returns (allowed, seconds to wait, reason)."""
        with self._lock:
            now = self._clock()
            if (today := self._today()) != self._day:
                self._day, self._day_count = today, 0
            if self._day_count >= self.per_day:
                return False, 3600, "The daily question limit has been reached. Please try again tomorrow."
            window = self._hits.setdefault(client, deque())
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= self.per_minute:
                wait = max(1, int(60 - (now - window[0])) + 1)
                return False, wait, "You're asking too quickly. Please wait a moment and try again."
            window.append(now)
            self._day_count += 1
            return True, 0, ""
