"""Settings loaded from .env. Fails fast, naming every missing or placeholder value."""

import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER_MARKERS = ("YOUR-", "[YOUR", "REGION", "you@example.com")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Required
    openai_api_key: str
    supabase_db_url: str
    tavily_api_key: str

    # Models
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    # Chosen by benchmark, see docs/decisions.md: nano tiers over-clarify (33-38/41 routing accuracy),
    # gpt-4.1-mini scores 41/41 at 0.9s. reasoning_effort only applies to gpt-5* models.
    small_model: str = "gpt-4.1-mini"
    large_model: str = "gpt-5.4-mini"
    large_model_reasoning_effort: str = "low"

    # Tavily credit guards
    tavily_daily_call_cap: int = Field(20, ge=0)
    tavily_max_results: int = Field(4, ge=1, le=10)
    tavily_search_depth: str = "basic"
    tavily_use_fixtures: bool = False
    web_result_cache_hours: int = Field(12, ge=0)
    web_fetch_top_n: int = Field(4, ge=0, le=8)  # fetch this many result pages ourselves; 0 = use Tavily snippets only
    web_fetch_timeout_seconds: float = Field(7.0, gt=0)  # interactive fetch; the bulk ingester waits longer
    web_context_tokens: int = Field(12000, ge=500)       # total budget for a web answer's sources
    web_page_max_tokens: int = Field(3500, ge=200)       # cap for any single page
    web_max_pages: int = Field(6, ge=1, le=12)           # total pages fetched across all parts of a question
    web_article_ttl_days: int = Field(14, ge=1)  # how long web-fallback pages stay in the corpus

    # Scraper
    scraper_user_agent: str = "vingo-rag-bot/0.1 (personal portfolio project)"
    scraper_request_delay_seconds: float = Field(2.0, ge=0)
    scraper_timeout_seconds: float = Field(20.0, gt=0)
    scraper_max_articles_per_source: int = Field(30, ge=1)
    chunk_size_tokens: int = Field(400, ge=50)
    chunk_overlap_tokens: int = Field(50, ge=0)

    # Retrieval / agent
    retrieval_top_k: int = Field(12, ge=1)
    context_top_n: int = Field(4, ge=1)            # index answers stay tight; 3 loses two detail queries
    max_chunks_per_article: int = Field(1, ge=1)   # stop one page monopolising the context
    neighbour_chunks: int = Field(1, ge=0, le=2)   # chunks either side of a hit: 0 drops detail recall 8/8 -> 1/8
    max_sub_queries: int = Field(3, ge=1, le=4)
    context_relative_floor: float = Field(0.65, ge=0, le=1)  # drop hits scoring below this fraction of the top hit
    confidence_threshold: float = Field(0.53, ge=0, le=1)  # see docs/decisions.md
    confidence_gray_margin: float = Field(0.08, ge=0, le=0.3)  # scores within +/- this of the threshold get a small-model check
    max_eval_retries: int = Field(1, ge=0, le=3)
    eval_enabled: bool = True
    eval_sim_pass: float = Field(0.60, ge=0, le=1)  # every answer sentence at least this similar to a source sentence: pass without an LLM (see docs/decisions.md)
    eval_sim_fail: float = Field(0.0, ge=0, le=1)  # any sentence below this is ungrounded without asking an LLM (0 = never)

    # App. Limits are per server process; see ratelimit.py.
    cors_origins: str = "http://localhost:5173"
    # How many reverse proxies sit in front of the API (0 when reached directly, 1 on Railway).
    # Used to find the real client address in X-Forwarded-For; see api.client_address.
    trusted_proxy_hops: int = Field(0, ge=0, le=5)
    rate_limit_per_minute: int = Field(10, ge=1)  # chat turns per client address per minute
    daily_chat_cap: int = Field(300, ge=1)        # chat turns across all clients per UTC day

    @field_validator("openai_api_key", "supabase_db_url", "tavily_api_key")
    @classmethod
    def _not_placeholder(cls, v: str) -> str:
        v = v.strip()
        if not v or any(m in v for m in PLACEHOLDER_MARKERS):
            raise ValueError("empty or still a placeholder")
        return v

    @field_validator("tavily_search_depth")
    @classmethod
    def _depth(cls, v: str) -> str:
        if v not in ("basic", "advanced"):
            raise ValueError("must be 'basic' or 'advanced'")
        return v


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as e:
        lines = [
            f"  - {'.'.join(str(p) for p in err['loc']).upper()}: "
            f"{'missing' if err['type'] == 'missing' else err['msg']}"
            for err in e.errors()
        ]
        sys.exit("Config error. Fix these in .env (see .env.example):\n" + "\n".join(lines))
