-- Response caching was descoped, so these were never used. answer_cache was never queried and
-- app_state held only the ingest_generation counter whose sole purpose was invalidating that cache.
-- (web_search_cache is a different thing, reuse of Tavily results, and stays.)
drop table if exists public.answer_cache;
drop table if exists public.app_state;
