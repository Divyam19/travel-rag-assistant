# Repo audit: vingo_rag (travelrag)

A hybrid RAG travel assistant: a Python package (`src/travelrag`, about 3,300 lines in 33 files) with a
LangGraph agent, a scraper and cron ingester, a Postgres/pgvector store, an answer checker, a FastAPI
service, and a React chat UI (about 300 lines). It is small, consistent and unusually well measured.
The structure is sound: the client/server contract matches exactly, SQL is parametrised everywhere and
a copy-paste scan finds zero clones. The problems are at the seams: two logic bugs in the multi-part
path, one dangerous unused credential, a broken first-run setup, and a configured-but-unbuilt cache.

Method note: this repo is **not under version control**, so the churn analysis was not possible and
"what got rewritten repeatedly" is unknown. Evidence below is from the filesystem, AST scripts, a
clean-venv install and direct tests.

## Feature wiring status

| Feature | Client / caller | Backend | Data layer | Status |
|---|---|---|---|---|
| Scrape and ingest (9 sources, cron) | n/a | ✅ | ✅ | Complete |
| Vector retrieval and context building | n/a | ✅ | ✅ | Complete (`context.py` untested) |
| Agent: clarify, retrieve, gate, web fallback | ✅ | ⚠️ two bugs (C1, C2) | ✅ | Working, degraded on multi-part |
| Answer checker (Tier 1 and 2) | ✅ | ✅ | n/a | Complete |
| Streaming chat UI | ✅ | ✅ | ✅ | Complete, contract verified |
| Token/usage accounting | n/a | ✅ | ✅ | Complete |
| **Answer cache** (exact and semantic) | ❌ | ❌ no code | ⚠️ table only | **Scaffolding only** (S3) |
| **Conversation compression** | ❌ | ❌ | n/a | Not built (history is sliced to 6 messages) |
| README, demo script, architecture diagram | n/a | n/a | n/a | **Missing** |

## Critical: breaks or exposes something now

**C1. Judge is shown the wrong sub-query on multi-part questions.** `src/travelrag/agent.py:207`
```
weakest = parts[scores.index(min(scores)) - 1] if parts and len(scores) == len(parts) + 1 else None
```
`scores[0]` is the main query, `scores[1:]` are the parts. When the main query is the weakest, the
index is `-1`, which silently selects the **last** part. Reproduced: `scores=[0.40,0.70,0.80]` judges
`parts[-1]`. Effect: the gray-band relevance check answers a question about the wrong text.
Fix: `i = scores.index(min(scores)); weakest = state["standalone_query"] if i == 0 else parts[i - 1]`.

**C2. The second part of a multi-part question is starved of pages.** `agent.py:248-260` extends
`to_fetch` part by part and then keeps `to_fetch[: s.web_max_pages]` (6). Two parts returning 4 results
each leave part 1 with 4 pages and part 2 with 2, and a third part gets none. This defeats the feature.
Fix: interleave round-robin (`zip_longest`) before the cap, so each part gets an equal share.

**C3. An unused, maximum-privilege credential lives in `.env` and a stray backup.** `.env` holds a
Supabase **service_role** key (bypasses row-level security). `config.py:24` declares it and nothing
reads it (`grep -rn supabase_service_role_key src` finds only the declaration). A copy sits in
`.env.bak.1789904260`. There is no git repo, so `.gitignore` has never protected anything. The key
also appeared in a terminal during this audit, so treat it as exposed. Fix: delete the field
(`config.py:23-24`), remove both lines and the `.bak`, rotate the key in Supabase.

**C4. `make setup` then `make test` fails on a fresh clone.** `Makefile` `setup` runs
`pip install -e .`, but `pytest` is only in the `dev` extra (`pyproject.toml:23`). Reproduced in a clean
venv: `ModuleNotFoundError: No module named 'pytest'`. This breaks the "runs locally in under 5 minutes"
goal. Fix: `pip install -e ".[dev]"`.

**C5. No version control.** `git rev-parse` reports "not a git repository". All 5,250 lines, the
calibration data and the decisions log have no history or backup. Do this before anything else in "Fix
order", and only after removing C3.

## Structural: works today, drifts as it grows

**S1. The citation entity has five hand-maintained shapes with different field names.**
`rag.py:24` (`Source`: `source`, `source_type`, `chunk_id`), `api.py:79` (renames to `site`,
`origin`, drops `chunk_id`), `agent_check.py:78-81` (serialises 10 fields to JSONL),
`eval_check.py:61-63` (rebuilds `Source` by position from that JSONL), `frontend/src/types.ts:3-10`.
`chunk_id` was added to three of them by hand and the JSONL layout has already broken evaluation twice.
Fix: give `Source` `to_dict()` and `from_dict()` in `rag.py`, use them in `agent_check` and `eval_check`,
and keep the `site`/`origin` rename in exactly one function in `api.py`.

**S2. The adversarial cases hard-code phrases of model output.** `data/adversarial_cases.json` replaces
strings like `"64 business-class seats"`, which vanish whenever a model rewords an answer; this aborted
the whole validation three times. The harness now skips unappliable cases (`eval_check.py`), but the
data is still brittle. Anchor cases on figures only, as `adv04` now does.

**S3. A whole feature is configured and documented but has no code.** `config.py:72-74`
(`cache_enabled`, `cache_semantic_threshold`, `cache_ttl_hours`), `db/migrations/0001_init.sql`
(`answer_cache` table plus HNSW index, never queried), `.env.example` cache block, and `.env` lines all
imply a cache. Decide: build it, or delete all four so nobody assumes it works.

**S4. Dead and phantom config.** Read by nothing: `config.py:59` `web_context_top_n` (also in `.env`
and `.env.example`; the real web budget is `web_context_tokens`/`web_max_pages`), `config.py:77-78`
`api_host`/`api_port` (the port is hard-coded in `Makefile`, so setting `API_HOST` does nothing),
`config.py:23-24` `supabase_*`, and `VITE_API_URL` in `.env.example` (the frontend uses the dev proxy).
Conversely `EVAL_ENABLED`, `EVAL_SIM_PASS`, `EVAL_SIM_FAIL` (`config.py:60-62`) are missing from
`.env.example`, although `docs/decisions.md` tells users to set `EVAL_SIM_PASS`.

**S5. `probe.py` re-implements the fetcher's robots.txt and throttling.** `probe.py:28-41` vs
`fetch.py:40-54`. It also judges an article "usable" at 150 words (`probe.py:24`) while the ingester
accepts 120 (`ingest.py:28`), so the probe's report does not predict what ingestion will do.
Fix: build the probe on `Fetcher` and import the threshold.

**S6. The multi-part path has no tests, and neither does the code the biggest measured gain rests on.**
`context.py` (148 lines; neighbour expansion took detail recall from 1/8 to 8/8) has no test file, nor do
`agent.merge_ranked`, `agent.web_fallback` or `web.fetch_pages`. C1 and C2 would have been caught by
two small tests. Add them alongside the C1/C2 fixes.

**S7. Unauthenticated endpoint that spends money.** `api.py:110` `POST /api/chat` has no auth or rate
limit; each call makes several OpenAI calls and up to 3 Tavily searches. Only Tavily has a daily cap
(`config.py` `tavily_daily_call_cap`). Bound to localhost by default, so not urgent, but it becomes
critical the moment it is deployed or tunnelled. Add a per-IP limit and an OpenAI spend cap first.

## Dead weight: safe to delete

- `src/travelrag/web.py:120-157` `store_results`: superseded by `_index_pages`, no callers (verified by grep).
- `src/travelrag/web.py:233` `flush_indexing`: no callers.
- `web.py:183` `fetch_pages(scores=...)`: no caller passes it, and the comment on `web.py:173`
  ("Tavily's own relevance score") is false; `parse_results` never keeps a score.
- `config.py:59`, `:77-78`, `:23-24` and their `.env` / `.env.example` lines (see S4).
- `tests/conftest.py`: a docstring and nothing else.
- `.env.bak.1789904260` (see C3).
- `answer_cache` table and `config.py:72-74`, unless the cache is built next (see S3).

## Deliberately not flagged

Zero clones by `jscpd` across Python, TypeScript and CSS; client and server agree on routes, event names
and payload fields; all SQL is parametrised (the two `execute` hits are a static string and the
migration file); every broad `except Exception` either logs, reports or re-raises as a typed error; no
TODO/FIXME/stubs; CORS is an explicit origin list; config defaults match `.env.example` apart from
formatting (`2` vs `2.0`). I also tried to defeat the fact-checker by planting an instruction in its
evidence text. It **did not work** (the fabricated claim was still flagged), so the unescaped evidence
in `evaluation.judge` is a hardening opportunity, not a defect. Skipped as too small: `embed_query` /
`embed_texts` near-twins (`llm.py:44-55`), the sentence regex repeated in `evaluation.py`/`chunking.py`,
importing private `_get_pool` in `api.py:24`, and `ingest.py:92` loading every content hash per source
(about 540 rows today; watch it if web write-back grows the table).

## Fix order

1. **C3, then C5:** remove and rotate the service-role key and delete the `.bak`, then `git init` and
   make the first commit. Doing it in this order keeps the secret out of history for good.
2. **C4:** one-line `Makefile` fix, verified in a clean venv.
3. **C1, C2, S6 together:** two small fixes plus tests for `merge_ranked`, the round-robin allocation
   and `context.py`. They share the same multi-part code path.
4. **S3/S4 and the dead-code list:** decide about the cache first (it determines what to delete),
   then remove dead config and `store_results` and add the missing `EVAL_*` lines to `.env.example`.
5. **S1, S2, S5:** consolidate the citation entity, re-anchor the adversarial cases, rebuild the probe
   on `Fetcher`. Each step leaves the repo working.
6. **S7 and the README:** rate limiting before any deployment; the README, architecture diagram and
   demo script are the remaining Phase 8 work.

## Resolution

Every finding above was addressed in order, one commit each. Verified by running the tests, by
mutation-checking the two logic fixes against the original code, and by a clean-clone install.

| Finding | Status | Commit |
|---|---|---|
| C3 unused service-role key | Removed from config, `.env`, `.env.example`; backup deleted. **Rotate the key in Supabase** (it appeared in a terminal during the audit) | a49efd5 |
| C5 no version control | `git init`; secret values scanned for and absent from every tracked file before the first commit | a49efd5 |
| C4 `make setup` lacked test deps | Installs `.[dev]`; verified in a fresh clone | 72005d2 |
| C1 judge asked about the wrong part | `weakest_part()`; regression test fails against the old arithmetic | 2f737f5 |
| C2 second part starved of pages | `interleave()` before the cap; regression test fails against concatenation | 2f737f5 |
| S6 no tests for `context.py` or the multi-part path | 14 new tests | 2f737f5 |
| S3/S4 unbuilt cache and dead config | Cache descoped and removed; migration 0003 drops `answer_cache` and `app_state`; dead settings and `store_results` deleted; `EVAL_*` added to `.env.example` | d4bca7b |
| S1 five shapes of the citation entity | One `Source.to_dict/from_dict`; API and frontend use the model's own field names | 7ff06df |
| S2 brittle adversarial cases | Alternative phrasings anchored on figures; unappliable cases skipped and reported | 7ff06df |
| S5 duplicate fetcher logic in `probe.py` | Rebuilt on `Fetcher`; shares the ingester's threshold, asserted by a test | 7ff06df |
| S7 unauthenticated spending endpoint | Per-client and daily limits, 429 with `Retry-After`; verified live | 7f641dd |
| Missing README | Written with diagram, setup, demo script and measured results | dede26d |

Not done, deliberately: the judge's unescaped evidence text (an injection attempt failed, so it is
hardening, not a defect), and the small items listed under "Deliberately not flagged".
Two limits remain by design: rate limits are in memory per process, and there is still no login.
