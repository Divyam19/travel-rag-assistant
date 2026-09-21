# Travel Assistant

A hybrid RAG chatbot for travel news. It answers from a curated index of scraped articles, asks a
clarifying question when a request is ambiguous, falls back to a live web search when the index is thin,
and checks every answer against its sources before showing it.

Built as a portfolio project, so the interesting parts are the measurements behind each design choice.
Those are in [`docs/decisions.md`](docs/decisions.md), with the numbers.

## How it works

### At a glance

```mermaid
flowchart LR
    U["User question"] --> A["Agent<br/>(LangGraph)"]
    A -->|"ambiguous"| C["Clarifying question"]
    C --> U
    A -->|"searchable"| R["Retriever<br/>(vector index)"]
    R -->|"confident"| G["Grounded draft answer"]
    R -->|"thin or unsure"| W["Live web search"]
    W --> G
    G --> E["Evaluation layer"]
    E -->|"pass"| F["Final answer, streamed<br/>to the React UI"]
    E -->|"flagged"| G
```

### The full decision flow

Every diamond is a real branch in `src/travelrag/agent.py`. Score bands assume the defaults
(`CONFIDENCE_THRESHOLD=0.53`, `CONFIDENCE_GRAY_MARGIN=0.08`).

```mermaid
flowchart TD
    Q["User message<br/>+ last 6 turns"] --> RL{"Within the rate limit?<br/>10 per minute per client<br/>300 per day overall"}
    RL -->|"no"| R429["HTTP 429 + Retry-After<br/>costs nothing"]
    RL -->|"yes"| AN["ANALYZE<br/>small model, temperature 0<br/>one JSON decision"]

    AN --> D1{"Missing a key detail?<br/>(which place, which card)"}
    D1 -->|"yes"| CL(["Ask ONE clarifying question<br/>END. The reply is merged<br/>with this turn next time"])
    D1 -->|"no"| D2{"About travel?"}
    D2 -->|"no"| OT(["Polite decline, no search<br/>END"])
    D2 -->|"yes"| SQ["Rewrite as a standalone query<br/>split into up to 3 parts<br/>flag if time-sensitive"]

    SQ --> RT["RETRIEVE<br/>embed each part, top 12 chunks each<br/>score every part separately"]
    RT --> WK["Confidence = the WEAKEST part's best score<br/>a strong part cannot vouch for a weak one"]
    WK --> BAND{"Score band"}
    BAND -->|"0.61 or more<br/>(0.69 with several parts)"| CONF["Confident"]
    BAND -->|"below 0.45<br/>(below 0.37 with several parts)"| THIN["Thin"]
    BAND -->|"in between"| GJ["Small model checks the top 3 excerpts<br/>against the weakest part"]
    GJ -->|"sufficient"| CONF
    GJ -->|"insufficient"| THIN

    CONF --> CTX["Index context<br/>drop hits under 65% of the best<br/>1 chunk per article + its neighbours<br/>merge, keep 4"]

    THIN --> WS["WEB FALLBACK<br/>search every part in parallel"]
    WS --> WG{"Search allowed?<br/>1 reuse a cached result, 12h<br/>2 else daily cap of 20 calls<br/>3 else live Tavily search"}
    WG -->|"blocked or failed<br/>for every part"| DG{"Index partly relevant?<br/>best score 0.45 or more"}
    DG -->|"yes"| CTXD["Use the index context<br/>and say web search was unavailable"]
    DG -->|"no"| NF(["Say nothing reliable was found<br/>END"])
    WG -->|"results"| FP["Fetch pages concurrently<br/>top 4 in full, 7s timeout, robots.txt obeyed<br/>others use the search snippet<br/>6 pages max, shared between parts"]
    FP --> PG{"Any usable page?"}
    PG -->|"no"| NF
    PG -->|"yes"| WCTX["Web context<br/>short pages whole, long pages<br/>keep their best window<br/>12,000 tokens max"]
    FP -.->|"in the background"| IDX[("Index the pages<br/>expire in 14 days,<br/>1 day if time-sensitive")]

    CTX --> GEN
    CTXD --> GEN
    WCTX --> GEN
    GEN["GENERATE<br/>large model, streamed word by word<br/>facts must be cited, arrangement is free<br/>estimates allowed, never converts currency"]
    GEN --> EV["EVALUATE<br/>see the checker below"]
    EV -->|"pass"| OK(["Answer + numbered citations<br/>END"])
    EV -->|"states an entry rule"| DI(["Answer + not-legal-advice note<br/>END"])
    EV -->|"flagged, first time"| RW["Discard the streamed draft<br/>rewrite once with the rejected<br/>statements listed"]
    RW --> GEN
    EV -->|"flagged again"| CV(["Answer + a note naming exactly<br/>what could not be verified<br/>END"])
```

### The answer checker

Cheap checks run first and settle most answers. A model only sees the specific claims those checks
could not settle.

```mermaid
flowchart TD
    DR["Draft answer + the exact sources<br/>it was written from"] --> FS{"Contains characters from<br/>another alphabet?"}
    FS -->|"yes"| FL1["FLAGGED<br/>no model call"]
    FS -->|"no"| T1["TIER 1, free (one embedding call)<br/>each claim sentence vs. source sentences<br/>every figure and name must appear in the sources"]
    T1 --> K{"Every claim at least 0.60 similar,<br/>no unverifiable figure or name,<br/>no advisory level, no visa topic?"}
    K -->|"yes"| ER1{"States an entry rule?<br/>'you need a visa'"}
    ER1 -->|"yes"| DI["DISCLAIMER"]
    ER1 -->|"no"| PA["PASS"]
    K -->|"no"| SEL["Pick up to 6 suspicious claims, worst first<br/>1 figure or name not in the sources<br/>2 advisory level or entry rule<br/>3 weakest similarity"]
    SEL --> T2["TIER 2, small model<br/>verify only those claims, each with its<br/>4 closest source lines and any line<br/>that contains its figures"]
    T2 --> PR{"Problems found?"}
    PR -->|"yes"| FL2["FLAGGED<br/>problems are the offending sentences"]
    PR -->|"none"| ER2{"States an entry rule?"}
    ER2 -->|"yes"| DI
    ER2 -->|"no"| PA
    T2 -->|"reply unreadable"| FB["Fall back on Tier 1:<br/>flagged if it saw something suspicious,<br/>otherwise pass"]
```

Outcomes feed the last branch of the decision flow above: pass, disclaimer, one rewrite, then a
caveat naming what could not be verified.

### How the index is filled

```mermaid
flowchart LR
    CR["cron, every 4 hours"] --> SRC["For each of 9 sources"]
    SRC --> FD["Read the RSS or Atom feed"]
    FD --> KN{"URL already stored?"}
    KN -->|"yes"| SK["Skip before any request"]
    KN -->|"no"| FT{"Page fetchable?"}
    FT -->|"yes"| PG["Fetch politely<br/>robots.txt, 2s between requests<br/>extract with trafilatura"]
    FT -->|"blocked (State Dept)"| FX["Use the text in the feed entry"]
    PG --> MN{"Long enough?<br/>120 words, 30 from a feed"}
    FX --> MN
    MN -->|"no"| LG["Log the failure, carry on"]
    MN -->|"yes"| HS{"Same content seen before?"}
    HS -->|"yes"| SK
    HS -->|"no"| CH["Chunk: 400 tokens, 50 overlap"]
    CH --> EM["Embed"] --> ST[("Supabase pgvector<br/>one transaction per article")]
    CR --> PU["Delete expired web pages"]
```

A failed article or a failed source never stops the run, and re-running is safe.

### Where each threshold lives

| Setting | Default | Controls |
|---|---|---|
| `CONFIDENCE_THRESHOLD` / `CONFIDENCE_GRAY_MARGIN` | 0.53 / 0.08 | The confident, unsure and thin score bands (the margin doubles for multi-part questions) |
| `CONTEXT_RELATIVE_FLOOR`, `MAX_CHUNKS_PER_ARTICLE`, `NEIGHBOUR_CHUNKS`, `CONTEXT_TOP_N` | 0.65, 1, 1, 4 | What goes into an index answer's prompt |
| `MAX_SUB_QUERIES`, `WEB_MAX_PAGES`, `WEB_CONTEXT_TOKENS`, `WEB_FETCH_TIMEOUT_SECONDS` | 3, 6, 12000, 7 | The web fallback's breadth and budget |
| `TAVILY_DAILY_CALL_CAP`, `WEB_RESULT_CACHE_HOURS` | 20, 12 | Web search credits |
| `EVAL_SIM_PASS`, `MAX_EVAL_RETRIES` | 0.60, 1 | How much the checker trusts free checks, and how many rewrites it allows |
| `RATE_LIMIT_PER_MINUTE`, `DAILY_CHAT_CAP` | 10, 300 | Protection for the paid APIs |

### The layers

| Layer | What it is |
|---|---|
| Ingestion | RSS discovery, `trafilatura` extraction, robots.txt-aware fetcher, dedupe by URL and content hash, cron every 4 hours |
| Store | Supabase Postgres with `pgvector` (HNSW). Chunks of 400 tokens with 50 overlap |
| Agent | LangGraph state machine (`src/travelrag/agent.py`) |
| Checker | Sentence-level similarity plus figure and name checks, then a small-model judge on suspicious claims only (`evaluation.py`) |
| API | FastAPI, streaming answers over server-sent events, per-client rate limit |
| UI | React + TypeScript + Vite: live progress, streamed answer, citations, how-it-was-answered trace |

Sources: Skift, The Points Guy, Simple Flying, Matador, UK FCDO travel advice, US State Department
advisories, The Hindu (travel), Economic Times (aviation), Travel and Tour World.

## Run it

You need Python 3.12, Node 20+, a Supabase project, an OpenAI key and a Tavily key.

```bash
cp .env.example .env         # fill in OPENAI_API_KEY, SUPABASE_DB_URL, TAVILY_API_KEY
make setup                   # venv and dependencies (about 30 seconds)
make migrate                 # create the tables
make doctor                  # checks config, database, pgvector and OpenAI
make ingest                  # first fill of the index (see the note below)
make api                     # terminal 1: backend on :8000
make web-install && make web # terminal 2: UI on http://localhost:5173
make test                    # 74 tests
make cron-install            # optional: re-scrape every 4 hours
```

The first `make ingest` takes roughly ten minutes, because the scraper waits two seconds between
requests to each site and embeds about 1,700 chunks. For a quick demo-sized index run
`.venv/bin/python -m travelrag.ingest --limit 10` instead. Re-running is safe: known articles are
skipped before any request is made.

## Deploy to Railway

One container serves both the API and the built UI, so a deployment is a single service on a single
origin (no CORS, no second proxy). The `Dockerfile` builds the React app, then the API image serves it.

```bash
railway init --name travel-rag-assistant
railway add --service web
# secrets go in via stdin so they never appear in a process list or shell history:
railway variable set OPENAI_API_KEY --stdin --skip-deploys --service web    # then paste, Ctrl-D
railway variable set SUPABASE_DB_URL --stdin --skip-deploys --service web
railway variable set TAVILY_API_KEY  --stdin --skip-deploys --service web
railway variable set TRUSTED_PROXY_HOPS --stdin --skip-deploys --service web <<< "2"
railway up --service web            # uploads the folder (respects .gitignore, so .env stays local)
railway domain --service web        # public https URL
```

Things worth knowing:

- **`TRUSTED_PROXY_HOPS=2` is measured, not assumed.** Railway sends `X-Forwarded-For: client,
  edge-proxy` and overwrites anything the caller supplied, so a forged header cannot get through. With
  the value 1 the rate limiter picked Railway's edge address and bucketed visitors by edge node.
- **One replica, in Singapore** (`railway.json`), close to the Supabase database. The rate limiter
  keeps its counters in memory, so a second replica would silently double every limit.
- The service is public and has no login. Spend is bounded by 10 questions a minute per visitor, 300 a
  day overall, and Tavily's own daily cap. Lower `DAILY_CHAT_CAP` for a demo you do not want busy.
- `railway.json` is the older config format. Railway keeps it working until 2026-12-01; migrate with
  `railway config migrate`.
- The index is filled from your machine (`make ingest`) or cron; the deployed service only reads it
  and adds short-lived web pages.

## Try it: a five-minute demo

| Ask | What you should see |
|---|---|
| "Is it safe to travel to Iraq right now?" | Streams in about 3 seconds, cites the State Department advisory, badge "From the travel index", "Checked against sources" |
| "What's the best time to visit?" | One clarifying question ("Which destination?"). Reply "Japan" and it merges the two turns into one search |
| "Do I need a visa to visit Vietnam as a US citizen?" | Answers, then adds the not-legal-advice note: the checker adds it whenever an answer states an entry rule |
| "Tell me about flights from Kanpur to Delhi and Bangalore" | Splits into two searches; if the index lacks either route it goes to the web and covers both, in INR |
| "What is 2+2?" | Declines politely without searching, so it spends no Tavily credit |

Open **How this was answered** under any reply to see each step, the token count and anything the
checker flagged.

## What was measured

| | |
|---|---|
| Right path on the 41-query golden set | 40 / 41 (the miss is a strict-ranking check, not a wrong answer) |
| Detail questions where the answer text must be in the retrieved chunk | 7 / 8 with neighbour chunks, 1 / 8 without |
| Corrupted answers the checker catches | 13 / 16 |
| Clean answers it wrongly flags | 2 / 34 |
| Chat tokens per query | 3,733 (a plain retrieve-and-answer baseline is 3,033, with no checking or routing) |
| Time to first text | about 3.5 s for an index answer, about 13 s for a live three-part web search |

The three corrupted answers it misses are built only from words the sources already contain: a number
swapped for another real number from the same article, a reworded ratio, and an invented perk. Catching
them means running the judge on every answer, which is a cost decision. The pass threshold
(`EVAL_SIM_PASS`) is the dial, and `docs/decisions.md` shows the trade-off across thresholds.

## Design decisions worth defending

- **Ground facts, not shape.** Every price, date and name must come from a cited source, but arranging
  cited facts into an itinerary is allowed, and a labelled estimate may add sourced figures up. It never
  converts currencies. Getting the checker to judge facts and not shape took a rewrite: asked "is every
  statement supported?", it flagged "Day 1: arrive in Osaka".
- **Retrieval brings back neighbours.** The chunk that matches a question is rarely the one that answers
  it; the table sits next to the prose. Adding the adjacent chunks lifted detail recall from 1/8 to 8/8.
- **Free checks first.** Sentence similarity and figure checks run for the price of one embedding call.
  The model only sees the claims those checks found suspicious, with their four closest source lines,
  which cut judge cost by 70%.
- **The weakest part of a question decides.** A confident answer to half a question must not hide a half
  the index knows nothing about.
- **Models chosen by benchmark**, not reputation. `gpt-4.1-mini` routes and judges; the nano tiers were
  fast but scored 33 to 38 of 41 on routing because they over-clarify. `gpt-5.4-mini` writes answers.

## Known limits

- Rate limits are held in memory, per process. Fine for one server, not for several workers.
- No login. The rate limit and Tavily's daily cap bound spend, but do not expose the API publicly
  without adding authentication.
- Dollar costs are not quoted: prices for the two current models are not verified in `pricing.py`.
  Token counts are measured and stored per call.
- The US State Department blocks article pages, so its advisories come from the RSS feed text.
- Tavily is capped at 20 calls per day (`TAVILY_DAILY_CALL_CAP`); past that the assistant answers from
  the index and says so.
- Answers are English only.

## Layout

```
src/travelrag/   agent, evaluation, context, web fallback, ingest, api, config, llm, db
db/migrations/   numbered SQL migrations (make migrate)
data/            golden queries, adversarial cases, recorded Tavily responses for free replays
frontend/        React UI
tests/           74 tests. None calls OpenAI or Tavily; they need a filled .env, the Tavily-cap tests use your database
docs/            decisions.md (measurements and reasoning), repo-audit.md (a full code audit)
```

Evaluation tooling, run as modules: `agent_check` (all golden queries), `eval_check` (corrupted-answer
test), `baseline`, `retrieval_report`, `chunk_experiment`, `probe`, `stats`. Set
`TAVILY_USE_FIXTURES=true` to replay recorded web searches for free.
