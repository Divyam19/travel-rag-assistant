# Design decisions, with the numbers behind them

Reproduce with `python -m travelrag.retrieval_report` and `python -m travelrag.chunk_experiment`
(golden set: `data/golden_queries.json`, 41 queries).

## Corpus (2026-09-20)

287 articles, 746 chunks from six sources. 217 are US State Dept country advisories (feed text,
because article pages return 403); the rest are news and guides from Skift, The Points Guy,
Simple Flying, Matador and the UK FCDO. Median article is 483 tokens, but the long-form news
sources run 1,700 to 2,300.

## Chunk size: 400 tokens, 50 overlap

Five configs re-chunked and re-embedded in memory; 8 "detail" queries that require the retrieved
chunk itself to contain the answer text.

| size/overlap | chunks | avg tokens | detail recall@1 | detail MRR | context tokens (top 4) |
|---|---|---|---|---|---|
| 150/20 | 1865 | 125 | 0.75 | 0.88 | 499 |
| 250/40 | 1145 | 212 | 0.62 | 0.78 | 847 |
| **400/50** | 746 | 319 | **0.88** | **0.94** | **1276** |
| 600/80 | 543 | 439 | 0.75 | 0.81 | 1755 |
| 900/100 | 429 | 546 | 0.88 | 0.94 | 2186 |

All sizes reach recall@4 = 1.00, so any of them finds the right passage in the top 4. 400/50
ties the best ranking quality while sending 42% fewer context tokens than 900/100.
Caveat: 8 queries, so the gaps between configs are one or two queries; treat this as a
tie-break on cost, not a strong statistical result.

## Confidence threshold: 0.53 on top-1 cosine similarity

Top-1 similarity, 400/50 index:

| group | lowest / highest |
|---|---|
| answerable (18) | min 0.624, median 0.738 |
| detail (8) | min 0.565 |
| out of index (10) | max 0.495, median 0.374 |

0.53 sits between the weakest answerable (0.565) and the strongest out-of-index (0.495).
The margin is only about 0.07, and the answerable queries were written from article titles, so
real paraphrased queries will score lower. A bare threshold is therefore fragile. Plan: treat
scores in a gray band (roughly 0.45 to 0.60) as "unsure" and let the small model judge relevance
of the top chunks before deciding to fall back to web search.

Traps behaving as intended: "visa to Vietnam" retrieved the Vietnam safety advisory at 0.495
(below threshold, so it goes to fallback), and "taxi from Heathrow" matched the Heathrow
net-zero article at only 0.325.

Ambiguous queries ("Is it safe?") scored 0.35 to 0.43, so they would also fall below the
threshold; clarification must be decided before retrieval, not inferred from scores.

## Baseline cost (Phase 3): retrieve top 8, large model, nothing else

`python -m travelrag.baseline` runs all 41 golden queries with `run_label='baseline'`; token counts
come from the API's usage field and are stored per call in `llm_calls`. Both models were
`gpt-4o-mini` for this run.

| measure | value |
|---|---|
| answer calls | 41 |
| prompt tokens (total / mean / range) | 119,978 / 2,926 / 1,423 to 3,331 |
| completion tokens (total / mean) | 4,401 / 107 |
| embedding tokens (query embedding, total) | 463 |
| estimated cost (see pricing.py caveat) | ~$0.021 total, ~$0.0005 per query |

What this shows, and what the efficiency pass should move:

- **Prompt tokens dominate**: 27x more prompt than completion tokens. Trimming context is the
  biggest lever (top 4 instead of 8 is roughly half).
- **Duplicated context**: the 8 chunks come from only 4.0 distinct articles on average, and in 18
  of 41 queries 5 or more chunks came from a single article. A reranker or per-article cap would
  keep coverage while cutting tokens.
- **Wasted calls**: the 10 out-of-index queries each spent about 2,800 prompt tokens to reply "the
  sources do not contain this". The confidence check exists to route those to web search without
  paying for a full-context answer first. The 5 ambiguous queries got confident guesses instead
  of a clarifying question (m01 answered about Texas Hill Country; m03 asserted visa
  requirements for India and Venezuela, one of them stretching its source).
- Answers were otherwise grounded and cited; a suspicious "August 2025" in a07 turned out to be in
  the Skift article text itself.

Caveat for the final numbers: cheap-vs-large routing savings only show up if the large model
differs from the small one. Baseline the dollar figures again with the real demo `LARGE_MODEL`.

## Agent (Phase 4)

`python -m travelrag.agent_check` runs all 41 golden queries through the LangGraph agent
(analyze, retrieve, assess, web fallback, generate) and checks the path each one took.

| category | passed |
|---|---|
| answerable | 18/18 |
| out of index | 10/10 (7 web fallbacks, 1 answered from an earlier fallback page, 2 off-topic) |
| ambiguous | 5/5 (clarifying question, no retrieval) |
| detail | 7/8 |

The one detail miss (d07) is a strict-check artifact: the answer quotes the source correctly, but
the chunk holding that text ranked second and the check requires it first.

### Cost against the baseline (same 41 queries, both models gpt-4o-mini)

| | tokens/query | est. cost, 41 queries |
|---|---|---|
| baseline (top 8 chunks, always answer) | 3,033 | $0.0206 |
| agent | 1,984 (-35%) | $0.0152 (-26%) |

Where the saving comes from: context is 4 chunks instead of 8, and the 7 clarify/off-topic queries
cost about 450 tokens each instead of about 3,000. Every query now also pays for an analysis
call (about 400 tokens). No model-price saving yet, since the small and large models are the same.
Tavily: 7 live searches in this run (1 credit each), reruns are free via recorded fixtures.

### Routing decisions and what it took

- **Clarify and rewrite in one small-model call**, at temperature 0 so a message always takes the
  same path. Checking routing over all 41 queries exposed real failures the first prompt hid:
  10 wrong routes (airline finance and engine questions flagged off-topic, fully specified card
  questions over-clarified), then 5, then 1, then 0 after the prompt described what the index
  covers and gave counter-examples. Caveat: the prompt was tuned on this same set, so 0 wrong
  routes is a fit, not an independent test.
- **Gray band**: top-1 scores within 0.08 of the 0.53 threshold go to a small-model relevance check
  instead of a bare cutoff. Fixed the "Do I need a visa to visit Vietnam" case: the safety
  advisory scored 0.495, the judge said insufficient, web search ran.
- **Off-topic is refused before retrieval** rather than sent to web search (deviation from the PRD's
  "off-topic goes to web fallback"): it saves a Tavily credit and keeps unrelated pages out of the corpus.
- **Web pages are written back** into the corpus (`source_type='web'`, 14 day TTL) without bumping
  the cache generation. Calibration scripts exclude them so thresholds stay reproducible.

### Known weaknesses (motivating Phase 5 and 6)

- **Stale facts stated as current.** "Is there a rail strike in France this week?" was answered
  "Yes, on Tuesday, January 13, 2026" from an old page, while a September 2026 page was also in
  the sources. Time-sensitive questions need the current date in the prompt and a recency filter.
- **Flat visa claims.** The Vietnam answer says "Yes, US citizens require a visa" with no advice to
  confirm officially, despite the prompt. This is the case the Tier 2 judge must catch.
- **Duplicate context.** The top 4 chunks are often from one article (all four for Delta One Suites).
- **Web sources are third-party.** Fee and price answers cite aggregator sites, not the airline.

### Incident: pooled connections and search_path

The 6543 (transaction) pooler shares server connections between clients. One was left with an
empty `search_path`, so unqualified table names and the `vector` type stopped resolving and a run
crashed mid-way. The data was untouched. `db.connect()` now pins `search_path` on every new connection.

## Evaluation layer (Phase 5)

`python -m travelrag.eval_check` grades 16 deliberately corrupted answers (`data/adversarial_cases.json`)
and 34 real answers from the golden run. Each corrupted case takes a real answer and either changes a
fact, appends an invented claim, or adds a definite legal claim. The real answers must not be flagged.

### Design

- **Tier 1, no LLM call.** Answer sentences are embedded together with the context's own sentences
  (one embedding request) and each is matched to its closest context sentence. Numbers in the
  answer must appear in the sources, and so must capitalised names. Anything that fails, or scores
  under the pass threshold, is sent to Tier 2.
- **Tier 2, small model, strict rubric.** Unsupported claims, contradicted claims, and whether the
  answer states legal or visa requirements as definite fact. Also always run for visa/immigration topics.
- **Tier 3 in the agent.** Flagged: regenerate once with the offending statements listed. Still
  flagged: show the answer with a note naming exactly what could not be verified (no blanket
  refusal). Legal claim: append a fixed disclaimer.

### What the first design got wrong (worth knowing before the interview)

1. Comparing an answer sentence with a whole 300-token chunk does not discriminate. Genuine
   paraphrases scored 0.25 to 0.5, the same as invented sentences, so 76% of clean answers were
   flagged. Comparing sentence to sentence fixed most of it (clean answers' median lowest score
   0.43 to 0.69).
2. The number check had two bugs: it ignored titles (where "Level 3" lives) and matched small
   integers to the word before them. Fixed; small integers above 12 must appear as standalone
   numbers, integers up to 12 are ignored except advisory levels.

### Results (pass threshold 0.60)

| | |
|---|---|
| corrupted answers caught | 14 of 16 |
| real answers wrongly flagged | 3 of 34 (9%); 2 more got the visa disclaimer, as intended |
| real answers that reached the judge | 15 of 34 (44%) |
| judge cost (35 calls in the last full run) | about $0.0095 for 49 gradings |

Not caught: a number swapped for a different real number from the same article (2,500 to 5,000)
and an invented perk built from words that occur elsewhere in the sources ("free lifetime Global
Entry membership"). Sentence similarity and lexical checks cannot see either; only a judge run on
every answer would, and that is the price of the "most answers skip the judge" design.
The names check cost 12 points of extra judge calls (32% to 44%) for one extra catch, a marginal trade.

### The pass threshold is a cost/safety dial

Share of answers sent past Tier 1 to the judge, by threshold:

| threshold | real answers | corrupted answers |
|---|---|---|
| 0.50 | 29% | 69% |
| 0.55 | 32% | 75% |
| **0.60** | **44%** | **88%** |
| 0.65 | 50% | 94% |
| 0.70 | 62% | 94% |
| 0.75 | 74% | 100% |
| 0.80 | 79% | 100% |

`EVAL_SIM_PASS` sets it. 0.60 is the cost-conscious default; 0.75 catches everything in this test
set at nearly double the judge calls.

### Agent cost with evaluation, all 41 golden queries (both models gpt-4o-mini)

| | est. cost | chat tokens/query |
|---|---|---|
| baseline: top 8 chunks, no checks | $0.0206 | 3,033 |
| agent, Phase 4 (routing, no evaluation) | $0.0152 | 1,984 |
| agent with evaluation | $0.0198 | 2,565 |

Evaluation adds about 30% over the Phase 4 agent (judge, sentence embeddings, retries); the whole
system with safety checks still costs about 4% less than the naive chain. Paths: 34 index, 2 off-topic,
5 clarify; 40 of 41 as expected (d07 is the strict-ranking artifact from Phase 4).

### Other Phase 5 fixes

- **Stale facts.** The analyzer marks time-sensitive questions; they search with a one-month recency
  filter, are stored for 1 day instead of 14, and the answer prompt carries today's date. The France
  rail-strike question now answers from a September 2026 tracker ("no strikes today, next is the
  easyJet France pilots strike on 24 to 25 September"), checked against the stored page text.
- **Latency.** A plain answer took 16.8 s; the profile showed about half of it was opening database
  connections (1.7 s each to the remote pooler), 4 of them just to log token usage. A warm connection
  pool plus background usage logging brought it to 5.3 s (clarify: 0.9 s, judge path: 6.3 s).
  The remaining time is mostly the OpenAI calls.
- **Connection string.** The transaction pooler (port 6543) is supported but the session pooler
  (5432) is more robust, since one server connection is pinned per client.

## API and frontend (Phase 7)

- **Streaming.** `POST /api/chat` returns server-sent events: one `status` per finished graph node
  (with what the next step is), then a single `result` or `error`. Server-sent events fit because the
  data flows one way; `fetch` reads the stream, since `EventSource` cannot POST.
- **Stateless server.** The UI sends its last six messages with each request and the analyzer folds
  a clarifying question and its answer into one search. No sessions to store or expire.
- **Errors.** Failures become a generic message; details stay in the server log.
  Request sizes are capped (message 1,000 characters, 12 history messages of 4,000 each).
- **Honest labels.** Web pages saved by an earlier search live in the index, so the badge reads
  "From saved web pages" when every source is one, not "From the travel index".
- **Verified.** 9 API tests (streaming, error hiding, input limits); the real UI was driven in a
  browser: clarify, then follow-up reply, then a visa question with the legal note.

Known gaps: several of the four cited sources can be chunks of the same page (the per-article cap
is planned for the efficiency pass); no authentication or rate limiting beyond the daily Tavily cap,
so do not expose the API publicly; a turn takes about 5 to 8 seconds, mostly model calls.

## Rebuild after the first real conversation (Phase 6)

Four turns of real use exposed problems the golden set never would. What follows is what was
measured, changed, and measured again.

### The failure that started it

Asked for a 7-day Japan itinerary with costs, the assistant replied "the sources do not provide a
specific cost estimate" — while its top source was titled *7-Day Japan Itinerary: Osaka, Kyoto &
Tokyo*. A Kanpur flights question cited the same trip.com page four times, from 2015, and silently
dropped half the question. Both took 30+ seconds.

### Retrieval was fetching the wrong half of the page

The single most valuable measurement of this phase. Detail queries, where the answer text must be
in the retrieved chunk:

| chunks per article | neighbours | context tokens | detail recall |
|---|---|---|---|
| 2 | 0 | 626 | **1/8** |
| 2 | 1 | 2066 | 8/8 |
| **1** | **1** | **1718** | **8/8** |

Retrieving a chunk and not its neighbours misses the answer seven times out of eight: the prose
that matches the question and the table that answers it are adjacent, and we were keeping only the
former. Context is now built by capping each article to one chunk, pulling in the chunks either
side, and merging contiguous runs into a single citation. That alone fixed the "sources don't say"
answers and the four-identical-cards list. A relative floor (drop hits below 0.65 of the top score)
trims noise with no recall loss.

### Where the 28 seconds went

Measured per stage rather than guessed:

| stage | share of a fresh web turn |
|---|---|
| write-back: chunk, embed, insert pages | 9.4s (34%) |
| vector searches | 5.4s (19%) |
| model calls | 4.3s (15%) |
| Tavily search | 4.0s (14%) |
| page fetching | 2.4s (9%) |

The web fallback was writing every page to Postgres and then *re-querying Postgres* for chunks of
pages it already held in memory. It now answers from the fetched text and indexes in the
background. Two further findings, both invisible without measurement:

- Each pooled connection cost a ~200ms round trip because the pool verified it on every
  acquisition, and a turn opens several. Removing the check and pre-warming the pool took a plain
  answer from 16.8s to about 5s.
- A booking site took **20 seconds to serve robots.txt**, using the bulk-ingest timeout on the
  interactive path, and a fresh fetcher per sub-query re-fetched it every time. One shared fetcher
  with the interactive timeout, plus searching every part concurrently, took a three-part web
  question from **131s to 28.6s**.

### Answers now stream

`/api/chat` emits `token` events as the model writes, so text appears in about 3.5s on an index
answer and 13s on a live three-part web search, instead of after the whole turn. A `restart` event
tells the UI to discard a draft the checker rejected, so a rewrite never concatenates with the
draft it replaces. The `result` event stays authoritative, because the checker can append a
disclaimer after the last token.

### Multi-part questions

The analyzer now splits a question into at most three searchable parts, each part is retrieved
separately and the results interleaved, and the web fallback searches every part. Crucially, the
**weakest** part decides confidence: a well-covered half used to mask an uncovered one, which is
how "Kanpur to Delhi and Bangalore" became an answer about Bangalore only. Parts are also held to
a stricter bar (double the gray margin), because a single vector can look confident on a topical
near-miss — "flights Kanpur to Delhi" matched an article about a Bhopal-Delhi diversion at 0.62.

### Planning, and what grounding now means

The assistant composes itineraries and rough budgets. Grounding applies to facts, not to shape:
every price and duration must be cited, but arranging cited facts into a day-by-day plan is not
invention. It may add sourced figures into a total if it labels it an estimate, and it never
converts currencies — it reports the currency the sources use.

This forced the judge to be rebuilt. Asked "is every statement supported?", it flagged "Day 1:
arrive in Osaka" four times out of four. Asked instead to *extract checkable claims* (a price, a
date, a name, a rule) and verify those, it passes a clean itinerary and still catches a price
changed from JPY 14,110 to 9,200 and a fabricated onward flight. Statements about what the sources
do not contain are explicitly not claims.

Three real bugs surfaced while tuning the checker, each found by measurement rather than reading:

1. The estimate exemption skipped a whole line, so "approximately 3.1 million" shielded a
   falsified "18.4%" on the same line. It now covers only the figure just after the marker.
2. The name check used an ASCII-only pattern, so "Cancún" became "Canc" and was reported as an
   invented name, as were "Potosí", "Yucatán" and "I'm". It now folds accents and possessives.
3. `eval_check` re-retrieved context instead of using what the answer was written from, so it was
   grading answers against sources the model never saw — 17 of 34 clean answers "failed". The
   context each answer used is now stored with it. This is the integration point that was flagged
   as risky before the work started, and it broke exactly as predicted.

### Where it landed

| | before | after |
|---|---|---|
| golden queries taking the right path | 40/41 | **41/41** (8/8 detail, was 7/8) |
| corrupted answers caught | 14/16 | 15/16 |
| clean answers wrongly flagged | 3/34 | 7/34 |
| fresh multi-part web turn | 131s | 28.6s |
| plain index answer | 16.8s | ~5s, first text ~3.5s |

The false-flag rate is the one number that moved the wrong way, and it is the honest cost of a
stricter judge plus longer, more detailed answers. A false flag appends a "could not verify" note;
it does not refuse the answer.

### Models, chosen by benchmark rather than reputation

Routing accuracy over the 41 golden queries, all warm and at temperature 0:

| model | routing correct | routing latency |
|---|---|---|
| gpt-4.1-nano | 33/41 | 0.83s |
| gpt-5-nano | 38/41 | 0.86s |
| gpt-5.4-mini | 40/41 | 0.85s |
| **gpt-4.1-mini** | **41/41** | 0.90s |
| gpt-4o-mini | 41/41 (39/41 on the revised prompt) | 1.06s |

The nano tiers are fast and cheap and cannot be trusted with this decision: they over-clarify
fully specified questions. `gpt-4.1-mini` handles routing and judging; `gpt-5.4-mini` at low
reasoning effort writes the answers (43 reasoning tokens on a typical answer, 1.4s). Note that
`gpt-5.4-mini` rejects `reasoning_effort="minimal"`, and gpt-5 models reject `temperature`.

Cost per query is **not** quoted for this phase: `pricing.py` has no verified rates for either
model, and `cost_usd` deliberately returns None rather than a plausible-looking wrong number.
Token counts are measured and stored per call in `llm_calls`.

### Indian coverage

The corpus was US/UK-centric, so Indian routes, fares and destinations always fell through to web
search. Added The Hindu travel (59 articles), Economic Times aviation (50) and Travel and Tour
World. Air India, IndiGo, Tamil Nadu, Delhi airport and Northeast India questions now answer from
the curated index at 0.6+ similarity, spending no Tavily credits. Times of India's travel feed IDs
are dead or point at Health; Condé Nast Traveller India and Business Standard block us.

## Checker regressions and the token pass (final numbers; supersede the table above)

After the rebuild the checker regressed and the token bill was higher than asked for. Both were
measured and fixed; several fixes were bugs in my own earlier work.

### Bugs found by measuring, not by reading

1. **Bullet answers were invisible to the sentence check.** `split_sentences` demanded a full stop,
   so "Fan diameter of the LEAP-1B engine: 72.0 inches" yielded zero claims, which scored a vacuous
   perfect 1.00 similarity and skipped every check. A falsified figure passed. Bullets now count,
   and "nothing extractable" is no longer reported as a perfect score.
2. **Advisory levels slipped through.** A context holding several countries' advisories contains
   "Level 2" somewhere, so a Level 4 country falsely reported as Level 2 passed the number check.
   Any answer quoting a level now goes to the judge.
3. **The visa disclaimer depended on the model.** Once the answer model was told to add "please
   confirm officially", the judge concluded the answer was already safe and the disclaimer
   stopped appearing. A definite entry rule ("US citizens need a visa") is now detected in code
   and always carries the disclaimer, judge or no judge.
4. **A stray Arabic word in an English answer** ("...not safe بسبب the war") came from the new
   answer model at low reasoning effort (1 of 41 answers, none in 82 earlier). The prompt now
   pins English and the checker rejects any answer containing another script without a model call.
5. **I shifted my own calibration**: appending source dates to evidence lines before embedding
   them moved similarities across the tuned 0.60 threshold. Dates are now display-only.
6. **The checker graded its own boilerplate**: the appended "Note: entry and visa rules..." and
   statements like "the source is dated 2026-09-17" were scored as claims. Offers of further help
   ("If you want, I can turn this into a timeline") were too, and were flagged as unsupported.
7. **Adversarial cases were brittle.** Each hard-coded a phrase from a model answer, so any
   regeneration could abort the whole validation (three times). Cases now list alternative
   phrasings, and one that cannot be applied is reported as skipped.

### Judge cost: verify claims, do not re-read the sources

The judge used to re-read the full sources the answer model had already read: 2,788 tokens per
call, 36% of all chat tokens. It now receives only the claims Tier 1 found suspicious (an
unverifiable figure or name, an advisory level, an entry rule, or a weak match), each with the four
closest source lines, the source date, and any line that literally contains a figure the claim
quotes (semantic search is poor at table rows). It replies with claim numbers.

| | before | after |
|---|---|---|
| judge tokens per call | 2,788 | **831** (-70%) |
| share of chat tokens | 36% | 11% |
| answers reaching the judge | 65% | 47% |
| chat tokens per query (41 golden) | 5,439 | **3,733** (-31%) |
| baseline for reference (no checks, no routing) | 3,033 | 3,033 |

### Where the checker stands

| | whole-context judge | focused judge |
|---|---|---|
| corrupted answers caught | 15/16 | 13/16 |
| clean answers wrongly flagged | 4/34 | 2/34 |
| judge tokens per call | 2,788 | 831 |

The focused judge gives up two catches for a 70% cheaper judge. The three misses, all on cases
where the corruption is built only from vocabulary the sources already contain:

- **adv04**: a number swapped for a different real number from the same article (2,500 to 5,000).
- **adv05**: a small-integer ratio reworded ("1 to 1 to 4 to 3" becomes "3 to 2"). Integers up to
  12 are ignored by the number check because they match almost any text.
- **adv08**: an invented perk ("free lifetime Global Entry membership") whose words all appear
  elsewhere in the article, so no cheap signal marks it suspicious and it never reaches the judge.

Closing them means running a judge on every answer, which is the opposite of the cost goal. The
pass threshold is the dial: at 0.75 every corrupted case is sent to the judge, at the price of 94%
of clean answers going too.

### Token spend, 41 golden queries

| purpose | tokens | share |
|---|---|---|
| answer (gpt-5.4-mini) | 104,036 | 68% |
| analyze (gpt-4.1-mini) | 30,624 | 20% |
| judge | 16,624 | 11% |
| confidence check | 1,800 | 1% |

Remaining levers, not taken: the routing prompt is about 650 tokens (below the 1,024 that OpenAI
caches automatically) and is tuned on this set, so shrinking it risks the routing accuracy that
took several rounds to reach. Index context is already at its measured floor: three chunks lose
two detail queries.


## Scope decision: no response cache

The original plan called for a two-layer answer cache (exact and semantic) in Postgres. It was
descoped, and everything that only existed to serve it was removed in one pass: the `cache_*`
settings, the `answer_cache` table, and the `ingest_generation` counter (its sole job was
invalidating cached answers when the corpus changed). Migration 0003 drops both tables; 0001 stays as
history. `web_search_cache`, which reuses Tavily results for 12 hours to save credits, is unrelated
and stays.

Also removed as dead: `api_host`/`api_port` (the port is set in the Makefile, so the settings did
nothing), `web_context_top_n` (the web budget is `web_context_tokens` and `web_max_pages`), the unused
Supabase REST credentials, and the superseded `store_results` write-back path.
