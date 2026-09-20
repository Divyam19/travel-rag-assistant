-- Core schema: articles, chunks (pgvector), answer cache, web-search cache,
-- ingest bookkeeping, Tavily usage counter.

create extension if not exists vector with schema extensions;

-- ---------------------------------------------------------------- articles
create table public.articles (
  id            bigint generated always as identity primary key,
  url           text        not null,
  url_hash      text        not null unique,
  content_hash  text        not null unique,
  source        text        not null,
  source_type   text        not null default 'feed' check (source_type in ('feed', 'web')),
  title         text        not null,
  body          text        not null,
  published_at  timestamptz,
  fetched_at    timestamptz not null default now(),
  expires_at    timestamptz            -- null = keep forever; set for web-fallback articles
);

create index articles_source_published_idx on public.articles (source, published_at desc);
create index articles_expires_at_idx on public.articles (expires_at) where expires_at is not null;

-- ------------------------------------------------------------------ chunks
create table public.chunks (
  id           bigint generated always as identity primary key,
  article_id   bigint  not null references public.articles (id) on delete cascade,
  chunk_index  int     not null,
  content      text    not null,
  token_count  int     not null,
  embedding    extensions.vector(1536) not null,
  unique (article_id, chunk_index)      -- also serves as the FK index on article_id
);

create index chunks_embedding_hnsw_idx
  on public.chunks using hnsw (embedding extensions.vector_cosine_ops);

-- Top-k nearest chunks with article metadata. similarity = 1 - cosine distance.
-- Note: the expiry/source filters are applied after the ANN scan, so heavy filtering can
-- return fewer than match_count rows. Fine at this scale.
create function public.match_chunks(
  query_embedding extensions.vector(1536),
  match_count     int     default 8,
  include_web     boolean default true
)
returns table (
  chunk_id     bigint,
  article_id   bigint,
  content      text,
  similarity   float8,
  title        text,
  url          text,
  source       text,
  source_type  text,
  published_at timestamptz
)
language sql stable
set search_path = public, extensions
as $$
  select c.id, c.article_id, c.content,
         1 - (c.embedding <=> query_embedding),
         a.title, a.url, a.source, a.source_type, a.published_at
  from chunks c
  join articles a on a.id = c.article_id
  where (a.expires_at is null or a.expires_at > now())
    and (include_web or a.source_type = 'feed')
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

-- ------------------------------------------------------ ingest bookkeeping
-- Single-row-per-key counters. 'ingest_generation' is bumped whenever the corpus changes,
-- which invalidates cached answers built on an older corpus.
create table public.app_state (
  key    text   primary key,
  value  bigint not null
);
insert into public.app_state (key, value) values ('ingest_generation', 0);

create table public.ingest_runs (
  id            bigint generated always as identity primary key,
  source        text        not null,
  started_at    timestamptz not null default now(),
  finished_at   timestamptz,
  discovered    int not null default 0,
  fetched       int not null default 0,
  extracted     int not null default 0,
  inserted      int not null default 0,
  skipped_dupe  int not null default 0,
  failed        int not null default 0,
  error         text
);
create index ingest_runs_source_started_idx on public.ingest_runs (source, started_at desc);

-- ------------------------------------------------------------------- caches
create table public.answer_cache (
  id                bigint generated always as identity primary key,
  query_hash        text        not null unique,   -- sha256 of normalized query (exact layer)
  query_text        text        not null,
  query_embedding   extensions.vector(1536) not null,  -- semantic layer
  answer            text        not null,
  citations         jsonb       not null default '[]',
  ingest_generation bigint      not null,
  created_at        timestamptz not null default now(),
  expires_at        timestamptz not null
);
create index answer_cache_embedding_hnsw_idx
  on public.answer_cache using hnsw (query_embedding extensions.vector_cosine_ops);
create index answer_cache_expires_at_idx on public.answer_cache (expires_at);

create table public.web_search_cache (
  query_hash  text        primary key,
  query_text  text        not null,
  results     jsonb       not null,
  created_at  timestamptz not null default now()
);

-- One row per UTC day; the app increments it before every Tavily call and refuses at the cap.
create table public.tavily_usage (
  day    date primary key,
  calls  int  not null default 0
);

-- -------------------------------------------------------------- security
-- Supabase exposes the public schema over its REST API. Enable RLS with no policies so the
-- anon/authenticated keys cannot read or write anything; the backend connects as the
-- database owner via SUPABASE_DB_URL, which bypasses RLS.
alter table public.articles         enable row level security;
alter table public.chunks           enable row level security;
alter table public.app_state        enable row level security;
alter table public.ingest_runs      enable row level security;
alter table public.answer_cache     enable row level security;
alter table public.web_search_cache enable row level security;
alter table public.tavily_usage     enable row level security;

revoke execute on function public.match_chunks(extensions.vector, int, boolean)
  from public, anon, authenticated;
