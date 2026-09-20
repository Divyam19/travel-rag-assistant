-- Per-call token accounting, so cost claims can be backed by measured numbers.
-- run_label groups calls into an experiment (e.g. 'baseline', 'optimized').
create table public.llm_calls (
  id                bigint generated always as identity primary key,
  created_at        timestamptz not null default now(),
  run_label         text        not null,
  query_id          text,
  purpose           text        not null,   -- 'answer', 'embed_query', later 'route', 'judge', ...
  model             text        not null,
  prompt_tokens     int         not null default 0,   -- embeddings: total tokens
  completion_tokens int         not null default 0
);
create index llm_calls_run_label_idx on public.llm_calls (run_label, created_at);

alter table public.llm_calls enable row level security;
