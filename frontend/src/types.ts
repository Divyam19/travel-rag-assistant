export type Path = 'index' | 'web' | 'clarify' | 'off_topic' | 'not_found'

export interface Source {
  n: number
  title: string
  url: string
  site: string
  origin: 'feed' | 'web'
  published_at: string | null
  similarity: number
}

export interface Evaluation {
  verdict: 'pass' | 'disclaimer' | 'flagged'
  tier: number
  legal: boolean
  min_similarity: number
  problems: string[]
}

export interface ChatResult {
  answer: string
  path: Path
  sources: Source[]
  evaluation: Evaluation | null
  trace: string[]
  usage: { prompt_tokens: number; completion_tokens: number; web_calls: number }
  elapsed_seconds: number
}

export interface StatusEvent {
  node: string
  summary: string[]
  next: string | null
}

export interface Stats {
  articles: number
  curated_articles: number
  web_articles: number
  chunks: number
  last_ingest: string | null
}

export interface HistoryMessage {
  role: 'user' | 'assistant'
  content: string
}

export type Turn =
  | { id: number; role: 'user'; text: string }
  | { id: number; role: 'assistant'; text: string; result?: ChatResult; error?: boolean }
