import { useState } from 'react'
import Markdown from 'react-markdown'
import type { ChatResult } from '../types'

/** Describe where the answer came from. Web pages saved by an earlier search live in the index too. */
function originLabel(result: ChatResult): string {
  switch (result.path) {
    case 'web':
      return 'From a web search'
    case 'index': {
      const web = result.sources.filter((s) => s.source_type === 'web').length
      if (web === 0) return 'From the travel index'
      return web === result.sources.length ? 'From saved web pages' : 'From the index and saved web pages'
    }
    case 'clarify':
      return 'Needs one more detail'
    case 'off_topic':
      return 'Outside my topic'
    case 'not_found':
      return 'Nothing reliable found'
  }
}

const VERDICT_LABELS = {
  pass: 'Checked against sources',
  disclaimer: 'Checked, with a legal note',
  flagged: 'Partly unverified',
} as const

/** Turn "[1]" citations into links to the source cards; leave real markdown links alone. */
function linkCitations(text: string): string {
  return text.replace(/\[(\d+)\](?!\()/g, '[[$1]](#source-$1)')
}

export function AnswerCard({ result, text, streaming }: { result?: ChatResult; text: string; streaming?: boolean }) {
  const [active, setActive] = useState<number | null>(null)
  const sources = result?.sources ?? []
  const evaluation = result?.evaluation

  return (
    <article className={streaming ? 'answer streaming' : 'answer'} aria-busy={streaming || undefined}>
      {result && (
        <div className="badges">
          <span className={`badge path-${result.path}`}>{originLabel(result)}</span>
          {evaluation && (
            <span className={`badge verdict-${evaluation.verdict}`} title={`Tier ${evaluation.tier} check`}>
              {VERDICT_LABELS[evaluation.verdict]}
            </span>
          )}
        </div>
      )}

      <div className="prose">
        <Markdown
          components={{
            a({ href, children }) {
              const match = href?.match(/^#source-(\d+)$/)
              if (match) {
                const n = Number(match[1])
                return (
                  <button className="cite" onClick={() => setActive(n)} aria-label={`Show source ${n}`}>
                    {n}
                  </button>
                )
              }
              return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
            },
            p({ children }) {
              const first = Array.isArray(children) ? children[0] : children
              const isNote = typeof first === 'string' && first.startsWith('Note:')
              return <p className={isNote ? 'note' : undefined}>{children}</p>
            },
          }}
        >
          {linkCitations(text)}
        </Markdown>
      </div>

      {sources.length > 0 && (
        <ol className="sources" aria-label="Sources">
          {sources.map((s) => (
            <li key={s.n} className={active === s.n ? 'source active' : 'source'}>
              <span className="source-n">{s.n}</span>
              <div>
                <a href={s.url} target="_blank" rel="noopener noreferrer">{s.title}</a>
                <div className="source-meta">
                  <span className={`chip origin-${s.source_type}`}>{s.source_type === 'web' ? 'Web' : 'Index'}</span>
                  {s.source}
                  {s.published_at && ` · ${new Date(s.published_at).toLocaleDateString()}`}
                  {` · match ${s.similarity.toFixed(2)}`}
                </div>
              </div>
            </li>
          ))}
        </ol>
      )}

      {result && (
        <details className="trace">
          <summary>How this was answered</summary>
          <ul>
            {result.trace.map((line, i) => <li key={i}>{line}</li>)}
          </ul>
          {evaluation && evaluation.problems.length > 0 && (
            <p className="trace-problems">Flagged by the checker: {evaluation.problems.join(' · ')}</p>
          )}
          <p className="trace-stats">
            {result.elapsed_seconds}s · {result.usage.prompt_tokens + result.usage.completion_tokens} tokens ·{' '}
            {result.usage.web_calls} live web search{result.usage.web_calls === 1 ? '' : 'es'}
          </p>
        </details>
      )}
    </article>
  )
}
