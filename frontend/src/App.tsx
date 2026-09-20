import { useEffect, useRef, useState } from 'react'
import { fetchStats, streamChat } from './api'
import { AnswerCard } from './components/AnswerCard'
import { Composer } from './components/Composer'
import { Progress } from './components/Progress'
import type { HistoryMessage, Stats, Turn } from './types'

const HISTORY_TURNS = 6

const SUGGESTIONS = [
  { text: 'Is it safe to travel to Iraq right now?', note: 'Answered from the curated index' },
  { text: "What's the best time to visit?", note: 'Asks a clarifying question' },
  { text: 'Do I need a visa to visit Vietnam as a US citizen?', note: 'Live web search and a visa note' },
  { text: 'Why are ultra-low-cost airlines struggling with jet fuel prices?', note: 'Airline industry news' },
]

interface ProgressState {
  done: string[]
  current: string | null
  /** The answer as it is being written. Replaced by the final text when the turn ends. */
  draft: string
}

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [progress, setProgress] = useState<ProgressState | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)
  const nextId = useRef(1)
  const abort = useRef<AbortController | null>(null)
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => { void fetchStats().then(setStats) }, [])
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [turns, progress])

  const busy = progress !== null

  const send = async (text: string) => {
    const history: HistoryMessage[] = turns
      .filter((t) => !(t.role === 'assistant' && t.error))
      .slice(-HISTORY_TURNS)
      .map((t) => ({ role: t.role, content: t.text }))
    setTurns((t) => [...t, { id: nextId.current++, role: 'user', text }])
    setProgress({ done: [], current: 'Reading your question', draft: '' })
    const controller = new AbortController()
    abort.current = controller

    const addAnswer = (turn: Omit<Extract<Turn, { role: 'assistant' }>, 'id' | 'role'>) =>
      setTurns((t) => [...t, { id: nextId.current++, role: 'assistant', ...turn }])
    try {
      await streamChat(text, history, {
        onStatus: (s) => setProgress((p) => ({ ...(p ?? { draft: '' }), done: [...(p?.done ?? []), ...s.summary], current: s.next })),
        onToken: (text) => setProgress((p) => (p ? { ...p, draft: p.draft + text } : p)),
        onRestart: () => setProgress((p) => (p ? { ...p, draft: '' } : p)),
        onResult: (result) => addAnswer({ text: result.answer, result }),
        onError: (message) => addAnswer({ text: message, error: true }),
      }, controller.signal)
    } catch (error) {
      if (!(error instanceof DOMException && error.name === 'AbortError')) {
        addAnswer({ text: 'I could not reach the server. Is the API running?', error: true })
      }
    } finally {
      setProgress(null)
      abort.current = null
    }
  }

  const lastQuestion = [...turns].reverse().find((t) => t.role === 'user')

  return (
    <div className="app">
      <header>
        <h1>Travel Assistant</h1>
        <button className="secondary" onClick={() => setTurns([])} disabled={busy || turns.length === 0}>
          New chat
        </button>
      </header>

      <main>
        {turns.length === 0 && !busy && (
          <section className="welcome">
            <p>Ask about travel news, entry advisories, flights and destinations. Answers cite their sources and are checked against them.</p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s.text} onClick={() => void send(s.text)}>
                  <span>{s.text}</span>
                  <small>{s.note}</small>
                </button>
              ))}
            </div>
          </section>
        )}

        {turns.map((turn) =>
          turn.role === 'user' ? (
            <div key={turn.id} className="bubble user">{turn.text}</div>
          ) : turn.error ? (
            <div key={turn.id} className="bubble error" role="alert">
              {turn.text}
              {lastQuestion && !busy && (
                <button className="secondary" onClick={() => void send(lastQuestion.text)}>Try again</button>
              )}
            </div>
          ) : (
            <AnswerCard key={turn.id} text={turn.text} result={turn.result} />
          ),
        )}

        {progress?.draft && <AnswerCard text={progress.draft} streaming />}
        {progress && <Progress done={progress.done} current={progress.current} />}
        <div ref={bottom} />
      </main>

      <footer>
        <Composer busy={busy} onSend={(t) => void send(t)} onStop={() => abort.current?.abort()} />
        {stats && (
          <p className="corpus">
            Index: {stats.curated_articles} curated articles
            {stats.web_articles > 0 && ` + ${stats.web_articles} recent web pages`} · {stats.chunks} passages
          </p>
        )}
      </footer>
    </div>
  )
}
