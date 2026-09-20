export function Progress({ done, current }: { done: string[]; current: string | null }) {
  return (
    <div className="progress" role="status" aria-live="polite">
      <ul>
        {done.map((line, i) => <li key={i} className="done">{line}</li>)}
        {current && <li className="current"><span className="spinner" aria-hidden="true" />{current}…</li>}
      </ul>
    </div>
  )
}
