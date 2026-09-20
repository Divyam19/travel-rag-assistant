import { useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'

const MAX_LENGTH = 1000

interface Props {
  busy: boolean
  onSend: (text: string) => void
  onStop: () => void
}

export function Composer({ busy, onSend, onStop }: Props) {
  const [text, setText] = useState('')

  const submit = (event?: FormEvent) => {
    event?.preventDefault()
    const trimmed = text.trim()
    if (!trimmed || busy) return
    onSend(trimmed)
    setText('')
  }

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <form className="composer" onSubmit={submit}>
      <label htmlFor="message" className="sr-only">Ask about travel</label>
      <textarea
        id="message"
        value={text}
        maxLength={MAX_LENGTH}
        rows={1}
        placeholder="Ask about travel news, advisories, flights, destinations…"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        autoFocus
      />
      {busy ? (
        <button type="button" className="secondary" onClick={onStop}>Stop</button>
      ) : (
        <button type="submit" disabled={!text.trim()}>Send</button>
      )}
    </form>
  )
}
