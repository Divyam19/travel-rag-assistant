import type { ChatResult, HistoryMessage, Stats, StatusEvent } from './types'

interface Handlers {
  onStatus: (status: StatusEvent) => void
  onToken: (text: string) => void
  onRestart: () => void
  onResult: (result: ChatResult) => void
  onError: (message: string) => void
}

const GENERIC_ERROR = 'Something went wrong while answering. Please try again.'

/** POST a chat turn and dispatch the server-sent events as they arrive. */
export async function streamChat(
  message: string,
  history: HistoryMessage[],
  handlers: Handlers,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history }),
    signal,
  })
  if (response.status === 429) {
    const { detail } = await response.json().catch(() => ({ detail: '' }))
    handlers.onError(detail || 'Too many requests. Please wait a moment and try again.')
    return
  }
  if (!response.ok || !response.body) {
    handlers.onError(response.status === 422 ? 'That message is too long or malformed.' : GENERIC_ERROR)
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const blocks = buffer.split('\n\n')
    buffer = blocks.pop() ?? ''
    for (const block of blocks) dispatch(block, handlers)
  }
}

function dispatch(block: string, handlers: Handlers): void {
  let event = ''
  let data = ''
  for (const line of block.split('\n')) {
    if (line.startsWith('event: ')) event = line.slice(7)
    else if (line.startsWith('data: ')) data += line.slice(6)
  }
  if (!event || !data) return
  const payload = JSON.parse(data)
  if (event === 'status') handlers.onStatus(payload)
  else if (event === 'token') handlers.onToken(payload.text ?? '')
  else if (event === 'restart') handlers.onRestart()
  else if (event === 'result') handlers.onResult(payload)
  else if (event === 'error') handlers.onError(payload.message ?? GENERIC_ERROR)
}

export async function fetchStats(): Promise<Stats | null> {
  try {
    const response = await fetch('/api/stats')
    return response.ok ? await response.json() : null
  } catch {
    return null
  }
}
