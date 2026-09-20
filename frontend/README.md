# Travel Assistant UI

React + TypeScript + Vite chat interface for the Travel RAG backend.

    make web-install   # once
    make api           # terminal 1: FastAPI on :8000
    make web           # terminal 2: UI on http://localhost:5173

The dev server proxies `/api` to the backend (`vite.config.ts`), so there is no CORS setup to do.

- `src/api.ts` posts a chat turn and parses the server-sent events (`status` while the agent works,
  then one `result` or `error`).
- `src/components/AnswerCard.tsx` renders the answer as markdown, turns `[1]` citations into links to
  the source cards, and shows where the answer came from and how it was checked.
- `src/components/Progress.tsx` shows the live steps of the current turn.

The server keeps no session: the UI sends its last six messages with each request.
