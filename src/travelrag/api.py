"""HTTP API for the chat UI. Run with: uvicorn travelrag.api:app

POST /api/chat streams server-sent events:
  `status`  the agent finished a step, and what it is doing next
  `token`   a piece of the answer, as the model writes it
  `restart` discard the answer so far: the checker rejected the draft and it is being rewritten
  `result`  the finished answer with sources, verdict and usage (or `error`)
The answer in `result` is authoritative: the checker may append a disclaimer after the last token.
The client sends recent history each time; the server keeps no session state.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Iterator, Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import AgentResult, stream_agent
from .config import ROOT, get_settings
from .db import _get_pool, connect
from .ratelimit import RateLimiter

log = logging.getLogger("travelrag.api")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[Message] = Field(default_factory=list, max_length=12)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _get_pool()  # open the first database connection now so the first chat turn is not slow
    yield


app = FastAPI(title="Travel RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=[o.strip() for o in get_settings().cors_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)


limiter = RateLimiter(get_settings().rate_limit_per_minute, get_settings().daily_chat_cap)


def client_address(peer: str | None, forwarded_for: str | None, hops: int) -> str:
    """The address to rate-limit on.

    Reached directly (hops=0) the peer is the client. Behind `hops` trusted reverse proxies the peer
    is the proxy, and every visitor would share one bucket, so use X-Forwarded-For instead. Each
    proxy appends the address it saw, so the entry `hops` places from the RIGHT is the one our own
    proxy recorded and cannot be forged. The left-hand entries are whatever the client sent, which
    is why the leftmost value must never be used.
    """
    if hops > 0 and forwarded_for:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if len(parts) >= hops:
            return parts[-hops]
    return peer or "unknown"


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def next_step(node: str, update: dict) -> str | None:
    """What the agent does after `node`, for the progress line. None when the turn is over."""
    if node == "analyze":
        return None if update.get("path") else "Searching the travel index"
    if node == "retrieve":
        return "Checking whether the index answers this"
    if node == "assess":
        return "Writing the answer" if update.get("confident") else "Searching the web"
    if node == "web_fallback":
        return None if update.get("path") == "not_found" else "Writing the answer"
    if node == "generate":
        return "Checking the answer against its sources"
    if node == "evaluate":
        return "Rewriting more strictly" if update.get("retry") else None
    return None


def result_payload(result: AgentResult, elapsed: float) -> dict:
    ev = result.evaluation
    return {
        "answer": result.answer,
        "path": result.path,
        "sources": [s.to_dict(include_text=False) for s in result.sources],
        "evaluation": None if ev is None else {"verdict": ev.verdict, "tier": ev.tier, "legal": ev.legal,
                                               "min_similarity": round(ev.min_sim, 3), "problems": ev.problems[:3]},
        "trace": result.trace,
        "usage": {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                  "web_calls": result.web_calls},
        "elapsed_seconds": round(elapsed, 1),
    }


def chat_events(request: ChatRequest) -> Iterator[str]:
    started = time.time()
    history = [m.model_dump() for m in request.history]
    yield sse("status", {"node": "start", "summary": [], "next": "Reading your question"})
    try:
        for kind, *rest in stream_agent(request.message, history, run_label="api"):
            if kind == "node":
                node, update = rest
                if node == "evaluate" and update.get("retry"):
                    yield sse("restart", {})  # the streamed draft is being thrown away
                yield sse("status", {"node": node, "summary": update.get("trace", []), "next": next_step(node, update)})
            elif kind == "token":
                yield sse("token", {"text": rest[0]})
            else:
                yield sse("result", result_payload(rest[0], time.time() - started))
    except Exception:  # noqa: BLE001 - show the user a generic message, keep details in the server log
        log.exception("chat turn failed")
        yield sse("error", {"message": "Something went wrong while answering. Please try again."})


@app.post("/api/chat")
def chat(request: ChatRequest, http: Request):
    address = client_address(http.client.host if http.client else None,
                             http.headers.get("x-forwarded-for"), get_settings().trusted_proxy_hops)
    allowed, wait, reason = limiter.check(address)
    if not allowed:
        return JSONResponse({"detail": reason}, status_code=429, headers={"Retry-After": str(wait)})
    return StreamingResponse(chat_events(request), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/stats")
def stats() -> dict:
    """Corpus size for the UI footer."""
    with connect() as conn:
        by_type = dict(conn.execute("select source_type, count(*) from articles group by 1").fetchall())
        chunks = conn.execute("select count(*) from chunks").fetchone()[0]
        last = conn.execute("select max(finished_at) from ingest_runs").fetchone()[0]
    return {"articles": sum(by_type.values()), "curated_articles": by_type.get("feed", 0),
            "web_articles": by_type.get("web", 0), "chunks": chunks, "last_ingest": last}


FRONTEND_DIST = ROOT / "frontend" / "dist"


def mount_frontend(application: FastAPI, dist=FRONTEND_DIST) -> bool:
    """Serve the built UI from the same origin as the API, so a deployment is one service and the
    browser needs no CORS. Registered last so every /api route wins. Returns whether it mounted;
    in development the Vite server serves the UI and there is no dist folder."""
    if not (dist / "index.html").is_file():
        return False
    application.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return True


mount_frontend(app)
