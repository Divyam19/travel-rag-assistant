import json

import pytest
from fastapi.testclient import TestClient

from travelrag import api
from travelrag.agent import AgentResult
from travelrag.evaluation import EvalResult
from travelrag.ratelimit import RateLimiter
from travelrag.rag import Source

client = TestClient(api.app)  # no `with`: skips the lifespan, so no database connection is opened


@pytest.fixture(autouse=True)
def fresh_limiter(monkeypatch):
    """Every test gets its own generous limiter, so the shared module-level one never leaks state
    between tests and a growing suite cannot trip the per-minute limit by accident."""
    monkeypatch.setattr(api, "limiter", RateLimiter(per_minute=1000, per_day=100000))


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def fake_result() -> AgentResult:
    source = Source(1, "Qatar - Level 3", "https://x.test/q", "statedept", None, 0.7123, "text", "feed", 5)
    return AgentResult("Qatar is Level 3 [1].", "index", [source], ["analyze: ok"], 100, 20, 0,
                       EvalResult("pass", 1, 0.81))


def test_chat_streams_status_then_result(monkeypatch):
    def fake_stream(question, history, run_label):
        assert question == "Is Qatar safe?" and history == [{"role": "user", "content": "hi"}]
        yield "node", "analyze", {"trace": ["analyze: searching"]}
        yield "node", "assess", {"confident": True, "trace": ["assess: confident"]}
        yield "result", fake_result()

    monkeypatch.setattr(api, "stream_agent", fake_stream)
    response = client.post("/api/chat", json={"message": "Is Qatar safe?", "history": [{"role": "user", "content": "hi"}]})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [e for e, _ in events] == ["status", "status", "status", "result"]
    assert events[1][1]["next"] == "Searching the travel index"
    assert events[2][1]["next"] == "Writing the answer"
    result = events[3][1]
    assert result["path"] == "index" and result["sources"][0]["similarity"] == 0.712
    # The browser gets the model's own field names, and no chunk text.
    assert set(result["sources"][0]) == {"n", "title", "url", "source", "published_at", "similarity", "source_type"}
    assert result["evaluation"]["verdict"] == "pass"


def test_clarifying_turn_has_no_next_step(monkeypatch):
    monkeypatch.setattr(api, "stream_agent", lambda *a, **k: iter([
        ("node", "analyze", {"path": "clarify", "trace": ["analyze: needs clarification"]}),
        ("result", AgentResult("Which destination?", "clarify", [], [], 10, 5, 0))]))
    events = parse_sse(client.post("/api/chat", json={"message": "best time to visit?"}).text)
    assert events[1][1]["next"] is None
    assert events[2][1]["evaluation"] is None


def test_failure_becomes_a_generic_error_event(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret database password in message")
        yield

    monkeypatch.setattr(api, "stream_agent", boom)
    events = parse_sse(client.post("/api/chat", json={"message": "hello"}).text)
    assert events[-1][0] == "error"
    assert "secret" not in json.dumps(events[-1][1])


@pytest.mark.parametrize("body", [
    {"message": ""},
    {"message": "x" * 1001},
    {"message": "hi", "history": [{"role": "system", "content": "x"}]},
    {"message": "hi", "history": [{"role": "user", "content": "x"}] * 13},
    {"message": "hi", "history": [{"role": "user", "content": "x" * 4001}]},
])
def test_invalid_requests_are_rejected(body):
    assert client.post("/api/chat", json=body).status_code == 422


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_answer_tokens_are_streamed_and_a_rejected_draft_is_restarted(monkeypatch):
    monkeypatch.setattr(api, "stream_agent", lambda *a, **k: iter([
        ("token", "Qatar is "),
        ("token", "Level 4."),
        ("node", "evaluate", {"retry": True, "trace": ["evaluate: flagged"]}),
        ("token", "Qatar is Level 3."),
        ("node", "evaluate", {"trace": ["evaluate: pass"]}),
        ("result", fake_result()),
    ]))
    events = parse_sse(client.post("/api/chat", json={"message": "Qatar?"}).text)
    kinds = [e for e, _ in events]
    assert kinds.count("token") == 3
    assert "restart" in kinds
    # The restart must arrive before the replacement text, or the client would concatenate drafts.
    assert kinds.index("restart") < len(kinds) - kinds[::-1].index("token") - 1
    assert events[-1][0] == "result"
    assert events[-1][1]["answer"] == "Qatar is Level 3 [1]."


def test_source_round_trips_through_its_dict_form():
    from datetime import datetime
    original = Source(2, "T", "https://x.test", "site", datetime(2026, 9, 18, 10, 30), 0.71234, "body text", "web", 42)
    restored = Source.from_dict(original.to_dict())
    assert restored == Source(2, "T", "https://x.test", "site", datetime(2026, 9, 18, 10, 30), 0.712, "body text", "web", 42)
    assert "content" not in original.to_dict(include_text=False) and "chunk_id" not in original.to_dict(include_text=False)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_a_client_is_limited_per_minute_and_recovers_when_the_window_passes():
    from travelrag.ratelimit import RateLimiter
    clock = FakeClock()
    limiter = RateLimiter(per_minute=3, per_day=100, clock=clock)
    assert [limiter.check("a")[0] for _ in range(4)] == [True, True, True, False]
    allowed, wait, reason = limiter.check("a")
    assert not allowed and 1 <= wait <= 61 and "too quickly" in reason
    clock.now += 61
    assert limiter.check("a")[0]


def test_clients_are_limited_independently():
    from travelrag.ratelimit import RateLimiter
    limiter = RateLimiter(per_minute=1, per_day=100, clock=FakeClock())
    assert limiter.check("a")[0] and not limiter.check("a")[0]
    assert limiter.check("b")[0]


def test_the_daily_cap_applies_across_clients_and_resets_the_next_day():
    from travelrag.ratelimit import RateLimiter
    day = ["2026-09-20"]
    limiter = RateLimiter(per_minute=100, per_day=2, clock=FakeClock(), today=lambda: day[0])
    assert limiter.check("a")[0] and limiter.check("b")[0]
    allowed, _, reason = limiter.check("c")
    assert not allowed and "daily" in reason
    day[0] = "2026-09-21"
    assert limiter.check("c")[0]


def test_a_rate_limited_request_gets_429_with_retry_after_and_never_reaches_the_agent(monkeypatch):
    monkeypatch.setattr(api.limiter, "check", lambda client: (False, 17, "Slow down."))
    called = []
    monkeypatch.setattr(api, "stream_agent", lambda *a, **k: called.append(1) or iter([]))
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "17" and response.json() == {"detail": "Slow down."}
    assert called == []  # no OpenAI or Tavily spend
