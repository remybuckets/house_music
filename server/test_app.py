"""
Verifies the HTTP layer WITHOUT Bedrock or a real corpus, by swapping in a
fake engine. Proves routing, Pydantic validation, and startup wiring.

    pip install fastapi httpx pytest   # then: pytest test_app.py
"""
import json

import pytest
from fastapi.testclient import TestClient

import app as app_module


class FakeEngine:
    size = 86

    @classmethod
    def load(cls, *a, **k):
        return cls()

    def chat(self, messages, k=3):
        # Echo the last question so we can assert it routed correctly.
        return f"(stub answer to: {messages[-1].content})", 0.812, 3

    def stream_chat(self, messages, k=3):
        # Mirror the real engine's event protocol: meta first, deltas, done.
        # Echo the question (split into tokens) so we can assert routing.
        yield {"type": "meta", "top_score": 0.812, "used_chunks": 3}
        for token in f"(stub answer to: {messages[-1].content})".split(" "):
            yield {"type": "delta", "text": token + " "}
        yield {"type": "done"}


# Swap the real engine for the fake BEFORE the app's lifespan runs.
app_module.RagEngine = FakeEngine


@pytest.fixture
def client():
    # Context-manager form is what actually triggers lifespan startup.
    with TestClient(app_module.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "chunks": 86}


def test_chat_happy_path(client):
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "who invented house?"}]})
    assert r.status_code == 200
    body = r.json()
    assert "who invented house?" in body["answer"]
    assert body["top_score"] == 0.812
    assert body["used_chunks"] == 3


def test_chat_rejects_bad_role(client):
    # Pydantic should reject a role outside user/assistant with a 422.
    r = client.post("/chat", json={"messages": [{"role": "wizard", "content": "hi"}]})
    assert r.status_code == 422


def test_chat_rejects_missing_messages(client):
    r = client.post("/chat", json={})
    assert r.status_code == 422


# --- /chat/stream (SSE) ---------------------------------------------------

def _parse_sse(text):
    """Pull the JSON payloads out of `data: {...}` SSE frames, in order."""
    return [
        json.loads(line[len("data: "):])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def test_stream_content_type(client):
    r = client.post("/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]


def test_stream_event_protocol(client):
    # meta first, done last, only deltas in between.
    r = client.post("/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]})
    events = _parse_sse(r.text)
    assert events[0]["type"] == "meta"
    assert events[-1]["type"] == "done"
    assert all(e["type"] == "delta" for e in events[1:-1])


def test_stream_meta_surfaces_retrieval_info(client):
    r = client.post("/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]})
    meta = _parse_sse(r.text)[0]
    assert meta["top_score"] == 0.812
    assert meta["used_chunks"] == 3


def test_stream_deltas_reassemble_and_route(client):
    # Concatenating the deltas should reproduce the (echoed) answer, proving
    # the last question routed through to the engine.
    r = client.post("/chat/stream", json={"messages": [{"role": "user", "content": "who invented house?"}]})
    deltas = [e["text"] for e in _parse_sse(r.text) if e["type"] == "delta"]
    assert "who invented house?" in "".join(deltas)


def test_stream_rejects_bad_role(client):
    # Same Pydantic model as /chat, so validation should reject before streaming.
    r = client.post("/chat/stream", json={"messages": [{"role": "wizard", "content": "hi"}]})
    assert r.status_code == 422