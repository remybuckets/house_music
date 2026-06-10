"""
HouseMusic.ai - Step 3a: FastAPI backend.

Loads the RAG engine once at startup and exposes it over HTTP. Stateless:
the client sends the full conversation each request (the same shape a Next.js
chat front end will hold in state), so the server keeps no per-user memory.

    pip install fastapi uvicorn boto3 numpy
    uvicorn app:app --reload

    curl -s localhost:8000/health
    curl -s -X POST localhost:8000/chat \
      -H 'content-type: application/json' \
      -d '{"messages":[{"role":"user","content":"Who invented house music?"}]}'
"""
from contextlib import asynccontextmanager
from typing import Literal
import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from engine import RagEngine

engine: RagEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the engine ONCE, not per request. Backend is chosen by
    # environment: if DATABASE_URL is set, retrieve from Postgres/pgvector;
    # otherwise fall back to the legacy in-memory pickle cache.
    global engine
    if os.environ.get("DATABASE_URL"):
        engine = RagEngine.load_pg()
    else:
        engine = RagEngine.load()
    yield
    # (Shutdown cleanup would go here.)


app = FastAPI(title="HouseMusic.ai", lifespan=lifespan)

# Allow the Next.js dev server (localhost:3000) to call this API directly from
# the browser. Override with CORS_ORIGINS (comma-separated) in other envs.
_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class ChatResponse(BaseModel):
    answer: str
    top_score: float
    used_chunks: int


@app.get("/health")
def health():
    return {"status": "ok", "chunks": engine.size if engine else 0}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer, top_score, n = engine.chat(req.messages)
    return ChatResponse(answer=answer, top_score=top_score, used_chunks=n)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Streaming counterpart of /chat, as Server-Sent Events. Each line is

        data: {json}\n\n

    where {json} is one of the events yielded by RagEngine.stream_chat:
      - {"type":"meta","top_score":...,"used_chunks":...}  (first)
      - {"type":"delta","text":"..."}                      (zero or more)
      - {"type":"done"}                                    (last)

    A Next.js front end reads these with EventSource / fetch + ReadableStream,
    appending each delta's text to the live answer.
    """

    def event_stream():
        for event in engine.stream_chat(req.messages):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )