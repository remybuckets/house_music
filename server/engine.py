"""
HouseMusic.ai - the RAG engine, decoupled from any interface AND from storage.

This is the production-grade refactor: all the retrieval + generation logic
that used to live inside the step2 scripts now lives here as a class with no
knowledge of REPLs, HTTP, or anything else. The FastAPI app (app.py) imports
it; a CLI could import it; tests can import it. One brain, many front doors.

Storage is now behind a VectorStore (see vector_store.py): the engine holds a
store and asks it for nearest chunks, so the backend can be the in-memory
NumPy scan (legacy pickle) or pgvector without the engine changing. Two entry
points:
    RagEngine.load()     - legacy: read corpus_cache.pkl into an InMemoryStore
    RagEngine.load_pg()  - production: connect to Postgres/pgvector
"""
import json
import pickle

import boto3
import numpy as np

from vector_store import InMemoryStore, PgVectorStore

EMBED_MODEL = "amazon.titan-embed-text-v2:0"
CHAT_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
CACHE_PATH = "corpus_cache.pkl"

SYSTEM = (
    "You are HouseMusic.ai, a friendly and knowledgeable guide to house music - "
    "its history, its clubs, and its DJs. Answer using ONLY the context provided "
    "with each question. If the context does not contain the answer, say you "
    "don't have that in your knowledge yet rather than inventing facts. Keep "
    "answers conversational and concise."
)


class RagEngine:
    def __init__(self, store, bedrock):
        self.store = store
        self.bedrock = bedrock

    @classmethod
    def load(cls, cache_path=CACHE_PATH, region="us-east-1"):
        """
        Legacy path: load the cached corpus into an in-memory store + a Bedrock
        client. Call this ONCE at startup. Kept so the pickle workflow and the
        existing tests keep working unchanged.
        """
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        bedrock = boto3.client("bedrock-runtime", region_name=region)
        store = InMemoryStore(cache["docs"], cache["vectors"])
        return cls(store, bedrock)

    @classmethod
    def load_pg(cls, dsn=None, region="us-east-1"):
        """
        Production path: back retrieval with Postgres/pgvector. Corpus must
        already be ingested (see ingest.py). dsn defaults to $DATABASE_URL.
        """
        bedrock = boto3.client("bedrock-runtime", region_name=region)
        return cls(PgVectorStore(dsn), bedrock)

    @property
    def size(self):
        return self.store.size

    def embed(self, text):
        resp = self.bedrock.invoke_model(
            modelId=EMBED_MODEL,
            body=json.dumps({"inputText": text}),
        )
        return np.array(json.loads(resp["body"].read())["embedding"], dtype=np.float32)

    def retrieve(self, query, k=3):
        """Top-k chunks by cosine similarity, as (score, chunk) pairs.

        Embedding still happens here (one Bedrock call per query); the nearest-
        neighbour search is delegated to the store, so the same code works for
        the in-memory scan and for pgvector.
        """
        q = self.embed(query)
        return self.store.search(q, k)

    def chat(self, messages, k=3):
        """
        messages: list of objects with .role ('user'/'assistant') and .content (str),
        oldest first, last one being the new user question.

        Returns (answer, top_score, n_chunks). top_score is surfaced so callers
        can later gate on it (e.g. refuse below ~0.45) - not enforced here yet.
        """
        convo, top_score, n = self._build_convo(messages, k)

        resp = self.bedrock.converse(
            modelId=CHAT_MODEL,
            system=[{"text": SYSTEM}],
            messages=convo,
            inferenceConfig={"maxTokens": 400, "temperature": 0.2},
        )
        answer = resp["output"]["message"]["content"][0]["text"]
        return answer, top_score, n

    def _build_convo(self, messages, k):
        """Shared retrieval + prompt assembly for chat() and stream_chat()."""
        query = messages[-1].content
        hits = self.retrieve(query, k=k)
        context = "\n\n".join(chunk for _, chunk in hits)
        # Prior turns pass through as-is; context is injected only into the
        # current turn, keeping history lean (same pattern as the REPL).
        convo = [{"role": m.role, "content": [{"text": m.content}]} for m in messages[:-1]]
        convo.append(
            {"role": "user", "content": [{"text": f"Context:\n{context}\n\nQuestion: {query}"}]}
        )
        top_score = hits[0][0] if hits else 0.0
        return convo, top_score, len(hits)

    def stream_chat(self, messages, k=3):
        """
        Streaming counterpart of chat(). Same retrieval + grounding prompt, but
        uses Bedrock `converse_stream` and yields events as they arrive:

            {"type": "meta",  "top_score": float, "used_chunks": int}   # once, first
            {"type": "delta", "text": str}                              # zero or more
            {"type": "done"}                                            # once, last

        The meta event is emitted before any token so callers (e.g. an SSE
        endpoint) can surface retrieval info up front, mirroring the
        (answer, top_score, n_chunks) tuple that chat() returns.
        """
        convo, top_score, n = self._build_convo(messages, k)
        yield {"type": "meta", "top_score": top_score, "used_chunks": n}

        resp = self.bedrock.converse_stream(
            modelId=CHAT_MODEL,
            system=[{"text": SYSTEM}],
            messages=convo,
            inferenceConfig={"maxTokens": 400, "temperature": 0.2},
        )
        for event in resp["stream"]:
            block = event.get("contentBlockDelta")
            if block:
                text = block["delta"].get("text", "")
                if text:
                    yield {"type": "delta", "text": text}
        yield {"type": "done"}