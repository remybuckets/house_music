# HouseMusic.ai

A retrieval-augmented (RAG) chatbot that answers questions about house music —
its history, clubs, and DJs — grounded in a small Wikipedia corpus.

- **`server/`** — Python FastAPI backend. Embeds questions with Amazon Titan,
  retrieves the closest corpus chunks, and answers with Claude Haiku via AWS
  Bedrock. Replies stream back as Server-Sent Events.
- **`client/`** — Next.js (App Router) chat UI that renders the streamed reply
  token by token.

```
house_music/
├── client/                 # Next.js 16 + React 19 + Tailwind v4 front end
│   └── app/
│       ├── page.tsx        # chat screen
│       └── lib/chat.ts     # SSE streaming client for /chat/stream
└── server/                 # FastAPI + Bedrock RAG backend
    ├── app.py              # HTTP API (/health, /chat, /chat/stream)
    ├── engine.py           # RagEngine: retrieval + generation
    ├── vector_store.py     # InMemoryStore (pickle) / PgVectorStore (Postgres)
    ├── ingest.py           # embed corpus into Postgres/pgvector
    ├── corpus_cache.pkl    # pre-embedded corpus for the no-database mode
    ├── docker-compose.yml  # local Postgres + pgvector
    └── step*.py            # earlier standalone pipeline scripts (see below)
```

## Prerequisites

- Python 3.11+
- Node.js 20+
- AWS credentials with Bedrock access in `us-east-1` to
  `amazon.titan-embed-text-v2:0` and Claude Haiku 4.5
  (`aws configure` or the usual `AWS_*` environment variables)
- Docker (optional, only for the Postgres/pgvector mode)

## Quick start

### 1. Backend

```bash
cd server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload          # http://localhost:8000
```

If `DATABASE_URL` is not set, the server loads the bundled `corpus_cache.pkl`
into memory, so no database is needed.

### 2. Frontend

In a second terminal:

```bash
cd client
npm install
npm run dev                       # http://localhost:3000
```

Open http://localhost:3000 and ask something like *"Who was Frankie Knuckles?"*

## Using Postgres + pgvector (optional)

For a larger corpus, keep the vectors in Postgres instead of the pickle:

```bash
cd server
docker compose up -d
export DATABASE_URL=postgresql://house:house@localhost:5432/housemusic
python ingest.py                  # embeds new/changed chunks; safe to re-run
uvicorn app:app --reload          # now retrieves from pgvector
```

Re-running `ingest.py` skips chunks that are already in the database (they are
matched by content hash) and removes chunks that are no longer in the corpus.

## Configuration

| Variable               | Where  | Default                  | Purpose                                     |
| ---------------------- | ------ | ------------------------ | ------------------------------------------- |
| `DATABASE_URL`         | server | *(unset)*                | Use pgvector instead of `corpus_cache.pkl`  |
| `CORS_ORIGINS`         | server | `http://localhost:3000`  | Comma-separated origins allowed to call API |
| `NEXT_PUBLIC_API_BASE` | client | `http://localhost:8000`  | Backend URL used by the browser             |

## API

| Method | Path           | Description                                   |
| ------ | -------------- | --------------------------------------------- |
| GET    | `/health`      | `{"status":"ok","chunks":N}`                  |
| POST   | `/chat`        | Full answer as JSON                           |
| POST   | `/chat/stream` | Answer as Server-Sent Events                  |

The request body for both chat endpoints is the whole conversation, oldest
message first. The server is stateless:

```json
{ "messages": [{ "role": "user", "content": "Who invented house music?" }] }
```

`/chat/stream` sends `data: {json}\n\n` frames in this order: one
`{"type":"meta","top_score":…,"used_chunks":…}`, then any number of
`{"type":"delta","text":"…"}`, then `{"type":"done"}`.

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Who invented house music?"}]}'
```

## Tests

```bash
cd server && pytest               # no AWS or database needed (fakes are used)
cd client && npm run lint
```

## Rebuilding the corpus

The corpus comes from the Wikipedia pages listed in `TOPICS` in
`server/step1_corpus.py`. After changing that list:

```bash
cd server
python step1_corpus.py            # check that every topic has enough chunks
python step2_embed.py             # re-embed into corpus_cache.pkl (in-memory mode)
python ingest.py                  # or re-ingest into Postgres (pgvector mode)
```

`step2_retrieve.py` (inspect retrieval scores) and `step2c_chat.py` (a terminal
chat REPL) are earlier standalone versions of the pipeline. They're still handy
for debugging retrieval without the web stack.

## Front-end notes

- **Styling:** Tailwind CSS v4 has no `tailwind.config` file. Theme colors,
  fonts and animations are defined in the `@theme` block in
  `client/app/globals.css`.
- **Fonts:** Space Grotesk and Inter load via a `<link>` in
  `client/app/layout.tsx` and fall back to system fonts, so the build still
  succeeds when Google Fonts is unreachable.
