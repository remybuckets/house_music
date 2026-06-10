# HouseMusic.ai — web

Next.js (App Router) streaming chat UI for the HouseMusic.ai RAG backend.
It POSTs the conversation to the FastAPI `/chat/stream` endpoint and renders
the reply token-by-token as Server-Sent Events arrive.

## Run it

The backend must be running first (from the `server` folder):

```bash
# in server/  — with DATABASE_URL set and the corpus ingested
uvicorn app:app --reload      # serves on http://localhost:8000
```

Then the frontend:

```bash
# in web/
npm install
npm run dev                   # serves on http://localhost:3000
```

Open http://localhost:3000.

## How the pieces fit

- `app/page.tsx` — the chat screen. Holds the conversation in React state,
  shows the user + assistant turns, and streams the reply in live. The
  pulsing equalizer is the "now playing / thinking" indicator.
- `app/lib/chat.ts` — the streaming client. `streamChat()` POSTs to
  `/chat/stream` and invokes `onMeta` / `onDelta` / `onDone` as frames arrive.
  `parseSSEChunk()` parses the `data: {json}\n\n` frames and correctly handles
  a frame split across two network reads.
- The request body is `{ messages: [{role, content}, ...] }` — the full
  conversation, oldest first — exactly the stateless shape the backend expects.

## Config

The API base URL defaults to `http://localhost:8000`. Override it with an
environment variable when the backend lives elsewhere:

```bash
NEXT_PUBLIC_API_BASE=https://api.example.com npm run dev
```

## CORS

Because the browser calls the FastAPI server directly (cross-origin), the
backend enables CORS for `http://localhost:3000`. To allow other origins, set
`CORS_ORIGINS` (comma-separated) in the backend environment.

## Styling (Tailwind v4)

This uses Tailwind CSS v4, which has no `tailwind.config` file — the theme
(colors, fonts, animations) is defined in a `@theme` block at the top of
`app/globals.css`, and PostCSS loads the `@tailwindcss/postcss` plugin. To add
or change a color, edit the `--color-*` variables in `globals.css`; the
matching `bg-*` / `text-*` utilities are generated automatically.

## Fonts

Fonts (Space Grotesk, Inter) load via a `<link>` in `app/layout.tsx` with a
system-font fallback in `globals.css`, so the build never fails when Google
Fonts is unreachable. Self-host them for a fully offline production build.