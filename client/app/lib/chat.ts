/**
 * Streaming client for the FastAPI /chat/stream endpoint.
 *
 * The backend speaks Server-Sent Events: each frame is `data: {json}\n\n`,
 * where the JSON is one of (matching RagEngine.stream_chat):
 *
 *   { type: "meta",  top_score: number, used_chunks: number }   // once, first
 *   { type: "delta", text: string }                             // zero or more
 *   { type: "done" }                                            // once, last
 *
 * EventSource can't POST, so we read the body stream with fetch + a reader and
 * parse frames by hand. Callbacks fire as events arrive, so the UI can render
 * tokens live.
 */
export type StreamMeta = { type: "meta"; top_score: number; used_chunks: number };
export type StreamDelta = { type: "delta"; text: string };
export type StreamDone = { type: "done" };
export type StreamEvent = StreamMeta | StreamDelta | StreamDone;

export type ChatMessage = { role: "user" | "assistant"; content: string };

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type StreamHandlers = {
  onMeta?: (meta: StreamMeta) => void;
  onDelta?: (text: string) => void;
  onDone?: () => void;
};

/**
 * Parse a buffer of SSE text into events, returning the events found plus any
 * trailing partial frame that hasn't been terminated by a blank line yet.
 * Exported so it can be unit-tested without a network.
 */
export function parseSSEChunk(buffer: string): {
  events: StreamEvent[];
  rest: string;
} {
  const events: StreamEvent[] = [];
  // Frames are separated by a blank line.
  const parts = buffer.split("\n\n");
  const rest = parts.pop() ?? ""; // last element may be an incomplete frame
  for (const part of parts) {
    const line = part.split("\n").find((l) => l.startsWith("data: "));
    if (!line) continue;
    const json = line.slice("data: ".length);
    try {
      events.push(JSON.parse(json) as StreamEvent);
    } catch {
      // Ignore malformed frames rather than killing the stream.
    }
  }
  return { events, rest };
}

/**
 * Send the conversation and stream the reply. Resolves when the stream ends.
 * Pass an AbortSignal to cancel an in-flight request (e.g. a Stop button).
 */
export async function streamChat(
  messages: ChatMessage[],
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
    signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`Chat request failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const { events, rest } = parseSSEChunk(buffer);
    buffer = rest;
    for (const ev of events) {
      if (ev.type === "meta") handlers.onMeta?.(ev);
      else if (ev.type === "delta") handlers.onDelta?.(ev.text);
      else if (ev.type === "done") handlers.onDone?.();
    }
  }
}