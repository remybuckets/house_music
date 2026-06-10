"use client";

import { useEffect, useRef, useState } from "react";
import { streamChat, type ChatMessage } from "./lib/chat";

type Turn = ChatMessage & {
  // Retrieval signal surfaced from the meta event, for assistant turns.
  topScore?: number;
  usedChunks?: number;
};

const SUGGESTIONS = [
  "Where did house music get its name?",
  "Who was Frankie Knuckles?",
  "What happened at the Paradise Garage?",
];

// The streaming "now playing" equalizer — the page's signature element.
function Equalizer() {
  return (
    <span className="inline-flex h-4 items-end gap-[3px]" aria-hidden>
      {[0, 1, 2, 3].map((i) => (
        <span
          key={i}
          className="w-[3px] origin-bottom rounded-full bg-amber animate-pulsebar"
          style={{ height: "100%", animationDelay: `${i * 0.12}s` }}
        />
      ))}
    </span>
  );
}

export default function Home() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Keep the latest turn in view as tokens arrive.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [turns]);

  async function send(text: string) {
    const question = text.trim();
    if (!question || streaming) return;

    setError(null);
    setInput("");

    // The wire format: full conversation, user + assistant turns, oldest first.
    const history: ChatMessage[] = turns.map((t) => ({
      role: t.role,
      content: t.content,
    }));
    const outgoing: ChatMessage[] = [...history, { role: "user", content: question }];

    // Optimistically show the user turn + an empty assistant turn to fill in.
    setTurns((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "" },
    ]);
    setStreaming(true);

    try {
      await streamChat(outgoing, {
        onMeta: (meta) =>
          setTurns((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = {
              ...last,
              topScore: meta.top_score,
              usedChunks: meta.used_chunks,
            };
            return next;
          }),
        onDelta: (chunk) =>
          setTurns((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, content: last.content + chunk };
            return next;
          }),
      });
    } catch {
      setError("Couldn't reach the booth. Is the backend running on :8000?");
      // Drop the empty assistant turn we added.
      setTurns((prev) => {
        const next = [...prev];
        if (next.length && next[next.length - 1].role === "assistant" && !next[next.length - 1].content) {
          next.pop();
        }
        return next;
      });
    } finally {
      setStreaming(false);
    }
  }

  const empty = turns.length === 0;

  return (
    <main className="mx-auto flex h-dvh max-w-3xl flex-col px-5">
      {/* Header */}
      <header className="flex items-baseline justify-between border-b border-haze py-5">
        <h1 className="font-display text-2xl font-bold tracking-tight">
          HouseMusic<span className="text-amber">.ai</span>
        </h1>
        <p className="font-display text-xs uppercase tracking-[0.2em] text-muted">
          grounded in the record
        </p>
      </header>

      {/* Transcript */}
      <div ref={scrollRef} className="flex-1 space-y-5 overflow-y-auto py-6">
        {empty && (
          <div className="animate-risein pt-6">
            <p className="font-display text-3xl font-medium leading-snug">
              Ask about the music that
              <br />
              <span className="text-amber">started in the warehouse.</span>
            </p>
            <p className="mt-3 max-w-md text-sm leading-relaxed text-muted">
              Its history, its clubs, its DJs — answered only from sourced
              material. If it&apos;s not in the record, it&apos;ll say so rather
              than make something up.
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="rounded-full border border-haze px-3.5 py-1.5 text-sm text-bone transition-colors hover:border-amber hover:text-amber focus:outline-none focus-visible:ring-2 focus-visible:ring-amber"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((t, i) => {
          const isUser = t.role === "user";
          const isLast = i === turns.length - 1;
          const isStreamingHere = streaming && isLast && t.role === "assistant";
          return (
            <div
              key={i}
              className={`flex animate-risein ${isUser ? "justify-end" : "justify-start"}`}
            >
              <div
                className={
                  isUser
                    ? "max-w-[85%] rounded-2xl rounded-br-sm bg-magenta/15 px-4 py-2.5 text-bone ring-1 ring-magenta/30"
                    : "max-w-[85%] rounded-2xl rounded-bl-sm bg-booth px-4 py-3 text-bone ring-1 ring-haze"
                }
              >
                {!isUser && (
                  <div className="mb-1.5 flex items-center gap-2">
                    <span className="font-display text-[11px] uppercase tracking-[0.18em] text-muted">
                      HouseMusic.ai
                    </span>
                    {isStreamingHere && !t.content && <Equalizer />}
                  </div>
                )}
                <p className="whitespace-pre-wrap text-[15px] leading-relaxed">
                  {t.content}
                  {isStreamingHere && t.content && (
                    <span className="ml-0.5 inline-block h-4 w-[2px] -translate-y-[1px] animate-pulse bg-amber align-middle" />
                  )}
                </p>
                {!isUser && t.usedChunks !== undefined && t.content && (
                  <p className="mt-2 font-display text-[11px] tracking-wide text-muted">
                    {t.usedChunks} sources · match {t.topScore?.toFixed(2)}
                  </p>
                )}
              </div>
            </div>
          );
        })}

        {error && (
          <p className="rounded-lg border border-magenta/40 bg-magenta/10 px-4 py-2.5 text-sm text-bone">
            {error}
          </p>
        )}
      </div>

      {/* Composer */}
      <div className="border-t border-haze py-4">
        <div className="flex items-end gap-2 rounded-xl bg-booth p-2 ring-1 ring-haze focus-within:ring-amber">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
              }
            }}
            rows={1}
            placeholder="Ask about a DJ, a club, a track…"
            className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] text-bone placeholder:text-muted focus:outline-none"
          />
          <button
            onClick={() => send(input)}
            disabled={streaming || !input.trim()}
            className="rounded-lg bg-amber px-4 py-2 font-display text-sm font-bold text-ink transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-bone"
          >
            {streaming ? "…" : "Send"}
          </button>
        </div>
        <p className="mt-2 text-center text-[11px] text-muted">
          Enter to send · Shift+Enter for a new line
        </p>
      </div>
    </main>
  );
}