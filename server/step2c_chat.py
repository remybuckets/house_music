"""
HouseMusic.ai - Step 2c: end-to-end grounded chat.

Retrieve relevant chunks, feed them to Haiku as context, and answer - while
keeping conversation history so follow-up questions work. This is the first
point where you actually talk to the bot.

    python step2_embed.py      # build corpus_cache.pkl first (if not already)
    python step2c_chat.py      # then chat

Needs AWS credentials configured.
"""
from step2_embed import bedrock
from step2_retrieve import load_cache, retrieve

CHAT_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

SYSTEM = (
    "You are HouseMusic.ai, a friendly and knowledgeable guide to house music - "
    "its history, its clubs, and its DJs. Answer using ONLY the context provided "
    "with each question. If the context does not contain the answer, say you "
    "don't have that in your knowledge yet rather than inventing facts. Keep "
    "answers conversational and concise."
)


def chat_turn(query, docs, vectors, history, k=3):
    """One grounded turn: retrieve, prompt with context, remember the exchange."""
    hits = retrieve(query, docs, vectors, k=k)
    context = "\n\n".join(chunk for _, chunk in hits)

    # The current turn carries the retrieved context; history stays lean (no
    # context) so it doesn't balloon turn over turn.
    grounded_turn = {
        "role": "user",
        "content": [{"text": f"Context:\n{context}\n\nQuestion: {query}"}],
    }

    resp = bedrock.converse(
        modelId=CHAT_MODEL,
        system=[{"text": SYSTEM}],
        messages=history + [grounded_turn],
        inferenceConfig={"maxTokens": 400, "temperature": 0.2},
    )
    answer = resp["output"]["message"]["content"][0]["text"]

    # Store the RAW question (without context) plus the answer, so memory works
    # but the prompt doesn't accumulate every chunk we've ever retrieved.
    history.append({"role": "user", "content": [{"text": query}]})
    history.append({"role": "assistant", "content": [{"text": answer}]})
    return answer, hits


if __name__ == "__main__":
    docs, vectors = load_cache()
    history = []
    print("HouseMusic.ai - ask about house music. ('quit' to exit)\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"quit", "exit"}:
            break
        if not q:
            continue
        answer, hits = chat_turn(q, docs, vectors, history)
        top = hits[0][0] if hits else 0.0
        print(f"\nbot > {answer}")
        print(f"      [grounded on {len(hits)} chunks, top score {top:.3f}]\n")