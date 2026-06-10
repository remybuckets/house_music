"""
HouseMusic.ai - Step 1: build and inspect the corpus.

This script ONLY builds the document corpus from Wikipedia and reports on its
health. No embeddings, no Bedrock - that is Step 2. Run it, read the counts,
and fix any topic that comes back empty or thin before moving on.

    pip install requests
    python step1_corpus.py
"""
import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "housemusic-ai/0.1 (learning project)"}
MIN_PARA_LEN = 80  # paragraphs shorter than this are dropped as noise


# Tight, high-confidence corpus. Expand once retrieval works.
TOPICS = [
    "House music",
    "Chicago house",
    "Acid house",
    "Warehouse (nightclub)",   # the club house music is named after
    "Frankie Knuckles",
    "Larry Levan",
    "Paradise Garage",
    "Amnesia (nightclub)",
]


def fetch_wiki(topic):
    """Return (resolved_title, full_plaintext) for a Wikipedia page."""
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": topic,
        "format": "json",
        "explaintext": True,
        "redirects": True,
    }
    resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    page = next(iter(resp.json()["query"]["pages"].values()))
    if "missing" in page:          # the title does not resolve to a real page
        return topic, ""
    return page.get("title", topic), page.get("extract", "")


def chunk(text):
    """Split a page into paragraph-sized chunks, dropping short noise."""
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) > MIN_PARA_LEN]


def build_docs(topics):
    docs = []
    print(f"{'topic':<24}{'resolved title':<26}{'chars':>8}{'chunks':>8}")
    print("-" * 66)
    for t in topics:
        try:
            title, content = fetch_wiki(t)
            chunks = chunk(content)
            docs.extend(chunks)
            flag = "  <-- EMPTY/THIN" if len(chunks) < 2 else ""
            print(f"{t:<24}{title:<26}{len(content):>8}{len(chunks):>8}{flag}")
        except Exception as e:
            print(f"{t:<24}{'!! ERROR':<26}{'':>8}{'':>8}  {e}")
    print("-" * 66)
    print(f"total chunks: {len(docs)}")
    return docs


# Questions the TIGHT corpus should be able to answer. If a probe finds no
# good chunk here, no amount of embedding magic will fix it - it's a corpus gap.
PROBES = [
    "Where did house music get its name?",
    "Who was Frankie Knuckles?",
    "What is acid house?",
    "What famous nightclub is in Ibiza?",
]

STOP = {"the", "a", "an", "is", "was", "are", "were", "of", "in", "on", "at",
        "to", "for", "what", "who", "where", "when", "why", "how", "did",
        "does", "do", "get", "its", "famous", "and", "or"}


def keyword_probe(docs, questions):
    """No-embedding sanity check: does a chunk plausibly hold each answer?"""
    print("\nAnswerability check (keyword overlap, no embeddings):")
    for q in questions:
        terms = [w.strip("?.,").lower() for w in q.split()]
        terms = [w for w in terms if w and w not in STOP]
        best, best_score = "", 0
        for d in docs:
            score = sum(1 for t in terms if t in d.lower())
            if score > best_score:
                best, best_score = d, score
        print(f"\nQ: {q}")
        print(f"   matched {best_score}/{len(terms)} key terms")
        print("   ->", (best[:140].replace("\n", " ") + " ...") if best else "(no match)")


if __name__ == "__main__":
    docs = build_docs(TOPICS)
    keyword_probe(docs, PROBES)