"""
HouseMusic.ai - Step 2b: semantic retrieval with cosine similarity.

Loads the cached vectors from step2_embed.py (run that first), embeds a query,
and ranks chunks by cosine similarity. Then re-runs the SAME four probes from
Step 1 so you can see semantic retrieval vs. the keyword baseline.

    python step2_embed.py      # first, to build corpus_cache.pkl
    python step2_retrieve.py   # then this

Needs AWS credentials configured.
"""
import pickle

import numpy as np

from step2_embed import embed, CACHE_PATH

# Same probes we used for the keyword baseline in Step 1.
PROBES = [
    "Where did house music get its name?",
    "Who was Frankie Knuckles?",
    "What is acid house?",
    "What famous nightclub is in Ibiza?",
]


def load_cache():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    return cache["docs"], cache["vectors"]


def retrieve(query, docs, vectors, k=3):
    """Return the top-k chunks by cosine similarity, with scores."""
    q = embed(query)
    # cosine = dot product over the product of norms, vectorized over all chunks
    sims = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q) + 1e-9)
    top = np.argsort(sims)[::-1][:k]
    return [(float(sims[i]), docs[i]) for i in top]


if __name__ == "__main__":
    docs, vectors = load_cache()
    print(f"Loaded {len(docs)} chunks x {vectors.shape[1]} dims.\n")

    for q in PROBES:
        print(f"Q: {q}")
        for rank, (score, chunk) in enumerate(retrieve(q, docs, vectors), 1):
            preview = chunk[:120].replace("\n", " ")
            print(f"  {rank}. [{score:.3f}] {preview} ...")
        print()