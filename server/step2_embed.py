"""
HouseMusic.ai - Step 2a: embed the corpus once and cache it to disk.

Embedding every chunk costs a Bedrock call, so we do it ONCE and save the
vectors. Later runs reload from disk - UNLESS the corpus changed, in which
case it re-embeds automatically (the cache is keyed to the corpus contents).

    python step2_embed.py     # first run:  embeds, then saves
    python step2_embed.py     # second run: loads from cache, instant

Needs AWS credentials configured, same as your original script.
Sits next to step1_corpus.py and imports the corpus from it.
"""
import json
import hashlib
import os
import pickle
import time

import boto3
import numpy as np
from botocore.exceptions import ClientError

from step1_corpus import build_docs, TOPICS

EMBED_MODEL = "amazon.titan-embed-text-v2:0"
CACHE_PATH = "corpus_cache.pkl"

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")


def embed(text):
    resp = bedrock.invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps({"inputText": text}),
    )
    return np.array(json.loads(resp["body"].read())["embedding"], dtype=np.float32)


def embed_with_retry(text, tries=5):
    """Bedrock can throttle on rapid calls; back off and retry."""
    for attempt in range(tries):
        try:
            return embed(text)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code not in ("ThrottlingException", "TooManyRequestsException"):
                raise
            wait = 2 ** attempt
            print(f"  throttled, waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("still throttled after retries")


def corpus_hash(docs):
    """A fingerprint of the corpus so a changed corpus invalidates the cache."""
    h = hashlib.sha256()
    for d in docs:
        h.update(d.encode("utf-8"))
    return h.hexdigest()


def load_or_embed(docs):
    h = corpus_hash(docs)
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        if cache.get("hash") == h:
            print(f"Loaded {len(cache['vectors'])} vectors from cache ({CACHE_PATH}).")
            return cache["docs"], cache["vectors"]
        print("Corpus changed since last run - re-embedding.")

    print(f"Embedding {len(docs)} chunks (one Bedrock call each)...")
    start = time.time()
    vectors = []
    for i, d in enumerate(docs, 1):
        vectors.append(embed_with_retry(d))
        if i % 10 == 0 or i == len(docs):
            print(f"  {i}/{len(docs)}")
    vectors = np.vstack(vectors)
    elapsed = time.time() - start

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(
            {"hash": h, "docs": docs, "vectors": vectors, "model": EMBED_MODEL}, f
        )
    print(f"Embedded and cached in {elapsed:.1f}s -> {CACHE_PATH}")
    return docs, vectors


if __name__ == "__main__":
    docs = build_docs(TOPICS)
    docs, vectors = load_or_embed(docs)
    print(
        f"\nVectors ready: {vectors.shape[0]} chunks x {vectors.shape[1]} dims "
        f"({vectors.dtype})"
    )