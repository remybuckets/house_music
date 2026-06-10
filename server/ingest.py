"""
HouseMusic.ai - Step 3c: ingest the corpus into Postgres/pgvector.

This is the embed-once-at-ingest path that replaces step2_embed.py's pickle.
It builds the corpus the same way step1 does, embeds each chunk with Titan
(via Bedrock), and upserts into the `chunks` table. Idempotent: each chunk is
keyed by a content hash, so re-running only embeds chunks that are new or
changed - the same "don't pay twice" property the pickle cache had, now at
row granularity.

    pip install "psycopg[binary,pool]" pgvector boto3 numpy requests

    # bring up Postgres+pgvector (see docker-compose.yml)
    docker compose up -d
    export DATABASE_URL=postgresql://house:house@localhost:5432/housemusic
    # AWS creds configured for Bedrock embeddings
    python ingest.py            # embeds new/changed chunks, upserts
    python ingest.py            # second run is a near-no-op (all hashes match)

Schema (created if absent):
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE chunks (
        id           bigserial PRIMARY KEY,
        content      text NOT NULL,
        content_hash text UNIQUE NOT NULL,
        embedding    vector(1024) NOT NULL
    );
    CREATE INDEX ... USING ivfflat (embedding vector_cosine_ops);
"""
import hashlib
import json
import os

import boto3
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from engine import EMBED_MODEL  # reuse the one source of truth for the model id

# step1_corpus.py owns corpus construction (TOPICS, build_docs, chunk, fetch_wiki).
# Reuse it rather than duplicating the Wikipedia logic here.
from step1_corpus import TOPICS, build_docs

EMBED_DIM = 1024

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    content      text NOT NULL,
    content_hash text UNIQUE NOT NULL,
    embedding    vector(1024) NOT NULL
);
"""

# ivfflat needs the table populated before it's worth building; create it after
# the first insert. lists=100 is a reasonable default for the planned corpus
# size (tune as the corpus grows).
INDEX = (
    "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
    "ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed(bedrock, text: str) -> np.ndarray:
    resp = bedrock.invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps({"inputText": text}),
    )
    return np.array(json.loads(resp["body"].read())["embedding"], dtype=np.float32)


def main(dsn: str | None = None, region: str = "us-east-1"):
    dsn = dsn or os.environ["DATABASE_URL"]
    bedrock = boto3.client("bedrock-runtime", region_name=region)

    docs = build_docs(TOPICS)  # list[str], one entry per chunk (prints a health table)
    print(f"corpus: {len(docs)} chunks")

    with psycopg.connect(dsn, autocommit=True) as conn:
        register_vector(conn)
        conn.execute(SCHEMA)

        # Which chunks are already stored (by hash)? Skip those - that's the
        # idempotent / don't-re-embed property.
        existing = {
            row[0] for row in conn.execute("SELECT content_hash FROM chunks").fetchall()
        }

        new, skipped = 0, 0
        for chunk in docs:
            h = _hash(chunk)
            if h in existing:
                skipped += 1
                continue
            vec = embed(bedrock, chunk)
            conn.execute(
                "INSERT INTO chunks (content, content_hash, embedding) "
                "VALUES (%s, %s, %s) ON CONFLICT (content_hash) DO NOTHING",
                (chunk, h, vec),
            )
            new += 1

        # Optionally prune rows whose chunks are no longer in the corpus, so the
        # store tracks the corpus exactly (mirrors the hash-keyed cache intent).
        current_hashes = {_hash(c) for c in docs}
        if current_hashes:
            placeholders = ",".join(["%s"] * len(current_hashes))
            removed = conn.execute(
                f"DELETE FROM chunks WHERE content_hash NOT IN ({placeholders})",
                tuple(current_hashes),
            ).rowcount
        else:
            removed = 0

        conn.execute(INDEX)
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    print(f"embedded {new} new, skipped {skipped} unchanged, pruned {removed}.")
    print(f"chunks table now holds {total} rows.")


if __name__ == "__main__":
    main()