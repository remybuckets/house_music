"""
HouseMusic.ai - Step 3c: the vector store, decoupled from the engine.

The step2 pipeline kept the whole corpus in a pickle and scanned it with an
in-memory NumPy cosine loop. That is fine for 86 chunks; it does not scale to
the planned thousands. This module puts a seam in front of that decision:

    VectorStore        - the interface RagEngine depends on
    InMemoryStore      - the old NumPy behaviour, behind the interface
                         (keeps tests + the pickle path working, zero DB needed)
    PgVectorStore      - Postgres + pgvector, the production target

RagEngine never imports a concrete store directly for retrieval - it holds a
VectorStore and calls .search(query_vector, k). Swapping pgvector for Qdrant
later is a new class here, untouched engine.

== Similarity convention (important) ==
The whole project treats "score" as COSINE SIMILARITY in [-1, 1], higher =
closer (confident answers ~0.77-0.81, out-of-scope ~0.37). pgvector's `<=>`
operator returns cosine DISTANCE (= 1 - similarity). PgVectorStore converts
back to similarity (1 - distance) so `top_score` means the same thing no
matter which backend is live. Do not leak distance past this module.
"""
from __future__ import annotations

import os
from typing import Protocol

import numpy as np

EMBED_DIM = 1024  # Amazon Titan Text Embeddings v2


class VectorStore(Protocol):
    """What RagEngine needs from a store. Concrete stores implement this."""

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[float, str]]:
        """Return up to k (cosine_similarity, chunk_text) pairs, best first."""
        ...

    @property
    def size(self) -> int:
        """Number of chunks currently stored."""
        ...


class InMemoryStore:
    """
    The original behaviour, behind the interface: vectors held in a NumPy
    matrix, cosine similarity computed by a full scan. Used by tests and by
    the legacy pickle path. No external dependencies beyond NumPy.
    """

    def __init__(self, docs: list[str], vectors: np.ndarray):
        self.docs = docs
        self.vectors = np.asarray(vectors, dtype=np.float32)
        # Precompute chunk norms once (same production habit as before).
        self._norms = np.linalg.norm(self.vectors, axis=1)

    @property
    def size(self) -> int:
        return len(self.docs)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[float, str]]:
        q = np.asarray(query_vector, dtype=np.float32)
        sims = self.vectors @ q / (self._norms * np.linalg.norm(q) + 1e-9)
        top = np.argsort(sims)[::-1][:k]
        return [(float(sims[i]), self.docs[i]) for i in top]


class PgVectorStore:
    """
    Postgres + pgvector. Stores one row per chunk:

        chunks(id, content, content_hash, embedding vector(1024))

    Search uses the cosine-distance operator `<=>` with an ivfflat index, and
    converts distance back to similarity so callers see the same score scale
    as InMemoryStore. Connection is pooled and opened lazily.

    Requires: pip install "psycopg[binary,pool]" pgvector
    Env: DATABASE_URL (e.g. postgresql://house:house@localhost:5432/housemusic)
    """

    def __init__(self, dsn: str | None = None, pool=None):
        # Allow injecting a pool (tests / custom config); else build from DSN.
        if pool is not None:
            self._pool = pool
        else:
            from psycopg_pool import ConnectionPool

            dsn = dsn or os.environ["DATABASE_URL"]
            # open=False + explicit open avoids surprises at import time.
            self._pool = ConnectionPool(dsn, open=True, min_size=1, max_size=4)

    @property
    def size(self) -> int:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT count(*) FROM chunks").fetchone()
        return int(row[0])

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[float, str]]:
        # pgvector wants a list; register_vector (in ingest) handles np arrays
        # too, but a plain list keeps this method dependency-light.
        vec = np.asarray(query_vector, dtype=np.float32).tolist()
        # `<=>` = cosine distance. similarity = 1 - distance.
        sql = (
            "SELECT content, 1 - (embedding <=> %s::vector) AS similarity "
            "FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s"
        )
        with self._pool.connection() as conn:
            rows = conn.execute(sql, (vec, vec, k)).fetchall()
        return [(float(sim), content) for content, sim in rows]