"""
Vector-store tests. The in-memory store is exercised directly. The pgvector
store is exercised against a FAKE psycopg pool, so this runs with no Postgres:
it proves the SQL shape, the parameter passing, and - critically - that cosine
DISTANCE from pgvector is converted back to cosine SIMILARITY before it leaves
the store (the score scale the rest of the app depends on).

    pip install numpy pytest   # then: pytest test_vector_store.py
"""
import numpy as np
import pytest

from vector_store import InMemoryStore, PgVectorStore


# --- InMemoryStore --------------------------------------------------------

def test_inmemory_ranks_by_cosine_similarity():
    docs = ["east", "north"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store = InMemoryStore(docs, vectors)

    # A query pointing "east" should rank the east chunk first with sim ~1.0.
    hits = store.search(np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert store.size == 2
    assert hits[0][1] == "east"
    assert hits[0][0] == pytest.approx(1.0, abs=1e-6)
    assert hits[1][1] == "north"
    assert hits[1][0] == pytest.approx(0.0, abs=1e-6)


def test_inmemory_respects_k():
    docs = ["a", "b", "c"]
    vectors = np.eye(3, dtype=np.float32)
    store = InMemoryStore(docs, vectors)
    hits = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=2)
    assert len(hits) == 2


# --- PgVectorStore (fake pool, no real Postgres) --------------------------

class FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return self

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0]


class FakeConn:
    def __init__(self, rows):
        self.cursor = FakeCursor(rows)

    def execute(self, sql, params=None):
        return self.cursor.execute(sql, params)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakePool:
    """Mimics psycopg_pool.ConnectionPool's .connection() context manager."""

    def __init__(self, rows):
        self._rows = rows
        self.conn = FakeConn(rows)

    def connection(self):
        return self.conn


def test_pgvector_converts_distance_to_similarity():
    # SQL already computes similarity = 1 - distance and returns
    # (content, similarity) rows. The store re-emits them as (score, content)
    # to match InMemoryStore and what RagEngine.retrieve unpacks.
    rows = [("east", 0.99), ("north", 0.01)]
    store = PgVectorStore(pool=FakePool(rows))

    hits = store.search(np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert hits == [
        (pytest.approx(0.99), "east"),
        (pytest.approx(0.01), "north"),
    ]
    # And the score is a plain float, best-first.
    assert isinstance(hits[0][0], float)
    assert hits[0][0] > hits[1][0]


def test_pgvector_search_sql_uses_cosine_and_limit():
    rows = [("east", 0.99)]
    pool = FakePool(rows)
    store = PgVectorStore(pool=pool)
    store.search(np.array([1.0, 0.0], dtype=np.float32), k=5)

    sql, params = pool.conn.cursor.executed[-1]
    # similarity = 1 - distance, cosine operator, parameterised LIMIT.
    assert "1 - (embedding <=> %s::vector)" in sql
    assert "ORDER BY embedding <=> %s::vector" in sql
    assert "LIMIT %s" in sql
    # query vector passed as a plain list (twice: select + order), k last.
    assert params[-1] == 5
    assert isinstance(params[0], list)


def test_pgvector_size_counts_rows():
    store = PgVectorStore(pool=FakePool([(42,)]))
    assert store.size == 42