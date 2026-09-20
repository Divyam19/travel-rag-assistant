"""Postgres connections for Supabase.

connect() hands out warm pooled connections: opening a fresh one to the remote pooler costs over a
second (TLS, auth, setup), which used to dominate a chat turn. connect(with_vectors=False) opens a
plain one-off connection; migrate and doctor use it because the vector extension may not exist yet.
"""

import threading

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .config import get_settings

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def _pin_search_path(conn: psycopg.Connection) -> None:
    # Behind Supabase's transaction pooler (port 6543) server connections are shared between
    # clients, so one that another client left with a different search_path can be handed to us.
    # Our SQL uses unqualified names and the vector type, so pin it (and commit so it sticks).
    # The session pooler (port 5432) pins one server connection per client and avoids this.
    conn.execute("set search_path = public, extensions")
    conn.commit()


def _configure(conn: psycopg.Connection) -> None:
    _pin_search_path(conn)
    register_vector(conn)


def _reset(conn: psycopg.Connection) -> None:
    conn.autocommit = False  # callers may switch autocommit on; hand the next caller a default connection


def _get_pool() -> ConnectionPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            pool = ConnectionPool(
                get_settings().supabase_db_url, min_size=4, max_size=12, open=False,
                # prepare_threshold=None: prepared statements break behind the pooler.
                kwargs={"prepare_threshold": None}, configure=_configure, reset=_reset,
                # No `check`: it costs a round trip on every acquisition, and the database is
                # ~200ms away, which showed up as seconds per turn. Long max_idle keeps them warm.
                max_idle=600, max_lifetime=1800,
            )
            pool.open(wait=True, timeout=30)
            _pool = pool
    return _pool


def connect(with_vectors: bool = True):
    """Use as `with connect() as conn:`; commits on success, rolls back on error."""
    if with_vectors:
        return _get_pool().connection()
    conn = psycopg.connect(get_settings().supabase_db_url, prepare_threshold=None)
    _pin_search_path(conn)
    return conn
