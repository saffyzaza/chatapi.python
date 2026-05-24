import psycopg2
import psycopg2.pool
from contextlib import contextmanager
from src.config import get_settings

_sync_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def get_sync_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _sync_pool
    if _sync_pool is None:
        s = get_settings()
        _sync_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=20,
            host=s.DB_HOST,
            port=s.DB_PORT,
            database=s.DB_NAME,
            user=s.DB_USER,
            password=s.DB_PASSWORD,
        )
    return _sync_pool


def close_sync_pool() -> None:
    global _sync_pool
    if _sync_pool is not None:
        _sync_pool.closeall()
        _sync_pool = None


@contextmanager
def get_db_connection():
    pool = get_sync_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


def query_db(sql: str, params: tuple | list | None = None) -> list[dict]:
    """Execute a SELECT query and return rows as list of dicts."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            if cur.description is None:
                return []
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def execute_db(sql: str, params: tuple | list | None = None) -> int:
    """Execute a write query and return rowcount."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount
