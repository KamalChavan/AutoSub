"""Zero-dependency SQLite content-hash cache for ASR chunks and LLM translation batches.

Uses only Python's standard library (`sqlite3`, `hashlib`, `json`, `threading`) so AutoSub keeps its
slim standalone bundle while avoiding duplicate paid API calls on retries, post-processing tweaks,
or dual-ASR additions.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from src.config import get_cache_config


def sha256_file(path: Path, *, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hex digest of a file using constant memory."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """Compute SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_cache_key(namespace: str, payload: Any) -> str:
    """Build a deterministic SHA-256 cache key from a canonical JSON-serializable payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


class ResultCache:
    """Thread-safe persistent SQLite content-hash cache with optional TTL expiration."""

    def __init__(
        self,
        db_path: Path,
        *,
        enabled: bool = True,
        ttl_days: float = 30.0,
    ) -> None:
        self.db_path = Path(db_path)
        self.enabled = bool(enabled)
        self.ttl_seconds = max(0.0, float(ttl_days)) * 86400.0
        self._lock = threading.Lock()
        self._initialized = False
        self._broken = False

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error:
            pass
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        if self._initialized:
            return
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                cache_key TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_accessed_at REAL NOT NULL,
                meta_json TEXT NOT NULL DEFAULT '{}',
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_ns_created ON entries(namespace, created_at)"
        )
        conn.commit()
        self._initialized = True

    def get(self, namespace: str, cache_key: str) -> dict[str, Any] | None:
        """Fetch a cached JSON dict if present and unexpired; returns None on miss or SQLite error."""
        if not self.enabled or self._broken:
            return None
        now = time.time()
        with self._lock:
            if not self.db_path.is_file():
                return None
            conn: sqlite3.Connection | None = None
            try:
                conn = self._connect()
                self._ensure_schema(conn)
                row = conn.execute(
                    "SELECT created_at, payload_json FROM entries WHERE cache_key = ? AND namespace = ?",
                    (cache_key, namespace),
                ).fetchone()
                if row is None:
                    return None
                created_at, payload_json = float(row[0]), str(row[1])
                if self.ttl_seconds > 0 and (now - created_at) > self.ttl_seconds:
                    conn.execute("DELETE FROM entries WHERE cache_key = ?", (cache_key,))
                    conn.commit()
                    return None
                data = json.loads(payload_json)
                if not isinstance(data, dict):
                    return None
                conn.execute(
                    "UPDATE entries SET last_accessed_at = ? WHERE cache_key = ?",
                    (now, cache_key),
                )
                conn.commit()
                return data
            except (sqlite3.Error, OSError, json.JSONDecodeError):
                return None
            finally:
                if conn is not None:
                    conn.close()

    def set(
        self,
        namespace: str,
        cache_key: str,
        value: dict[str, Any],
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Persist a JSON dict under `(namespace, cache_key)`; silently no-ops on storage errors."""
        if not self.enabled or self._broken:
            return
        now = time.time()
        payload_json = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        meta_json = json.dumps(meta or {}, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            conn: sqlite3.Connection | None = None
            try:
                conn = self._connect()
                self._ensure_schema(conn)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO entries
                        (cache_key, namespace, created_at, last_accessed_at, meta_json, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (cache_key, namespace, now, now, meta_json, payload_json),
                )
                conn.commit()
            except (sqlite3.Error, OSError):
                self._broken = True
            finally:
                if conn is not None:
                    conn.close()

    def prune_expired(self) -> int:
        """Delete expired cache entries according to `ttl_seconds`; returns number of rows removed."""
        if self.ttl_seconds <= 0 or not self.db_path.is_file():
            return 0
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            conn: sqlite3.Connection | None = None
            try:
                conn = self._connect()
                self._ensure_schema(conn)
                cur = conn.execute("DELETE FROM entries WHERE created_at < ?", (cutoff,))
                conn.commit()
                return max(0, cur.rowcount)
            except (sqlite3.Error, OSError):
                return 0
            finally:
                if conn is not None:
                    conn.close()

    def stats(self) -> dict[str, Any]:
        """Return cache path, disk usage, and entry counts grouped by namespace."""
        if not self.db_path.is_file():
            return {
                "path": str(self.db_path),
                "exists": False,
                "size_bytes": 0,
                "total": 0,
                "by_namespace": {},
                "expired_pruned": 0,
            }
        pruned = self.prune_expired()
        with self._lock:
            conn: sqlite3.Connection | None = None
            by_ns: dict[str, int] = {}
            try:
                conn = self._connect()
                self._ensure_schema(conn)
                for ns, cnt in conn.execute(
                    "SELECT namespace, COUNT(*) FROM entries GROUP BY namespace ORDER BY namespace"
                ).fetchall():
                    by_ns[str(ns)] = int(cnt)
            except (sqlite3.Error, OSError):
                pass
            finally:
                if conn is not None:
                    conn.close()
        size_bytes = self.db_path.stat().st_size if self.db_path.is_file() else 0
        return {
            "path": str(self.db_path),
            "exists": True,
            "size_bytes": size_bytes,
            "total": sum(by_ns.values()),
            "by_namespace": by_ns,
            "expired_pruned": pruned,
        }

    def clear(self, namespace: str | None = None) -> int:
        """Delete all entries (or only a specific namespace) and reclaim disk space."""
        if not self.db_path.is_file():
            return 0
        with self._lock:
            conn: sqlite3.Connection | None = None
            try:
                conn = self._connect()
                self._ensure_schema(conn)
                if namespace and namespace != "all":
                    cur = conn.execute("DELETE FROM entries WHERE namespace = ?", (namespace,))
                else:
                    cur = conn.execute("DELETE FROM entries")
                deleted = max(0, cur.rowcount)
                conn.commit()
                try:
                    conn.execute("VACUUM")
                except sqlite3.Error:
                    pass
                return deleted
            except (sqlite3.Error, OSError):
                return 0
            finally:
                if conn is not None:
                    conn.close()


def build_result_cache(
    defaults: dict[str, Any],
    *,
    use_cache_override: bool | None = None,
) -> ResultCache | None:
    """Create a configured `ResultCache` from engine config and CLI override, or None if disabled."""
    cfg = get_cache_config(defaults)
    enabled = bool(cfg["enabled"]) if use_cache_override is None else bool(use_cache_override)
    if not enabled:
        return None
    return ResultCache(
        cfg["path"],
        enabled=True,
        ttl_days=float(cfg["ttl_days"]),
    )
