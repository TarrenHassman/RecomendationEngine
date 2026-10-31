"""Short-lived Redis cache of ranked post ids per user (ids only; no post data is stored).

Each entry holds the ranked ids plus a signature of the inputs (pantry, following, candidate ids),
so a request with different inputs is a miss rather than a stale hit.
"""
from __future__ import annotations

import hashlib
import json
import os

import redis

TTL_SECONDS = int(os.environ.get("REC_CACHE_TTL", "300"))
MAX_CANDIDATES = 100

_client: redis.Redis | None = None


def client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
                                       decode_responses=True)
    return _client


def _key(uid: str) -> str:
    return f"rec:{uid}"


def signature(pantry, following, candidate_ids) -> str:
    payload = json.dumps([sorted(pantry), sorted(following), sorted(candidate_ids)])
    return hashlib.sha256(payload.encode()).hexdigest()


def get(uid: str, sig: str, k: int) -> list[str] | None:
    raw = client().get(_key(uid))
    if raw is None:
        return None
    entry = json.loads(raw)
    return entry["ids"][:k] if entry["sig"] == sig else None


def put(uid: str, sig: str, ids: list[str], ttl: int = TTL_SECONDS) -> None:
    client().set(_key(uid), json.dumps({"sig": sig, "ids": ids[:MAX_CANDIDATES]}), ex=ttl)


def remove_post(uid: str, post_id: str) -> None:
    """Drop a post the user just interacted with, keeping the remaining TTL."""
    r = client()
    raw = r.get(_key(uid))
    if raw is None:
        return
    entry = json.loads(raw)
    entry["ids"] = [i for i in entry["ids"] if i != post_id]
    r.set(_key(uid), json.dumps(entry), keepttl=True)


def invalidate(uid: str) -> None:
    client().delete(_key(uid))
