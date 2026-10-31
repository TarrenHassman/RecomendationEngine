"""Inference API. Run: uvicorn api:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import os
import threading
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel

import trainer
from recommender import (MODEL_PATH, Recommender, load_model, post_from_dict, user_from_dict)


class _ModelCache:
    """Reloads the model file when the trainer has replaced it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._model = None

    def get(self):
        mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        with self._lock:
            if mtime != self._mtime:
                self._model = load_model() if mtime is not None else None
                self._mtime = mtime
            return self._model


class UserIn(BaseModel):
    id: str
    pantry: list[str] = []
    following: list[str] = []
    liked_posts: list[str] = []
    disliked_posts: list[str] = []


class PostIn(BaseModel):
    id: str
    author_id: str = ""
    ingredients: list[str] = []
    created_at: datetime | None = None
    like_count: int = 0


class RankRequest(BaseModel):
    user: UserIn
    candidates: list[PostIn]


class EventIn(BaseModel):
    action: Literal["like", "dislike"]
    ts: datetime | None = None
    user: UserIn
    post: PostIn


cache = _ModelCache()
app = FastAPI(title="Food recommendation API")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": cache.get() is not None}


@app.post("/recommendations", response_model=list[str])
def recommendations(req: RankRequest, k: int = Query(20, ge=1, le=100)):
    """Rank the supplied candidates; returns post ids for the client to fetch from Firebase."""
    user = user_from_dict(req.user.model_dump())
    posts = [post_from_dict(p.model_dump()) for p in req.candidates]
    return Recommender(cache.get()).rank(user, posts, k)


@app.post("/events", status_code=202)
def post_event(ev: EventIn):
    """Queue a like/dislike for the next incremental training run."""
    trainer.publish_event(ev.model_dump(mode="json"))
    return {"queued": True}
