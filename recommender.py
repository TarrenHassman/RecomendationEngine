"""Stateless post recommendation engine for a nutrition/food app.

The server stores no posts or user data: callers supply the user's current ingredients and
candidate posts per request, and training data arrives via the RabbitMQ queue (trainer.py).
Only the model file is persisted.

Payload shapes (dicts / JSON):
  user: {id, pantry[], following[], liked_posts[], disliked_posts[]}   pantry = current ingredients
  post: {id, author_id, ingredients[], created_at (ISO-8601), like_count}
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

MODEL_PATH = os.environ.get("RECOMMENDER_MODEL_PATH", "model.joblib")
FEATURE_NAMES = ["pantry_overlap", "pantry_missing", "recency", "popularity", "following"]
HALF_LIFE_HOURS = 48.0

# Used until a trained model exists.
DEFAULT_WEIGHTS = np.array([0.6, -0.2, 0.2, 0.1, 0.05])


@dataclass
class User:
    id: str
    pantry: set[str] = field(default_factory=set)
    following: set[str] = field(default_factory=set)
    liked_posts: set[str] = field(default_factory=set)
    disliked_posts: set[str] = field(default_factory=set)


@dataclass
class Post:
    id: str
    author_id: str
    ingredients: set[str]
    created_at: datetime
    like_count: int = 0


def _norm(items) -> set[str]:
    return {str(i).strip().lower() for i in items or [] if str(i).strip()}


def parse_time(value) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value is None:
        return datetime.now(timezone.utc)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def user_from_dict(d: dict) -> User:
    return User(
        id=d["id"],
        pantry=_norm(d.get("pantry")),
        following=set(d.get("following", [])),
        liked_posts=set(d.get("liked_posts", [])),
        disliked_posts=set(d.get("disliked_posts", [])),
    )


def post_from_dict(d: dict) -> Post:
    return Post(
        id=d["id"],
        author_id=d.get("author_id", ""),
        ingredients=_norm(d.get("ingredients")),
        created_at=parse_time(d.get("created_at")),
        like_count=int(d.get("like_count", 0)),
    )


def features(user: User, post: Post, now: datetime) -> np.ndarray:
    """Must stay in the same order as FEATURE_NAMES."""
    n = len(post.ingredients)
    have = len(post.ingredients & user.pantry)
    overlap = have / n if n else 0.0
    missing = min(n - have, 10) / 10
    age_h = max((now - post.created_at).total_seconds() / 3600, 0.0)
    recency = 0.5 ** (age_h / HALF_LIFE_HOURS)
    popularity = min(np.log1p(max(post.like_count, 0)) / np.log1p(1000), 1.0)
    following = 1.0 if post.author_id in user.following else 0.0
    return np.array([overlap, missing, recency, popularity, following])


def new_model() -> SGDClassifier:
    return SGDClassifier(loss="log_loss", alpha=1e-4, random_state=0)


def load_model(path: str = MODEL_PATH) -> SGDClassifier | None:
    return joblib.load(path) if os.path.exists(path) else None


def save_model(model: SGDClassifier, path: str = MODEL_PATH) -> None:
    tmp = f"{path}.tmp"
    joblib.dump(model, tmp)
    os.replace(tmp, path)  # atomic so readers never see a partial file


class Recommender:
    def __init__(self, model: SGDClassifier | None = None):
        self.model = model

    def _scores(self, X: np.ndarray) -> np.ndarray:
        if self.model is not None and hasattr(self.model, "coef_"):
            return self.model.decision_function(X)
        return X @ DEFAULT_WEIGHTS

    def rank(self, user: User, candidates: list[Post], k: int = 20,
             now: datetime | None = None) -> list[str]:
        """Return post ids, best first."""
        now = now or datetime.now(timezone.utc)
        seen = user.liked_posts | user.disliked_posts
        pool = [p for p in candidates if p.id not in seen and p.author_id != user.id]
        if not pool:
            return []
        scores = self._scores(np.vstack([features(user, p, now) for p in pool]))
        return [pool[i].id for i in np.argsort(-scores)[:k]]
