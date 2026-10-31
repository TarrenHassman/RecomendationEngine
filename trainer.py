"""Incremental trainer: drains interaction events from RabbitMQ and updates the model.

Event (JSON), self-contained so nothing is looked up or stored server-side:
  {"action": "like" | "dislike", "ts": ISO-8601 (optional),
   "user": {id, pantry[], following[]}, "post": {id, author_id, ingredients[], created_at, like_count}}
Delivery is at-least-once: messages are acked only after the updated model is saved.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pika

from recommender import (features, load_model, new_model, parse_time, post_from_dict, save_model,
                         user_from_dict)

log = logging.getLogger(__name__)

QUEUE = os.environ.get("RABBITMQ_QUEUE", "recommender.events")
LABELS = {"like": 1, "dislike": 0}


def _connect() -> pika.BlockingConnection:
    return pika.BlockingConnection(pika.URLParameters(os.environ["RABBITMQ_URL"]))


def publish_event(event: dict) -> None:
    """Queue a like/dislike event (shape above)."""
    if event.get("action") not in LABELS:
        raise ValueError(f"unknown action {event.get('action')!r}")
    conn = _connect()
    try:
        ch = conn.channel()
        ch.queue_declare(queue=QUEUE, durable=True)
        ch.basic_publish(exchange="", routing_key=QUEUE, body=json.dumps(event),
                         properties=pika.BasicProperties(delivery_mode=2))
    finally:
        conn.close()


def _featurize(ev: dict) -> tuple[np.ndarray, int]:
    ts = parse_time(ev.get("ts") or datetime.now(timezone.utc))
    return features(user_from_dict(ev["user"]), post_from_dict(ev["post"]), ts), LABELS[ev["action"]]


def run_batch(max_messages: int = 1000) -> int:
    """Consume up to max_messages events, partial_fit the model, and save. Returns events used."""
    conn = _connect()
    try:
        ch = conn.channel()
        ch.queue_declare(queue=QUEUE, durable=True)

        last_tag = None
        n_msgs = 0
        X, y = [], []
        for _ in range(max_messages):
            method, _props, body = ch.basic_get(queue=QUEUE, auto_ack=False)
            if method is None:
                break
            last_tag = method.delivery_tag
            n_msgs += 1
            try:
                x, label = _featurize(json.loads(body))
                X.append(x)
                y.append(label)
            except (ValueError, KeyError, TypeError):
                log.warning("dropping malformed message: %r", body)

        if last_tag is None:
            return 0

        if X:
            model = load_model() or new_model()
            if hasattr(model, "classes_"):
                model.partial_fit(np.vstack(X), np.array(y))
            else:
                model.partial_fit(np.vstack(X), np.array(y), classes=np.array([0, 1]))
            save_model(model)

        # Ack only after the model is durably saved.
        ch.basic_ack(delivery_tag=last_tag, multiple=True)
        log.info("consumed %d messages, trained on %d", n_msgs, len(X))
        return len(X)
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_batch()
