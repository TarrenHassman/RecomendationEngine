"""Airflow DAG: periodically drain the RabbitMQ events queue into the recommender model.

Requires on the Airflow workers: RABBITMQ_URL, RECOMMENDER_MODEL_PATH (shared storage with the
API if on different machines), and this project's directory on PYTHONPATH.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _train(**_):
    from trainer import run_batch

    return run_batch(max_messages=5000)


with DAG(
    dag_id="recommender_incremental_training",
    start_date=datetime(2026, 1, 1),
    schedule=timedelta(minutes=15),
    catchup=False,
    max_active_runs=1,  # only one consumer, so partial_fit updates never race on the model file
    default_args={"retries": 2, "retry_delay": timedelta(minutes=2)},
) as dag:
    PythonOperator(task_id="train_from_queue", python_callable=_train)
