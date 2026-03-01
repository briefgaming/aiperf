# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty
from typing import Any

import orjson


@dataclass(frozen=True)
class OTelStreamingFanoutConfig:
    """Configuration for the dedicated OTel/MLflow streaming fanout process."""

    endpoint_url: str
    request_timeout_seconds: float
    export_interval_millis: int
    export_timeout_millis: int
    resource_attributes: dict[str, str]
    mlflow_tracking_uri: str | None
    mlflow_experiment: str
    mlflow_run_name: str | None
    mlflow_tags: dict[str, str]
    benchmark_id: str | None
    metadata_file: Path


def _write_live_mlflow_metadata(
    *,
    metadata_file: Path,
    tracking_uri: str,
    experiment: str,
    run_id: str,
    run_name: str | None,
    benchmark_id: str | None,
) -> None:
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tracking_uri": tracking_uri,
        "experiment": experiment,
        "run_id": run_id,
        "run_name": run_name,
        "benchmark_id": benchmark_id,
        "live_streaming": True,
        "stream_started_at_ns": time.time_ns(),
    }
    metadata_file.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def run_otel_streaming_fanout(
    event_queue: Queue[Any],
    config: OTelStreamingFanoutConfig,
) -> None:
    """Run OTel + MLflow live streaming fanout in a dedicated process."""
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource

    logger = logging.getLogger(__name__)

    resource = Resource.create(config.resource_attributes)
    exporter = OTLPMetricExporter(
        endpoint=config.endpoint_url,
        timeout=config.request_timeout_seconds,
    )
    reader = PeriodicExportingMetricReader(
        exporter,
        export_interval_millis=config.export_interval_millis,
        export_timeout_millis=config.export_timeout_millis,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    meter = meter_provider.get_meter("aiperf.records")

    histograms: dict[str, Any] = {}
    counters: dict[str, Any] = {}
    up_down_counters: dict[str, Any] = {}

    mlflow_state: dict[str, Any] | None = None
    if config.mlflow_tracking_uri:
        try:
            import mlflow
            from mlflow.entities import Metric
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment(config.mlflow_experiment)
            run = mlflow.start_run(run_name=config.mlflow_run_name)
            if config.mlflow_tags:
                mlflow.set_tags(config.mlflow_tags)
            if config.benchmark_id:
                mlflow.set_tag("benchmark_id", config.benchmark_id)
            run_id = run.info.run_id
            _write_live_mlflow_metadata(
                metadata_file=config.metadata_file,
                tracking_uri=config.mlflow_tracking_uri,
                experiment=config.mlflow_experiment,
                run_id=run_id,
                run_name=config.mlflow_run_name,
                benchmark_id=config.benchmark_id,
            )
            mlflow_state = {
                "module": mlflow,
                "client": MlflowClient(),
                "metric_cls": Metric,
                "run_id": run_id,
                "step": 0,
                "buffer": [],
            }
        except Exception as exc:
            logger.warning(f"MLflow live streaming disabled in fanout process: {exc!r}")
            mlflow_state = None

    def _append_mlflow_metric(metric_name: str, metric_value: float) -> None:
        if mlflow_state is None:
            return
        mlflow_state["buffer"].append((f"live.{metric_name}", float(metric_value)))

    def _flush_mlflow_metrics() -> None:
        if mlflow_state is None:
            return
        buffered: list[tuple[str, float]] = mlflow_state["buffer"]
        if not buffered:
            return
        now_ms = int(time.time() * 1000)
        metrics = []
        metric_cls = mlflow_state["metric_cls"]
        for metric_name, metric_value in buffered:
            metrics.append(
                metric_cls(
                    metric_name,
                    metric_value,
                    now_ms,
                    mlflow_state["step"],
                )
            )
            mlflow_state["step"] += 1
        mlflow_state["buffer"] = []
        try:
            mlflow_state["client"].log_batch(
                run_id=mlflow_state["run_id"],
                metrics=metrics,
                params=[],
                tags=[],
            )
        except Exception as exc:
            logger.warning(f"Failed to log live MLflow metrics batch: {exc!r}")

    poll_timeout_sec = max(config.export_interval_millis / 1000.0, 0.1)

    try:
        while True:
            try:
                event = event_queue.get(timeout=poll_timeout_sec)
            except Empty:
                _flush_mlflow_metrics()
                continue

            event_type = event.get("type")
            payload = event.get("payload", {})

            if event_type == "histogram_record":
                try:
                    metric_name = payload["metric_name"]
                    if metric_name not in histograms:
                        histograms[metric_name] = meter.create_histogram(
                            name=metric_name,
                            unit=payload["unit"],
                            description=payload["description"],
                        )
                    histograms[metric_name].record(
                        payload["value"], payload["attributes"]
                    )
                    _append_mlflow_metric(metric_name, payload["value"])
                except Exception as exc:
                    logger.warning(
                        f"Invalid histogram fanout payload received: {exc!r}"
                    )
                continue

            if event_type == "counter_add":
                try:
                    metric_name = payload["metric_name"]
                    if metric_name not in counters:
                        counters[metric_name] = meter.create_counter(
                            name=metric_name,
                            unit=payload["unit"],
                            description=payload["description"],
                        )
                    counters[metric_name].add(payload["value"], payload["attributes"])
                    _append_mlflow_metric(metric_name, payload["value"])
                except Exception as exc:
                    logger.warning(f"Invalid counter fanout payload received: {exc!r}")
                continue

            if event_type == "up_down_counter_add":
                try:
                    metric_name = payload["metric_name"]
                    if metric_name not in up_down_counters:
                        up_down_counters[metric_name] = meter.create_up_down_counter(
                            name=metric_name,
                            unit=payload["unit"],
                            description=payload["description"],
                        )
                    up_down_counters[metric_name].add(
                        payload["value"], payload["attributes"]
                    )
                    _append_mlflow_metric(metric_name, payload["value"])
                except Exception as exc:
                    logger.warning(
                        f"Invalid up-down-counter fanout payload received: {exc!r}"
                    )
                continue

            if event_type == "flush":
                meter_provider.force_flush(timeout_millis=config.export_timeout_millis)
                _flush_mlflow_metrics()
                continue

            if event_type == "shutdown":
                meter_provider.force_flush(timeout_millis=config.export_timeout_millis)
                _flush_mlflow_metrics()
                break
    finally:
        meter_provider.shutdown()
        if mlflow_state is not None:
            try:
                mlflow_state["module"].end_run()
            except Exception as exc:
                logger.warning(f"Failed to close live MLflow run cleanly: {exc!r}")
