# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from pathlib import Path
from queue import Empty
from typing import Any

import orjson
import pytest

from aiperf.post_processors.otel_streaming_fanout import (
    OTelStreamingFanoutConfig,
    run_otel_streaming_fanout,
)


class _SequenceQueue:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = list(events)

    def get(self, timeout: float | None = None) -> dict[str, Any]:
        if not self._events:
            raise Empty
        return self._events.pop(0)


def _install_fake_otel_modules(
    monkeypatch: pytest.MonkeyPatch,
    state: dict[str, Any],
) -> None:
    def _add_module(name: str, *, package: bool = True) -> types.ModuleType:
        module = types.ModuleType(name)
        if package:
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    opentelemetry = _add_module("opentelemetry")
    exporter = _add_module("opentelemetry.exporter")
    otlp = _add_module("opentelemetry.exporter.otlp")
    proto = _add_module("opentelemetry.exporter.otlp.proto")
    http = _add_module("opentelemetry.exporter.otlp.proto.http")
    metric_exporter = _add_module(
        "opentelemetry.exporter.otlp.proto.http.metric_exporter",
        package=False,
    )
    sdk = _add_module("opentelemetry.sdk")
    sdk_metrics = _add_module("opentelemetry.sdk.metrics")
    sdk_metrics_export = _add_module("opentelemetry.sdk.metrics.export", package=False)
    sdk_resources = _add_module("opentelemetry.sdk.resources", package=False)

    opentelemetry.exporter = exporter
    exporter.otlp = otlp
    otlp.proto = proto
    proto.http = http
    http.metric_exporter = metric_exporter

    opentelemetry.sdk = sdk
    sdk.metrics = sdk_metrics
    sdk.resources = sdk_resources
    sdk_metrics.export = sdk_metrics_export

    class FakeHistogram:
        def __init__(self, name: str) -> None:
            self.name = name
            self.records: list[tuple[float, dict[str, Any]]] = []

        def record(self, value: float, attributes: dict[str, Any]) -> None:
            self.records.append((value, attributes))

    class FakeCounter:
        def __init__(self, name: str) -> None:
            self.name = name
            self.adds: list[tuple[float, dict[str, Any]]] = []

        def add(self, value: float, attributes: dict[str, Any]) -> None:
            self.adds.append((value, attributes))

    class FakeMeter:
        def __init__(self) -> None:
            self.histograms: dict[str, FakeHistogram] = {}
            self.counters: dict[str, FakeCounter] = {}
            self.up_down_counters: dict[str, FakeCounter] = {}

        def create_histogram(
            self, name: str, unit: str, description: str
        ) -> FakeHistogram:
            histogram = FakeHistogram(name)
            self.histograms[name] = histogram
            return histogram

        def create_counter(self, name: str, unit: str, description: str) -> FakeCounter:
            counter = FakeCounter(name)
            self.counters[name] = counter
            return counter

        def create_up_down_counter(
            self, name: str, unit: str, description: str
        ) -> FakeCounter:
            up_down = FakeCounter(name)
            self.up_down_counters[name] = up_down
            return up_down

    class FakeMeterProvider:
        def __init__(self, resource: object, metric_readers: list[object]) -> None:
            state["resource"] = resource
            state["metric_readers"] = metric_readers
            state["force_flush_calls"] = []
            state["shutdown_calls"] = 0
            state["meter"] = FakeMeter()

        def get_meter(self, name: str) -> FakeMeter:
            state["meter_name"] = name
            return state["meter"]

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            state["force_flush_calls"].append(timeout_millis)
            return True

        def shutdown(self) -> None:
            state["shutdown_calls"] += 1

    class FakeReader:
        def __init__(
            self,
            exporter: object,
            export_interval_millis: int,
            export_timeout_millis: int,
        ) -> None:
            state["reader_export_interval_millis"] = export_interval_millis
            state["reader_export_timeout_millis"] = export_timeout_millis
            state["reader_exporter"] = exporter

    class FakeResource:
        @staticmethod
        def create(attributes: dict[str, str]) -> dict[str, Any]:
            return {"attributes": attributes}

    class FakeExporter:
        def __init__(self, endpoint: str, timeout: float) -> None:
            state["exporter_endpoint"] = endpoint
            state["exporter_timeout"] = timeout

    metric_exporter.OTLPMetricExporter = FakeExporter
    sdk_metrics.MeterProvider = FakeMeterProvider
    sdk_metrics_export.PeriodicExportingMetricReader = FakeReader
    sdk_resources.Resource = FakeResource


def _install_fake_mlflow_modules(
    monkeypatch: pytest.MonkeyPatch,
    state: dict[str, Any],
) -> None:
    mlflow_module = types.ModuleType("mlflow")
    entities_module = types.ModuleType("mlflow.entities")
    tracking_module = types.ModuleType("mlflow.tracking")

    class FakeMetric:
        def __init__(self, key: str, value: float, timestamp: int, step: int) -> None:
            self.key = key
            self.value = value
            self.timestamp = timestamp
            self.step = step

    class FakeClient:
        def log_batch(
            self,
            *,
            run_id: str,
            metrics: list[FakeMetric],
            params: list[Any],
            tags: list[Any],
        ) -> None:
            state["log_batch_calls"].append(
                {"run_id": run_id, "metrics": metrics, "params": params, "tags": tags}
            )

    class FakeRun:
        def __init__(self, run_id: str) -> None:
            self.info = types.SimpleNamespace(run_id=run_id)

    def set_tracking_uri(uri: str) -> None:
        state["tracking_uri"] = uri

    def set_experiment(name: str) -> None:
        state["experiment"] = name

    def start_run(run_name: str | None = None) -> FakeRun:
        state["run_name"] = run_name
        return FakeRun("live-run-1")

    def set_tags(tags: dict[str, str]) -> None:
        state["tags"] = tags

    def set_tag(key: str, value: str) -> None:
        state.setdefault("single_tags", {})[key] = value

    def end_run() -> None:
        state["end_run_called"] = True

    mlflow_module.set_tracking_uri = set_tracking_uri  # type: ignore[attr-defined]
    mlflow_module.set_experiment = set_experiment  # type: ignore[attr-defined]
    mlflow_module.start_run = start_run  # type: ignore[attr-defined]
    mlflow_module.set_tags = set_tags  # type: ignore[attr-defined]
    mlflow_module.set_tag = set_tag  # type: ignore[attr-defined]
    mlflow_module.end_run = end_run  # type: ignore[attr-defined]
    entities_module.Metric = FakeMetric  # type: ignore[attr-defined]
    tracking_module.MlflowClient = FakeClient  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(sys.modules, "mlflow.entities", entities_module)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking_module)


def _build_config(
    tmp_path: Path, *, endpoint_url: str | None
) -> OTelStreamingFanoutConfig:
    return OTelStreamingFanoutConfig(
        endpoint_url=endpoint_url,
        request_timeout_seconds=5.0,
        export_interval_millis=100,
        export_timeout_millis=1000,
        resource_attributes={"service.name": "aiperf"},
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment="aiperf-tests",
        mlflow_run_name="live-test",
        mlflow_tags={"team": "perf"},
        benchmark_id="bench-1",
        metadata_file=tmp_path / "mlflow_export.json",
    )


def test_run_fanout_processes_events_for_otel_and_mlflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    otel_state: dict[str, Any] = {}
    mlflow_state: dict[str, Any] = {"log_batch_calls": []}
    _install_fake_otel_modules(monkeypatch, otel_state)
    _install_fake_mlflow_modules(monkeypatch, mlflow_state)

    queue = _SequenceQueue(
        [
            {
                "type": "histogram_record",
                "payload": {
                    "metric_name": "aiperf.request_latency_ns",
                    "unit": "ns",
                    "description": "latency",
                    "value": 123.0,
                    "attributes": {"aiperf.worker.id": "worker-1"},
                },
            },
            {
                "type": "counter_add",
                "payload": {
                    "metric_name": "aiperf.requests.completed",
                    "unit": "1",
                    "description": "completed",
                    "value": 1.0,
                    "attributes": {"aiperf.worker.id": "worker-1"},
                },
            },
            {"type": "flush", "payload": {}},
            {"type": "shutdown", "payload": {}},
        ]
    )
    config = _build_config(tmp_path, endpoint_url="http://collector:4318/v1/metrics")

    run_otel_streaming_fanout(queue, config)

    meter = otel_state["meter"]
    assert "aiperf.request_latency_ns" in meter.histograms
    assert "aiperf.requests.completed" in meter.counters
    assert otel_state["force_flush_calls"]
    assert otel_state["shutdown_calls"] == 1

    assert mlflow_state["log_batch_calls"]
    logged_metric_keys = [
        metric.key
        for call in mlflow_state["log_batch_calls"]
        for metric in call["metrics"]
    ]
    assert "live.aiperf.request_latency_ns" in logged_metric_keys
    assert "live.aiperf.requests.completed" in logged_metric_keys
    assert mlflow_state["end_run_called"] is True

    metadata = orjson.loads((tmp_path / "mlflow_export.json").read_bytes())
    assert metadata["run_id"] == "live-run-1"
    assert metadata["live_streaming"] is True


def test_run_fanout_without_otel_sink_still_logs_mlflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mlflow_state: dict[str, Any] = {"log_batch_calls": []}
    _install_fake_mlflow_modules(monkeypatch, mlflow_state)

    queue = _SequenceQueue(
        [
            {
                "type": "counter_add",
                "payload": {
                    "metric_name": "aiperf.requests.completed",
                    "unit": "1",
                    "description": "completed",
                    "value": 1.0,
                    "attributes": {},
                },
            },
            {"type": "shutdown", "payload": {}},
        ]
    )
    config = _build_config(tmp_path, endpoint_url=None)

    run_otel_streaming_fanout(queue, config)

    assert mlflow_state["log_batch_calls"]
    assert mlflow_state["end_run_called"] is True


def test_run_fanout_invalid_payload_logs_warning_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    otel_state: dict[str, Any] = {}
    _install_fake_otel_modules(monkeypatch, otel_state)

    queue = _SequenceQueue(
        [
            {
                "type": "histogram_record",
                "payload": {
                    # Missing required fields on purpose.
                    "metric_name": "aiperf.invalid_payload",
                },
            },
            {"type": "shutdown", "payload": {}},
        ]
    )
    config = OTelStreamingFanoutConfig(
        endpoint_url="http://collector:4318/v1/metrics",
        request_timeout_seconds=5.0,
        export_interval_millis=100,
        export_timeout_millis=1000,
        resource_attributes={"service.name": "aiperf"},
        mlflow_tracking_uri=None,
        mlflow_experiment="aiperf-tests",
        mlflow_run_name=None,
        mlflow_tags={},
        benchmark_id=None,
        metadata_file=tmp_path / "mlflow_export.json",
    )

    run_otel_streaming_fanout(queue, config)

    assert "Invalid histogram fanout payload received" in caplog.text
    assert otel_state["shutdown_calls"] == 1
