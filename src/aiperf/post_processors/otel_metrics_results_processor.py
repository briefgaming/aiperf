# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
from typing import Any

from aiperf.common.config import UserConfig
from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.models import CreditPhaseStats, MetricResult
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor
from aiperf.post_processors.strategies import (
    MetricResultsStrategy,
    OTelResultData,
    OTelResultsStrategyProtocol,
    TimingResultsStrategy,
)


class OTelMetricsResultsProcessor(BaseMetricsProcessor):
    """Streams per-record metrics to an OpenTelemetry collector."""

    def __init__(
        self,
        service_id: str | None,
        user_config: UserConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(user_config=user_config, **kwargs)
        self.service_id = service_id or "records-manager"
        self._otel_metrics_urls = user_config.otel_metrics_urls
        if not self._otel_metrics_urls:
            self.info("OTel metrics streaming is disabled (set --otel-url to enable)")
            raise PostProcessorDisabled(
                "OTel metrics streaming is disabled (set --otel-url to enable)"
            )
        try:
            import opentelemetry.exporter.otlp.proto.http.metric_exporter  # noqa: F401
            import opentelemetry.sdk.metrics  # noqa: F401
        except ImportError as exc:
            self.warning(
                "OpenTelemetry metrics dependencies are not installed. "
                "Install with: uv add opentelemetry-sdk "
                "opentelemetry-exporter-otlp-proto-http. "
                f"ImportError={exc!r}. python_executable={sys.executable}"
            )
            raise PostProcessorDisabled(
                "OpenTelemetry metrics dependencies are not installed"
            ) from exc

        self._meter_provider: Any | None = None
        self._meter: Any | None = None
        self._histogram_instruments: dict[str, Any] = {}
        self._counter_instruments: dict[str, Any] = {}
        self._up_down_counter_instruments: dict[str, Any] = {}
        self._timing_counter_state: dict[tuple[CreditPhase, str], int] = {}
        self._timing_gauge_state: dict[tuple[CreditPhase, str], float] = {}
        self._instrument_lock = asyncio.Lock()
        self._result_strategies: list[OTelResultsStrategyProtocol] = []
        if self.user_config.otel_stream_metrics_enabled:
            self._result_strategies.append(MetricResultsStrategy(self))
        if self.user_config.otel_stream_timing_enabled:
            self._result_strategies.append(TimingResultsStrategy(self))
        if not self._result_strategies:
            raise PostProcessorDisabled(
                "OTel telemetry selection disabled all stream domains."
            )
        self._export_timeout_millis = self._to_millis(
            Environment.OTEL.REQUEST_TIMEOUT_SECONDS,
            minimum=1000,
        )
        self.info("Initialized OTelMetricsResultsProcessor")

    @on_init
    async def _initialize_meter_provider(self) -> None:
        """Initialize the OpenTelemetry meter provider."""
        self.info("Initializing OpenTelemetry meter provider")
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import (
                PeriodicExportingMetricReader,
            )
            from opentelemetry.sdk.resources import Resource
        except ImportError as exc:
            self.warning(
                "OpenTelemetry metrics dependencies are not installed. "
                f"ImportError={exc!r}. python_executable={sys.executable}"
            )
            raise PostProcessorDisabled(
                "OpenTelemetry metrics dependencies are not installed. Install with: "
                "uv add opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
            ) from exc

        resource = Resource.create(self._build_resource_attributes())
        exporter = OTLPMetricExporter(
            endpoint=self._otel_metrics_urls[0],
            timeout=Environment.OTEL.REQUEST_TIMEOUT_SECONDS,
        )

        reader = PeriodicExportingMetricReader(
            exporter,
            export_interval_millis=self._to_millis(
                Environment.OTEL.FLUSH_INTERVAL_SECONDS,
                minimum=100,
            ),
            export_timeout_millis=self._export_timeout_millis,
        )
        self._meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[reader],
        )
        self._meter = self._meter_provider.get_meter("aiperf.records")
        self.info(f"OTel metrics streaming enabled: {self._otel_metrics_urls[0]}")

    async def process_result(self, record_data: OTelResultData) -> None:
        """Record metric data for export via the OpenTelemetry SDK."""
        if self._meter is None:
            return

        for strategy in self._result_strategies:
            if strategy.supports(record_data):
                await strategy.process(record_data)
                return

        self.debug(
            lambda: (
                f"Skipping unsupported OTel result payload type: {type(record_data)}"
            )
        )

    async def flush(self, *, force: bool = False) -> None:
        """Force a flush of pending SDK metrics exports."""
        if self._meter_provider is None:
            return
        await asyncio.to_thread(
            self._meter_provider.force_flush,
            timeout_millis=self._export_timeout_millis,
        )

    async def summarize(self) -> list[MetricResult]:
        return []

    @on_stop
    async def _flush_and_shutdown(self) -> None:
        """Final flush before shutdown and close SDK resources."""
        try:
            await self.flush(force=True)
        except Exception as exc:
            self.warning(f"Failed to flush metrics: {exc}")
        finally:
            if self._meter_provider is not None:
                await asyncio.to_thread(self._meter_provider.shutdown)
                self._meter_provider = None
                self._meter = None

    async def _get_or_create_histogram(self, metric_name: str) -> Any:
        """Create or reuse a histogram instrument for a metric name."""
        if metric_name in self._histogram_instruments:
            return self._histogram_instruments[metric_name]

        async with self._instrument_lock:
            if metric_name in self._histogram_instruments:
                return self._histogram_instruments[metric_name]
            if self._meter is None:
                raise RuntimeError("OTel meter is not initialized")
            instrument = self._meter.create_histogram(
                name=f"aiperf.{metric_name}",
                unit=self._metric_unit(metric_name),
                description=f"AIPerf streaming metric: {metric_name}",
            )
            self._histogram_instruments[metric_name] = instrument
            return instrument

    async def _get_or_create_counter(
        self, metric_name: str, unit: str, description: str
    ) -> Any:
        """Create or reuse a counter instrument."""
        if metric_name in self._counter_instruments:
            return self._counter_instruments[metric_name]

        async with self._instrument_lock:
            if metric_name in self._counter_instruments:
                return self._counter_instruments[metric_name]
            if self._meter is None:
                raise RuntimeError("OTel meter is not initialized")
            instrument = self._meter.create_counter(
                name=metric_name,
                unit=unit,
                description=description,
            )
            self._counter_instruments[metric_name] = instrument
            return instrument

    async def _get_or_create_up_down_counter(
        self, metric_name: str, unit: str, description: str
    ) -> Any:
        """Create or reuse an up-down counter instrument."""
        if metric_name in self._up_down_counter_instruments:
            return self._up_down_counter_instruments[metric_name]

        async with self._instrument_lock:
            if metric_name in self._up_down_counter_instruments:
                return self._up_down_counter_instruments[metric_name]
            if self._meter is None:
                raise RuntimeError("OTel meter is not initialized")
            instrument = self._meter.create_up_down_counter(
                name=metric_name,
                unit=unit,
                description=description,
            )
            self._up_down_counter_instruments[metric_name] = instrument
            return instrument

    def _build_resource_attributes(self) -> dict[str, str]:
        """Build OTLP resource attributes shared across all streamed metrics."""
        attributes: dict[str, str] = {}
        attributes["service.name"] = "aiperf"
        attributes["service.instance.id"] = self.service_id
        if self.user_config.benchmark_id is not None:
            attributes["aiperf.benchmark.id"] = self.user_config.benchmark_id
        attributes["aiperf.endpoint.type"] = str(self.user_config.endpoint.type)
        attributes["aiperf.model.name"] = self.user_config.endpoint.model_names[0]
        return attributes

    def _build_record_attributes(self, record: MetricRecordsData) -> dict[str, Any]:
        """Build OTLP metric attributes for an individual metric record."""
        metadata = record.metadata
        attributes: dict[str, Any] = {}
        attributes["aiperf.worker.id"] = metadata.worker_id
        attributes["aiperf.record_processor.id"] = metadata.record_processor_id
        attributes["aiperf.benchmark_phase"] = str(metadata.benchmark_phase)
        if metadata.session_num is not None:
            attributes["aiperf.session_num"] = metadata.session_num
        if metadata.turn_index is not None:
            attributes["aiperf.turn_index"] = metadata.turn_index
        attributes["aiperf.was_cancelled"] = metadata.was_cancelled
        attributes["aiperf.has_error"] = record.error is not None
        return attributes

    def _build_timing_attributes(self, stats: CreditPhaseStats) -> dict[str, Any]:
        """Build OTLP metric attributes for phase-level timing metrics."""
        attributes: dict[str, Any] = {}
        attributes["aiperf.benchmark_phase"] = str(stats.phase)
        if stats.total_expected_requests is not None:
            attributes["aiperf.total_expected_requests"] = stats.total_expected_requests
        if stats.expected_duration_sec is not None:
            attributes["aiperf.expected_duration_sec"] = stats.expected_duration_sec
        if stats.expected_num_sessions is not None:
            attributes["aiperf.expected_num_sessions"] = stats.expected_num_sessions
        return attributes

    def _calculate_timing_counter_delta(
        self, *, metric_name: str, phase: CreditPhase, current_value: int
    ) -> int:
        """Calculate delta from cumulative timing counters."""
        key = (phase, metric_name)
        previous_value = self._timing_counter_state.get(key)
        self._timing_counter_state[key] = current_value
        if previous_value is None:
            return current_value
        if current_value < previous_value:
            self.warning(
                f"Timing counter reset detected for {metric_name} ({phase}). "
                f"current={current_value}, previous={previous_value}"
            )
            return current_value
        return current_value - previous_value

    def _calculate_timing_gauge_delta(
        self, *, metric_name: str, phase: CreditPhase, current_value: float
    ) -> float:
        """Calculate delta required to represent the latest gauge-like snapshot."""
        key = (phase, metric_name)
        previous_value = self._timing_gauge_state.get(key)
        self._timing_gauge_state[key] = current_value
        if previous_value is None:
            return current_value
        return current_value - previous_value

    def _coerce_metric_values(self, metric_name: str, metric_value: Any) -> list[float]:
        """Convert metric value to numeric values suitable for histograms."""
        if isinstance(metric_value, bool):
            return []
        if isinstance(metric_value, int | float):
            return [float(metric_value)]
        if isinstance(metric_value, list):
            numeric_values = [
                float(value)
                for value in metric_value
                if isinstance(value, int | float) and not isinstance(value, bool)
            ]
            if not numeric_values:
                return []
            return numeric_values
        self.debug(
            lambda: (
                f"Skipping unsupported OTel metric value type for "
                f"'{metric_name}': {type(metric_value)}"
            )
        )
        return []

    def _metric_unit(self, metric_name: str) -> str:
        """Return a unit string for a metric name."""
        if metric_name.endswith("_ns"):
            return "ns"
        return "1"

    def _timing_unit(self, metric_name: str) -> str:
        """Return a unit string for timing metrics."""
        if metric_name.endswith("_sec"):
            return "s"
        return "1"

    def _to_millis(self, seconds: float, *, minimum: int) -> int:
        """Convert seconds to milliseconds with a minimum bound."""
        return max(minimum, int(seconds * 1000))
