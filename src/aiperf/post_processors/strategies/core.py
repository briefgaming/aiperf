# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from aiperf.common.enums import CreditPhase
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.models import CreditPhaseStats

OTelResultData = MetricRecordsData | CreditPhaseStats


@runtime_checkable
class OTelResultsStrategyProtocol(Protocol):
    """Protocol for OTel result processing strategies."""

    def supports(self, record_data: OTelResultData) -> bool: ...

    async def process(self, record_data: OTelResultData) -> None: ...


@runtime_checkable
class OTelStrategyContextProtocol(Protocol):
    """Protocol implemented by the OTel processor to support strategy execution."""

    async def _get_or_create_histogram(self, metric_name: str) -> Any: ...

    async def _get_or_create_counter(
        self, metric_name: str, unit: str, description: str
    ) -> Any: ...

    async def _get_or_create_up_down_counter(
        self, metric_name: str, unit: str, description: str
    ) -> Any: ...

    def _build_record_attributes(self, record: MetricRecordsData) -> dict[str, Any]: ...

    def _build_timing_attributes(self, stats: CreditPhaseStats) -> dict[str, Any]: ...

    def _coerce_metric_values(
        self, metric_name: str, metric_value: Any
    ) -> list[float]: ...

    def _calculate_timing_counter_delta(
        self, *, metric_name: str, phase: CreditPhase, current_value: int
    ) -> int: ...

    def _calculate_timing_gauge_delta(
        self, *, metric_name: str, phase: CreditPhase, current_value: float
    ) -> float: ...

    def _timing_unit(self, metric_name: str) -> str: ...
