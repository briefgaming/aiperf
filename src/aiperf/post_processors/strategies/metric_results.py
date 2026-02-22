# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.post_processors.strategies.core import (
    OTelResultData,
    OTelResultsStrategyProtocol,
    OTelStrategyContextProtocol,
)


class MetricResultsStrategy(OTelResultsStrategyProtocol):
    """Streams per-request metric records as histogram observations."""

    def __init__(self, context: OTelStrategyContextProtocol) -> None:
        self._context = context

    def supports(self, record_data: OTelResultData) -> bool:
        return isinstance(record_data, MetricRecordsData)

    async def process(self, record_data: OTelResultData) -> None:
        if not isinstance(record_data, MetricRecordsData):
            return

        attributes = self._context._build_record_attributes(record_data)
        for metric_name, metric_value in record_data.metrics.items():
            numeric_values = self._context._coerce_metric_values(
                metric_name, metric_value
            )
            if not numeric_values:
                continue
            instrument = await self._context._get_or_create_histogram(metric_name)
            for value in numeric_values:
                instrument.record(value, attributes)
