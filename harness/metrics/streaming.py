"""
Streaming metrics: Playback stall simulation.

Simulates a player that starts at first audio chunk and plays in real time.
Tracks buffered_end and detects stalls when chunks arrive late.
"""

from typing import Any, List, Optional

from harness.client import AskResult
from harness.metrics.base import Metric, MetricResult, mean


class StallMetrics(Metric):
    """
    Stall simulation metrics.
    
    Computes mean total stall time (ms) and mean stall count per answer.
    """
    
    @property
    def name(self) -> str:
        return "stall_ms_mean"
    
    @property
    def direction(self) -> str:
        return "lower"
    
    def compute(
        self,
        records: List[AskResult],
        dataset: Any,
        config: dict
    ) -> MetricResult:
        successful = [r for r in records if r.success]
        
        if not successful:
            return MetricResult(
                name=self.name,
                value=None,
                direction=self.direction,
                details={"reason": "no successful questions"}
            )
        
        stall_ms_values = [r.total_stall_ms for r in successful]
        stall_count_values = [r.stall_count for r in successful]
        
        value = mean(stall_ms_values)
        stall_count_mean = mean(stall_count_values)
        
        return MetricResult(
            name=self.name,
            value=value,
            direction=self.direction,
            details={
                "n": len(successful),
                "stall_ms_values": stall_ms_values,
                "stall_count_values": stall_count_values,
                "stall_count_mean": stall_count_mean
            }
        )


class StallCountMean(Metric):
    """Mean stall count per answer (reported but not weighted)."""
    
    @property
    def name(self) -> str:
        return "stall_count_mean"
    
    @property
    def direction(self) -> str:
        return "lower"
    
    def compute(
        self,
        records: List[AskResult],
        dataset: Any,
        config: dict
    ) -> MetricResult:
        successful = [r for r in records if r.success]
        
        if not successful:
            return MetricResult(
                name=self.name,
                value=None,
                direction=self.direction,
                details={"reason": "no successful questions"}
            )
        
        stall_count_values = [r.stall_count for r in successful]
        value = mean(stall_count_values)
        
        return MetricResult(
            name=self.name,
            value=value,
            direction=self.direction,
            details={"n": len(successful), "values": stall_count_values}
        )