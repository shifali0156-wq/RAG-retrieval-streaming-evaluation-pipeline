"""
Latency and streaming metrics: TTFA, retrieval time, total time, overlap ratio.

Computed only on successful questions.
"""

from typing import Any, List, Optional

from harness.client import AskResult
from harness.metrics.base import Metric, MetricResult, percentile, mean


class TTFA(Metric):
    """Time to First Audio (ms) - median across questions."""
    
    def __init__(self, p: int = 50):
        self._p = p
    
    @property
    def name(self) -> str:
        return f"ttfa_p{self._p}_ms"
    
    @property
    def direction(self) -> str:
        return "lower"
    
    def compute(
        self,
        records: List[AskResult],
        dataset: Any,
        config: dict
    ) -> MetricResult:
        successful = [r for r in records if r.success and r.t_first_audio > 0]
        
        if not successful:
            return MetricResult(
                name=self.name,
                value=None,
                direction=self.direction,
                details={"reason": "no successful questions"}
            )
        
        values = [(r.t_first_audio - r.t_send) * 1000 for r in successful]
        value = percentile(values, self._p)
        
        return MetricResult(
            name=self.name,
            value=value,
            direction=self.direction,
            details={"n": len(successful), "values_ms": values}
        )


class RetrievalTime(Metric):
    """Retrieval time (ms) - median across questions."""
    
    def __init__(self, p: int = 50):
        self._p = p
    
    @property
    def name(self) -> str:
        return f"retrieval_p{self._p}_ms"
    
    @property
    def direction(self) -> str:
        return "lower"
    
    def compute(
        self,
        records: List[AskResult],
        dataset: Any,
        config: dict
    ) -> MetricResult:
        successful = [r for r in records if r.success and r.t_retrieved > 0]
        
        if not successful:
            return MetricResult(
                name=self.name,
                value=None,
                direction=self.direction,
                details={"reason": "no successful questions"}
            )
        
        values = [(r.t_retrieved - r.t_send) * 1000 for r in successful]
        value = percentile(values, self._p)
        
        return MetricResult(
            name=self.name,
            value=value,
            direction=self.direction,
            details={"n": len(successful), "values_ms": values}
        )


class TotalTime(Metric):
    """Total answer time (ms) - median across questions."""
    
    def __init__(self, p: int = 50):
        self._p = p
    
    @property
    def name(self) -> str:
        return f"total_p{self._p}_ms"
    
    @property
    def direction(self) -> str:
        return "lower"
    
    def compute(
        self,
        records: List[AskResult],
        dataset: Any,
        config: dict
    ) -> MetricResult:
        successful = [r for r in records if r.success and r.t_done > 0]
        
        if not successful:
            return MetricResult(
                name=self.name,
                value=None,
                direction=self.direction,
                details={"reason": "no successful questions"}
            )
        
        values = [(r.t_done - r.t_send) * 1000 for r in successful]
        value = percentile(values, self._p)
        
        return MetricResult(
            name=self.name,
            value=value,
            direction=self.direction,
            details={"n": len(successful), "values_ms": values}
        )


class OverlapRatio(Metric):
    """
    Streaming overlap ratio = (t_done - t_first_audio) / (t_done - t_send).
    
    Higher = audio streamed progressively; near 0 = burst after full generation.
    """
    
    @property
    def name(self) -> str:
        return "overlap_ratio_mean"
    
    @property
    def direction(self) -> str:
        return "higher"
    
    def compute(
        self,
        records: List[AskResult],
        dataset: Any,
        config: dict
    ) -> MetricResult:
        successful = [
            r for r in records 
            if r.success and r.t_first_audio > 0 and r.t_done > r.t_send
        ]
        
        if not successful:
            return MetricResult(
                name=self.name,
                value=None,
                direction=self.direction,
                details={"reason": "no successful questions with valid timing"}
            )
        
        ratios = []
        for r in successful:
            total = r.t_done - r.t_send
            if total > 0:
                overlap = (r.t_done - r.t_first_audio) / total
                ratios.append(overlap)
        
        value = mean(ratios)
        
        return MetricResult(
            name=self.name,
            value=value,
            direction=self.direction,
            details={"n": len(ratios), "values": ratios}
        )


# Pre-instantiated metrics for common percentiles
class TTFA_P50(TTFA):
    def __init__(self):
        super().__init__(50)


class TTFA_P95(TTFA):
    def __init__(self):
        super().__init__(95)


class Retrieval_P50(RetrievalTime):
    def __init__(self):
        super().__init__(50)


class Total_P50(TotalTime):
    def __init__(self):
        super().__init__(50)