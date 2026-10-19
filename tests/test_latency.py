"""
Tests for latency metrics: overlap ratio, TTFA, retrieval, total time.
"""

import pytest
from harness.client import AskResult, TimedEvent
from harness.contract import RetrievedEvent, AudioEvent, DoneEvent
from harness.metrics.latency import TTFA_P50, TTFA_P95, Retrieval_P50, Total_P50, OverlapRatio
from harness.metrics.base import MetricResult


def make_result(
    question_id: str,
    t_send: float,
    t_retrieved: float,
    t_first_audio: float,
    t_done: float,
    success: bool = True
) -> AskResult:
    """Create a mock AskResult with specified timings."""
    result = AskResult(question_id=question_id, success=success)
    result.t_send = t_send
    result.t_retrieved = t_retrieved
    result.t_first_audio = t_first_audio
    result.t_done = t_done
    
    # Add minimal events for validation
    if success:
        result.events = [
            TimedEvent(event=RetrievedEvent(type="retrieved", passage_ids=["p01"]), timestamp=t_retrieved),
            TimedEvent(event=AudioEvent(type="audio", data="dummy"), timestamp=t_first_audio),
            TimedEvent(event=DoneEvent(type="done"), timestamp=t_done),
        ]
    return result


def make_dataset() -> dict:
    return {"questions": [{"id": "q1", "gold_passage_ids": ["p01"]}], "passages": []}


class TestLatencyMetrics:
    """Tests for latency and streaming metrics."""
    
    def test_ttfa_p50(self):
        """TTFA p50 is median time to first audio."""
        records = [
            make_result("q1", 0.0, 0.1, 0.5, 2.0),   # TTFA = 500ms
            make_result("q2", 0.0, 0.1, 1.0, 3.0),   # TTFA = 1000ms
            make_result("q3", 0.0, 0.1, 1.5, 4.0),   # TTFA = 1500ms
        ]
        dataset = make_dataset()
        
        metric = TTFA_P50()
        result = metric.compute(records, dataset, {})
        
        assert result.value == 1000.0  # median of [500, 1000, 1500]
    
    def test_ttfa_p95(self):
        """TTFA p95 is 95th percentile."""
        records = [
            make_result(f"q{i}", 0.0, 0.1, 0.1 * i, 2.0) for i in range(1, 21)
        ]  # TTFA: 100, 200, ..., 2000ms
        dataset = make_dataset()
        
        metric = TTFA_P95()
        result = metric.compute(records, dataset, {})
        
        # 95th percentile of 20 values with numpy linear interpolation:
        # index = 0.95 * (20-1) = 18.05 -> interpolate between 1900 and 2000
        # value = 1900 * 0.95 + 2000 * 0.05 = 1905
        assert abs(result.value - 1905.0) < 0.001
    
    def test_retrieval_p50(self):
        """Retrieval time p50."""
        records = [
            make_result("q1", 0.0, 0.05, 0.5, 2.0),   # retrieval = 50ms
            make_result("q2", 0.0, 0.10, 1.0, 3.0),   # retrieval = 100ms
            make_result("q3", 0.0, 0.20, 1.5, 4.0),   # retrieval = 200ms
        ]
        dataset = make_dataset()
        
        metric = Retrieval_P50()
        result = metric.compute(records, dataset, {})
        
        assert result.value == 100.0
    
    def test_total_p50(self):
        """Total answer time p50."""
        records = [
            make_result("q1", 0.0, 0.1, 0.5, 1.0),   # total = 1000ms
            make_result("q2", 0.0, 0.1, 1.0, 3.0),   # total = 3000ms
            make_result("q3", 0.0, 0.1, 1.5, 5.0),   # total = 5000ms
        ]
        dataset = make_dataset()
        
        metric = Total_P50()
        result = metric.compute(records, dataset, {})
        
        assert result.value == 3000.0
    
    def test_overlap_ratio_progressive(self):
        """Progressive streaming: high overlap ratio."""
        # t_first_audio at 500ms, t_done at 2000ms, t_send at 0
        # overlap = (2000-500) / (2000-0) = 1500/2000 = 0.75
        records = [
            make_result("q1", 0.0, 0.1, 0.5, 2.0),
            make_result("q2", 0.0, 0.1, 0.3, 1.5),
        ]
        dataset = make_dataset()
        
        metric = OverlapRatio()
        result = metric.compute(records, dataset, {})
        
        # q1: (2.0-0.5)/2.0 = 0.75
        # q2: (1.5-0.3)/1.5 = 0.8
        # mean = 0.775
        assert abs(result.value - 0.775) < 0.001
    
    def test_overlap_ratio_burst(self):
        """Burst delivery: low overlap ratio (near 0)."""
        # Audio arrives just before done
        records = [
            make_result("q1", 0.0, 0.1, 1.9, 2.0),
            make_result("q2", 0.0, 0.1, 2.9, 3.0),
        ]
        dataset = make_dataset()
        
        metric = OverlapRatio()
        result = metric.compute(records, dataset, {})
        
        # q1: (2.0-1.9)/2.0 = 0.05
        # q2: (3.0-2.9)/3.0 = 0.033
        # mean ~ 0.042
        assert result.value < 0.1
    
    def test_failed_questions_excluded(self):
        """Failed questions are excluded from latency metrics."""
        records = [
            make_result("q1", 0.0, 0.1, 0.5, 2.0, success=True),
            make_result("q2", 0.0, 0.1, 1.0, 3.0, success=False),
        ]
        dataset = make_dataset()
        
        metric = TTFA_P50()
        result = metric.compute(records, dataset, {})
        
        assert result.value == 500.0  # Only q1 counted
    
    def test_empty_records_returns_none(self):
        """Empty records returns None."""
        records = []
        dataset = make_dataset()
        
        metric = TTFA_P50()
        result = metric.compute(records, dataset, {})
        
        assert result.value is None
        assert result.details["reason"] == "no successful questions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
