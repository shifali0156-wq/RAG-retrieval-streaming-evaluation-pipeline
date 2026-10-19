"""
Tests for streaming metrics: Stall simulation.
"""

import pytest
from harness.client import AskResult, TimedEvent
from harness.contract import RetrievedEvent, AudioEvent, DoneEvent
from harness.metrics.streaming import StallMetrics, StallCountMean
from harness.metrics.base import MetricResult


def make_result_with_audio(
    question_id: str,
    chunk_arrivals: list,  # list of (arrival_time_s, chunk_duration_s)
    t_send: float = 0.0,
    t_retrieved: float = 0.1,
) -> AskResult:
    """Create a mock AskResult with audio events at specified times."""
    result = AskResult(question_id=question_id, success=True)
    result.t_send = t_send
    result.t_retrieved = t_retrieved
    
    events = []
    # Retrieved event
    events.append(TimedEvent(
        event=RetrievedEvent(type="retrieved", passage_ids=["p01"]),
        timestamp=t_retrieved
    ))
    
    # Audio events
    for arrival, duration in chunk_arrivals:
        # Create dummy PCM data of appropriate duration
        num_bytes = int(duration * 48000)  # 48000 bytes/sec
        pcm = b'\x00\x00' * (num_bytes // 2)
        import base64
        b64 = base64.b64encode(pcm).decode('ascii')
        
        events.append(TimedEvent(
            event=AudioEvent(type="audio", data=b64),
            timestamp=t_send + arrival
        ))
    
    # Done event
    last_arrival = chunk_arrivals[-1][0] if chunk_arrivals else 0
    events.append(TimedEvent(
        event=DoneEvent(type="done"),
        timestamp=t_send + last_arrival + 0.01
    ))
    
    result.events = events
    result.compute_derived_metrics(stall_min_ms=200.0)
    return result


def make_dataset() -> dict:
    return {"questions": [{"id": "q1", "gold_passage_ids": ["p01"]}], "passages": []}


class TestStallSimulation:
    """Tests for playback stall detection."""
    
    def test_no_stall_continuous_streaming(self):
        """Continuous chunks arriving before buffer empties = no stalls."""
        # Chunks arrive every 50ms, each chunk is 100ms of audio
        # So buffer always has 50ms+ remaining when next chunk arrives
        chunk_arrivals = [
            (0.15, 0.1),   # first audio at 150ms, 100ms duration
            (0.20, 0.1),   # arrives at 200ms, buffer had 50ms left
            (0.25, 0.1),   # arrives at 250ms, buffer had 50ms left
            (0.30, 0.1),   # arrives at 300ms, buffer had 50ms left
        ]
        result = make_result_with_audio("q1", chunk_arrivals)
        
        assert result.stall_count == 0
        assert result.total_stall_ms == 0.0
    
    def test_stall_detected(self):
        """Gap between chunks larger than chunk duration = stall."""
        # First chunk 100ms at 150ms
        # Second chunk at 500ms (350ms gap) - buffer empty at 250ms
        # Stall = 500ms - 250ms = 250ms
        chunk_arrivals = [
            (0.15, 0.1),   # arrives 150ms, 100ms audio -> buffer until 250ms
            (0.50, 0.1),   # arrives 500ms, buffer empty since 250ms -> 250ms stall
        ]
        result = make_result_with_audio("q1", chunk_arrivals)
        
        # Stall gap = 0.50 - 0.25 = 0.25s = 250ms
        assert result.stall_count == 1  # 250ms >= 200ms threshold
        assert abs(result.total_stall_ms - 250.0) < 1.0
    
    def test_stall_below_threshold_not_counted(self):
        """Stall < 200ms adds to total_stall_ms but doesn't increment stall_count."""
        chunk_arrivals = [
            (0.15, 0.1),   # buffer until 250ms
            (0.30, 0.1),   # arrives 300ms, buffer empty since 250ms -> 50ms stall
        ]
        result = make_result_with_audio("q1", chunk_arrivals)
        
        assert result.stall_count == 0  # 50ms < 200ms threshold
        assert abs(result.total_stall_ms - 50.0) < 1.0
    
    def test_multiple_stalls(self):
        """Multiple separate stalls are counted correctly."""
        chunk_arrivals = [
            (0.15, 0.1),   # buffer until 250ms
            (0.50, 0.1),   # stall 250ms (arrives 500ms, buffer empty at 250ms)
            (0.70, 0.1),   # buffer until 800ms, arrives 700ms -> 100ms stall (below threshold)
            (1.20, 0.1),   # stall 400ms (arrives 1200ms, buffer empty at 800ms)
        ]
        result = make_result_with_audio("q1", chunk_arrivals)
        
        # Total stall = 250 + 100 + 400 = 750ms
        # Stall count = 2 (250ms >= 200ms and 400ms >= 200ms, but 100ms < 200ms)
        assert result.stall_count == 2
        assert abs(result.total_stall_ms - 750.0) < 1.0  # 250 + 100 + 400
    
    def test_stall_metrics_aggregation(self):
        """StallMetrics computes mean across questions."""
        # Question 1: no stalls
        r1 = make_result_with_audio("q1", [(0.15, 0.1), (0.20, 0.1), (0.25, 0.1)])
        # Question 2: one stall of 300ms
        r2 = make_result_with_audio("q2", [(0.15, 0.1), (0.55, 0.1)])
        
        records = [r1, r2]
        dataset = make_dataset()
        
        metric = StallMetrics()
        result = metric.compute(records, dataset, {})
        
        # Mean stall_ms = (0 + 300) / 2 = 150
        assert abs(result.value - 150.0) < 1.0
        
        metric_count = StallCountMean()
        result_count = metric_count.compute(records, dataset, {})
        
        # Mean stall_count = (0 + 1) / 2 = 0.5
        assert abs(result_count.value - 0.5) < 0.01
    
    def test_empty_chunks_ignored(self):
        """Empty audio chunks are ignored in stall simulation."""
        result = make_result_with_audio("q1", [(0.15, 0.1)])
        # Manually add an empty audio event
        from harness.contract import AudioEvent
        result.events.insert(-1, TimedEvent(
            event=AudioEvent(type="audio", data=""),
            timestamp=0.2
        ))
        result.compute_derived_metrics(stall_min_ms=200.0)
        
        # Should still work
        assert result.stall_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
