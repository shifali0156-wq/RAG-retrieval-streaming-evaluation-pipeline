"""
Tests for retrieval metrics: Recall@k and MRR.
"""

import pytest
from dataclasses import dataclass
from typing import List, Optional

from harness.client import AskResult, TimedEvent
from harness.contract import RetrievedEvent
from harness.metrics.retrieval import RecallAt1, RecallAt3, RecallAt5, MRR
from harness.metrics.base import MetricResult


def make_result(question_id: str, passage_ids: List[str], success: bool = True) -> AskResult:
    """Create a mock AskResult with a retrieved event."""
    result = AskResult(question_id=question_id, success=success)
    if success:
        event = RetrievedEvent(type="retrieved", passage_ids=passage_ids)
        result.events = [TimedEvent(event=event, timestamp=0.0)]
    return result


def make_dataset(questions: List[dict]) -> dict:
    """Create a mock dataset."""
    return {"questions": questions, "passages": []}


class TestRecallAtK:
    """Tests for Recall@k metrics."""
    
    def test_recall_at_1_perfect(self):
        """All gold passages in top-1."""
        records = [
            make_result("q1", ["p01"]),
            make_result("q2", ["p02"]),
        ]
        dataset = make_dataset([
            {"id": "q1", "gold_passage_ids": ["p01"]},
            {"id": "q2", "gold_passage_ids": ["p02"]},
        ])
        
        metric = RecallAt1()
        result = metric.compute(records, dataset, {})
        
        assert result.value == 1.0
        assert result.name == "recall_at_1"
    
    def test_recall_at_1_partial(self):
        """Some gold passages in top-1."""
        records = [
            make_result("q1", ["p01"]),  # hit
            make_result("q2", ["p03"]),  # miss (gold is p02)
        ]
        dataset = make_dataset([
            {"id": "q1", "gold_passage_ids": ["p01"]},
            {"id": "q2", "gold_passage_ids": ["p02"]},
        ])
        
        metric = RecallAt1()
        result = metric.compute(records, dataset, {})
        
        assert result.value == 0.5
    
    def test_recall_at_3_with_gold_in_top3(self):
        """Gold passage appears in top-3 but not top-1."""
        records = [
            make_result("q1", ["p02", "p01", "p03"]),  # gold p01 at rank 2
        ]
        dataset = make_dataset([
            {"id": "q1", "gold_passage_ids": ["p01"]},
        ])
        
        metric = RecallAt3()
        result = metric.compute(records, dataset, {})
        
        assert result.value == 1.0
    
    def test_recall_at_5_multiple_gold(self):
        """Multiple gold passages, any in top-5 counts."""
        records = [
            make_result("q1", ["p05", "p01", "p02", "p03", "p04"]),  # gold p01, p02 at ranks 2,3
        ]
        dataset = make_dataset([
            {"id": "q1", "gold_passage_ids": ["p01", "p02"]},
        ])
        
        metric = RecallAt5()
        result = metric.compute(records, dataset, {})
        
        assert result.value == 1.0
    
    def test_unanswerable_question_excluded(self):
        """Unanswerable questions (empty gold) are excluded from recall."""
        records = [
            make_result("q1", ["p01"]),      # answerable
            make_result("q2", ["p02"]),      # unanswerable
        ]
        dataset = make_dataset([
            {"id": "q1", "gold_passage_ids": ["p01"]},
            {"id": "q2", "gold_passage_ids": []},  # unanswerable
        ])
        
        metric = RecallAt5()
        result = metric.compute(records, dataset, {})
        
        # Only q1 counted, q2 excluded
        assert result.value == 1.0
        assert result.details["per_question"]["q1"]["hit"] is True
        assert "q2" not in result.details["per_question"]
    
    def test_failed_question_excluded(self):
        """Failed but answerable questions count as misses (0 recall)."""
        records = [
            make_result("q1", ["p01"]),      # success
            make_result("q2", [], success=False),  # failed
        ]
        dataset = make_dataset([
            {"id": "q1", "gold_passage_ids": ["p01"]},
            {"id": "q2", "gold_passage_ids": ["p02"]},
        ])
        
        metric = RecallAt5()
        result = metric.compute(records, dataset, {})
        
        # q1 hit, q2 miss (failed) -> 1/2 = 0.5
        assert result.value == 0.5
        # Both answerable questions should be in details
        assert "q1" in result.details["per_question"]
        assert "q2" in result.details["per_question"]
        assert result.details["per_question"]["q1"]["hit"] is True
        assert result.details["per_question"]["q2"]["hit"] is False
    
    def test_no_answerable_questions(self):
        """All questions unanswerable returns None."""
        records = [
            make_result("q1", ["p01"]),
        ]
        dataset = make_dataset([
            {"id": "q1", "gold_passage_ids": []},
        ])
        
        metric = RecallAt5()
        result = metric.compute(records, dataset, {})
        
        assert result.value is None
        assert result.details["reason"] == "no answerable questions"
    
    def test_truncation_to_10(self):
        """Retrieved lists are truncated to max 10."""
        records = [
            make_result("q1", [f"p{i}" for i in range(15)]),  # 15 passages
        ]
        dataset = make_dataset([
            {"id": "q1", "gold_passage_ids": ["p11"]},  # at position 11 (0-indexed 10)
        ])
        
        metric = RecallAt5()
        result = metric.compute(records, dataset, {})
        
        # p11 is at index 10, truncated to 10 means only indices 0-9 kept
        # So p11 should NOT be in top-5
        assert result.value == 0.0


class TestMRR:
    """Tests for Mean Reciprocal Rank."""
    
    def test_mrr_perfect_rank_1(self):
        """Gold at rank 1 gives RR=1."""
        records = [make_result("q1", ["p01"])]
        dataset = make_dataset([{"id": "q1", "gold_passage_ids": ["p01"]}])
        
        metric = MRR()
        result = metric.compute(records, dataset, {})
        
        assert result.value == 1.0
    
    def test_mrr_rank_3(self):
        """Gold at rank 3 gives RR=1/3."""
        records = [make_result("q1", ["p02", "p03", "p01"])]
        dataset = make_dataset([{"id": "q1", "gold_passage_ids": ["p01"]}])
        
        metric = MRR()
        result = metric.compute(records, dataset, {})
        
        assert abs(result.value - 1/3) < 0.001
    
    def test_mrr_no_gold_in_list(self):
        """No gold in retrieved list gives RR=0."""
        records = [make_result("q1", ["p02", "p03", "p04"])]
        dataset = make_dataset([{"id": "q1", "gold_passage_ids": ["p01"]}])
        
        metric = MRR()
        result = metric.compute(records, dataset, {})
        
        assert result.value == 0.0
    
    def test_mrr_multiple_questions(self):
        """MRR averages across questions."""
        records = [
            make_result("q1", ["p01"]),           # RR=1
            make_result("q2", ["p02", "p03"]),    # RR=0 (gold p01 not in list)
            make_result("q3", ["p05", "p01"]),    # RR=1/2
        ]
        dataset = make_dataset([
            {"id": "q1", "gold_passage_ids": ["p01"]},
            {"id": "q2", "gold_passage_ids": ["p01"]},
            {"id": "q3", "gold_passage_ids": ["p01"]},
        ])
        
        metric = MRR()
        result = metric.compute(records, dataset, {})
        
        expected = (1.0 + 0.0 + 0.5) / 3
        assert abs(result.value - expected) < 0.001
    
    def test_mrr_excludes_unanswerable(self):
        """Unanswerable questions excluded from MRR."""
        records = [
            make_result("q1", ["p01"]),
            make_result("q2", ["p02"]),
        ]
        dataset = make_dataset([
            {"id": "q1", "gold_passage_ids": ["p01"]},
            {"id": "q2", "gold_passage_ids": []},
        ])
        
        metric = MRR()
        result = metric.compute(records, dataset, {})
        
        assert result.value == 1.0  # Only q1 counted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
