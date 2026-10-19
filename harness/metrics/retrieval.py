"""
Retrieval quality metrics: Recall@k and MRR.

Computed only on answerable questions (those with non-empty gold_passage_ids).
Uses the ranked "retrieved" list from the first run, truncated to max 10.
"""

from typing import Any, List, Optional, Set

from harness.client import AskResult
from harness.metrics.base import Metric, MetricResult, percentile, mean


class RecallAtK(Metric):
    """Recall@k: fraction of questions where ANY gold passage appears in top-k."""
    
    def __init__(self, k: int):
        self._k = k
    
    @property
    def name(self) -> str:
        return f"recall_at_{self._k}"
    
    @property
    def direction(self) -> str:
        return "higher"
    
    def compute(
        self,
        records: List[AskResult],
        dataset: Any,
        config: dict
    ) -> MetricResult:
        questions = dataset.get("questions", [])
        answerable = [q for q in questions if q.get("gold_passage_ids")]
        
        if not answerable:
            return MetricResult(
                name=self.name,
                value=None,
                direction=self.direction,
                details={"reason": "no answerable questions"}
            )
        
        # Build lookup: question_id -> retrieved passage_ids (from first run)
        # Records are already median-aggregated per question
        retrieved_map = {}
        for r in records:
            if r.success and r.events:
                for te in r.events:
                    if hasattr(te.event, 'passage_ids'):
                        retrieved_map[r.question_id] = te.event.passage_ids[:10]  # truncate to 10
                        break
        
        hits = 0
        details = {"per_question": {}}
        
        for q in answerable:
            qid = q["id"]
            gold_set: Set[str] = set(q.get("gold_passage_ids", []))
            retrieved = retrieved_map.get(qid, [])
            retrieved_set = set(retrieved[:self._k])
            
            hit = len(gold_set & retrieved_set) > 0
            if hit:
                hits += 1
            
            details["per_question"][qid] = {
                "gold": list(gold_set),
                "retrieved_top_k": retrieved[:self._k],
                "hit": hit
            }
        
        value = hits / len(answerable)
        
        return MetricResult(
            name=self.name,
            value=value,
            direction=self.direction,
            details=details
        )


class MRR(Metric):
    """Mean Reciprocal Rank: mean of 1/rank of first gold passage (0 if none)."""
    
    @property
    def name(self) -> str:
        return "mrr"
    
    @property
    def direction(self) -> str:
        return "higher"
    
    def compute(
        self,
        records: List[AskResult],
        dataset: Any,
        config: dict
    ) -> MetricResult:
        questions = dataset.get("questions", [])
        answerable = [q for q in questions if q.get("gold_passage_ids")]
        
        if not answerable:
            return MetricResult(
                name=self.name,
                value=None,
                direction=self.direction,
                details={"reason": "no answerable questions"}
            )
        
        retrieved_map = {}
        for r in records:
            if r.success and r.events:
                for te in r.events:
                    if hasattr(te.event, 'passage_ids'):
                        retrieved_map[r.question_id] = te.event.passage_ids[:10]
                        break
        
        reciprocal_ranks = []
        details = {"per_question": {}}
        
        for q in answerable:
            qid = q["id"]
            gold_set: Set[str] = set(q.get("gold_passage_ids", []))
            retrieved = retrieved_map.get(qid, [])
            
            rr = 0.0
            rank = None
            for i, pid in enumerate(retrieved):
                if pid in gold_set:
                    rank = i + 1
                    rr = 1.0 / rank
                    break
            
            reciprocal_ranks.append(rr)
            details["per_question"][qid] = {
                "gold": list(gold_set),
                "retrieved": retrieved,
                "first_gold_rank": rank,
                "reciprocal_rank": rr
            }
        
        value = mean(reciprocal_ranks)
        
        return MetricResult(
            name=self.name,
            value=value,
            direction=self.direction,
            details=details
        )


class RecallAt1(RecallAtK):
    def __init__(self):
        super().__init__(1)


class RecallAt3(RecallAtK):
    def __init__(self):
        super().__init__(3)


class RecallAt5(RecallAtK):
    def __init__(self):
        super().__init__(5)