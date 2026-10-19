"""
Runner: executes warmup, multiple runs, computes medians per question.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from harness.client import TeamClient, AskRequest, AskResult
from harness.contract import parse_event


@dataclass
class QuestionRecord:
    """Aggregated record for a single question across runs."""
    question_id: str
    question_text: str
    gold_passage_ids: List[str]
    category: str
    runs: List[AskResult] = field(default_factory=list)
    
    # Median values across runs (computed after all runs)
    median: Optional[AskResult] = None
    
    def compute_median(self) -> None:
        """Compute median metrics across successful runs."""
        successful_runs = [r for r in self.runs if r.success]
        
        if not successful_runs:
            # All runs failed - create a failed median result
            self.median = AskResult(
                question_id=self.question_id,
                success=False,
                error="All runs failed"
            )
            return
        
        # For median, we need to pick the median run for each numeric field
        # Sort by each metric and pick middle
        def median_of(values: List[float]) -> float:
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            if n % 2 == 1:
                return sorted_vals[n // 2]
            return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        
        # Use the first successful run as base, then overwrite with medians
        base = successful_runs[0]
        self.median = AskResult(
            question_id=self.question_id,
            events=base.events,  # Use events from first successful run
            success=True
        )
        
        # Compute medians for timing fields
        self.median.t_send = median_of([r.t_send for r in successful_runs])
        self.median.t_retrieved = median_of([r.t_retrieved for r in successful_runs])
        self.median.t_first_audio = median_of([r.t_first_audio for r in successful_runs])
        self.median.t_done = median_of([r.t_done for r in successful_runs])
        self.median.stall_count = int(median_of([float(r.stall_count) for r in successful_runs]))
        self.median.total_stall_ms = median_of([r.total_stall_ms for r in successful_runs])
        
        # For retrieval, use the first successful run's retrieved list
        for te in base.events:
            if hasattr(te.event, 'passage_ids'):
                self.median.events = [te]  # Just keep the retrieved event
                break
        
        # Mark as successful
        self.median.success = True


class Runner:
    """
    Orchestrates evaluation runs against a team server.
    
    Flow:
    1. Warmup (2 requests, not scored)
    2. For each run (default 3):
       - Run all questions sequentially
       - Collect AskResult per question
    3. Compute median per question across runs
    4. Return aggregated records for metrics computation
    """
    
    def __init__(
        self,
        client: TeamClient,
        questions: List[Dict[str, Any]],
        runs: int = 3,
        warmup: int = 2,
        stall_min_ms: float = 200.0
    ):
        self.client = client
        self.questions = questions
        self.runs = runs
        self.warmup = warmup
        self.stall_min_ms = stall_min_ms
        
        # Records: question_id -> QuestionRecord
        self.records: Dict[str, QuestionRecord] = {}
        for q in questions:
            self.records[q["id"]] = QuestionRecord(
                question_id=q["id"],
                question_text=q["question"],
                gold_passage_ids=q.get("gold_passage_ids", []),
                category=q.get("category", "unknown")
            )
    
    def run_warmup(self) -> None:
        """Run warmup requests (not scored)."""
        print(f"Running {self.warmup} warmup requests...")
        for i in range(self.warmup):
            q = self.questions[i % len(self.questions)]
            request = AskRequest(question_id=q["id"], question=q["question"])
            self.client.ask(request)
        print("Warmup complete.")
    
    def run_all(self) -> List[QuestionRecord]:
        """
        Execute all runs and return list of QuestionRecords with medians computed.
        """
        # Warmup
        self.run_warmup()
        
        # Main runs
        for run_idx in range(self.runs):
            print(f"Run {run_idx + 1}/{self.runs}...")
            for q in self.questions:
                request = AskRequest(question_id=q["id"], question=q["question"])
                result = self.client.ask(request)
                
                # Compute derived metrics with stall_min_ms from config
                if result.success:
                    result.compute_derived_metrics(self.stall_min_ms)
                
                self.records[q["id"]].runs.append(result)
        
        # Compute medians
        print("Computing medians...")
        for record in self.records.values():
            record.compute_median()
        
        return list(self.records.values())
    
    def get_median_results(self) -> List[AskResult]:
        """Get list of median AskResults for metrics computation."""
        return [r.median for r in self.records.values() if r.median is not None]