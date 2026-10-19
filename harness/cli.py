"""
CLI argument parsing and command dispatch.
"""

import argparse
import sys
from typing import List, Dict, Any

from harness.client import TeamClient, create_client
from harness.runner import Runner
from harness.scoring import load_config, compute_scores, compute_final_score
from harness.report import (
    print_terminal_table, write_results_json, load_results_from_dir, write_leaderboard
)
from harness.metrics.retrieval import RecallAt1, RecallAt3, RecallAt5, MRR
from harness.metrics.latency import TTFA_P50, TTFA_P95, Retrieval_P50, Total_P50, OverlapRatio
from harness.metrics.streaming import StallMetrics, StallCountMean
from harness.report import RunResult
from datetime import datetime
import json


def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """Load passages and questions from dataset directory."""
    passages = []
    questions = []
    
    passages_path = f"{dataset_path}/passages.jsonl"
    questions_path = f"{dataset_path}/questions.jsonl"
    
    with open(passages_path, 'r') as f:
        for line in f:
            if line.strip():
                passages.append(json.loads(line))
    
    with open(questions_path, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    return {"passages": passages, "questions": questions}


def create_all_metrics(config: Dict[str, Any]) -> List:
    """Create all metric instances."""
    return [
        RecallAt1(),
        RecallAt3(),
        RecallAt5(),
        MRR(),
        TTFA_P50(),
        TTFA_P95(),
        Retrieval_P50(),
        Total_P50(),
        OverlapRatio(),
        StallMetrics(),
        StallCountMean(),
    ]


def run_evaluation(args: argparse.Namespace) -> int:
    """Run the evaluation harness."""
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset['passages'])} passages, {len(dataset['questions'])} questions")
    
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    print(f"Connecting to team server at {args.team_url}...")
    client = create_client(args.team_url, {
        "request_timeout_s": config.request_timeout_s
    })
    
    if not client.health_check():
        print(f"ERROR: Health check failed for {args.team_url}/health")
        return 1
    
    print("Health check passed.")
    
    # Create runner
    runner = Runner(
        client=client,
        questions=dataset["questions"],
        runs=args.runs,
        warmup=2,
        stall_min_ms=config.stall_min_ms
    )
    
    # Run evaluation
    print(f"Starting evaluation: {args.runs} runs, {len(dataset['questions'])} questions each")
    question_records = runner.run_all()
    median_results = runner.get_median_results()
    
    # Compute metrics
    print("Computing metrics...")
    metrics = create_all_metrics(config.metrics)
    metric_results = []
    for metric in metrics:
        result = metric.compute(median_results, dataset, config.metrics)
        metric_results.append(result)
    
    # Compute scores
    scored_metrics = compute_scores(metric_results, config)
    
    # Compute success rate
    total_questions = len(dataset["questions"])
    successful = sum(1 for r in median_results if r.success)
    success_rate = successful / total_questions if total_questions > 0 else 0.0
    
    # Final score
    final_score = compute_final_score(scored_metrics, success_rate)
    
    # Prepare metric values dict
    metric_values = {mr.name: mr.value for mr in metric_results}
    metric_scores = {sm.name: sm.score for sm in scored_metrics}
    
    # Prepare question records for JSON output
    question_records_json = []
    for qr in question_records:
        runs_data = []
        for run in qr.runs:
            runs_data.append({
                "success": run.success,
                "error": run.error,
                "t_send": run.t_send,
                "t_retrieved": run.t_retrieved,
                "t_first_audio": run.t_first_audio,
                "t_done": run.t_done,
                "ttfa_ms": (run.t_first_audio - run.t_send) * 1000 if run.t_first_audio > 0 else None,
                "retrieval_ms": (run.t_retrieved - run.t_send) * 1000 if run.t_retrieved > 0 else None,
                "total_ms": (run.t_done - run.t_send) * 1000 if run.t_done > 0 else None,
                "stall_count": run.stall_count,
                "total_stall_ms": run.total_stall_ms,
                "events": [
                    {"type": te.event.type, "timestamp": te.timestamp}
                    for te in run.events
                ]
            })
        
        median = qr.median
        question_records_json.append({
            "question_id": qr.question_id,
            "question_text": qr.question_text,
            "gold_passage_ids": qr.gold_passage_ids,
            "category": qr.category,
            "runs": runs_data,
            "median": {
                "success": median.success if median else False,
                "error": median.error if median else "No median",
                "ttfa_ms": (median.t_first_audio - median.t_send) * 1000 if median and median.t_first_audio > 0 else None,
                "retrieval_ms": (median.t_retrieved - median.t_send) * 1000 if median and median.t_retrieved > 0 else None,
                "total_ms": (median.t_done - median.t_send) * 1000 if median and median.t_done > 0 else None,
                "stall_count": median.stall_count if median else None,
                "total_stall_ms": median.total_stall_ms if median else None,
                "retrieved_passage_ids": [
                    te.event.passage_ids for te in median.events 
                    if hasattr(te.event, 'passage_ids')
                ] if median else []
            }
        })
    
    # Build config dict for output
    config_dict = {
        "stall_min_ms": config.stall_min_ms,
        "request_timeout_s": config.request_timeout_s,
        "metrics": {
            name: {
                "weight": m.weight,
                "threshold": m.threshold,
                "target": m.target,
                "direction": m.direction
            }
            for name, m in config.metrics.items()
        }
    }
    
    # Create result object
    result = RunResult(
        team_name=args.team_name,
        team_url=args.team_url,
        timestamp=datetime.now().isoformat(),
        config=config_dict,
        question_records=question_records_json,
        metric_values=metric_values,
        metric_scores=metric_scores,
        final_score=final_score,
        success_rate=success_rate
    )
    
    # Print terminal table
    print_terminal_table(args.team_name, metric_results, scored_metrics, final_score, success_rate)
    
    # Write JSON output
    output_path = write_results_json(result, "results")
    print(f"Results written to {output_path}")
    
    return 0


def run_leaderboard(args: argparse.Namespace) -> int:
    """Generate leaderboard from results directory."""
    results = load_results_from_dir(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return 1
    
    output_path = write_leaderboard(results, args.results_dir)
    print(f"Leaderboard written to {output_path}")
    
    # Also print to terminal
    with open(output_path, 'r') as f:
        print(f.read())
    
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harness",
        description="Voice-driven RAG Assistant Evaluation Harness"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # run command
    run_parser = subparsers.add_parser("run", help="Run evaluation against a team server")
    run_parser.add_argument("--team-url", required=True, help="Team server base URL (e.g., http://localhost:8000)")
    run_parser.add_argument("--team-name", required=True, help="Team name for results")
    run_parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    run_parser.add_argument("--config", required=True, help="Path to config YAML")
    run_parser.add_argument("--runs", type=int, default=3, help="Number of runs (default: 3)")
    
    # leaderboard command
    lb_parser = subparsers.add_parser("leaderboard", help="Generate leaderboard from results")
    lb_parser.add_argument("--results-dir", default="results", help="Results directory (default: results)")
    
    args = parser.parse_args()
    
    if args.command == "run":
        return run_evaluation(args)
    elif args.command == "leaderboard":
        return run_leaderboard(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
