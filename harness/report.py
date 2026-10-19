"""
Reporting: terminal table, JSON output, leaderboard markdown.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

from harness.client import AskResult
from harness.scoring import ScoredMetric, ProblemConfig
from harness.metrics.base import MetricResult


@dataclass
class RunResult:
    """Complete results for a single team run."""
    team_name: str
    team_url: str
    timestamp: str
    config: Dict[str, Any]
    question_records: List[Dict[str, Any]]  # Per-question, per-run raw data
    metric_values: Dict[str, Optional[float]]  # Aggregated metric values
    metric_scores: Dict[str, float]  # Normalized scores per metric
    final_score: float
    success_rate: float
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def format_metric_value(value: Optional[float], name: str) -> str:
    """Format a metric value for display."""
    if value is None:
        return "N/A"
    if "ms" in name or "stall" in name:
        return f"{value:.1f}"
    if "ratio" in name or "recall" in name or name == "mrr":
        return f"{value:.3f}"
    return f"{value:.3f}"


def print_terminal_table(
    team_name: str,
    metric_results: List[MetricResult],
    scored_metrics: List[ScoredMetric],
    final_score: float,
    success_rate: float
) -> None:
    """Print a readable table to terminal."""
    print(f"\n{'='*70}")
    print(f"  RESULTS: {team_name}")
    print(f"{'='*70}")
    print(f"  Final Score: {final_score:.2f} / 100")
    print(f"  Success Rate: {success_rate:.1%}")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'Value':>12} {'Score':>8} {'Weight':>8} {'Dir':>6}")
    print(f"  {'-'*60}")
    
    for sm in scored_metrics:
        value_str = format_metric_value(sm.value, sm.name)
        print(f"  {sm.name:<25} {value_str:>12} {sm.score:>7.3f} {sm.weight:>7.2f} {sm.direction:>6}")
    
    # Also show unweighted metrics
    weighted_names = {sm.name for sm in scored_metrics}
    for mr in metric_results:
        if mr.name not in weighted_names:
            value_str = format_metric_value(mr.value, mr.name)
            print(f"  {mr.name:<25} {value_str:>12} {'N/A':>7} {'N/A':>8} {mr.direction:>6}")
    
    print(f"{'='*70}\n")


def write_results_json(result: RunResult, output_dir: str) -> str:
    """Write results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{result.team_name}.json")
    with open(filepath, 'w') as f:
        f.write(result.to_json())
    return filepath


def load_results_from_dir(results_dir: str) -> List[RunResult]:
    """Load all result JSON files from a directory."""
    results = []
    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            results.append(RunResult(**data))
    return results


def generate_leaderboard_markdown(results: List[RunResult]) -> str:
    """Generate leaderboard markdown table."""
    if not results:
        return "# Leaderboard\n\nNo results found.\n"
    
    # Sort by final score descending
    results = sorted(results, key=lambda r: r.final_score, reverse=True)
    
    # Get all metric names from the first result's metric_scores
    metric_names = list(results[0].metric_scores.keys())
    
    lines = [
        "# Leaderboard",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "| Rank | Team | Score | Success Rate | " + " | ".join(metric_names) + " |",
        "|------|------|-------|--------------|" + "|".join(["---------"] * len(metric_names)) + "|"
    ]
    
    for i, r in enumerate(results, 1):
        metric_values = " | ".join(
            format_metric_value(r.metric_values.get(name), name) 
            for name in metric_names
        )
        lines.append(
            f"| {i} | {r.team_name} | {r.final_score:.2f} | {r.success_rate:.1%} | {metric_values} |"
        )
    
    return "\n".join(lines)


def write_leaderboard(results: List[RunResult], output_dir: str) -> str:
    """Write leaderboard markdown to file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "leaderboard.md")
    markdown = generate_leaderboard_markdown(results)
    with open(filepath, 'w') as f:
        f.write(markdown)
    return filepath