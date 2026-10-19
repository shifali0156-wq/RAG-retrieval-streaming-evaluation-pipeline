"""
Scoring engine: config loading/validation, metric normalization, aggregation.
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from harness.metrics.base import MetricResult


@dataclass
class MetricConfig:
    """Configuration for a single metric."""
    weight: float
    threshold: float
    target: float
    direction: str  # "higher" or "lower"
    
    def __post_init__(self):
        if self.direction not in ("higher", "lower"):
            raise ValueError(f"Invalid direction: {self.direction}. Must be 'higher' or 'lower'")


@dataclass
class ProblemConfig:
    """Complete problem configuration."""
    stall_min_ms: int
    request_timeout_s: int
    metrics: Dict[str, MetricConfig]
    
    def validate_weights(self) -> None:
        """Validate that weights sum to 1.0."""
        total_weight = sum(m.weight for m in self.metrics.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Metric weights must sum to 1.0, got {total_weight}. "
                f"Weights: { {k: v.weight for k, v in self.metrics.items()} }"
            )


def load_config(path: str) -> ProblemConfig:
    """Load and validate configuration from YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    problem_data = data.get("problem_4", {})
    
    metrics = {}
    for name, m_config in problem_data.get("metrics", {}).items():
        metrics[name] = MetricConfig(
            weight=m_config["weight"],
            threshold=m_config["threshold"],
            target=m_config["target"],
            direction=m_config["direction"]
        )
    
    config = ProblemConfig(
        stall_min_ms=problem_data.get("stall_min_ms", 200),
        request_timeout_s=problem_data.get("request_timeout_s", 30),
        metrics=metrics
    )
    
    config.validate_weights()
    return config


def normalize_score(
    value: Optional[float],
    threshold: float,
    target: float,
    direction: str
) -> float:
    """
    Normalize a metric value to [0, 1] score.
    
    Higher is better: clip((x - threshold) / (target - threshold), 0, 1)
    Lower is better:  clip((threshold - x) / (threshold - target), 0, 1)
    
    If x is null, score 0.
    If threshold == target, score 1 if x meets the target else 0.
    """
    if value is None:
        return 0.0
    
    if threshold == target:
        if direction == "higher":
            return 1.0 if value >= target else 0.0
        else:
            return 1.0 if value <= target else 0.0
    
    if direction == "higher":
        score = (value - threshold) / (target - threshold)
    else:  # lower
        score = (threshold - value) / (threshold - target)
    
    return max(0.0, min(1.0, score))


@dataclass
class ScoredMetric:
    """A metric with its computed value and normalized score."""
    name: str
    value: Optional[float]
    score: float
    weight: float
    direction: str
    threshold: float
    target: float
    details: dict = field(default_factory=dict)


def compute_scores(
    metric_results: List[MetricResult],
    config: ProblemConfig
) -> List[ScoredMetric]:
    """Compute normalized scores for all metrics."""
    scored = []
    
    # Build lookup of metric results by name
    result_map = {mr.name: mr for mr in metric_results}
    
    for name, m_config in config.metrics.items():
        result = result_map.get(name)
        value = result.value if result else None
        
        score = normalize_score(value, m_config.threshold, m_config.target, m_config.direction)
        
        scored.append(ScoredMetric(
            name=name,
            value=value,
            score=score,
            weight=m_config.weight,
            direction=m_config.direction,
            threshold=m_config.threshold,
            target=m_config.target,
            details=result.details if result else {}
        ))
    
    return scored


def compute_final_score(scored_metrics: List[ScoredMetric], success_rate: float) -> float:
    """
    Compute final score: 100 * sum(weight * metric_score) * success_rate.
    """
    weighted_sum = sum(m.weight * m.score for m in scored_metrics)
    return 100.0 * weighted_sum * success_rate