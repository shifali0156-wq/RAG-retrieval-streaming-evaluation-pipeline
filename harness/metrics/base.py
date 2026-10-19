"""
Base metric interface for the evaluation harness.

All metrics implement this interface for consistent computation and reporting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

from harness.client import AskResult


@dataclass
class MetricResult:
    """Result of a metric computation."""
    name: str
    value: Optional[float]  # None if not computable
    direction: str  # "higher" or "lower"
    details: dict = None  # Additional info for debugging/reporting
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class Metric(ABC):
    """Abstract base class for all metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric name (e.g., 'recall_at_5')."""
        pass
    
    @property
    @abstractmethod
    def direction(self) -> str:
        """'higher' if larger values are better, 'lower' if smaller values are better."""
        pass
    
    @abstractmethod
    def compute(
        self,
        records: List[AskResult],
        dataset: Any,
        config: dict
    ) -> MetricResult:
        """
        Compute the metric from run records.
        
        Args:
            records: List of AskResult from all runs (median already taken per question)
            dataset: Parsed dataset (passages + questions)
            config: Configuration dict with metric-specific params
            
        Returns:
            MetricResult with value and metadata
        """
        pass


def percentile(values: List[float], p: float) -> Optional[float]:
    """Compute percentile using numpy-like linear interpolation."""
    if not values:
        return None
    try:
        import numpy as np
        return float(np.percentile(values, p))
    except ImportError:
        # Fallback without numpy
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n == 1:
            return sorted_vals[0]
        idx = (p / 100) * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        weight = idx - lower
        return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def mean(values: List[float]) -> Optional[float]:
    """Compute mean, returning None for empty list."""
    if not values:
        return None
    return sum(values) / len(values)