"""
Tests for scoring: normalization, clipping, threshold==target, failed questions.
"""

import pytest
from harness.scoring import (
    normalize_score, compute_scores, compute_final_score,
    load_config, ProblemConfig, MetricConfig
)
from harness.metrics.base import MetricResult


class TestNormalizeScore:
    """Tests for score normalization function."""
    
    def test_higher_direction_basic(self):
        """Higher is better: linear interpolation between threshold and target."""
        # threshold=0.5, target=0.9
        # value=0.5 -> score=0
        # value=0.7 -> score=0.5
        # value=0.9 -> score=1
        # value=1.0 -> score=1 (clipped)
        assert normalize_score(0.5, 0.5, 0.9, "higher") == 0.0
        assert abs(normalize_score(0.7, 0.5, 0.9, "higher") - 0.5) < 0.001
        assert normalize_score(0.9, 0.5, 0.9, "higher") == 1.0
        assert normalize_score(1.0, 0.5, 0.9, "higher") == 1.0
    
    def test_lower_direction_basic(self):
        """Lower is better: linear interpolation."""
        # threshold=3000, target=1200
        # value=3000 -> score=0
        # value=2100 -> score=0.5
        # value=1200 -> score=1
        # value=1000 -> score=1 (clipped)
        assert normalize_score(3000, 3000, 1200, "lower") == 0.0
        assert abs(normalize_score(2100, 3000, 1200, "lower") - 0.5) < 0.001
        assert normalize_score(1200, 3000, 1200, "lower") == 1.0
        assert normalize_score(1000, 3000, 1200, "lower") == 1.0
    
    def test_null_value_scores_zero(self):
        """None value always scores 0."""
        assert normalize_score(None, 0.5, 0.9, "higher") == 0.0
        assert normalize_score(None, 3000, 1200, "lower") == 0.0
    
    def test_threshold_equals_target_higher(self):
        """threshold==target with higher direction."""
        # threshold=target=0.8
        # value >= 0.8 -> score=1
        # value < 0.8 -> score=0
        assert normalize_score(0.8, 0.8, 0.8, "higher") == 1.0
        assert normalize_score(0.9, 0.8, 0.8, "higher") == 1.0
        assert normalize_score(0.7, 0.8, 0.8, "higher") == 0.0
    
    def test_threshold_equals_target_lower(self):
        """threshold==target with lower direction."""
        # threshold=target=1000
        # value <= 1000 -> score=1
        # value > 1000 -> score=0
        assert normalize_score(1000, 1000, 1000, "lower") == 1.0
        assert normalize_score(900, 1000, 1000, "lower") == 1.0
        assert normalize_score(1100, 1000, 1000, "lower") == 0.0
    
    def test_clipping_higher(self):
        """Values beyond target are clipped to 1."""
        assert normalize_score(0.95, 0.5, 0.9, "higher") == 1.0
        assert normalize_score(1.0, 0.5, 0.9, "higher") == 1.0
    
    def test_clipping_lower(self):
        """Values below target are clipped to 1."""
        assert normalize_score(1000, 3000, 1200, "lower") == 1.0
        assert normalize_score(0, 3000, 1200, "lower") == 1.0
    
    def test_below_threshold_higher(self):
        """Values below threshold score 0."""
        assert normalize_score(0.4, 0.5, 0.9, "higher") == 0.0
        assert normalize_score(0.0, 0.5, 0.9, "higher") == 0.0
    
    def test_above_threshold_lower(self):
        """Values above threshold score 0."""
        assert normalize_score(4000, 3000, 1200, "lower") == 0.0
        assert normalize_score(10000, 3000, 1200, "lower") == 0.0


class TestComputeScores:
    """Tests for compute_scores aggregation."""
    
    def test_compute_scores_basic(self):
        """Basic score computation with multiple metrics."""
        metric_results = [
            MetricResult(name="recall_at_5", value=0.8, direction="higher", details={}),
            MetricResult(name="mrr", value=0.6, direction="higher", details={}),
            MetricResult(name="ttfa_p50_ms", value=2000, direction="lower", details={}),
        ]
        
        config = ProblemConfig(
            stall_min_ms=200,
            request_timeout_s=30,
            metrics={
                "recall_at_5": MetricConfig(0.5, 0.5, 0.9, "higher"),
                "mrr": MetricConfig(0.3, 0.3, 0.75, "higher"),
                "ttfa_p50_ms": MetricConfig(0.2, 3000, 1200, "lower"),
            }
        )
        
        scored = compute_scores(metric_results, config)
        
        assert len(scored) == 3
        # recall_at_5: (0.8-0.5)/(0.9-0.5) = 0.75
        assert abs(scored[0].score - 0.75) < 0.001
        # mrr: (0.6-0.3)/(0.75-0.3) = 0.667
        assert abs(scored[1].score - 0.667) < 0.001
        # ttfa: (3000-2000)/(3000-1200) = 0.556
        assert abs(scored[2].score - 0.556) < 0.001


class TestComputeFinalScore:
    """Tests for final score computation."""
    
    def test_final_score_basic(self):
        """Final score = 100 * sum(weight * score) * success_rate."""
        from harness.scoring import ScoredMetric
        
        scored = [
            ScoredMetric("m1", 0.8, 0.8, 0.5, "higher", 0.5, 0.9),
            ScoredMetric("m2", 0.6, 0.6, 0.5, "higher", 0.3, 0.75),
        ]
        
        # sum(weight*score) = 0.5*0.8 + 0.5*0.6 = 0.4 + 0.3 = 0.7
        # success_rate = 1.0
        # final = 100 * 0.7 * 1.0 = 70
        final = compute_final_score(scored, 1.0)
        assert final == 70.0
    
    def test_final_score_with_failures(self):
        """Success rate reduces final score."""
        from harness.scoring import ScoredMetric
        
        scored = [
            ScoredMetric("m1", 0.8, 0.8, 1.0, "higher", 0.5, 0.9),
        ]
        
        # Perfect metric score but 80% success rate
        final = compute_final_score(scored, 0.8)
        assert final == 64.0  # 100 * 0.8 * 0.8
    
    def test_final_score_zero_success(self):
        """Zero success rate gives zero final score."""
        from harness.scoring import ScoredMetric
        
        scored = [
            ScoredMetric("m1", 0.8, 0.8, 1.0, "higher", 0.5, 0.9),
        ]
        
        final = compute_final_score(scored, 0.0)
        assert final == 0.0


class TestConfigValidation:
    """Tests for config validation."""
    
    def test_weights_must_sum_to_one(self):
        """Config validation rejects weights not summing to 1."""
        config = ProblemConfig(
            stall_min_ms=200,
            request_timeout_s=30,
            metrics={
                "m1": MetricConfig(0.5, 0.5, 0.9, "higher"),
                "m2": MetricConfig(0.3, 0.3, 0.75, "higher"),  # sum = 0.8
            }
        )
        
        with pytest.raises(ValueError, match="sum to 1.0"):
            config.validate_weights()
    
    def test_valid_weights_pass(self):
        """Valid weights pass validation."""
        config = ProblemConfig(
            stall_min_ms=200,
            request_timeout_s=30,
            metrics={
                "m1": MetricConfig(0.5, 0.5, 0.9, "higher"),
                "m2": MetricConfig(0.5, 0.3, 0.75, "higher"),
            }
        )
        
        config.validate_weights()  # Should not raise
    
    def test_invalid_direction_raises(self):
        """Invalid direction raises error."""
        with pytest.raises(ValueError, match="Invalid direction"):
            MetricConfig(0.5, 0.5, 0.9, "invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
