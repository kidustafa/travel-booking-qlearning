"""
Reward Normalization Module (RNM).

Converts heterogeneous rewards (USD, points, time) into a unified utility metric
using min-max normalization followed by weighted aggregation.

This is the key contribution enabling multi-objective optimization in the
travel booking domain.
"""

from typing import Dict, Optional
import numpy as np

from config import REWARD_RANGES, DEFAULT_WEIGHTS, logger

__all__ = ['RewardNormalizer']


class RewardNormalizer:
    """
    Aggregates and normalizes multi-objective rewards.

    Uses min-max normalization to scale each reward component to [-1, 1],
    then applies weighted aggregation to produce a single scalar reward.

    Attributes
    ----------
    weights : Dict[str, float]
        Weights for each reward component (cash, points, time).
    ranges : Dict[str, Dict[str, float]]
        Min/max ranges for normalization of each component.

    Parameters
    ----------
    weights : Dict[str, float], optional
        Custom weights for reward aggregation. Must sum to ~1.0.
        Default uses balanced weights (0.4, 0.4, 0.2).
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        """Initialize normalizer with weighting scheme."""
        if weights is None:
            weights = DEFAULT_WEIGHTS.copy()

        self.weights = weights
        self.ranges = REWARD_RANGES

        # Validate weights sum to approximately 1.0
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to ~1.0, got {total_weight:.4f}. "
                f"Weights: {weights}"
            )

    def normalize_component(self, value: float, component_type: str) -> float:
        """
        Apply min-max scaling to a single reward component.

        Parameters
        ----------
        value : float
            Raw reward value to normalize.
        component_type : str
            Type of reward: 'cash', 'points', or 'time'.

        Returns
        -------
        float
            Normalized value in [-1, 1].

        Raises
        ------
        KeyError
            If component_type is not recognized.
        """
        if component_type not in self.ranges:
            raise KeyError(
                f"Unknown component type: {component_type}. "
                f"Valid types: {list(self.ranges.keys())}"
            )

        r_min = self.ranges[component_type]["min"]
        r_max = self.ranges[component_type]["max"]

        if r_max == r_min:
            return 0.0

        # Min-max scaling to [-1, 1]
        normalized = 2.0 * (value - r_min) / (r_max - r_min) - 1.0

        # Clip to [-1, 1] to handle out-of-range values
        normalized = np.clip(normalized, -1.0, 1.0)

        return float(normalized)

    def aggregate_rewards(
        self,
        r_cash: float,
        r_points: float,
        r_time: float
    ) -> float:
        """
        Aggregate multi-objective rewards into single scalar.

        Parameters
        ----------
        r_cash : float
            Cash reward (negative = cost incurred).
        r_points : float
            Points reward (positive = value from points redemption).
        r_time : float
            Time reward (negative = days until departure penalty).

        Returns
        -------
        float
            Weighted aggregate reward in approximately [-1, 1].
        """
        # Normalize each component
        r_cash_norm = self.normalize_component(r_cash, "cash")
        r_points_norm = self.normalize_component(r_points, "points")
        r_time_norm = self.normalize_component(r_time, "time")

        # Weighted sum
        total_reward = (
            self.weights["cash"] * r_cash_norm +
            self.weights["points"] * r_points_norm +
            self.weights["time"] * r_time_norm
        )

        return float(total_reward)

    def get_component_contributions(
        self,
        r_cash: float,
        r_points: float,
        r_time: float
    ) -> Dict[str, float]:
        """
        Get individual weighted contributions of each component.

        Useful for debugging and understanding reward composition.

        Parameters
        ----------
        r_cash : float
            Cash reward.
        r_points : float
            Points reward.
        r_time : float
            Time reward.

        Returns
        -------
        Dict[str, float]
            Dictionary with normalized and weighted contribution of each component.
        """
        r_cash_norm = self.normalize_component(r_cash, "cash")
        r_points_norm = self.normalize_component(r_points, "points")
        r_time_norm = self.normalize_component(r_time, "time")

        return {
            "cash_normalized": r_cash_norm,
            "cash_weighted": self.weights["cash"] * r_cash_norm,
            "points_normalized": r_points_norm,
            "points_weighted": self.weights["points"] * r_points_norm,
            "time_normalized": r_time_norm,
            "time_weighted": self.weights["time"] * r_time_norm,
            "total": self.aggregate_rewards(r_cash, r_points, r_time),
        }

    def __repr__(self) -> str:
        return f"RewardNormalizer(weights={self.weights})"
