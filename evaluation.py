"""
Evaluation metrics and testing procedures.

Provides utilities for comparing the Q-learning agent against baseline
policies and computing statistical summaries.
"""

from typing import Dict, List, Any
import numpy as np

from config import TEST_EPISODES, logger
from baselines import (
    GreedyCashPolicy, GreedyPointsPolicy, TimeAwareHeuristic,
    run_baseline_episode
)

__all__ = ['EvaluationMetrics']


class EvaluationMetrics:
    """
    Compute evaluation metrics comparing learned and baseline policies.

    Provides methods for running comparative experiments and
    summarizing results with statistical measures.
    """

    @staticmethod
    def compare_policies(
        agent: Any,
        env: Any,
        num_episodes: int = TEST_EPISODES
    ) -> Dict[str, List[float]]:
        """
        Compare Q-learning agent against baselines.

        Parameters
        ----------
        agent : QLearningAgent
            Trained Q-learning agent.
        env : TravelBookingEnv
            Environment for evaluation.
        num_episodes : int, default=TEST_EPISODES
            Number of test episodes to run.

        Returns
        -------
        Dict[str, List[float]]
            Dictionary mapping policy names to lists of episode rewards.
        """
        results: Dict[str, List[float]] = {
            "qlearning": [],
            "greedy_cash": [],
            "greedy_points": [],
            "time_aware": [],
        }

        for episode in range(num_episodes):
            # Q-Learning
            q_reward = agent.test_episode(env)
            results["qlearning"].append(q_reward)

            # Greedy Cash
            gc_reward = run_baseline_episode(GreedyCashPolicy, env)
            results["greedy_cash"].append(gc_reward)

            # Greedy Points
            gp_reward = run_baseline_episode(GreedyPointsPolicy, env)
            results["greedy_points"].append(gp_reward)

            # Time-Aware
            ta_reward = run_baseline_episode(TimeAwareHeuristic, env)
            results["time_aware"].append(ta_reward)

        return results

    @staticmethod
    def summarize_results(results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Print and return summary statistics.

        Parameters
        ----------
        results : Dict[str, List[float]]
            Dictionary mapping policy names to reward lists.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Summary statistics for each policy.
        """
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)

        summaries: Dict[str, Dict[str, float]] = {}

        for policy_name, rewards in results.items():
            mean = float(np.mean(rewards))
            std = float(np.std(rewards))
            min_r = float(np.min(rewards))
            max_r = float(np.max(rewards))
            median = float(np.median(rewards))

            summaries[policy_name] = {
                "mean": mean,
                "std": std,
                "min": min_r,
                "max": max_r,
                "median": median,
            }

            print(f"\n{policy_name.upper()}")
            print(f"  Mean Reward: {mean:7.4f}")
            print(f"  Std Dev:     {std:7.4f}")
            print(f"  Median:      {median:7.4f}")
            print(f"  Min:         {min_r:7.4f}")
            print(f"  Max:         {max_r:7.4f}")

        # Compute improvements
        q_mean = np.mean(results["qlearning"])
        gc_mean = np.mean(results["greedy_cash"])
        gp_mean = np.mean(results["greedy_points"])
        ta_mean = np.mean(results["time_aware"])

        print("\n" + "-" * 70)
        print("IMPROVEMENTS OVER BASELINES (Q-Learning)")
        print("-" * 70)

        improvements: Dict[str, float] = {}

        if gc_mean != 0:
            improvement_gc = ((q_mean - gc_mean) / abs(gc_mean)) * 100
            improvements["vs_greedy_cash"] = improvement_gc
            print(f"  vs Greedy-Cash:    {improvement_gc:+6.2f}%")

        if gp_mean != 0:
            improvement_gp = ((q_mean - gp_mean) / abs(gp_mean)) * 100
            improvements["vs_greedy_points"] = improvement_gp
            print(f"  vs Greedy-Points:  {improvement_gp:+6.2f}%")

        if ta_mean != 0:
            improvement_ta = ((q_mean - ta_mean) / abs(ta_mean)) * 100
            improvements["vs_time_aware"] = improvement_ta
            print(f"  vs Time-Aware:     {improvement_ta:+6.2f}%")

        summaries["improvements"] = improvements

        return summaries

    @staticmethod
    def compute_confidence_interval(
        data: List[float],
        confidence: float = 0.95
    ) -> tuple:
        """
        Compute confidence interval for mean.

        Parameters
        ----------
        data : List[float]
            Data points.
        confidence : float, default=0.95
            Confidence level (e.g., 0.95 for 95% CI).

        Returns
        -------
        tuple
            (mean, lower_bound, upper_bound)
        """
        n = len(data)
        mean = np.mean(data)
        std_err = np.std(data, ddof=1) / np.sqrt(n)

        # Use t-distribution for small samples
        from scipy import stats
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)

        margin = t_val * std_err
        return (float(mean), float(mean - margin), float(mean + margin))

    @staticmethod
    def statistical_significance_test(
        rewards_a: List[float],
        rewards_b: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical significance test between two policies.

        Uses Welch's t-test for unequal variances.

        Parameters
        ----------
        rewards_a : List[float]
            Rewards from policy A.
        rewards_b : List[float]
            Rewards from policy B.
        alpha : float, default=0.05
            Significance level.

        Returns
        -------
        Dict[str, Any]
            Test results including t-statistic, p-value, and significance.
        """
        from scipy import stats

        t_stat, p_value = stats.ttest_ind(rewards_a, rewards_b, equal_var=False)

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "alpha": alpha,
            "mean_a": float(np.mean(rewards_a)),
            "mean_b": float(np.mean(rewards_b)),
            "difference": float(np.mean(rewards_a) - np.mean(rewards_b)),
        }
