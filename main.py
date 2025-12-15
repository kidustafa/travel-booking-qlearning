"""
Main training script for Q-Learning Travel Booking Optimizer.

Usage:
    python main.py

This script trains a Q-learning agent across multiple random seeds,
compares against baseline policies, and generates visualization plots.
"""

import os
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt

from config import (
    EPISODES, LOG_INTERVAL, RANDOM_SEEDS, PLOT_RESULTS,
    RESULTS_DIR, NUM_STATES, NUM_ACTIONS, logger
)
from environment import TravelBookingEnv
from agent import QLearningAgent
from evaluation import EvaluationMetrics


def train_agent(
    agent: QLearningAgent,
    env: TravelBookingEnv,
    num_episodes: int = EPISODES,
    log_interval: int = LOG_INTERVAL
) -> QLearningAgent:
    """
    Train Q-learning agent.

    Parameters
    ----------
    agent : QLearningAgent
        Agent to train.
    env : TravelBookingEnv
        Training environment.
    num_episodes : int, default=EPISODES
        Number of training episodes.
    log_interval : int, default=LOG_INTERVAL
        Logging frequency.

    Returns
    -------
    QLearningAgent
        Trained agent.
    """
    logger.info("Starting Q-learning training...")
    logger.info(
        f"Episodes: {num_episodes}, Learning Rate: {agent.alpha}, "
        f"Discount: {agent.gamma}"
    )
    logger.info(f"Epsilon Decay: {agent.epsilon:.4f} -> {agent.min_epsilon}")

    for episode in range(1, num_episodes + 1):
        reward = agent.train_episode(env)

        if episode % log_interval == 0:
            stats = agent.get_training_stats()
            logger.info(
                f"[EPISODE {episode:5d}] "
                f"Avg Reward: {stats['mean_episode_reward']:7.4f} | "
                f"Avg Length: {stats['mean_episode_length']:.1f} | "
                f"Epsilon: {stats['current_epsilon']:.4f}"
            )

    logger.info("Training complete!")
    return agent


def evaluate_agent(
    agent: QLearningAgent,
    env: TravelBookingEnv,
    seed: int = None
) -> Dict[str, List[float]]:
    """
    Evaluate trained agent against baselines.

    Parameters
    ----------
    agent : QLearningAgent
        Trained agent to evaluate.
    env : TravelBookingEnv
        Evaluation environment.
    seed : int, optional
        Random seed used (for logging).

    Returns
    -------
    Dict[str, List[float]]
        Evaluation results for each policy.
    """
    logger.info(f"Testing agent (seed={seed})...")
    results = EvaluationMetrics.compare_policies(agent, env)
    return results


def plot_training_curves(
    agent: QLearningAgent,
    save_path: str = None
) -> None:
    """
    Plot training curves.

    Parameters
    ----------
    agent : QLearningAgent
        Trained agent with episode history.
    save_path : str, optional
        Path to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Rewards
    rewards = agent.episode_rewards
    window = 100

    if len(rewards) >= window:
        smoothed = np.convolve(
            rewards,
            np.ones(window) / window,
            mode='valid'
        )

        ax1.plot(rewards, alpha=0.3, label='Raw')
        ax1.plot(
            np.arange(window - 1, len(rewards)),
            smoothed,
            label=f'{window}-episode MA'
        )
    else:
        ax1.plot(rewards, label='Raw')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Episode lengths
    lengths = agent.episode_lengths

    if len(lengths) >= window:
        smoothed_lengths = np.convolve(
            lengths,
            np.ones(window) / window,
            mode='valid'
        )

        ax2.plot(lengths, alpha=0.3, label='Raw')
        ax2.plot(
            np.arange(window - 1, len(lengths)),
            smoothed_lengths,
            label=f'{window}-episode MA'
        )
    else:
        ax2.plot(lengths, label='Raw')

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps per Episode')
    ax2.set_title('Episode Lengths Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Training curves saved to {save_path}")

    plt.close()


def plot_evaluation_results(
    all_results: Dict[str, List[float]],
    save_path: str = None
) -> None:
    """
    Box plot comparing policies.

    Parameters
    ----------
    all_results : Dict[str, List[float]]
        Results for each policy.
    save_path : str, optional
        Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    policy_names = list(all_results.keys())
    rewards_by_policy = [all_results[policy] for policy in policy_names]

    # Create box plot
    bp = ax.boxplot(rewards_by_policy, labels=policy_names, patch_artist=True)

    # Color the boxes
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Cumulative Normalized Utility')
    ax.set_title('Policy Comparison: Q-Learning vs Baselines')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Comparison plot saved to {save_path}")

    plt.close()


def main() -> None:
    """Main execution entry point."""

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print(" Q-LEARNING TRAVEL BOOKING OPTIMIZER")
    print(" WITH ALL 5 CRITICAL FIXES APPLIED")
    print("=" * 70)

    all_evaluation_results: Dict[str, List[float]] = {
        "qlearning": [],
        "greedy_cash": [],
        "greedy_points": [],
        "time_aware": [],
    }

    for seed_idx, seed in enumerate(RANDOM_SEEDS):
        print(f"\n{'=' * 70}")
        print(f"RUN {seed_idx + 1}/{len(RANDOM_SEEDS)} (seed={seed})")
        print(f"{'=' * 70}")

        # Create fresh environment and agent
        env = TravelBookingEnv(seed=seed, weight_scheme="balanced")
        agent = QLearningAgent(NUM_STATES, NUM_ACTIONS)

        # Train
        agent = train_agent(agent, env, num_episodes=EPISODES)

        # Evaluate
        results = evaluate_agent(agent, env, seed=seed)

        # Aggregate results
        for policy, rewards in results.items():
            all_evaluation_results[policy].extend(rewards)

        # Print summary for this run
        EvaluationMetrics.summarize_results(results)

        # Save Q-table
        qtable_path = os.path.join(RESULTS_DIR, f"qtable_seed{seed}.npy")
        agent.save_qtable(qtable_path)

        # Plot training curves
        if PLOT_RESULTS:
            plot_path = os.path.join(RESULTS_DIR, f"training_curves_seed{seed}.png")
            plot_training_curves(agent, save_path=plot_path)

    # Final comparison across all seeds
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS (Aggregated Across All Seeds)")
    print(f"{'=' * 70}")

    EvaluationMetrics.summarize_results(all_evaluation_results)

    # Plot final comparison
    if PLOT_RESULTS:
        plot_path = os.path.join(RESULTS_DIR, "final_comparison.png")
        plot_evaluation_results(all_evaluation_results, save_path=plot_path)

    print(f"\n[COMPLETE] Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
