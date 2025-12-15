"""
Tabular Q-Learning Agent.

Implements the classic Q-learning algorithm with epsilon-greedy exploration
and epsilon decay for smooth convergence.

Critical fix applied:
- FIX 4: Epsilon decay (0.3 -> 0.01) for smooth convergence
"""

from typing import Dict, Optional, Any
import numpy as np

from config import (
    LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_RATE_START,
    EXPLORATION_RATE_END, EXPLORATION_RATE_DECAY, logger
)

__all__ = ['QLearningAgent']


class QLearningAgent:
    """
    Tabular Q-learning agent for discrete MDP.

    Uses epsilon-greedy exploration during training with decaying epsilon
    to balance exploration and exploitation.

    The Q-learning update rule:
        Q(s,a) <- Q(s,a) + α[r + γ * max_a' Q(s', a') - Q(s,a)]

    Attributes
    ----------
    num_states : int
        Size of the state space.
    num_actions : int
        Number of available actions.
    alpha : float
        Learning rate for Q-value updates.
    gamma : float
        Discount factor for future rewards.
    epsilon : float
        Current exploration rate.
    epsilon_decay : float
        Decay multiplier applied after each episode.
    min_epsilon : float
        Minimum exploration rate floor.
    Q_table : np.ndarray
        Q-value table of shape (num_states, num_actions).

    Parameters
    ----------
    num_states : int
        Size of the state space.
    num_actions : int
        Number of available actions.
    learning_rate : float, default=0.1
        Step size for Q-value updates (α).
    discount_factor : float, default=0.99
        Importance of future rewards (γ).
    exploration_rate : float, default=0.3
        Initial exploration probability (ε).
    exploration_decay : float, default=0.995
        Decay multiplier per episode.
    min_exploration : float, default=0.01
        Minimum exploration floor.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        learning_rate: float = LEARNING_RATE,
        discount_factor: float = DISCOUNT_FACTOR,
        exploration_rate: float = EXPLORATION_RATE_START,
        exploration_decay: float = EXPLORATION_RATE_DECAY,
        min_exploration: float = EXPLORATION_RATE_END
    ) -> None:
        """Initialize Q-learning agent with FIX 4 epsilon decay."""
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate          # Decays over time (FIX 4)
        self.epsilon_decay = exploration_decay   # Decay rate (FIX 4)
        self.min_epsilon = min_exploration       # Minimum epsilon (FIX 4)

        # Initialize Q-table to zeros
        self.Q_table = np.zeros((num_states, num_actions))

        # Tracking
        self.episode_rewards: list = []
        self.episode_lengths: list = []

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action via epsilon-greedy policy.

        Parameters
        ----------
        state : int
            Current state index.
        training : bool, default=True
            If True, use epsilon-greedy. If False, always greedy.

        Returns
        -------
        int
            Selected action index.
        """
        if training and np.random.rand() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.num_actions)
        else:
            # Exploitation: greedy action (max Q-value)
            return int(np.argmax(self.Q_table[state, :]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        """
        Update Q-table using Q-learning update rule.

        Q(s,a) <- Q(s,a) + α[r + γ * max_a' Q(s', a') - Q(s,a)]

        Parameters
        ----------
        state : int
            Current state index.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : int
            Next state index.
        done : bool
            Whether the episode terminated.

        Returns
        -------
        float
            The TD error magnitude.
        """
        if done:
            # Terminal state: no future value
            max_next_q = 0.0
        else:
            # Non-terminal: use max Q-value of next state
            max_next_q = float(np.max(self.Q_table[next_state, :]))

        # Q-learning update
        current_q = self.Q_table[state, action]
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q

        self.Q_table[state, action] += self.alpha * td_error

        return abs(td_error)

    def train_episode(self, env: Any) -> float:
        """
        Run one training episode.

        FIX 4: Epsilon DECAYS after each episode.

        Parameters
        ----------
        env : TravelBookingEnv
            The environment to train in.

        Returns
        -------
        float
            Total episode reward.
        """
        state = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            # Select and execute action
            action = self.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            # Update Q-table
            self.update(state, action, reward, next_state, done)

            episode_reward += reward
            steps += 1
            state = next_state

        # FIX 4: DECAY EPSILON AFTER EACH EPISODE
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.min_epsilon)

        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(steps)

        return episode_reward

    def test_episode(self, env: Any) -> float:
        """
        Run one test episode (greedy policy, no learning).

        Parameters
        ----------
        env : TravelBookingEnv
            The environment to test in.

        Returns
        -------
        float
            Total episode reward.
        """
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # Select action greedily (no exploration)
            action = self.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            state = next_state

        return episode_reward

    def get_policy(self) -> np.ndarray:
        """
        Extract greedy policy from Q-table.

        Returns
        -------
        np.ndarray
            Array of best actions for each state.
        """
        return np.argmax(self.Q_table, axis=1)

    def save_qtable(self, filepath: str) -> None:
        """
        Save Q-table to file.

        Parameters
        ----------
        filepath : str
            Path to save the Q-table.
        """
        np.save(filepath, self.Q_table)
        logger.info(f"Q-table saved to {filepath}")

    def load_qtable(self, filepath: str) -> None:
        """
        Load Q-table from file.

        Parameters
        ----------
        filepath : str
            Path to load the Q-table from.
        """
        self.Q_table = np.load(filepath)

    def get_training_stats(self) -> Dict[str, float]:
        """
        Return training statistics.

        Returns
        -------
        Dict[str, float]
            Dictionary containing training statistics.
        """
        if not self.episode_rewards:
            return {
                "mean_episode_reward": 0.0,
                "mean_episode_length": 0.0,
                "max_episode_reward": 0.0,
                "num_episodes_trained": 0,
                "current_epsilon": self.epsilon,
            }

        # Use last 100 episodes for running average
        window = min(100, len(self.episode_rewards))

        return {
            "mean_episode_reward": float(np.mean(self.episode_rewards[-window:])),
            "mean_episode_length": float(np.mean(self.episode_lengths[-window:])),
            "max_episode_reward": float(np.max(self.episode_rewards)),
            "num_episodes_trained": len(self.episode_rewards),
            "current_epsilon": self.epsilon,  # Track epsilon decay (FIX 4)
        }

    def get_q_table_stats(self) -> Dict[str, float]:
        """
        Return statistics about the Q-table.

        Returns
        -------
        Dict[str, float]
            Statistics about Q-values.
        """
        non_zero = np.count_nonzero(self.Q_table)
        total = self.Q_table.size

        return {
            "non_zero_entries": non_zero,
            "total_entries": total,
            "coverage": non_zero / total,
            "mean_q": float(np.mean(self.Q_table)),
            "max_q": float(np.max(self.Q_table)),
            "min_q": float(np.min(self.Q_table)),
        }

    def __repr__(self) -> str:
        return (
            f"QLearningAgent(states={self.num_states}, actions={self.num_actions}, "
            f"alpha={self.alpha}, gamma={self.gamma}, epsilon={self.epsilon:.4f})"
        )
