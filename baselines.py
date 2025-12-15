"""
Baseline heuristic policies for comparison.

These policies provide benchmarks against which to evaluate the
Q-learning agent's performance.
"""

from typing import Any
import numpy as np

from config import (
    ACTION_BOOK_CASH, ACTION_BOOK_POINTS, ACTION_WAIT_3_DAYS,
    ACTION_WAIT_7_DAYS, LOYALTY_PROGRAMS
)

__all__ = [
    'GreedyCashPolicy', 'GreedyPointsPolicy', 'TimeAwareHeuristic',
    'run_baseline_episode'
]


class GreedyCashPolicy:
    """
    Always books immediately at current cash price.

    This naive baseline represents a traveler who books as soon as
    they decide to travel, without considering price trends or points.
    """

    @staticmethod
    def select_action(state: int, env: Any) -> int:
        """
        Select BookNow_Cash action.

        Parameters
        ----------
        state : int
            Current state index (unused).
        env : TravelBookingEnv
            Environment instance (unused).

        Returns
        -------
        int
            ACTION_BOOK_CASH
        """
        return ACTION_BOOK_CASH


class GreedyPointsPolicy:
    """
    Redeem points if balance sufficient; otherwise book with cash.

    This baseline represents a traveler who prioritizes using
    loyalty points when available.
    """

    @staticmethod
    def select_action(state: int, env: Any) -> int:
        """
        Select BookNow_Points if sufficient balance, else BookNow_Cash.

        Parameters
        ----------
        state : int
            Current state index (unused).
        env : TravelBookingEnv
            Environment instance with airline and points info.

        Returns
        -------
        int
            ACTION_BOOK_POINTS or ACTION_BOOK_CASH
        """
        airline = env.airline
        airline_program = f"{airline}_Miles"

        if airline_program not in LOYALTY_PROGRAMS:
            return ACTION_BOOK_CASH

        points_req = LOYALTY_PROGRAMS[airline_program]["domestic_short"]["points_required"]

        if env.points_balance >= points_req:
            return ACTION_BOOK_POINTS
        else:
            return ACTION_BOOK_CASH


class TimeAwareHeuristic:
    """
    Book if price is at good percentile; otherwise wait.

    This more sophisticated baseline considers price history
    and time remaining to make decisions.
    """

    # Price percentile threshold for booking
    GOOD_PRICE_THRESHOLD: float = 0.15

    @staticmethod
    def select_action(state: int, env: Any) -> int:
        """
        Book if price is low or deadline is near; otherwise wait.

        Parameters
        ----------
        state : int
            Current state index (unused).
        env : TravelBookingEnv
            Environment instance with price and time info.

        Returns
        -------
        int
            Selected action based on heuristic rules.
        """
        percentile = env.price_sim.get_price_percentile()

        # Book if price is in bottom 15% or deadline is tomorrow
        if percentile <= TimeAwareHeuristic.GOOD_PRICE_THRESHOLD:
            return ACTION_BOOK_CASH

        if env.days_until_departure <= 1:
            return ACTION_BOOK_CASH

        # Otherwise wait
        if env.days_until_departure > 7:
            return ACTION_WAIT_3_DAYS
        else:
            return ACTION_WAIT_3_DAYS


class RandomPolicy:
    """
    Selects actions uniformly at random.

    Useful as a lower-bound baseline to verify learning is occurring.
    """

    @staticmethod
    def select_action(state: int, env: Any) -> int:
        """
        Select a random action.

        Parameters
        ----------
        state : int
            Current state index (unused).
        env : TravelBookingEnv
            Environment instance (unused).

        Returns
        -------
        int
            Random action in [0, 3].
        """
        return np.random.randint(4)


def run_baseline_episode(policy_class: Any, env: Any) -> float:
    """
    Run one episode using a baseline policy.

    Parameters
    ----------
    policy_class : class
        Policy class with select_action static method.
    env : TravelBookingEnv
        Environment to run the episode in.

    Returns
    -------
    float
        Total episode reward.
    """
    state = env.reset()
    done = False
    episode_reward = 0.0

    while not done:
        action = policy_class.select_action(state, env)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        state = next_state

    return episode_reward
