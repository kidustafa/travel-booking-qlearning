"""
Travel Booking MDP Environment.

Implements the Markov Decision Process for travel booking decisions with
discretized state space and normalized multi-objective rewards.

Critical fixes applied:
- FIX 3: Use np.ravel_multi_index for safe state encoding
- FIX 5: Add -2.0 penalty for missing deadline

Supports both legacy GBM simulator and realistic price simulator.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from config import (
    DAYS_UNTIL_DEPARTURE_BINS, PRICE_BINS, POINTS_BALANCE_BINS,
    CASH_BUDGET_BINS, AIRLINES, NUM_STATES, MAX_PLANNING_HORIZON,
    ACTION_BOOK_CASH, ACTION_BOOK_POINTS, ACTION_WAIT_3_DAYS,
    ACTION_WAIT_7_DAYS, ACTION_NAMES, LOYALTY_PROGRAMS,
    INITIAL_CASH_MEAN, INITIAL_CASH_STD, INITIAL_POINTS_MEAN,
    INITIAL_POINTS_STD, WEIGHT_SCHEMES, logger
)
from simulator import FlightPriceSimulator
from realistic_simulator import RealisticFlightSimulator
from reward_normalizer import RewardNormalizer

__all__ = ['TravelBookingEnv']


class TravelBookingEnv:
    """
    Markov Decision Process for travel booking decisions.

    State Space (2,016 discrete states):
        - days_until_departure: 6 bins
        - current_price: 7 bins
        - points_balance: 4 bins
        - cash_budget: 4 bins
        - airline: 3 options

    Action Space (4 actions):
        - BookNow_Cash (0)
        - BookNow_Points (1)
        - Wait_3Days (2)
        - Wait_7Days (3)

    Reward: Normalized aggregation of cost, points, and time objectives.

    Attributes
    ----------
    price_sim : FlightPriceSimulator
        GBM-based flight price simulator.
    normalizer : RewardNormalizer
        Multi-objective reward normalizer.
    days_until_departure : int
        Days remaining until flight departure.
    cash_budget : float
        Available cash budget.
    points_balance : float
        Available loyalty points.
    current_price : float
        Current flight price.
    airline : str
        Selected airline for the trip.
    booked : bool
        Whether a booking has been made.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    weight_scheme : str, default="balanced"
        Reward weighting scheme: "balanced", "cost_optimized", or "rewards_optimized".
    use_realistic_simulator : bool, default=True
        If True, use RealisticFlightSimulator with real-world pricing patterns.
        If False, use legacy FlightPriceSimulator with simple GBM.
    """

    # Penalty for missing deadline without booking (FIX 5)
    DEADLINE_PENALTY: float = -2.0

    def __init__(
        self,
        seed: Optional[int] = None,
        weight_scheme: str = "balanced",
        use_realistic_simulator: bool = True
    ) -> None:
        """Initialize travel booking environment."""
        if seed is not None:
            np.random.seed(seed)

        self.use_realistic_simulator = use_realistic_simulator
        self.seed = seed

        # Simulator will be created in reset() with airline info
        self.price_sim = None
        self.normalizer = RewardNormalizer(weights=WEIGHT_SCHEMES[weight_scheme])

        # Current episode state
        self.days_until_departure: int = 0
        self.cash_budget: float = 0.0
        self.points_balance: float = 0.0
        self.current_price: float = 0.0
        self.airline: str = ""
        self.booked: bool = False

        # Tracking
        self.reward_history: List[float] = []
        self.action_history: List[int] = []
        self.price_history: List[float] = []

    def reset(self) -> int:
        """
        Reset environment to new episode.

        Returns
        -------
        int
            Initial encoded state.
        """
        # Random initial conditions
        self.days_until_departure = np.random.randint(1, MAX_PLANNING_HORIZON + 1)
        self.cash_budget = max(100, np.random.normal(INITIAL_CASH_MEAN, INITIAL_CASH_STD))
        self.points_balance = max(0, np.random.normal(INITIAL_POINTS_MEAN, INITIAL_POINTS_STD))
        self.airline = np.random.choice(AIRLINES)

        # Create/reset price simulator based on type
        if self.use_realistic_simulator:
            # Realistic simulator needs airline and days info
            self.price_sim = RealisticFlightSimulator(
                airline=self.airline,
                days_until_departure=self.days_until_departure,
                seed=None  # Don't re-seed, already done in __init__
            )
        else:
            # Legacy GBM simulator
            if self.price_sim is None:
                self.price_sim = FlightPriceSimulator(seed=self.seed)
            else:
                self.price_sim.reset()

        self.current_price = self.price_sim.current_price

        self.booked = False
        self.reward_history = []
        self.action_history = []
        self.price_history = [self.current_price]

        return self._encode_state()

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Execute action in environment.

        Parameters
        ----------
        action : int
            Action index (0-3).

        Returns
        -------
        Tuple[int, float, bool, Dict[str, Any]]
            (next_state, reward, done, info)
        """
        self.action_history.append(action)

        done = False
        info: Dict[str, Any] = {}
        reward = 0.0

        if action == ACTION_BOOK_CASH:
            reward = self._book_cash()
            done = True
            info["booking_type"] = "cash"
            info["price_paid"] = self.current_price

        elif action == ACTION_BOOK_POINTS:
            reward = self._book_points()
            done = True
            info["booking_type"] = "points"

        elif action == ACTION_WAIT_3_DAYS:
            reward = self._wait(3)

        elif action == ACTION_WAIT_7_DAYS:
            reward = self._wait(7)

        # FIX 5: Check deadline and apply penalty if not booked
        if self.days_until_departure <= 0:
            done = True

            # If we reached deadline without booking, severe penalty!
            if self.action_history[-1] not in [ACTION_BOOK_CASH, ACTION_BOOK_POINTS]:
                reward = self.DEADLINE_PENALTY
                info["terminal_reason"] = "deadline_missed_no_booking"
            else:
                info["terminal_reason"] = "booking_complete"

        self.reward_history.append(reward)
        next_state = self._encode_state()

        return next_state, reward, done, info

    def _book_cash(self) -> float:
        """
        Execute BookNow_Cash action.

        Returns
        -------
        float
            Normalized reward for the action.
        """
        cost = self.current_price
        r_cash = -cost
        r_points = 0
        r_time = -self.days_until_departure

        reward = self.normalizer.aggregate_rewards(r_cash, r_points, r_time)
        self.booked = True
        return reward

    def _book_points(self) -> float:
        """
        Execute BookNow_Points action.

        If sufficient points available, use them. Otherwise fall back to cash.

        Returns
        -------
        float
            Normalized reward for the action.
        """
        # Get points requirement for the airline
        airline_program = f"{self.airline}_Miles"
        if airline_program not in LOYALTY_PROGRAMS:
            # Fallback to cash if airline program not found
            return self._book_cash()

        points_req = LOYALTY_PROGRAMS[airline_program]["domestic_short"]["points_required"]

        if self.points_balance >= points_req:
            # Successfully use points
            points_val = LOYALTY_PROGRAMS[airline_program]["domestic_short"]["cash_equivalent"]
            r_cash = 0
            r_points = points_val
            self.points_balance -= points_req
        else:
            # Insufficient points, fall back to cash
            r_cash = -self.current_price
            r_points = 0

        r_time = -self.days_until_departure
        reward = self.normalizer.aggregate_rewards(r_cash, r_points, r_time)
        self.booked = True
        return reward

    def _wait(self, days: int) -> float:
        """
        Execute Wait action: advance time and update price.

        Parameters
        ----------
        days : int
            Number of days to wait.

        Returns
        -------
        float
            Normalized reward (small time penalty).
        """
        self.days_until_departure -= days

        # Step the simulator (realistic sim also decrements its internal days)
        self.price_sim.step(days=days)
        self.current_price = self.price_sim.current_price
        self.price_history.append(self.current_price)

        # Small time penalty for waiting
        r_cash = 0
        r_points = 0
        r_time = -max(0, self.days_until_departure)

        reward = self.normalizer.aggregate_rewards(r_cash, r_points, r_time)
        return reward

    def _encode_state(self) -> int:
        """
        Discretize and encode state into integer for Q-table.

        FIX 3: Uses numpy.ravel_multi_index for safe multi-dimensional indexing.
        This avoids off-by-one errors from manual calculation.

        Returns
        -------
        int
            Flattened state index in [0, NUM_STATES).
        """
        # Digitize each dimension
        days_bin = np.digitize(self.days_until_departure, DAYS_UNTIL_DEPARTURE_BINS) - 1
        price_bin = np.digitize(self.current_price, PRICE_BINS) - 1
        points_bin = np.digitize(self.points_balance, POINTS_BALANCE_BINS) - 1
        cash_bin = np.digitize(self.cash_budget, CASH_BUDGET_BINS) - 1
        airline_idx = AIRLINES.index(self.airline) if self.airline in AIRLINES else 0

        # Clip to valid ranges BEFORE creating tuple
        days_bin = int(np.clip(days_bin, 0, len(DAYS_UNTIL_DEPARTURE_BINS) - 1))
        price_bin = int(np.clip(price_bin, 0, len(PRICE_BINS) - 1))
        points_bin = int(np.clip(points_bin, 0, len(POINTS_BALANCE_BINS) - 1))
        cash_bin = int(np.clip(cash_bin, 0, len(CASH_BUDGET_BINS) - 1))

        # Create multi-dimensional index tuple
        state_tuple = (days_bin, price_bin, points_bin, cash_bin, airline_idx)

        # Define dimensions
        dims = (
            len(DAYS_UNTIL_DEPARTURE_BINS),
            len(PRICE_BINS),
            len(POINTS_BALANCE_BINS),
            len(CASH_BUDGET_BINS),
            len(AIRLINES)
        )

        # Use numpy's safe multi-index flattening
        try:
            state_idx = np.ravel_multi_index(state_tuple, dims)
        except ValueError as e:
            logger.warning(f"State encoding error: {e}")
            logger.warning(f"  state_tuple: {state_tuple}, dims: {dims}")
            state_idx = 0

        return int(state_idx)

    def get_episode_summary(self) -> Dict[str, Any]:
        """
        Return summary statistics for current episode.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing episode statistics.
        """
        total_reward = sum(self.reward_history)
        final_price = self.current_price
        initial_price = self.price_history[0] if self.price_history else 0

        return {
            "total_reward": total_reward,
            "num_steps": len(self.action_history),
            "final_price": final_price,
            "initial_price": initial_price,
            "price_change": final_price - initial_price,
            "actions": [ACTION_NAMES.get(a, "Unknown") for a in self.action_history],
            "booked": self.booked,
            "airline": self.airline,
        }

    def __repr__(self) -> str:
        return (
            f"TravelBookingEnv(days={self.days_until_departure}, "
            f"price=${self.current_price:.2f}, airline={self.airline})"
        )
