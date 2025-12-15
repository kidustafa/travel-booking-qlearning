"""
Synthetic flight price generator using Geometric Brownian Motion (GBM).

This module simulates realistic flight price trajectories that exhibit
the stochastic behavior observed in real airline pricing.

Reference:
    Heston, S. L. (1993). A closed-form solution for options with stochastic
    volatility with applications to bond and currency options.
    Review of Financial Studies, 6(2), 327-343.
"""

from typing import List, Optional
import numpy as np

from config import (
    GBM_DRIFT, GBM_VOLATILITY, PRICE_BASE_MEAN, PRICE_BASE_STD, logger
)

__all__ = ['FlightPriceSimulator']


class FlightPriceSimulator:
    """
    Generates synthetic flight prices following Geometric Brownian Motion.

    The price evolution follows:
        P_t = P_{t-1} * exp((μ - σ²/2)Δt + σ√Δt * Z_t)

    where:
        - μ (drift): Expected return rate
        - σ (volatility): Standard deviation of returns
        - Z_t: Standard normal random variable

    Attributes
    ----------
    initial_price : float
        The starting price for the current trajectory.
    current_price : float
        The current simulated price.
    price_history : List[float]
        History of all prices in the current trajectory.
    time_step : int
        Current time step (days elapsed).

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    """

    # Price bounds to keep simulations realistic
    PRICE_FLOOR: float = 80.0
    PRICE_CEILING: float = 600.0

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize price simulator with optional random seed."""
        if seed is not None:
            np.random.seed(seed)

        self.initial_price: float = 0.0
        self.current_price: float = 0.0
        self.price_history: List[float] = []
        self.time_step: int = 0

        self.reset()

    def reset(self) -> float:
        """
        Reset simulator to a new price trajectory.

        Returns
        -------
        float
            The new initial price.
        """
        self.initial_price = np.random.normal(PRICE_BASE_MEAN, PRICE_BASE_STD)
        self.initial_price = max(self.PRICE_FLOOR, self.initial_price)

        self.current_price = self.initial_price
        self.price_history = [self.current_price]
        self.time_step = 0

        return self.current_price

    def step(self, days: int = 1) -> float:
        """
        Advance price by given number of days using GBM.

        Parameters
        ----------
        days : int, default=1
            Number of days to advance the simulation.

        Returns
        -------
        float
            The new current price after stepping forward.
        """
        for _ in range(days):
            # GBM formula: drift-adjusted + diffusion term
            drift = GBM_DRIFT - 0.5 * GBM_VOLATILITY ** 2
            diffusion = GBM_VOLATILITY * np.random.randn()

            log_return = drift + diffusion
            self.current_price = self.current_price * np.exp(log_return)

            # Enforce price bounds
            self.current_price = np.clip(
                self.current_price,
                self.PRICE_FLOOR,
                self.PRICE_CEILING
            )

            self.price_history.append(self.current_price)
            self.time_step += 1

        return self.current_price

    def predict_price_n_days_ahead(self, days: int = 7) -> float:
        """
        Simple price prediction assuming prices stay constant on average.

        This naive predictor can be replaced with more sophisticated
        forecasting methods if needed.

        Parameters
        ----------
        days : int, default=7
            Number of days ahead to predict.

        Returns
        -------
        float
            Predicted price (currently just returns current price).
        """
        return self.current_price

    def get_price_percentile(self) -> float:
        """
        Return current price as percentile of historical range.

        Useful for determining if the current price is relatively
        high or low compared to observed prices.

        Returns
        -------
        float
            Percentile in [0, 1] where 0 is lowest observed, 1 is highest.
        """
        if len(self.price_history) < 2:
            return 0.5

        min_price = min(self.price_history)
        max_price = max(self.price_history)

        if max_price == min_price:
            return 0.5

        percentile = (self.current_price - min_price) / (max_price - min_price)
        return float(np.clip(percentile, 0, 1))

    def get_trajectory_stats(self) -> dict:
        """
        Get statistics about the current price trajectory.

        Returns
        -------
        dict
            Dictionary containing min, max, mean, std, and current price.
        """
        return {
            "min": min(self.price_history),
            "max": max(self.price_history),
            "mean": np.mean(self.price_history),
            "std": np.std(self.price_history),
            "current": self.current_price,
            "initial": self.initial_price,
            "time_steps": self.time_step,
        }

    def __repr__(self) -> str:
        return (
            f"FlightPriceSimulator(current=${self.current_price:.2f}, "
            f"initial=${self.initial_price:.2f}, steps={self.time_step})"
        )
