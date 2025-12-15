"""
Realistic Flight Price Simulator with Real-World Pricing Patterns.

This module generates synthetic flight prices that incorporate:
- Advance purchase curves (prices rise closer to departure)
- Day-of-week effects (cheaper Tuesdays, expensive Fridays)
- Demand spikes (simulated holidays/events)
- Flash sales (random discounts)
- Airline-specific pricing behaviors

The layered pricing model:
    Final Price = Base Price
                × Advance Purchase Multiplier
                × Day-of-Week Multiplier
                × Demand Spike Multiplier
                × Flash Sale Discount
                + GBM Noise (reduced volatility)
"""

from typing import List, Optional, Dict, Tuple
import numpy as np

from config import (
    ADVANCE_PURCHASE_CURVE, DAY_OF_WEEK_MULTIPLIERS,
    DEMAND_SPIKE_PROBABILITY, DEMAND_SPIKE_MULTIPLIER_RANGE,
    FLASH_SALE_PROBABILITY, FLASH_SALE_DISCOUNT_RANGE,
    FLASH_SALE_COOLDOWN_DAYS, AIRLINE_PROFILES,
    REALISTIC_PRICE_FLOOR, REALISTIC_PRICE_CEILING,
    MAX_PLANNING_HORIZON, logger
)

__all__ = ['RealisticFlightSimulator']


class RealisticFlightSimulator:
    """
    Generates synthetic flight prices with realistic pricing patterns.

    Incorporates multiple pricing factors that mirror real airline behavior:
    - Advance purchase effects
    - Day-of-week variations
    - Random demand spikes
    - Flash sale events
    - Airline-specific characteristics

    Attributes
    ----------
    airline : str
        The airline for this trajectory (affects base price, volatility).
    initial_days_out : int
        Days until departure at start of trajectory.
    current_price : float
        Current simulated price.
    price_history : List[float]
        History of all prices in trajectory.
    day_index : int
        Simulated day counter (for day-of-week effects).

    Parameters
    ----------
    airline : str
        Airline identifier (must be in AIRLINE_PROFILES).
    days_until_departure : int
        Initial days until departure.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        airline: str = "AirlineA",
        days_until_departure: int = 30,
        seed: Optional[int] = None
    ) -> None:
        """Initialize realistic price simulator."""
        if seed is not None:
            np.random.seed(seed)

        self.airline = airline
        self.initial_days_out = days_until_departure
        self.days_until_departure = days_until_departure

        # Get airline-specific profile
        if airline in AIRLINE_PROFILES:
            self.profile = AIRLINE_PROFILES[airline]
        else:
            # Default to mid-tier if airline not found
            self.profile = AIRLINE_PROFILES.get("AirlineC", {
                "base_price": 230.0,
                "volatility": 0.12,
                "flash_sale_freq": 0.03,
                "price_floor": 100.0,
                "price_ceiling": 550.0,
            })

        # State variables
        self.base_price: float = self.profile["base_price"]
        self.volatility: float = self.profile["volatility"]
        self.current_price: float = 0.0
        self.price_history: List[float] = []
        self.time_step: int = 0

        # Day-of-week tracking (random start day)
        self.day_index: int = np.random.randint(0, 7)

        # Demand spike state (determined at trajectory start)
        self.has_demand_spike: bool = np.random.random() < DEMAND_SPIKE_PROBABILITY
        self.demand_spike_multiplier: float = 1.0
        if self.has_demand_spike:
            self.demand_spike_multiplier = np.random.uniform(
                DEMAND_SPIKE_MULTIPLIER_RANGE[0],
                DEMAND_SPIKE_MULTIPLIER_RANGE[1]
            )

        # Flash sale state
        self.flash_sale_active: bool = False
        self.flash_sale_discount: float = 0.0
        self.days_since_flash_sale: int = FLASH_SALE_COOLDOWN_DAYS + 1

        # GBM state for noise
        self.gbm_factor: float = 1.0

        self.reset()

    def reset(self) -> float:
        """
        Reset simulator to a new price trajectory.

        Returns
        -------
        float
            The initial price.
        """
        # Reset days until departure
        self.days_until_departure = self.initial_days_out

        # Randomize starting day of week
        self.day_index = np.random.randint(0, 7)

        # Reset demand spike (new trajectory, new chance)
        self.has_demand_spike = np.random.random() < DEMAND_SPIKE_PROBABILITY
        if self.has_demand_spike:
            self.demand_spike_multiplier = np.random.uniform(
                DEMAND_SPIKE_MULTIPLIER_RANGE[0],
                DEMAND_SPIKE_MULTIPLIER_RANGE[1]
            )
        else:
            self.demand_spike_multiplier = 1.0

        # Reset flash sale state
        self.flash_sale_active = False
        self.flash_sale_discount = 0.0
        self.days_since_flash_sale = FLASH_SALE_COOLDOWN_DAYS + 1

        # Reset GBM noise factor
        self.gbm_factor = 1.0

        # Add small random variation to base price (+/- 10%)
        base_variation = np.random.uniform(0.90, 1.10)
        self.base_price = self.profile["base_price"] * base_variation

        # Calculate initial price
        self.current_price = self._calculate_price()
        self.price_history = [self.current_price]
        self.time_step = 0

        return self.current_price

    def step(self, days: int = 1) -> float:
        """
        Advance simulation by given number of days.

        Parameters
        ----------
        days : int, default=1
            Number of days to advance.

        Returns
        -------
        float
            The new current price.
        """
        for _ in range(days):
            # Advance day counters
            self.days_until_departure -= 1
            self.day_index = (self.day_index + 1) % 7
            self.time_step += 1
            self.days_since_flash_sale += 1

            # Update GBM noise factor
            self._update_gbm_noise()

            # Check for flash sale
            self._check_flash_sale()

            # Calculate new price with all factors
            self.current_price = self._calculate_price()
            self.price_history.append(self.current_price)

        return self.current_price

    def _calculate_price(self) -> float:
        """
        Calculate price using layered multiplier model.

        Returns
        -------
        float
            Calculated price with all factors applied.
        """
        price = self.base_price

        # 1. Advance Purchase Multiplier
        price *= self._get_advance_purchase_multiplier()

        # 2. Day-of-Week Multiplier
        price *= self._get_day_of_week_multiplier()

        # 3. Demand Spike Multiplier (if active for this trajectory)
        price *= self.demand_spike_multiplier

        # 4. Flash Sale Discount (if active)
        if self.flash_sale_active:
            price *= (1.0 - self.flash_sale_discount)

        # 5. GBM Noise
        price *= self.gbm_factor

        # Enforce bounds
        price = np.clip(
            price,
            self.profile.get("price_floor", REALISTIC_PRICE_FLOOR),
            self.profile.get("price_ceiling", REALISTIC_PRICE_CEILING)
        )

        return float(price)

    def _get_advance_purchase_multiplier(self) -> float:
        """
        Get price multiplier based on days until departure.

        Prices increase as departure approaches.

        Returns
        -------
        float
            Multiplier for advance purchase effect.
        """
        days = max(0, self.days_until_departure)

        # Find applicable threshold
        sorted_thresholds = sorted(ADVANCE_PURCHASE_CURVE.keys(), reverse=True)

        for threshold in sorted_thresholds:
            if days >= threshold:
                return ADVANCE_PURCHASE_CURVE[threshold]

        # If less than smallest threshold, use the 0-day multiplier
        return ADVANCE_PURCHASE_CURVE.get(0, 1.60)

    def _get_day_of_week_multiplier(self) -> float:
        """
        Get price multiplier based on day of week.

        Tuesday/Wednesday are cheaper, Friday is most expensive.

        Returns
        -------
        float
            Multiplier for day-of-week effect.
        """
        return DAY_OF_WEEK_MULTIPLIERS.get(self.day_index, 1.0)

    def _update_gbm_noise(self) -> None:
        """
        Update the GBM noise factor for daily price variation.

        Uses reduced volatility compared to pure GBM since other
        factors now drive systematic price changes.
        """
        # Reduced drift and volatility for noise component
        drift = 0.0001 - 0.5 * self.volatility ** 2
        diffusion = self.volatility * np.random.randn()

        log_return = drift + diffusion
        self.gbm_factor *= np.exp(log_return)

        # Keep noise factor bounded (0.7 to 1.3)
        self.gbm_factor = np.clip(self.gbm_factor, 0.7, 1.3)

    def _check_flash_sale(self) -> None:
        """
        Check if a flash sale should start or end.

        Flash sales are random discounts with cooldown periods.
        """
        # End existing flash sale after 1-2 days
        if self.flash_sale_active:
            if np.random.random() < 0.5:  # 50% chance to end each day
                self.flash_sale_active = False
                self.flash_sale_discount = 0.0
            return

        # Check if cooldown has passed
        if self.days_since_flash_sale < FLASH_SALE_COOLDOWN_DAYS:
            return

        # Use airline-specific flash sale frequency
        flash_prob = self.profile.get("flash_sale_freq", FLASH_SALE_PROBABILITY)

        # Random chance of new flash sale
        if np.random.random() < flash_prob:
            self.flash_sale_active = True
            self.flash_sale_discount = np.random.uniform(
                FLASH_SALE_DISCOUNT_RANGE[0],
                FLASH_SALE_DISCOUNT_RANGE[1]
            )
            self.days_since_flash_sale = 0

    def get_price_percentile(self) -> float:
        """
        Return current price as percentile of historical range.

        Returns
        -------
        float
            Percentile in [0, 1].
        """
        if len(self.price_history) < 2:
            return 0.5

        min_price = min(self.price_history)
        max_price = max(self.price_history)

        if max_price == min_price:
            return 0.5

        percentile = (self.current_price - min_price) / (max_price - min_price)
        return float(np.clip(percentile, 0, 1))

    def get_trajectory_stats(self) -> Dict:
        """
        Get statistics about the current price trajectory.

        Returns
        -------
        Dict
            Statistics including factors and price info.
        """
        return {
            "airline": self.airline,
            "initial_days_out": self.initial_days_out,
            "days_remaining": self.days_until_departure,
            "current_price": self.current_price,
            "initial_price": self.price_history[0] if self.price_history else 0,
            "min_price": min(self.price_history) if self.price_history else 0,
            "max_price": max(self.price_history) if self.price_history else 0,
            "mean_price": np.mean(self.price_history) if self.price_history else 0,
            "has_demand_spike": self.has_demand_spike,
            "demand_spike_multiplier": self.demand_spike_multiplier,
            "flash_sale_active": self.flash_sale_active,
            "current_day_of_week": self.day_index,
            "advance_purchase_mult": self._get_advance_purchase_multiplier(),
            "day_of_week_mult": self._get_day_of_week_multiplier(),
        }

    def get_current_factors(self) -> Dict[str, float]:
        """
        Get breakdown of current price factors.

        Useful for debugging and understanding price composition.

        Returns
        -------
        Dict[str, float]
            Individual multipliers currently applied.
        """
        return {
            "base_price": self.base_price,
            "advance_purchase_mult": self._get_advance_purchase_multiplier(),
            "day_of_week_mult": self._get_day_of_week_multiplier(),
            "demand_spike_mult": self.demand_spike_multiplier,
            "flash_sale_discount": self.flash_sale_discount if self.flash_sale_active else 0.0,
            "gbm_noise_factor": self.gbm_factor,
            "final_price": self.current_price,
        }

    def __repr__(self) -> str:
        factors = []
        if self.has_demand_spike:
            factors.append(f"spike={self.demand_spike_multiplier:.2f}")
        if self.flash_sale_active:
            factors.append(f"sale=-{self.flash_sale_discount:.0%}")

        factor_str = f", {', '.join(factors)}" if factors else ""

        return (
            f"RealisticFlightSimulator({self.airline}, "
            f"${self.current_price:.2f}, "
            f"days={self.days_until_departure}{factor_str})"
        )
