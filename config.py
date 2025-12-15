"""
Configuration and constants for Q-Learning Travel Booking Agent.

All hyperparameters defined here with critical fixes applied:
- FIX 1: Bins are ASCENDING for np.digitize
- FIX 2: Reward ranges widened to avoid clipping blind spot
- FIX 4: Added epsilon decay parameters
"""

from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

__all__ = [
    # Q-Learning params
    'LEARNING_RATE', 'DISCOUNT_FACTOR', 'EXPLORATION_RATE_START',
    'EXPLORATION_RATE_END', 'EXPLORATION_RATE_DECAY', 'EPISODES',
    'TEST_EPISODES', 'RANDOM_SEEDS',
    # State space
    'DAYS_UNTIL_DEPARTURE_BINS', 'PRICE_BINS', 'POINTS_BALANCE_BINS',
    'CASH_BUDGET_BINS', 'AIRLINES', 'NUM_STATES', 'NUM_ACTIONS',
    # Actions
    'ACTION_BOOK_CASH', 'ACTION_BOOK_POINTS', 'ACTION_WAIT_3_DAYS',
    'ACTION_WAIT_7_DAYS', 'ACTION_NAMES',
    # Rewards
    'REWARD_RANGES', 'WEIGHT_SCHEMES', 'DEFAULT_WEIGHTS',
    # Legacy GBM model
    'GBM_DRIFT', 'GBM_VOLATILITY', 'PRICE_BASE_MEAN', 'PRICE_BASE_STD',
    'MAX_PLANNING_HORIZON',
    # Realistic price model
    'ADVANCE_PURCHASE_CURVE', 'DAY_OF_WEEK_MULTIPLIERS',
    'DEMAND_SPIKE_PROBABILITY', 'DEMAND_SPIKE_MULTIPLIER_RANGE',
    'FLASH_SALE_PROBABILITY', 'FLASH_SALE_DISCOUNT_RANGE',
    'FLASH_SALE_COOLDOWN_DAYS', 'AIRLINE_PROFILES',
    'REALISTIC_PRICE_FLOOR', 'REALISTIC_PRICE_CEILING',
    # Loyalty programs
    'LOYALTY_PROGRAMS',
    # Simulation params
    'INITIAL_CASH_MEAN', 'INITIAL_CASH_STD',
    'INITIAL_POINTS_MEAN', 'INITIAL_POINTS_STD',
    # Logging
    'LOG_INTERVAL', 'PLOT_RESULTS', 'RESULTS_DIR', 'logger'
]

# ============================================================================
# Q-LEARNING HYPERPARAMETERS
# ============================================================================
LEARNING_RATE: float = 0.1           # α: Step size for Q-value updates
DISCOUNT_FACTOR: float = 0.99        # γ: Importance of future rewards

# FIX 4: Epsilon decay instead of fixed exploration
EXPLORATION_RATE_START: float = 0.3       # Start with 30% exploration
EXPLORATION_RATE_END: float = 0.01        # End with 1% exploration
EXPLORATION_RATE_DECAY: float = 0.995     # Decay by 0.5% per episode

EPISODES: int = 5000              # Number of training episodes
TEST_EPISODES: int = 1000         # Number of evaluation episodes
RANDOM_SEEDS: List[int] = [42, 123, 456, 789, 999]  # For statistical significance

# ============================================================================
# STATE SPACE DISCRETIZATION
# ============================================================================
# FIX 1: ALL bins are ASCENDING (required for np.digitize)
# WRONG: [30, 21, 14, 7, 3, 1]  <- DO NOT USE
# RIGHT: [1, 3, 7, 14, 21, 30]  <- USE THIS

DAYS_UNTIL_DEPARTURE_BINS: List[int] = [1, 3, 7, 14, 21, 30]     # ASCENDING
PRICE_BINS: List[int] = [100, 150, 200, 250, 300, 350, 400]      # ASCENDING
POINTS_BALANCE_BINS: List[int] = [0, 5000, 10000, 25000]         # ASCENDING
CASH_BUDGET_BINS: List[int] = [300, 500, 1000]                   # ASCENDING
AIRLINES: List[str] = ["AirlineA", "AirlineB", "AirlineC"]       # 3 airlines

# Total state space: 6 x 7 x 4 x 4 x 3 = 2,016 states
NUM_STATES: int = (
    len(DAYS_UNTIL_DEPARTURE_BINS) *
    len(PRICE_BINS) *
    len(POINTS_BALANCE_BINS) *
    len(CASH_BUDGET_BINS) *
    len(AIRLINES)
)
NUM_ACTIONS: int = 4  # BookNow_Cash, BookNow_Points, Wait_3Days, Wait_7Days

# Action indices
ACTION_BOOK_CASH: int = 0
ACTION_BOOK_POINTS: int = 1
ACTION_WAIT_3_DAYS: int = 2
ACTION_WAIT_7_DAYS: int = 3

ACTION_NAMES: Dict[int, str] = {
    ACTION_BOOK_CASH: "BookNow_Cash",
    ACTION_BOOK_POINTS: "BookNow_Points",
    ACTION_WAIT_3_DAYS: "Wait_3Days",
    ACTION_WAIT_7_DAYS: "Wait_7Days",
}

# ============================================================================
# REWARD NORMALIZATION PARAMETERS
# ============================================================================
# FIX 2: Reward ranges widened to cover simulator's full range
# Simulator price cap: $600
# Previous range: [-500, 0]  <- Creates blind spot for $500-600
# New range: [-650, 0]       <- Covers full range with buffer

REWARD_RANGES: Dict[str, Dict[str, float]] = {
    "cash": {"min": -850, "max": 0},      # Covers up to $800 flights (realistic sim)
    "points": {"min": 0, "max": 300},     # Points value in USD
    "time": {"min": -60, "max": 0},       # Days penalty
}

# Weighting schemes for multi-objective aggregation
WEIGHT_SCHEMES: Dict[str, Dict[str, float]] = {
    "balanced": {"cash": 0.4, "points": 0.4, "time": 0.2},
    "cost_optimized": {"cash": 0.6, "points": 0.2, "time": 0.2},
    "rewards_optimized": {"cash": 0.2, "points": 0.6, "time": 0.2},
}
DEFAULT_WEIGHTS: Dict[str, float] = WEIGHT_SCHEMES["balanced"]

# ============================================================================
# FLIGHT PRICE MODEL (Geometric Brownian Motion) - Legacy
# ============================================================================
GBM_DRIFT: float = 0.0005           # μ: Slight upward drift
GBM_VOLATILITY: float = 0.15        # σ: Daily volatility (~15%)
PRICE_BASE_MEAN: float = 250        # Mean base price (USD)
PRICE_BASE_STD: float = 80          # Standard deviation of base price
MAX_PLANNING_HORIZON: int = 60      # Max days to book ahead (extended for realistic sim)

# ============================================================================
# REALISTIC PRICE MODEL PARAMETERS
# ============================================================================

# Advance Purchase Curve: Prices increase as departure approaches
# Key thresholds (days out) -> multiplier
ADVANCE_PURCHASE_CURVE: Dict[int, float] = {
    60: 0.85,   # 60+ days out: 15% discount
    30: 0.92,   # 30-60 days: 8% discount
    14: 1.00,   # 14-30 days: baseline
    7: 1.15,    # 7-14 days: 15% premium
    3: 1.35,    # 3-7 days: 35% premium
    0: 1.60,    # 0-3 days: 60% premium (last minute)
}

# Day-of-Week Effects: Based on typical booking patterns
# 0=Monday, 1=Tuesday, ..., 6=Sunday
DAY_OF_WEEK_MULTIPLIERS: Dict[int, float] = {
    0: 1.05,    # Monday - business travel demand
    1: 0.92,    # Tuesday - cheapest day
    2: 0.95,    # Wednesday - second cheapest
    3: 1.02,    # Thursday - pre-weekend uptick
    4: 1.12,    # Friday - weekend departures premium
    5: 1.08,    # Saturday - leisure travel
    6: 1.00,    # Sunday - baseline
}

# Demand Spike Probability: Random high-demand events
# Simulates holidays, events, conferences without tracking real dates
DEMAND_SPIKE_PROBABILITY: float = 0.05      # 5% chance per trajectory
DEMAND_SPIKE_MULTIPLIER_RANGE: tuple = (1.15, 1.45)  # 15-45% price increase

# Flash Sales: Random discount events
FLASH_SALE_PROBABILITY: float = 0.03        # 3% chance per day
FLASH_SALE_DISCOUNT_RANGE: tuple = (0.15, 0.30)  # 15-30% discount
FLASH_SALE_COOLDOWN_DAYS: int = 5           # Min days between flash sales

# Airline-Specific Pricing Profiles
AIRLINE_PROFILES: Dict[str, Dict[str, float]] = {
    "AirlineA": {
        "base_price": 280.0,        # Legacy carrier - higher base
        "volatility": 0.08,         # Low volatility - stable pricing
        "flash_sale_freq": 0.01,    # Rare flash sales (1%)
        "price_floor": 150.0,
        "price_ceiling": 650.0,
    },
    "AirlineB": {
        "base_price": 180.0,        # Budget carrier - lower base
        "volatility": 0.20,         # High volatility - dynamic pricing
        "flash_sale_freq": 0.05,    # Frequent flash sales (5%)
        "price_floor": 60.0,
        "price_ceiling": 450.0,
    },
    "AirlineC": {
        "base_price": 230.0,        # Mid-tier carrier
        "volatility": 0.12,         # Medium volatility
        "flash_sale_freq": 0.03,    # Moderate flash sales (3%)
        "price_floor": 100.0,
        "price_ceiling": 550.0,
    },
}

# Global price bounds for realistic simulator
REALISTIC_PRICE_FLOOR: float = 60.0     # Flash sale on budget airline
REALISTIC_PRICE_CEILING: float = 800.0  # Holiday + last-minute + legacy

# ============================================================================
# LOYALTY PROGRAMS
# ============================================================================
LOYALTY_PROGRAMS: Dict[str, Any] = {
    "AirlineA_Miles": {
        "domestic_short": {
            "points_required": 12500,
            "cash_equivalent": 125,
        },
        "domestic_long": {
            "points_required": 25000,
            "cash_equivalent": 250,
        },
    },
    "AirlineB_Miles": {
        "domestic_short": {
            "points_required": 15000,
            "cash_equivalent": 150,
        },
        "domestic_long": {
            "points_required": 30000,
            "cash_equivalent": 300,
        },
    },
    "AirlineC_Miles": {
        "domestic_short": {
            "points_required": 10000,
            "cash_equivalent": 100,
        },
        "domestic_long": {
            "points_required": 20000,
            "cash_equivalent": 200,
        },
    },
    "AmexMR_Points": {
        "cash_equivalent_per_point": 0.01,
        "transfer_ratio": 1.0,
    },
}

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
INITIAL_CASH_MEAN: float = 1500
INITIAL_CASH_STD: float = 300
INITIAL_POINTS_MEAN: float = 12000
INITIAL_POINTS_STD: float = 5000

# ============================================================================
# LOGGING AND VISUALIZATION
# ============================================================================
LOG_INTERVAL: int = 500
PLOT_RESULTS: bool = True
RESULTS_DIR: str = "data/results/"
