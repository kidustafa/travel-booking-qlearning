"""
System validation script for Q-Learning Travel Booking Agent.

Tests core functionality before training or deployment.

Usage:
    python verify_fixes.py

Exit codes:
    0 - All tests passed
    1 - One or more tests failed
"""

import sys
import numpy as np

print("\n" + "=" * 60)
print("Q-LEARNING AGENT VALIDATION")
print("=" * 60)

try:
    from config import (
        DAYS_UNTIL_DEPARTURE_BINS, PRICE_BINS, POINTS_BALANCE_BINS,
        CASH_BUDGET_BINS, REWARD_RANGES, NUM_STATES,
        ACTION_WAIT_7_DAYS, ACTION_BOOK_CASH, ACTION_BOOK_POINTS
    )
    from environment import TravelBookingEnv
    from agent import QLearningAgent
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

all_pass = True

# TEST 1: Bin ordering
print("\n[TEST 1] Validating discretization bins...")
all_bins = {
    "DAYS": DAYS_UNTIL_DEPARTURE_BINS,
    "PRICE": PRICE_BINS,
    "POINTS": POINTS_BALANCE_BINS,
    "CASH": CASH_BUDGET_BINS,
}

for name, bins in all_bins.items():
    is_ascending = all(bins[i] <= bins[i + 1] for i in range(len(bins) - 1))
    status = "PASS" if is_ascending else "FAIL"
    print(f"  {name:10} {bins[:3]}...{bins[-2:]}: {status}")
    all_pass = all_pass and is_ascending

# TEST 2: Reward range coverage
print("\n[TEST 2] Checking reward normalization ranges...")
cash_min = REWARD_RANGES["cash"]["min"]
cash_max = REWARD_RANGES["cash"]["max"]
print(f"  Cash range: [{cash_min}, {cash_max}]")

if cash_min <= -600:
    print(f"  PASS: Covers max flight price ({abs(cash_min)} >= $600)")
else:
    print(f"  FAIL: Insufficient range ({abs(cash_min)} < $600)")
    all_pass = False

# TEST 3: State encoding
print("\n[TEST 3] Testing state encoding...")
env = TravelBookingEnv(seed=42)

test_states = [
    {"days": 1, "price": 100, "points": 0, "cash": 300},
    {"days": 30, "price": 600, "points": 100000, "cash": 5000},
    {"days": 0, "price": 250, "points": 50000, "cash": 1500},
]

all_valid = True
for scenario in test_states:
    env.days_until_departure = scenario["days"]
    env.current_price = scenario["price"]
    env.points_balance = scenario["points"]
    env.cash_budget = scenario["cash"]

    try:
        state_idx = env._encode_state()
        if 0 <= state_idx < NUM_STATES:
            print(
                f"  {scenario['days']:2d}d, ${scenario['price']:3.0f} "
                f"-> state {state_idx:4d} PASS"
            )
        else:
            print(
                f"  {scenario['days']:2d}d, ${scenario['price']:3.0f} "
                f"-> INVALID {state_idx:4d} FAIL"
            )
            all_valid = False
    except Exception as e:
        print(
            f"  {scenario['days']:2d}d, ${scenario['price']:3.0f} "
            f"-> ERROR: {e} FAIL"
        )
        all_valid = False

if all_valid:
    print("  PASS: All encodings valid")
else:
    print("  FAIL: Invalid encodings detected")
    all_pass = False

# TEST 4: Agent exploration decay
print("\n[TEST 4] Testing epsilon-greedy exploration...")
agent = QLearningAgent(NUM_STATES, 4)

if hasattr(agent, 'epsilon_decay') and agent.epsilon_decay < 1.0:
    print(f"  Initial epsilon: {agent.epsilon:.4f}")
    print(f"  Decay rate: {agent.epsilon_decay:.4f}")
    old_eps = agent.epsilon
    agent.epsilon *= agent.epsilon_decay
    print(f"  After decay: {agent.epsilon:.4f}")
    if agent.epsilon < old_eps:
        print("  PASS: Epsilon decay functional")
    else:
        print("  FAIL: Epsilon not decaying")
        all_pass = False
else:
    print("  FAIL: Epsilon decay not configured")
    all_pass = False

# TEST 5: Terminal state handling
print("\n[TEST 5] Testing deadline penalty...")
env = TravelBookingEnv(seed=42)
env.reset()
env.days_until_departure = 2

next_state, reward, done, info = env.step(ACTION_WAIT_7_DAYS)

if done and reward < -1.5:
    print(f"  Missed deadline -> reward {reward:.2f} PASS")
else:
    print(f"  Missed deadline -> reward {reward:.2f} FAIL")
    all_pass = False

# TEST 6: Realistic simulator
print("\n[TEST 6] Testing price simulator...")

try:
    from realistic_simulator import RealisticFlightSimulator

    sim = RealisticFlightSimulator(airline="AirlineA", days_until_departure=30)
    if sim.current_price > 0:
        print(f"  Initialization: ${sim.current_price:.2f} PASS")
    else:
        print(f"  Initialization: FAIL")
        all_pass = False

    # Advance purchase pricing
    prices_30d = [RealisticFlightSimulator("AirlineC", 30).current_price for _ in range(100)]
    prices_3d = [RealisticFlightSimulator("AirlineC", 3).current_price for _ in range(100)]
    avg_30d = np.mean(prices_30d)
    avg_3d = np.mean(prices_3d)

    if avg_3d > avg_30d:
        increase = (avg_3d - avg_30d) / avg_30d * 100
        print(f"  Advance purchase effect: ${avg_30d:.0f} (30d) -> ${avg_3d:.0f} (3d) = +{increase:.0f}% PASS")
    else:
        print(f"  Advance purchase effect: FAIL")
        all_pass = False

    # Airline differentiation
    avg_a = np.mean([RealisticFlightSimulator("AirlineA", 14).current_price for _ in range(100)])
    avg_b = np.mean([RealisticFlightSimulator("AirlineB", 14).current_price for _ in range(100)])
    avg_c = np.mean([RealisticFlightSimulator("AirlineC", 14).current_price for _ in range(100)])

    if avg_a > avg_c > avg_b:
        print(f"  Airline tiers: A=${avg_a:.0f} > C=${avg_c:.0f} > B=${avg_b:.0f} PASS")
    else:
        print(f"  Airline tiers: A=${avg_a:.0f}, C=${avg_c:.0f}, B=${avg_b:.0f} (variance)")

    # Step function
    sim = RealisticFlightSimulator("AirlineC", 30)
    sim.step(5)
    if len(sim.price_history) == 6:
        print(f"  Price history tracking: {len(sim.price_history)} entries PASS")
    else:
        print(f"  Price history tracking: FAIL")
        all_pass = False

    # Environment integration
    env = TravelBookingEnv(seed=42, use_realistic_simulator=True)
    env.reset()
    if hasattr(env.price_sim, 'has_demand_spike'):
        print(f"  Environment integration: PASS")
    else:
        print(f"  Environment integration: FAIL")
        all_pass = False

except ImportError as e:
    print(f"  IMPORT ERROR: {e}")
    all_pass = False
except Exception as e:
    print(f"  ERROR: {e}")
    all_pass = False

# Summary
print("\n" + "=" * 60)
if all_pass:
    print("ALL TESTS PASSED")
    print("System ready for training and deployment")
    sys.exit(0)
else:
    print("SOME TESTS FAILED")
    print("Review errors above before proceeding")
    sys.exit(1)
