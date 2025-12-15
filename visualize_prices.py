"""
Visualization script for validating realistic price simulator patterns.

Generates plots to verify:
1. Advance purchase curves (prices rise closer to departure)
2. Day-of-week effects (Tuesday cheapest, Friday most expensive)
3. Demand spike distribution
4. Flash sale occurrences
5. Airline-specific pricing differences
6. Sample price trajectories

Usage:
    python visualize_prices.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from config import (
    AIRLINES, AIRLINE_PROFILES, ADVANCE_PURCHASE_CURVE,
    DAY_OF_WEEK_MULTIPLIERS, RESULTS_DIR, MAX_PLANNING_HORIZON
)
from realistic_simulator import RealisticFlightSimulator


def plot_advance_purchase_curve(n_simulations: int = 500, save_path: str = None):
    """
    Plot average price vs days until departure.

    Should show prices increasing as departure approaches.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for airline in AIRLINES:
        days_out = list(range(1, MAX_PLANNING_HORIZON + 1))
        avg_prices = []

        for days in days_out:
            prices = []
            for _ in range(n_simulations):
                sim = RealisticFlightSimulator(
                    airline=airline,
                    days_until_departure=days
                )
                prices.append(sim.current_price)
            avg_prices.append(np.mean(prices))

        ax.plot(days_out, avg_prices, label=airline, linewidth=2)

    ax.set_xlabel('Days Until Departure', fontsize=12)
    ax.set_ylabel('Average Price ($)', fontsize=12)
    ax.set_title('Advance Purchase Curve by Airline', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # More days out on left

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_day_of_week_distribution(n_simulations: int = 1000, save_path: str = None):
    """
    Plot price distribution by day of week.

    Should show Tuesday/Wednesday lower, Friday higher.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_prices = {day: [] for day in range(7)}

    # Run many simulations and collect prices by day
    for _ in range(n_simulations):
        sim = RealisticFlightSimulator(
            airline="AirlineC",  # Use mid-tier for clearer patterns
            days_until_departure=30
        )

        # Step through 30 days
        for _ in range(30):
            day_prices[sim.day_index].append(sim.current_price)
            sim.step(1)

    # Create box plot
    data = [day_prices[d] for d in range(7)]
    bp = ax.boxplot(data, labels=day_names, patch_artist=True)

    # Color boxes
    colors = ['#ff9999' if d in [1, 2] else '#99ccff' if d == 4 else '#cccccc'
              for d in range(7)]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title('Price Distribution by Day of Week\n(Red=Cheaper, Blue=More Expensive)',
                 fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add theoretical multipliers as reference line
    theoretical = [DAY_OF_WEEK_MULTIPLIERS[d] * AIRLINE_PROFILES["AirlineC"]["base_price"]
                   for d in range(7)]
    ax.plot(range(1, 8), theoretical, 'ko--', label='Theoretical Baseline', alpha=0.5)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_demand_spike_distribution(n_simulations: int = 1000, save_path: str = None):
    """
    Plot distribution of prices with and without demand spikes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    normal_prices = []
    spike_prices = []

    for _ in range(n_simulations):
        sim = RealisticFlightSimulator(
            airline="AirlineC",
            days_until_departure=14
        )
        if sim.has_demand_spike:
            spike_prices.append(sim.current_price)
        else:
            normal_prices.append(sim.current_price)

    ax.hist(normal_prices, bins=30, alpha=0.6, label=f'Normal ({len(normal_prices)})',
            color='blue')
    ax.hist(spike_prices, bins=30, alpha=0.6, label=f'Demand Spike ({len(spike_prices)})',
            color='red')

    ax.axvline(np.mean(normal_prices), color='blue', linestyle='--',
               label=f'Normal Mean: ${np.mean(normal_prices):.0f}')
    ax.axvline(np.mean(spike_prices) if spike_prices else 0, color='red', linestyle='--',
               label=f'Spike Mean: ${np.mean(spike_prices):.0f}' if spike_prices else '')

    ax.set_xlabel('Price ($)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Price Distribution: Normal vs Demand Spike Periods', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_flash_sale_detection(n_trajectories: int = 50, save_path: str = None):
    """
    Plot sample trajectories highlighting flash sales.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    flash_sale_days = []

    for i in range(n_trajectories):
        sim = RealisticFlightSimulator(
            airline="AirlineB",  # Budget airline has most flash sales
            days_until_departure=30
        )

        prices = [sim.current_price]
        days = [30]
        flash_days = []
        flash_prices = []

        for day in range(29, 0, -1):
            sim.step(1)
            prices.append(sim.current_price)
            days.append(day)

            if sim.flash_sale_active:
                flash_days.append(day)
                flash_prices.append(sim.current_price)
                flash_sale_days.append(day)

        # Plot trajectory (thin gray line)
        ax.plot(days, prices, 'gray', alpha=0.2, linewidth=0.5)

        # Mark flash sales
        if flash_days:
            ax.scatter(flash_days, flash_prices, c='green', s=20, alpha=0.5)

    ax.set_xlabel('Days Until Departure', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title(f'Price Trajectories with Flash Sales (AirlineB - Budget)\n'
                 f'Green dots = Flash Sales ({len(flash_sale_days)} detected in {n_trajectories} trajectories)',
                 fontsize=14)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_airline_comparison(n_simulations: int = 500, save_path: str = None):
    """
    Compare price distributions across airlines.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, airline in enumerate(AIRLINES):
        prices_14d = []
        prices_7d = []
        prices_3d = []

        for _ in range(n_simulations):
            # 14 days out
            sim = RealisticFlightSimulator(airline=airline, days_until_departure=14)
            prices_14d.append(sim.current_price)

            # 7 days out
            sim = RealisticFlightSimulator(airline=airline, days_until_departure=7)
            prices_7d.append(sim.current_price)

            # 3 days out
            sim = RealisticFlightSimulator(airline=airline, days_until_departure=3)
            prices_3d.append(sim.current_price)

        ax = axes[idx]
        bp = ax.boxplot([prices_14d, prices_7d, prices_3d],
                        labels=['14 days', '7 days', '3 days'],
                        patch_artist=True)

        colors = ['#90EE90', '#FFD700', '#FF6B6B']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        profile = AIRLINE_PROFILES[airline]
        ax.set_title(f'{airline}\nBase: ${profile["base_price"]:.0f}, '
                     f'Vol: {profile["volatility"]:.0%}', fontsize=12)
        ax.set_ylabel('Price ($)' if idx == 0 else '')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Price Distribution by Airline and Days Until Departure', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_sample_trajectories(n_trajectories: int = 50, save_path: str = None):
    """
    Plot sample price trajectories for visual inspection.

    Uses 50 trajectories by default to ensure demand spikes (5% rate) are visible.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, airline in enumerate(AIRLINES):
        ax = axes[idx]

        spike_count = 0
        normal_count = 0

        for _ in range(n_trajectories):
            sim = RealisticFlightSimulator(
                airline=airline,
                days_until_departure=30
            )

            prices = [sim.current_price]
            days = [30]

            for d in range(29, 0, -1):
                sim.step(1)
                prices.append(sim.current_price)
                days.append(d)

            # Color based on whether demand spike
            if sim.has_demand_spike:
                color = 'red'
                alpha = 0.8
                linewidth = 1.5
                spike_count += 1
            else:
                color = 'blue'
                alpha = 0.2
                linewidth = 0.8
                normal_count += 1

            ax.plot(days, prices, color=color, alpha=alpha, linewidth=linewidth)

        ax.set_xlabel('Days Until Departure')
        ax.set_ylabel('Price ($)' if idx == 0 else '')
        ax.set_title(f'{airline}\n(Red: {spike_count} spikes, Blue: {normal_count} normal)')
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Sample Price Trajectories (Red=Demand Spike, Blue=Normal)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def print_validation_stats(n_simulations: int = 1000):
    """
    Print statistics to validate simulator behavior.
    """
    print("\n" + "=" * 60)
    print("REALISTIC SIMULATOR VALIDATION STATISTICS")
    print("=" * 60)

    # 1. Advance Purchase Effect
    print("\n1. ADVANCE PURCHASE EFFECT")
    print("-" * 40)
    for airline in AIRLINES:
        price_60d = np.mean([RealisticFlightSimulator(airline, 60).current_price
                            for _ in range(n_simulations)])
        price_3d = np.mean([RealisticFlightSimulator(airline, 3).current_price
                           for _ in range(n_simulations)])
        increase = (price_3d - price_60d) / price_60d * 100
        print(f"{airline}: ${price_60d:.0f} (60d) -> ${price_3d:.0f} (3d) = +{increase:.1f}%")

    # 2. Day-of-Week Effect
    print("\n2. DAY-OF-WEEK EFFECT (AirlineC, 14 days out)")
    print("-" * 40)
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_prices = {d: [] for d in range(7)}

    for _ in range(n_simulations):
        sim = RealisticFlightSimulator("AirlineC", 14)
        day_prices[sim.day_index].append(sim.current_price)

    tuesday_avg = np.mean(day_prices[1])
    friday_avg = np.mean(day_prices[4])
    for d in range(7):
        if day_prices[d]:
            avg = np.mean(day_prices[d])
            diff = (avg - tuesday_avg) / tuesday_avg * 100
            print(f"{day_names[d]}: ${avg:.0f} ({diff:+.1f}% vs Tue)")

    # 3. Demand Spike Rate
    print("\n3. DEMAND SPIKE RATE")
    print("-" * 40)
    spike_count = sum(1 for _ in range(n_simulations)
                      if RealisticFlightSimulator("AirlineC", 14).has_demand_spike)
    print(f"Observed: {spike_count}/{n_simulations} = {spike_count/n_simulations:.1%}")
    print(f"Expected: ~5%")

    # 4. Flash Sale Rate
    print("\n4. FLASH SALE RATE (30-day trajectories)")
    print("-" * 40)
    for airline in AIRLINES:
        flash_count = 0
        total_days = 0
        for _ in range(100):
            sim = RealisticFlightSimulator(airline, 30)
            for _ in range(30):
                sim.step(1)
                total_days += 1
                if sim.flash_sale_active:
                    flash_count += 1
        rate = flash_count / total_days * 100
        expected = AIRLINE_PROFILES[airline]["flash_sale_freq"] * 100
        print(f"{airline}: {rate:.2f}% observed (expected ~{expected:.1f}%)")

    print("\n" + "=" * 60)


def main():
    """Generate all validation plots."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Generating realistic simulator validation plots...")

    # Print stats first
    print_validation_stats(n_simulations=500)

    # Generate plots
    print("\nGenerating plots...")

    plot_advance_purchase_curve(
        n_simulations=300,
        save_path=os.path.join(RESULTS_DIR, "advance_purchase_curve.png")
    )

    plot_day_of_week_distribution(
        n_simulations=500,
        save_path=os.path.join(RESULTS_DIR, "day_of_week_distribution.png")
    )

    plot_demand_spike_distribution(
        n_simulations=1000,
        save_path=os.path.join(RESULTS_DIR, "demand_spike_distribution.png")
    )

    plot_flash_sale_detection(
        n_trajectories=50,
        save_path=os.path.join(RESULTS_DIR, "flash_sale_detection.png")
    )

    plot_airline_comparison(
        n_simulations=300,
        save_path=os.path.join(RESULTS_DIR, "airline_comparison.png")
    )

    plot_sample_trajectories(
        n_trajectories=10,
        save_path=os.path.join(RESULTS_DIR, "sample_trajectories.png")
    )

    print(f"\nAll plots saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
