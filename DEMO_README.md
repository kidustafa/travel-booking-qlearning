# Q-Learning Travel Booking - Interactive Demo

## Overview

Interactive CLI demo for trained Q-learning agent. Tests agent on 10 pre-defined booking scenarios.

## Features

- **Episode Walkthrough**: Step-by-step agent decisions
- **Scenario Testing**: Compare Q-Learning vs 3 baselines
- **Pre-Generated Scenarios**: 10 scenarios, no input needed
- **Policy Comparison**: Agent vs baseline performance

## Usage

```bash
cd travel-booking-qlearning
python demo.py
```

## Menu Options

### 1. Episode Walkthrough
Step-by-step decision visualization:
- Pick from 10 scenarios
- View state, actions, rewards
- Compare vs baselines

### 2. Scenario Testing
Run all scenarios, see aggregate results:
- Q-Learning vs 3 baselines
- Win stats and mean rewards
- Best policy per scenario

### 3. List All Scenarios
View all 10 scenarios with initial conditions

### 4. Change Agent Seed
Switch Q-tables (seeds: 42, 123, 456, 789, 999)

### 5. Exit
Quit demo

## Scenarios

1. **Early Bird**: 30d, $125, 15K pts - Advance booking
2. **Last-Minute**: 3d, $450, 8K pts - High pressure
3. **Mid-Range**: 14d, $280, 20K pts - Typical
4. **Points-Rich**: 7d, $320, 30K pts - Points optimization
5. **Cash-Limited**: 10d, $200, 2K pts - Budget constraint
6. **Flash Sale**: 5d, $95, 5K pts - Deal opportunity
7. **Deadline Pressure**: 1d, $550, 10K pts - Emergency
8. **Balanced Luxury**: 21d, $350, 25K pts - Premium
9. **Premium Dilemma**: 2d, $600, 0 pts - Cash-only
10. **Optimal Strategy**: 28d, $140, 12.5K pts - Timing test

## Requirements

- Trained Q-tables in `data/results/` (run `python main.py` first)
- All project dependencies installed (`pip install -r requirements.txt`)

## Example Output

### Scenario Testing Summary
```
Scenario                  | Q-Learning   | Greedy-Cash  | Greedy-Pts   | Time-Aware
----------------------------------------------------------------------------------------
 1. Early Bird Deal       |    +0.8667   |    -0.1176   |    +0.4000   |    -0.0960
 2. Last-Minute Booking   |    -0.2435   |    -0.2435   |    -0.2435   |    -2.0000
...

Win Statistics:
  Q-Learning:    Wins: 2/10 (20.0%) | Mean Reward: -0.0073
  Greedy-Cash:   Wins: 0/10 ( 0.0%) | Mean Reward: -0.1734
  Greedy-Points: Wins: 2/10 (20.0%) | Mean Reward: +0.0545
  Time-Aware:    Wins: 2/10 (20.0%) | Mean Reward: -0.6068
```

## Files

- `demo.py` - Main interactive CLI tool
- `demo_datasets.py` - Pre-defined scenario definitions
- `demo_formatter.py` - Output formatting utilities
- `DEMO_README.md` - This file

## Notes

- Fixed seeds for reproducibility
- Greedy mode (epsilon=0) for consistent decisions
- Rewards normalized to [-1, 1]
- Positive = good, Negative = poor
