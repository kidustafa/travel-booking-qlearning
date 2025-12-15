"""
Interactive CLI demo for Q-Learning Travel Booking Agent.
Runs trained agent on 10 pre-defined booking scenarios.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Any

from config import NUM_STATES, NUM_ACTIONS, ACTION_NAMES
from agent import QLearningAgent
from environment import TravelBookingEnv
from baselines import GreedyCashPolicy, GreedyPointsPolicy, TimeAwareHeuristic
from reward_normalizer import RewardNormalizer
from realistic_simulator import RealisticFlightSimulator
from demo_datasets import ScenarioDataset
from demo_formatter import ColorOutput, TableFormatter, ScenarioFormatter


class DemoRunner:
    """Runs interactive demo with trained agent."""

    def __init__(self):
        self.agent: QLearningAgent = None
        self.env: TravelBookingEnv = None
        self.scenarios: ScenarioDataset = ScenarioDataset()
        self.normalizer: RewardNormalizer = RewardNormalizer()
        self.current_seed = None

    def load_trained_agent(self, seed: int = 42) -> None:
        """Load pre-trained Q-table from disk."""
        print(f"\n{ColorOutput.OKCYAN}Loading trained agent (seed={seed})...{ColorOutput.ENDC}")

        try:
            self.agent = QLearningAgent(NUM_STATES, NUM_ACTIONS)
            qtable_path = f"data/results/qtable_seed{seed}.npy"
            self.agent.load_qtable(qtable_path)

            self.agent.epsilon = 0.0  # Greedy mode for demo

            self.current_seed = seed
            print(f"{ColorOutput.OKGREEN}[OK] Agent loaded successfully{ColorOutput.ENDC}")

        except FileNotFoundError:
            print(
                f"{ColorOutput.FAIL}[X] Error: Q-table not found at {qtable_path}{ColorOutput.ENDC}"
            )
            print(f"{ColorOutput.WARNING}Please run 'python main.py' first to train the agent.{ColorOutput.ENDC}")
            sys.exit(1)

    def initialize_scenario_state(
        self,
        env: TravelBookingEnv,
        scenario: Dict[str, Any]
    ) -> int:
        """Set env to scenario state (bypasses random reset for reproducibility)."""
        initial = scenario["initial_state"]

        # Set environment state
        env.days_until_departure = initial["days_until_departure"]
        env.current_price = initial["current_price"]
        env.points_balance = initial["points_balance"]
        env.cash_budget = initial["cash_budget"]
        env.airline = initial["airline"]
        env.booked = False

        # Initialize simulator with fixed seed
        env.price_sim = RealisticFlightSimulator(
            airline=env.airline,
            days_until_departure=env.days_until_departure,
            seed=42
        )
        env.price_sim.current_price = env.current_price

        # Reset tracking
        env.reward_history = []
        env.action_history = []
        env.price_history = [env.current_price]

        return env._encode_state()

    def run_episode_walkthrough(self, scenario_id: int) -> None:
        """Run single scenario with step-by-step display."""
        try:
            scenario = self.scenarios.get_scenario(scenario_id)
        except ValueError as e:
            print(f"{ColorOutput.FAIL}{e}{ColorOutput.ENDC}")
            return

        print(ScenarioFormatter.format_scenario_header(scenario))

        self.env = TravelBookingEnv(seed=42, use_realistic_simulator=True)
        state = self.initialize_scenario_state(self.env, scenario)

        done = False
        step = 1
        episode_rewards = []

        print(f"\n{ColorOutput.BOLD}Running Q-Learning Agent...{ColorOutput.ENDC}")
        print(f"{ColorOutput.OKCYAN}(Press Enter after each step to continue){ColorOutput.ENDC}")

        while not done:
            action = self.agent.select_action(state, training=False)

            current_state_dict = {
                'days': self.env.days_until_departure,
                'price': self.env.current_price,
                'points': self.env.points_balance,
                'cash': self.env.cash_budget,
                'airline': self.env.airline
            }

            next_state, reward, done, info = self.env.step(action)
            episode_rewards.append(reward)

            # Estimate reward components for display
            if action == 0:  # BookNow_Cash
                r_cash_norm, r_points_norm, r_time_norm = -0.5, 0.0, -0.1
            elif action == 1:  # BookNow_Points
                r_cash_norm, r_points_norm, r_time_norm = 0.0, 0.3, -0.1
            else:  # Wait
                r_cash_norm, r_points_norm, r_time_norm = 0.0, 0.0, -0.05

            reward_breakdown = {
                'cash': r_cash_norm,
                'points': r_points_norm,
                'time': r_time_norm
            }

            next_state_dict = {
                'days': self.env.days_until_departure,
                'price': self.env.current_price,
                'points': self.env.points_balance,
                'cash': self.env.cash_budget,
                'airline': self.env.airline
            }

            step_output = TableFormatter.format_episode_step(
                step=step,
                state=current_state_dict,
                action=action,
                reward=reward,
                next_state=next_state_dict,
                reward_breakdown=reward_breakdown,
                done=done
            )
            print(step_output)

            if not done:
                input(f"\n{ColorOutput.OKCYAN}Press Enter to continue...{ColorOutput.ENDC}")

            state = next_state
            step += 1

        total_reward = sum(episode_rewards)
        actions_taken = [ACTION_NAMES[a] for a in self.env.action_history]

        summary = ScenarioFormatter.format_episode_summary(
            scenario_id=scenario_id,
            policy_name="Q-Learning Agent",
            total_reward=total_reward,
            final_price=self.env.current_price,
            actions_taken=actions_taken,
            booked=self.env.booked,
            num_steps=len(self.env.action_history)
        )
        print(summary)

        print(f"\n{ColorOutput.OKCYAN}Running baseline policies for comparison...{ColorOutput.ENDC}")
        baseline_results = self._run_baselines_for_scenario(scenario)

        all_results = {
            'Q-Learning': {
                'total_reward': total_reward,
                'num_steps': len(self.env.action_history),
                'final_price': self.env.current_price,
                'booked': self.env.booked
            }
        }
        all_results.update(baseline_results)

        comparison = ScenarioFormatter.format_comparison_table(scenario_id, all_results)
        print(comparison)

    def _run_baselines_for_scenario(
        self,
        scenario: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Run all baseline policies on scenario."""
        baselines = {
            'Greedy-Cash': GreedyCashPolicy,
            'Greedy-Points': GreedyPointsPolicy,
            'Time-Aware': TimeAwareHeuristic
        }

        results = {}

        for policy_name, policy_class in baselines.items():
            env = TravelBookingEnv(seed=42, use_realistic_simulator=True)
            state = self.initialize_scenario_state(env, scenario)

            done = False
            episode_rewards = []

            while not done:
                action = policy_class.select_action(state, env)
                next_state, reward, done, info = env.step(action)
                episode_rewards.append(reward)
                state = next_state

            results[policy_name] = {
                'total_reward': sum(episode_rewards),
                'num_steps': len(env.action_history),
                'final_price': env.current_price,
                'booked': env.booked
            }

        return results

    def run_scenario_testing(self) -> None:
        """Run all scenarios and compare policies."""
        print(f"\n{ColorOutput.BOLD}{'=' * 100}{ColorOutput.ENDC}")
        print(
            f"{ColorOutput.HEADER}{ColorOutput.BOLD}"
            f"SCENARIO TESTING: Running all 10 scenarios"
            f"{ColorOutput.ENDC}"
        )
        print(f"{ColorOutput.BOLD}{'=' * 100}{ColorOutput.ENDC}\n")

        all_scenarios = self.scenarios.get_all_scenarios()
        all_results = {}

        for scenario in all_scenarios:
            scenario_id = scenario['scenario_id']
            print(
                f"{ColorOutput.OKCYAN}Running scenario {scenario_id}/10: "
                f"{scenario['name']}...{ColorOutput.ENDC}"
            )

            # Run Q-learning
            env = TravelBookingEnv(seed=42, use_realistic_simulator=True)
            state = self.initialize_scenario_state(env, scenario)

            done = False
            episode_rewards = []

            while not done:
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, info = env.step(action)
                episode_rewards.append(reward)
                state = next_state

            q_results = {
                'total_reward': sum(episode_rewards),
                'num_steps': len(env.action_history),
                'final_price': env.current_price,
                'booked': env.booked
            }

            # Run baselines
            baseline_results = self._run_baselines_for_scenario(scenario)

            # Combine results
            all_results[scenario_id] = {
                'Q-Learning': q_results,
                **baseline_results
            }

        # Display aggregated summary
        summary = ScenarioFormatter.format_scenario_testing_summary(all_results)
        print(summary)

    def run_interactive_menu(self) -> None:
        """Display menu and handle user input."""
        while True:
            print(f"\n{ColorOutput.BOLD}{'=' * 80}{ColorOutput.ENDC}")
            print(
                f"{ColorOutput.HEADER}{ColorOutput.BOLD}"
                f"  Q-LEARNING TRAVEL BOOKING AGENT - INTERACTIVE DEMO"
                f"{ColorOutput.ENDC}"
            )
            print(f"{ColorOutput.BOLD}{'=' * 80}{ColorOutput.ENDC}\n")

            print(f"{ColorOutput.OKCYAN}Available Demonstrations:{ColorOutput.ENDC}")
            print(f"  {ColorOutput.BOLD}1.{ColorOutput.ENDC} Episode Walkthrough        "
                  f"Show agent making decisions step-by-step")
            print(f"  {ColorOutput.BOLD}2.{ColorOutput.ENDC} Scenario Testing           "
                  f"Run all 10 scenarios and compare policies")
            print(f"  {ColorOutput.BOLD}3.{ColorOutput.ENDC} List All Scenarios         "
                  f"View available scenario summaries")
            print(f"  {ColorOutput.BOLD}4.{ColorOutput.ENDC} Change Agent Seed          "
                  f"Load different trained Q-table")
            print(f"  {ColorOutput.BOLD}5.{ColorOutput.ENDC} Exit\n")

            choice = input(f"{ColorOutput.OKGREEN}Select option (1-5): {ColorOutput.ENDC}").strip()

            if choice == '1':
                # Episode walkthrough
                self._handle_episode_walkthrough()

            elif choice == '2':
                # Scenario testing
                self.run_scenario_testing()

            elif choice == '3':
                # List scenarios
                self._list_scenarios()

            elif choice == '4':
                # Change seed
                self._change_agent_seed()

            elif choice == '5':
                # Exit
                print(f"\n{ColorOutput.OKGREEN}Thank you for using the demo!{ColorOutput.ENDC}\n")
                sys.exit(0)

            else:
                print(f"{ColorOutput.FAIL}Invalid choice. Please select 1-5.{ColorOutput.ENDC}")

    def _handle_episode_walkthrough(self) -> None:
        """Handle episode walkthrough menu option."""
        print(f"\n{ColorOutput.OKCYAN}Available Scenarios:{ColorOutput.ENDC}")
        summaries = self.scenarios.list_scenarios()
        for summary in summaries:
            print(f"  {summary}")

        scenario_input = input(
            f"\n{ColorOutput.OKGREEN}Enter scenario ID (1-10) or 'b' to go back: {ColorOutput.ENDC}"
        ).strip()

        if scenario_input.lower() == 'b':
            return

        try:
            scenario_id = int(scenario_input)
            self.run_episode_walkthrough(scenario_id)
        except ValueError:
            print(f"{ColorOutput.FAIL}Invalid input. Please enter a number 1-10.{ColorOutput.ENDC}")

    def _list_scenarios(self) -> None:
        """Display all scenarios."""
        print(f"\n{ColorOutput.BOLD}{'=' * 80}{ColorOutput.ENDC}")
        print(f"{ColorOutput.HEADER}Available Scenarios{ColorOutput.ENDC}")
        print(f"{ColorOutput.BOLD}{'=' * 80}{ColorOutput.ENDC}\n")

        for scenario in self.scenarios.get_all_scenarios():
            print(f"{ColorOutput.BOLD}{scenario['scenario_id']:2d}. {scenario['name']}{ColorOutput.ENDC}")
            print(f"    {scenario['description']}")
            state = scenario['initial_state']
            print(
                f"    {state['days_until_departure']:2d} days | "
                f"${state['current_price']:6.2f} | "
                f"{state['points_balance']:,} pts | "
                f"${state['cash_budget']:.2f} cash | "
                f"{state['airline']}\n"
            )

    def _change_agent_seed(self) -> None:
        """Change the loaded agent seed."""
        print(f"\n{ColorOutput.OKCYAN}Available seeds: 42, 123, 456, 789, 999{ColorOutput.ENDC}")
        print(f"Current seed: {self.current_seed}")

        seed_input = input(
            f"\n{ColorOutput.OKGREEN}Enter new seed or 'b' to go back: {ColorOutput.ENDC}"
        ).strip()

        if seed_input.lower() == 'b':
            return

        try:
            new_seed = int(seed_input)
            if new_seed in [42, 123, 456, 789, 999]:
                self.load_trained_agent(new_seed)
            else:
                print(
                    f"{ColorOutput.FAIL}Invalid seed. "
                    f"Please choose from: 42, 123, 456, 789, 999{ColorOutput.ENDC}"
                )
        except ValueError:
            print(f"{ColorOutput.FAIL}Invalid input. Please enter a valid seed number.{ColorOutput.ENDC}")


def main():
    """Main entry point."""
    print(f"{ColorOutput.HEADER}{ColorOutput.BOLD}")
    print("=" * 80)
    print("  Q-LEARNING TRAVEL BOOKING AGENT - INTERACTIVE DEMONSTRATION")
    print("=" * 80)
    print(f"{ColorOutput.ENDC}")

    # Initialize demo runner
    runner = DemoRunner()

    # Load default agent (seed 42)
    runner.load_trained_agent(seed=42)

    # Run interactive menu
    runner.run_interactive_menu()


if __name__ == "__main__":
    main()
