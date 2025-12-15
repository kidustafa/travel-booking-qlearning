"""
CLI formatting utilities for demo output.
Colored terminal output and formatted tables.
"""

from typing import Dict, List, Any
from config import ACTION_NAMES


class ColorOutput:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'      # Magenta
    OKBLUE = '\033[94m'      # Blue
    OKCYAN = '\033[96m'      # Cyan
    OKGREEN = '\033[92m'     # Green
    WARNING = '\033[93m'     # Yellow
    FAIL = '\033[91m'        # Red
    ENDC = '\033[0m'         # Reset
    BOLD = '\033[1m'         # Bold
    UNDERLINE = '\033[4m'    # Underline

    @staticmethod
    def disable():
        """Disable colors (for non-terminal output)."""
        ColorOutput.HEADER = ''
        ColorOutput.OKBLUE = ''
        ColorOutput.OKCYAN = ''
        ColorOutput.OKGREEN = ''
        ColorOutput.WARNING = ''
        ColorOutput.FAIL = ''
        ColorOutput.ENDC = ''
        ColorOutput.BOLD = ''
        ColorOutput.UNDERLINE = ''


class TableFormatter:
    """Format episode steps for display."""

    @staticmethod
    def format_episode_step(
        step: int,
        state: Dict[str, Any],
        action: int,
        reward: float,
        next_state: Dict[str, Any],
        reward_breakdown: Dict[str, float],
        done: bool
    ) -> str:
        """Format single episode step with state, action, and reward."""
        action_name = ACTION_NAMES.get(action, "Unknown")
        lines = []
        lines.append(f"\n{ColorOutput.BOLD}STEP {step}:{ColorOutput.ENDC}")
        lines.append(f"  {ColorOutput.OKCYAN}Current State:{ColorOutput.ENDC}")
        lines.append(
            f"    Days: {state['days']:2d}          | "
            f"Price: ${state['price']:7.2f}          | "
            f"Airline: {state['airline']}"
        )
        lines.append(
            f"    Points: {state['points']:6.0f}    | "
            f"Cash: ${state['cash']:7.2f}"
        )

        lines.append(f"\n  {ColorOutput.OKGREEN}Agent Decision:{ColorOutput.ENDC} {action_name}")
        lines.append(f"\n  {ColorOutput.WARNING}Reward Breakdown:{ColorOutput.ENDC}")
        lines.append(
            f"    Cash component:   {reward_breakdown['cash']:+7.4f} "
            f"(normalized from cash {'cost' if reward_breakdown['cash'] < 0 else 'gain'})"
        )
        lines.append(
            f"    Points component: {reward_breakdown['points']:+7.4f} "
            f"(normalized from points value)"
        )
        lines.append(
            f"    Time component:   {reward_breakdown['time']:+7.4f} "
            f"(normalized from time penalty)"
        )
        lines.append(f"    {'-' * 60}")
        reward_color = ColorOutput.OKGREEN if reward > 0 else ColorOutput.FAIL
        lines.append(
            f"    Total reward:     {reward_color}{reward:+7.4f}{ColorOutput.ENDC} "
            f"(normalized utility)"
        )

        if done:
            lines.append(f"\n  {ColorOutput.HEADER}Outcome: Episode Complete{ColorOutput.ENDC}")
            if "BookNow" in action_name:
                lines.append(f"    Flight booked successfully at ${state['price']:.2f}")
            elif next_state['days'] <= 0:
                lines.append(f"    {ColorOutput.FAIL}Deadline reached!{ColorOutput.ENDC}")
        else:
            price_change = next_state['price'] - state['price']
            price_pct = (price_change / state['price']) * 100 if state['price'] > 0 else 0
            change_color = ColorOutput.FAIL if price_change > 0 else ColorOutput.OKGREEN

            lines.append(f"\n  {ColorOutput.OKCYAN}Next State:{ColorOutput.ENDC}")
            lines.append(
                f"    Days: {next_state['days']:2d}          | "
                f"Price: ${next_state['price']:7.2f} "
                f"{change_color}({price_change:+.2f}, {price_pct:+.1f}%){ColorOutput.ENDC}"
            )

        return "\n".join(lines)


class ScenarioFormatter:
    """Format scenario headers and results."""

    @staticmethod
    def format_scenario_header(scenario: Dict[str, Any]) -> str:
        """Format scenario intro header."""
        state = scenario["initial_state"]

        lines = []
        lines.append("\n" + "=" * 80)
        lines.append(
            f"{ColorOutput.HEADER}{ColorOutput.BOLD}"
            f"SCENARIO {scenario['scenario_id']}: {scenario['name']}"
            f"{ColorOutput.ENDC}"
        )
        lines.append("=" * 80)
        lines.append(f"{ColorOutput.OKCYAN}Description:{ColorOutput.ENDC} {scenario['description']}")
        lines.append(f"\n{ColorOutput.BOLD}Initial Conditions:{ColorOutput.ENDC}")
        lines.append(f"  Days until departure: {state['days_until_departure']} days")
        lines.append(f"  Current price: ${state['current_price']:.2f}")
        lines.append(f"  Points balance: {state['points_balance']:,.0f} points")
        lines.append(f"  Cash available: ${state['cash_budget']:.2f}")
        lines.append(f"  Airline: {state['airline']}")
        lines.append("=" * 80)

        return "\n".join(lines)

    @staticmethod
    def format_episode_summary(
        scenario_id: int,
        policy_name: str,
        total_reward: float,
        final_price: float,
        actions_taken: List[str],
        booked: bool,
        num_steps: int
    ) -> str:
        """Format episode outcome summary."""
        lines = []
        lines.append(f"\n{ColorOutput.BOLD}{'-' * 80}{ColorOutput.ENDC}")
        lines.append(f"{ColorOutput.HEADER}Episode Summary - {policy_name}{ColorOutput.ENDC}")
        lines.append(f"{ColorOutput.BOLD}{'-' * 80}{ColorOutput.ENDC}")

        reward_color = ColorOutput.OKGREEN if total_reward > 0 else ColorOutput.FAIL
        lines.append(f"  Total Reward:     {reward_color}{total_reward:+7.4f}{ColorOutput.ENDC}")
        lines.append(f"  Actions Taken:    {' -> '.join(actions_taken)}")
        lines.append(f"  Episode Steps:    {num_steps}")
        lines.append(f"  Final Price:      ${final_price:.2f}")

        status_color = ColorOutput.OKGREEN if booked else ColorOutput.FAIL
        status_text = "SUCCESS (booked)" if booked else "FAILED (not booked)"
        lines.append(f"  Booking Status:   {status_color}{status_text}{ColorOutput.ENDC}")

        return "\n".join(lines)

    @staticmethod
    def format_comparison_table(
        scenario_id: int,
        results: Dict[str, Dict[str, Any]]
    ) -> str:
        """Format policy comparison table."""
        lines = []
        lines.append(f"\n{ColorOutput.BOLD}{'=' * 80}{ColorOutput.ENDC}")
        lines.append(f"{ColorOutput.HEADER}Policy Comparison{ColorOutput.ENDC}")
        lines.append(f"{ColorOutput.BOLD}{'=' * 80}{ColorOutput.ENDC}")

        lines.append(
            f"{'Policy':20s} | {'Reward':8s} | {'Steps':5s} | "
            f"{'Price':10s} | {'Booked':7s} | {'Status':10s}"
        )
        lines.append("-" * 80)

        best_reward = max(r['total_reward'] for r in results.values())

        sorted_policies = sorted(
            results.items(),
            key=lambda x: x[1]['total_reward'],
            reverse=True
        )

        for policy_name, result in sorted_policies:
            reward = result['total_reward']
            is_best = abs(reward - best_reward) < 0.001
            status = f"{ColorOutput.OKGREEN}BEST{ColorOutput.ENDC}" if is_best else ""

            reward_color = ColorOutput.OKGREEN if reward > 0 else ColorOutput.FAIL
            booked_str = "Yes" if result['booked'] else "No"

            lines.append(
                f"{policy_name:20s} | "
                f"{reward_color}{reward:+8.4f}{ColorOutput.ENDC} | "
                f"{result['num_steps']:5d} | "
                f"${result['final_price']:8.2f} | "
                f"{booked_str:7s} | "
                f"{status:10s}"
            )

        return "\n".join(lines)

    @staticmethod
    def format_scenario_testing_summary(
        all_results: Dict[int, Dict[str, Dict[str, Any]]]
    ) -> str:
        """Format aggregated results for all scenarios."""
        lines = []
        lines.append(f"\n{ColorOutput.BOLD}{'=' * 100}{ColorOutput.ENDC}")
        lines.append(
            f"{ColorOutput.HEADER}{ColorOutput.BOLD}"
            f"SCENARIO TESTING SUMMARY: All 10 Scenarios"
            f"{ColorOutput.ENDC}"
        )
        lines.append(f"{ColorOutput.BOLD}{'=' * 100}{ColorOutput.ENDC}\n")

        # Header
        lines.append(
            f"{'Scenario':25s} | {'Q-Learning':12s} | "
            f"{'Greedy-Cash':12s} | {'Greedy-Pts':12s} | {'Time-Aware':12s} | {'Best':15s}"
        )
        lines.append("-" * 100)

        # Collect stats
        policy_names = ['Q-Learning', 'Greedy-Cash', 'Greedy-Points', 'Time-Aware']
        policy_wins = {name: 0 for name in policy_names}

        # Rows
        for scenario_id in sorted(all_results.keys()):
            scenario_results = all_results[scenario_id]

            # Find best reward
            best_reward = max(r['total_reward'] for r in scenario_results.values())
            best_policies = [
                name for name, r in scenario_results.items()
                if abs(r['total_reward'] - best_reward) < 0.001
            ]

            # Count wins
            if len(best_policies) == 1:
                policy_wins[best_policies[0]] += 1
                best_text = best_policies[0]
            else:
                best_text = "TIED"

            # Get scenario name
            from demo_datasets import ScenarioDataset
            dataset = ScenarioDataset()
            scenario_name = dataset.get_scenario(scenario_id)['name']

            # Format row
            row = f"{scenario_id:2d}. {scenario_name:21s} |"
            for policy in policy_names:
                reward = scenario_results[policy]['total_reward']
                color = ColorOutput.OKGREEN if reward > 0 else ColorOutput.FAIL
                row += f" {color}{reward:+10.4f}{ColorOutput.ENDC} |"

            row += f" {best_text:15s}"
            lines.append(row)

        # Summary statistics
        lines.append("-" * 100)
        lines.append(f"\n{ColorOutput.BOLD}Win Statistics:{ColorOutput.ENDC}")

        for policy in policy_names:
            all_rewards = [
                all_results[sid][policy]['total_reward']
                for sid in all_results.keys()
            ]
            mean_reward = sum(all_rewards) / len(all_rewards)
            wins = policy_wins[policy]
            win_pct = (wins / len(all_results)) * 100

            color = ColorOutput.OKGREEN if wins > 0 else ""
            lines.append(
                f"  {color}{policy:20s}{ColorOutput.ENDC}: "
                f"Wins: {wins}/10 ({win_pct:4.1f}%) | "
                f"Mean Reward: {mean_reward:+7.4f}"
            )

        return "\n".join(lines)
