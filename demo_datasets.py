"""
Pre-defined scenarios for Q-Learning Travel Booking demo.
10 diverse booking situations for testing agent decisions.
"""

from typing import Dict, List, Any


class ScenarioDataset:
    """Container for 10 pre-defined demo scenarios."""

    def __init__(self):
        self.scenarios = [
            {
                "scenario_id": 1,
                "name": "Early Bird Deal",
                "description": "Book far in advance with favorable price and good balance",
                "initial_state": {
                    "days_until_departure": 30,
                    "current_price": 125.0,
                    "points_balance": 15000,
                    "cash_budget": 1500,
                    "airline": "AirlineB"
                }
            },
            {
                "scenario_id": 2,
                "name": "Last-Minute Booking",
                "description": "Expensive prices near departure deadline with limited points",
                "initial_state": {
                    "days_until_departure": 3,
                    "current_price": 450.0,
                    "points_balance": 8000,
                    "cash_budget": 800,
                    "airline": "AirlineA"
                }
            },
            {
                "scenario_id": 3,
                "name": "Mid-Range Standard",
                "description": "Typical booking window with moderate resources",
                "initial_state": {
                    "days_until_departure": 14,
                    "current_price": 280.0,
                    "points_balance": 20000,
                    "cash_budget": 1200,
                    "airline": "AirlineC"
                }
            },
            {
                "scenario_id": 4,
                "name": "Points-Rich Scenario",
                "description": "High points balance but limited cash, moderate time",
                "initial_state": {
                    "days_until_departure": 7,
                    "current_price": 320.0,
                    "points_balance": 30000,
                    "cash_budget": 400,
                    "airline": "AirlineA"
                }
            },
            {
                "scenario_id": 5,
                "name": "Cash-Limited Scenario",
                "description": "Very limited cash and points with reasonable time",
                "initial_state": {
                    "days_until_departure": 10,
                    "current_price": 200.0,
                    "points_balance": 2000,
                    "cash_budget": 300,
                    "airline": "AirlineB"
                }
            },
            {
                "scenario_id": 6,
                "name": "Flash Sale Window",
                "description": "Unusually low price - should the agent grab it or wait?",
                "initial_state": {
                    "days_until_departure": 5,
                    "current_price": 95.0,
                    "points_balance": 5000,
                    "cash_budget": 1000,
                    "airline": "AirlineB"
                }
            },
            {
                "scenario_id": 7,
                "name": "Deadline Pressure",
                "description": "Only 1 day left with high price and moderate resources",
                "initial_state": {
                    "days_until_departure": 1,
                    "current_price": 550.0,
                    "points_balance": 10000,
                    "cash_budget": 600,
                    "airline": "AirlineA"
                }
            },
            {
                "scenario_id": 8,
                "name": "Balanced Luxury",
                "description": "Premium flight with ample time and resources",
                "initial_state": {
                    "days_until_departure": 21,
                    "current_price": 350.0,
                    "points_balance": 25000,
                    "cash_budget": 2000,
                    "airline": "AirlineC"
                }
            },
            {
                "scenario_id": 9,
                "name": "Premium Dilemma",
                "description": "Expensive last-minute with no points - cash only option",
                "initial_state": {
                    "days_until_departure": 2,
                    "current_price": 600.0,
                    "points_balance": 0,
                    "cash_budget": 1500,
                    "airline": "AirlineA"
                }
            },
            {
                "scenario_id": 10,
                "name": "Optimal Strategy Test",
                "description": "Nearly optimal conditions - will agent find best timing?",
                "initial_state": {
                    "days_until_departure": 28,
                    "current_price": 140.0,
                    "points_balance": 12500,
                    "cash_budget": 800,
                    "airline": "AirlineB"
                }
            }
        ]

    def get_scenario(self, scenario_id: int) -> Dict[str, Any]:
        """Get scenario by ID (1-10)."""
        for scenario in self.scenarios:
            if scenario["scenario_id"] == scenario_id:
                return scenario

        raise ValueError(f"Invalid scenario_id: {scenario_id}. Must be 1-10.")

    def list_scenarios(self) -> List[str]:
        """Return formatted list of all scenarios."""
        summaries = []
        for scenario in self.scenarios:
            state = scenario["initial_state"]
            summary = (
                f"{scenario['scenario_id']:2d}. {scenario['name']:25s} | "
                f"{state['days_until_departure']:2d}d | "
                f"${state['current_price']:6.2f} | "
                f"{state['points_balance']:5.0f}pts | "
                f"${state['cash_budget']:4.0f}"
            )
            summaries.append(summary)

        return summaries

    def get_all_scenarios(self) -> List[Dict[str, Any]]:
        """Get all scenarios as a list."""
        return self.scenarios

    def __len__(self) -> int:
        """Return number of scenarios."""
        return len(self.scenarios)
