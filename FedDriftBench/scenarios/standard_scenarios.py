"""
Standard Benchmark Scenarios for FedDriftBench

Defines 4 standard testing scenarios:
- Scenario A: Homogeneous Drift (all clients drift simultaneously)
- Scenario B: Heterogeneous Drift (clients drift at different times)
- Scenario C: Partial Client Participation
- Scenario D: Staggered Drift with Dynamic Participation
"""

from typing import Dict, List, Tuple
from enum import Enum
import numpy as np


class ScenarioType(Enum):
    """Types of benchmark scenarios"""
    HOMOGENEOUS = "homogeneous"
    HETEROGENEOUS = "heterogeneous"
    PARTIAL_PARTICIPATION = "partial_participation"
    STAGGERED = "staggered"


class BenchmarkScenario:
    """Base class for benchmark scenarios"""

    def __init__(self, num_clients: int, num_rounds: int,
                 drift_start_round: int, drift_magnitude: float = 0.5):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.drift_start_round = drift_start_round
        self.drift_magnitude = drift_magnitude
        self.scenario_type = None

    def get_drift_schedule(self) -> Dict[int, Dict]:
        """
        Get drift schedule for all clients

        Returns:
            Dict mapping client_id to drift parameters
        """
        raise NotImplementedError

    def get_participation_schedule(self) -> Dict[int, List[int]]:
        """
        Get participation schedule (which clients participate in which rounds)

        Returns:
            Dict mapping round number to list of participating client IDs
        """
        # Default: all clients participate in all rounds
        return {
            round_num: list(range(self.num_clients))
            for round_num in range(self.num_rounds)
        }

    def get_scenario_description(self) -> str:
        """Get human-readable scenario description"""
        raise NotImplementedError


class HomogeneousDriftScenario(BenchmarkScenario):
    """
    Scenario A: Homogeneous Drift
    All clients experience drift at the same time
    """

    def __init__(self, num_clients: int = 10, num_rounds: int = 100,
                 drift_start_round: int = 50, drift_magnitude: float = 0.5):
        super().__init__(num_clients, num_rounds, drift_start_round, drift_magnitude)
        self.scenario_type = ScenarioType.HOMOGENEOUS

    def get_drift_schedule(self) -> Dict[int, Dict]:
        """All clients drift at the same round"""
        schedule = {}
        for client_id in range(self.num_clients):
            schedule[client_id] = {
                'start_round': self.drift_start_round,
                'end_round': self.num_rounds,
                'magnitude': self.drift_magnitude,
                'type': 'sudden'
            }
        return schedule

    def get_scenario_description(self) -> str:
        return (f"Homogeneous Drift: {self.num_clients} clients, "
                f"all drift at round {self.drift_start_round}")


class HeterogeneousDriftScenario(BenchmarkScenario):
    """
    Scenario B: Heterogeneous Drift
    Different clients experience drift at different times
    """

    def __init__(self, num_clients: int = 10, num_rounds: int = 100,
                 drift_start_round: int = 50, drift_magnitude: float = 0.5,
                 drift_spread: int = 20):
        super().__init__(num_clients, num_rounds, drift_start_round, drift_magnitude)
        self.scenario_type = ScenarioType.HETEROGENEOUS
        self.drift_spread = drift_spread

    def get_drift_schedule(self) -> Dict[int, Dict]:
        """Clients drift at different random times"""
        schedule = {}

        # Randomly assign drift start times within drift_spread
        drift_starts = np.random.randint(
            self.drift_start_round,
            min(self.drift_start_round + self.drift_spread, self.num_rounds),
            size=self.num_clients
        )

        for client_id in range(self.num_clients):
            schedule[client_id] = {
                'start_round': int(drift_starts[client_id]),
                'end_round': self.num_rounds,
                'magnitude': self.drift_magnitude,
                'type': 'sudden'
            }

        return schedule

    def get_scenario_description(self) -> str:
        return (f"Heterogeneous Drift: {self.num_clients} clients, "
                f"drift between rounds {self.drift_start_round}-"
                f"{self.drift_start_round + self.drift_spread}")


class PartialParticipationScenario(BenchmarkScenario):
    """
    Scenario C: Partial Client Participation
    Only subset of clients participate in each round
    """

    def __init__(self, num_clients: int = 10, num_rounds: int = 100,
                 drift_start_round: int = 50, drift_magnitude: float = 0.5,
                 participation_rate: float = 0.5):
        super().__init__(num_clients, num_rounds, drift_start_round, drift_magnitude)
        self.scenario_type = ScenarioType.PARTIAL_PARTICIPATION
        self.participation_rate = participation_rate

    def get_drift_schedule(self) -> Dict[int, Dict]:
        """All clients drift at same time"""
        schedule = {}
        for client_id in range(self.num_clients):
            schedule[client_id] = {
                'start_round': self.drift_start_round,
                'end_round': self.num_rounds,
                'magnitude': self.drift_magnitude,
                'type': 'sudden'
            }
        return schedule

    def get_participation_schedule(self) -> Dict[int, List[int]]:
        """Random subset of clients participate each round"""
        schedule = {}
        num_participants = max(1, int(self.num_clients * self.participation_rate))

        for round_num in range(self.num_rounds):
            participants = np.random.choice(
                self.num_clients,
                size=num_participants,
                replace=False
            ).tolist()
            schedule[round_num] = participants

        return schedule

    def get_scenario_description(self) -> str:
        return (f"Partial Participation: {self.num_clients} clients, "
                f"{self.participation_rate * 100}% participate each round, "
                f"drift at round {self.drift_start_round}")


class StaggeredDriftScenario(BenchmarkScenario):
    """
    Scenario D: Staggered Drift with Dynamic Participation
    Clients drift in groups at staggered times with varying participation
    """

    def __init__(self, num_clients: int = 10, num_rounds: int = 100,
                 drift_start_round: int = 40, drift_magnitude: float = 0.5,
                 num_groups: int = 3, stagger_interval: int = 10,
                 participation_rate: float = 0.6):
        super().__init__(num_clients, num_rounds, drift_start_round, drift_magnitude)
        self.scenario_type = ScenarioType.STAGGERED
        self.num_groups = num_groups
        self.stagger_interval = stagger_interval
        self.participation_rate = participation_rate

    def get_drift_schedule(self) -> Dict[int, Dict]:
        """Groups of clients drift at staggered times"""
        schedule = {}

        # Divide clients into groups
        clients_per_group = self.num_clients // self.num_groups
        remainder = self.num_clients % self.num_groups

        client_id = 0
        for group_idx in range(self.num_groups):
            group_size = clients_per_group + (1 if group_idx < remainder else 0)
            drift_start = self.drift_start_round + (group_idx * self.stagger_interval)

            for _ in range(group_size):
                schedule[client_id] = {
                    'start_round': drift_start,
                    'end_round': self.num_rounds,
                    'magnitude': self.drift_magnitude,
                    'type': 'sudden',
                    'group': group_idx
                }
                client_id += 1

        return schedule

    def get_participation_schedule(self) -> Dict[int, List[int]]:
        """Dynamic participation with varying rates"""
        schedule = {}
        num_participants = max(1, int(self.num_clients * self.participation_rate))

        for round_num in range(self.num_rounds):
            # Vary participation slightly around base rate
            variation = np.random.randint(-1, 2)
            round_participants = max(1, min(self.num_clients,
                                           num_participants + variation))

            participants = np.random.choice(
                self.num_clients,
                size=round_participants,
                replace=False
            ).tolist()
            schedule[round_num] = participants

        return schedule

    def get_scenario_description(self) -> str:
        return (f"Staggered Drift: {self.num_clients} clients in {self.num_groups} groups, "
                f"staggered by {self.stagger_interval} rounds, "
                f"{self.participation_rate * 100}% participation")


def get_all_scenarios(num_clients: int = 10, num_rounds: int = 100) -> List[BenchmarkScenario]:
    """
    Get all standard benchmark scenarios

    Args:
        num_clients: Number of clients
        num_rounds: Number of rounds

    Returns:
        List of scenario objects
    """
    scenarios = [
        HomogeneousDriftScenario(
            num_clients=num_clients,
            num_rounds=num_rounds,
            drift_start_round=50
        ),
        HeterogeneousDriftScenario(
            num_clients=num_clients,
            num_rounds=num_rounds,
            drift_start_round=50,
            drift_spread=20
        ),
        PartialParticipationScenario(
            num_clients=num_clients,
            num_rounds=num_rounds,
            drift_start_round=50,
            participation_rate=0.5
        ),
        StaggeredDriftScenario(
            num_clients=num_clients,
            num_rounds=num_rounds,
            drift_start_round=40,
            num_groups=3,
            stagger_interval=10,
            participation_rate=0.6
        )
    ]

    return scenarios
