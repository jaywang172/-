"""
Example: Custom Experiment Configuration
Demonstrates how to create custom experiments with FedDriftBench
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from experiments.run_benchmark import BenchmarkExperiment
from scenarios.standard_scenarios import HeterogeneousDriftScenario
from detectors.baseline.adwin_detector import FederatedADWIN
from detectors.baseline.ddm_detector import FederatedDDM
from detectors.baseline.kswin_detector import FederatedKSWIN


def custom_experiment_example():
    """
    Example showing how to run a custom experiment
    with specific parameters
    """
    print("\n" + "="*70)
    print("CUSTOM EXPERIMENT EXAMPLE")
    print("="*70 + "\n")

    # Custom configuration
    config = {
        'num_clients': 15,          # More clients
        'num_rounds': 120,          # Longer experiment
        'samples_per_round': 150,   # More data
        'local_epochs': 2,          # More local training
        'random_seed': 2024
    }

    print("Custom Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Set random seed
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    # Create custom scenario
    scenario = HeterogeneousDriftScenario(
        num_clients=config['num_clients'],
        num_rounds=config['num_rounds'],
        drift_start_round=60,       # Drift at middle
        drift_spread=30,            # Wide spread
        drift_magnitude=0.6         # Moderate drift
    )

    print(f"\nScenario: {scenario.get_scenario_description()}")

    # Create multiple detectors to compare
    detectors = {
        'ADWIN-Sensitive': FederatedADWIN(
            num_clients=config['num_clients'],
            delta=0.0005,  # More sensitive (lower delta)
            global_detector=True
        ),
        'ADWIN-Standard': FederatedADWIN(
            num_clients=config['num_clients'],
            delta=0.002,   # Standard sensitivity
            global_detector=True
        ),
        'DDM': FederatedDDM(
            num_clients=config['num_clients'],
            warning_level=2.0,
            drift_level=3.0,
            global_detector=True
        ),
        'KSWIN': FederatedKSWIN(
            num_clients=config['num_clients'],
            alpha=0.005,
            window_size=100,
            stat_size=30,
            global_detector=True
        )
    }

    # Run experiments
    experiment = BenchmarkExperiment(config)
    results = {}

    for detector_name, detector in detectors.items():
        print(f"\n{'='*70}")
        print(f"Testing: {detector_name}")
        print(f"{'='*70}")

        detector.reset_all()
        result = experiment.run_scenario_with_detector(
            scenario, detector_name, detector
        )
        results[detector_name] = result

        # Print summary
        metrics = result['detection_metrics']
        print(f"\nResults for {detector_name}:")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  Detection Delay: {metrics['avg_detection_delay']:.1f} rounds")

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    comparison = []
    for name, result in results.items():
        metrics = result['detection_metrics']
        comparison.append({
            'Detector': name,
            'F1': metrics['f1_score'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Delay': metrics['avg_detection_delay']
        })

    # Sort by F1 score
    comparison.sort(key=lambda x: x['F1'], reverse=True)

    print(f"\n{'Detector':<20} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Delay':>8}")
    print("-" * 70)
    for item in comparison:
        print(f"{item['Detector']:<20} {item['F1']:>8.3f} "
              f"{item['Precision']:>8.3f} {item['Recall']:>8.3f} "
              f"{item['Delay']:>8.1f}")

    print("\n" + "="*70)
    print("Custom experiment completed!")
    print("="*70 + "\n")


def scenario_comparison_example():
    """
    Example showing how to compare different scenarios
    with the same detector
    """
    print("\n" + "="*70)
    print("SCENARIO COMPARISON EXAMPLE")
    print("="*70 + "\n")

    config = {
        'num_clients': 8,
        'num_rounds': 80,
        'samples_per_round': 100,
        'local_epochs': 1,
        'random_seed': 42
    }

    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    # Import all scenario types
    from scenarios.standard_scenarios import (
        HomogeneousDriftScenario,
        HeterogeneousDriftScenario,
        PartialParticipationScenario,
        StaggeredDriftScenario
    )

    # Create scenarios
    scenarios = [
        HomogeneousDriftScenario(
            num_clients=config['num_clients'],
            num_rounds=config['num_rounds'],
            drift_start_round=40
        ),
        HeterogeneousDriftScenario(
            num_clients=config['num_clients'],
            num_rounds=config['num_rounds'],
            drift_start_round=40,
            drift_spread=20
        ),
        PartialParticipationScenario(
            num_clients=config['num_clients'],
            num_rounds=config['num_rounds'],
            drift_start_round=40,
            participation_rate=0.5
        ),
        StaggeredDriftScenario(
            num_clients=config['num_clients'],
            num_rounds=config['num_rounds'],
            drift_start_round=30,
            num_groups=3,
            stagger_interval=10
        )
    ]

    # Use ADWIN detector for all
    detector = FederatedADWIN(
        num_clients=config['num_clients'],
        delta=0.002,
        global_detector=True
    )

    experiment = BenchmarkExperiment(config)
    results = []

    for scenario in scenarios:
        print(f"\nTesting: {scenario.get_scenario_description()}")
        detector.reset_all()

        result = experiment.run_scenario_with_detector(
            scenario, 'ADWIN', detector
        )

        metrics = result['detection_metrics']
        results.append({
            'Scenario': scenario.scenario_type.value,
            'F1': metrics['f1_score'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Delay': metrics['avg_detection_delay']
        })

    # Print comparison
    print("\n" + "="*70)
    print("SCENARIO DIFFICULTY COMPARISON (with ADWIN)")
    print("="*70)

    print(f"\n{'Scenario':<25} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Delay':>8}")
    print("-" * 70)
    for item in results:
        print(f"{item['Scenario']:<25} {item['F1']:>8.3f} "
              f"{item['Precision']:>8.3f} {item['Recall']:>8.3f} "
              f"{item['Delay']:>8.1f}")

    print("\n" + "="*70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run custom experiments')
    parser.add_argument('--mode', choices=['detector', 'scenario', 'both'],
                       default='detector',
                       help='Experiment mode')

    args = parser.parse_args()

    if args.mode == 'detector' or args.mode == 'both':
        custom_experiment_example()

    if args.mode == 'scenario' or args.mode == 'both':
        scenario_comparison_example()
