"""
Quick Start Demo
Runs a minimal benchmark for quick testing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import json
from datetime import datetime

from core.federated_simulator import FederatedServer, SimpleNN
from core.drift_injector import DriftType
from detectors.baseline.adwin_detector import FederatedADWIN
from scenarios.standard_scenarios import HomogeneousDriftScenario
from datasets.data_generator import MNISTFederated, StreamDataGenerator
from experiments.run_benchmark import BenchmarkExperiment


def quick_demo():
    """Run a quick 5-minute demo"""
    print("\n" + "="*70)
    print("FEDDRIFTBENCH - QUICK START DEMO")
    print("="*70)
    print("\nThis is a minimal demo with reduced parameters for quick testing.")
    print("For full benchmark, run: python experiments/run_benchmark.py\n")

    # Minimal configuration for quick demo
    config = {
        'num_clients': 5,          # Reduced from 10
        'num_rounds': 30,          # Reduced from 100
        'samples_per_round': 50,   # Reduced from 100
        'local_epochs': 1,
        'random_seed': 42
    }

    # Set random seeds
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n" + "-"*70)
    print("Starting quick benchmark...")
    print("-"*70 + "\n")

    # Create simple scenario
    scenario = HomogeneousDriftScenario(
        num_clients=config['num_clients'],
        num_rounds=config['num_rounds'],
        drift_start_round=15  # Drift at half-way point
    )

    # Create detector
    detector = FederatedADWIN(
        num_clients=config['num_clients'],
        delta=0.002,
        global_detector=True
    )

    # Run experiment
    experiment = BenchmarkExperiment(config)
    result = experiment.run_scenario_with_detector(
        scenario=scenario,
        detector_name='ADWIN',
        detector=detector
    )

    # Print results
    print("\n" + "="*70)
    print("QUICK DEMO RESULTS")
    print("="*70)

    metrics = result['detection_metrics']
    print(f"\nScenario: {result['scenario']}")
    print(f"Detector: {result['detector']}")
    print(f"\nFinal Accuracy: {result['final_accuracy']:.4f}")
    print(f"Final Loss: {result['final_loss']:.4f}")
    print(f"\nDetection Metrics:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")
    print(f"  Avg Detection Delay: {metrics['avg_detection_delay']:.1f} rounds")
    print(f"  Total Detections: {metrics['total_detections']}")

    # Save quick demo results
    os.makedirs('results/benchmarks', exist_ok=True)
    filename = 'results/benchmarks/quick_demo_result.json'

    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj

    with open(filename, 'w') as f:
        json.dump(result, f, indent=2, default=convert)

    print(f"\nResults saved to: {filename}")
    print("\n" + "="*70)
    print("Demo complete! For full benchmark, run:")
    print("  python experiments/run_benchmark.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    quick_demo()
