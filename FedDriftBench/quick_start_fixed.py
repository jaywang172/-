"""
Quick Start Demo - FIXED VERSION with Better Parameters
Adjusts parameters for better drift detection
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


def quick_demo_fixed():
    """Run a quick demo with improved parameters for better detection"""
    print("\n" + "="*70)
    print("FEDDRIFTBENCH - QUICK START DEMO (FIXED VERSION)")
    print("="*70)
    print("\n✨ This version uses improved parameters for better drift detection.")
    print("Changes from original:")
    print("  - Stronger drift (magnitude 0.8 vs 0.3)")
    print("  - More sensitive detectors")
    print("  - More training epochs for better learning")
    print("  - Earlier drift injection for clearer signal\n")

    # Improved configuration
    config = {
        'num_clients': 5,
        'num_rounds': 40,          # Increased for more data
        'samples_per_round': 100,   # More samples per round
        'local_epochs': 3,          # More local training
        'random_seed': 42
    }

    # Set random seeds
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n" + "-"*70)
    print("Starting improved benchmark...")
    print("-"*70 + "\n")

    # Create scenario with stronger drift
    scenario = HomogeneousDriftScenario(
        num_clients=config['num_clients'],
        num_rounds=config['num_rounds'],
        drift_start_round=20,      # Earlier drift
        drift_magnitude=0.8        # Much stronger drift (was 0.5)
    )

    # Create more sensitive detector
    detector = FederatedADWIN(
        num_clients=config['num_clients'],
        delta=0.01,               # More sensitive (was 0.002, higher = more sensitive)
        global_detector=True
    )

    # Run experiment
    experiment = BenchmarkExperiment(config)

    print(f"🎯 Scenario: {scenario.get_scenario_description()}")
    print(f"🔍 Detector: ADWIN (delta=0.01, more sensitive)")
    print(f"💥 Drift: magnitude=0.8 (strong), starts at round 20\n")

    result = experiment.run_scenario_with_detector(
        scenario=scenario,
        detector_name='ADWIN-Sensitive',
        detector=detector
    )

    # Print results
    print("\n" + "="*70)
    print("IMPROVED DEMO RESULTS")
    print("="*70)

    metrics = result['detection_metrics']
    print(f"\n📊 Scenario: {result['scenario']}")
    print(f"🔍 Detector: {result['detector']}")
    print(f"\n🎯 Model Performance:")
    print(f"  Final Accuracy: {result['final_accuracy']:.4f}")
    print(f"  Final Loss: {result['final_loss']:.4f}")

    print(f"\n🔔 Detection Performance:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")
    print(f"  Avg Detection Delay: {metrics['avg_detection_delay']:.1f} rounds")
    print(f"  Total Detections: {metrics['total_detections']}")
    print(f"  True Positives: {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")

    # Analysis
    print("\n📈 Analysis:")
    if metrics['total_detections'] == 0:
        print("  ⚠️  No drift detected. Possible reasons:")
        print("     - Drift magnitude still too weak")
        print("     - Model not learning properly")
        print("     - Need more training rounds")
        print("     - Try increasing drift_magnitude to 1.0-2.0")
    elif metrics['f1_score'] > 0.7:
        print("  ✅ Excellent detection performance!")
    elif metrics['f1_score'] > 0.5:
        print("  ✓  Good detection performance")
    else:
        print("  ⚠️  Moderate detection performance")
        print("     - Consider adjusting detector sensitivity")

    # Save results
    os.makedirs('results/benchmarks', exist_ok=True)
    filename = 'results/benchmarks/quick_demo_fixed.json'

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

    print(f"\n💾 Results saved to: {filename}")

    print("\n" + "="*70)
    print("Next Steps:")
    print("  1. If detection worked: Run full benchmark with these parameters")
    print("  2. If still no detection: Try even stronger drift (magnitude=2.0)")
    print("  3. For research: Tune parameters based on your scenario")
    print("="*70 + "\n")


if __name__ == '__main__':
    quick_demo_fixed()
