"""
Improved Benchmark Runner with Better Detection Parameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from experiments.run_benchmark import BenchmarkExperiment
from detectors.baseline.adwin_detector import FederatedADWIN
from detectors.baseline.ddm_detector import FederatedDDM
from detectors.baseline.kswin_detector import FederatedKSWIN
from scenarios.standard_scenarios import get_all_scenarios


def main():
    """Main entry point with improved parameters"""
    print("\n" + "="*70)
    print("FEDDRIFTBENCH - IMPROVED BENCHMARK")
    print("="*70)
    print("\n🎯 This version uses optimized parameters for better detection:")
    print("  - Stronger drift injection (magnitude 0.8 vs 0.3)")
    print("  - More sensitive detectors")
    print("  - More training for better model performance")
    print("  - Longer rounds for clearer drift signals\n")

    # Improved configuration
    config = {
        'num_clients': 10,
        'num_rounds': 100,
        'samples_per_round': 150,   # More samples
        'local_epochs': 2,          # More local training
        'random_seed': 42
    }

    # Set random seeds
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Get scenarios with stronger drift
    scenarios = get_all_scenarios(
        num_clients=config['num_clients'],
        num_rounds=config['num_rounds']
    )

    # Adjust drift magnitude for all scenarios
    for scenario in scenarios:
        scenario.drift_magnitude = 0.8  # Stronger drift

    # Define more sensitive detectors
    detectors = {
        'ADWIN-Sensitive': FederatedADWIN(
            num_clients=config['num_clients'],
            delta=0.01,            # More sensitive (higher = more sensitive)
            global_detector=True
        ),
        'DDM-Sensitive': FederatedDDM(
            num_clients=config['num_clients'],
            warning_level=1.5,     # Lower threshold (more sensitive)
            drift_level=2.5,       # Lower threshold (more sensitive)
            global_detector=True
        ),
        'KSWIN-Sensitive': FederatedKSWIN(
            num_clients=config['num_clients'],
            alpha=0.05,            # Higher alpha (more sensitive)
            window_size=50,        # Smaller window (faster response)
            stat_size=20,          # Smaller stat window
            global_detector=True
        )
    }

    print("Detectors configured:")
    print("  - ADWIN-Sensitive: delta=0.01 (more sensitive)")
    print("  - DDM-Sensitive: warning=1.5, drift=2.5 (lower thresholds)")
    print("  - KSWIN-Sensitive: alpha=0.05, smaller windows")
    print()

    # Run benchmark
    experiment = BenchmarkExperiment(config)
    experiment.results['scenarios'] = {}

    for scenario in scenarios:
        scenario_name = scenario.scenario_type.value
        experiment.results['scenarios'][scenario_name] = {}

        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario_name.upper()}")
        print(f"{'='*70}\n")

        for detector_name, detector in detectors.items():
            # Reset detector
            detector.reset_all()

            # Run experiment
            import time
            start_time = time.time()
            result = experiment.run_scenario_with_detector(
                scenario, detector_name, detector
            )
            elapsed_time = time.time() - start_time

            result['elapsed_time'] = elapsed_time

            # Store result
            experiment.results['scenarios'][scenario_name][detector_name] = result

            # Print summary
            metrics = result['detection_metrics']
            print(f"\n✅ Completed in {elapsed_time:.2f}s")
            print(f"   F1: {metrics['f1_score']:.3f} | "
                  f"Precision: {metrics['precision']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | "
                  f"Detections: {metrics['total_detections']}")

            if metrics['f1_score'] == 0:
                print(f"   ⚠️  No drift detected - may need stronger parameters")

    # Save results
    experiment.save_results()

    # Print summary
    print("\n" + "="*70)
    print("IMPROVED BENCHMARK SUMMARY")
    print("="*70)

    for scenario_name, detectors_results in experiment.results['scenarios'].items():
        print(f"\n{scenario_name.upper()}")
        print("-" * 70)

        for detector_name, result in detectors_results.items():
            metrics = result['detection_metrics']
            status = "✅" if metrics['f1_score'] > 0.5 else "⚠️ " if metrics['f1_score'] > 0 else "❌"
            print(f"{status} {detector_name:18s} | "
                  f"F1: {metrics['f1_score']:.3f} | "
                  f"Precision: {metrics['precision']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | "
                  f"Delay: {metrics['avg_detection_delay']:.1f} rounds | "
                  f"Detections: {metrics['total_detections']}")

    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)

    # Analyze overall performance
    total_scenarios = len(experiment.results['scenarios'])
    total_detectors = len(detectors)
    working_count = 0

    for scenario_results in experiment.results['scenarios'].values():
        for result in scenario_results.values():
            if result['detection_metrics']['f1_score'] > 0:
                working_count += 1

    success_rate = working_count / (total_scenarios * total_detectors)

    print(f"\n📊 Overall Detection Success: {success_rate*100:.1f}% ({working_count}/{total_scenarios * total_detectors})")

    if success_rate == 0:
        print("\n⚠️  No drift detected across all experiments!")
        print("\nPossible solutions:")
        print("  1. Increase drift_magnitude to 1.5-2.0")
        print("  2. Check if model is learning (accuracy should improve initially)")
        print("  3. Use more training epochs (local_epochs=3-5)")
        print("  4. Increase samples_per_round to 200-300")
        print("  5. Make detectors even more sensitive")
    elif success_rate < 0.5:
        print("\n⚠️  Detection rate is low. Consider:")
        print("  - Further increase drift magnitude")
        print("  - Adjust detector sensitivity parameters")
        print("  - Check model convergence")
    else:
        print("\n✅ Good detection performance!")
        print("  - Results are ready for analysis")
        print("  - Generate visualizations with: python experiments/visualize_results.py")

    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
