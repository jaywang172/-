"""
Main Benchmark Experiment Runner
Integrates all components and runs complete experiments
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm

from core.federated_simulator import FederatedServer, SimpleNN
from core.drift_injector import DriftType, DriftInjector, MultiClientDriftScenario
from detectors.baseline.adwin_detector import FederatedADWIN
from detectors.baseline.ddm_detector import FederatedDDM
from detectors.baseline.kswin_detector import FederatedKSWIN
from scenarios.standard_scenarios import get_all_scenarios
from datasets.data_generator import MNISTFederated, StreamDataGenerator


class BenchmarkExperiment:
    """Main benchmark experiment orchestrator"""

    def __init__(self, config: Dict):
        """
        Initialize experiment

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.num_clients = config.get('num_clients', 10)
        self.num_rounds = config.get('num_rounds', 100)
        self.samples_per_round = config.get('samples_per_round', 100)
        self.local_epochs = config.get('local_epochs', 1)

        # Results storage
        self.results = {
            'config': config,
            'scenarios': {},
            'timestamp': datetime.now().isoformat()
        }

    def run_scenario_with_detector(self, scenario, detector_name: str,
                                   detector) -> Dict:
        """
        Run a single scenario with a specific detector

        Args:
            scenario: Benchmark scenario
            detector_name: Name of the detector
            detector: Detector instance

        Returns:
            Dict with experiment results
        """
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario.get_scenario_description()}")
        print(f"Detector: {detector_name}")
        print(f"{'='*60}\n")

        # Initialize federated learning
        model = SimpleNN(input_dim=784, hidden_dim=128, output_dim=10)
        server = FederatedServer(
            model=model,
            num_clients=self.num_clients,
            aggregation='fedavg'
        )

        # Load dataset
        dataset = MNISTFederated(
            num_clients=self.num_clients,
            data_dir='./data',
            non_iid=True,
            alpha=0.5
        )
        stream_gen = StreamDataGenerator(dataset, self.num_rounds)

        # Get drift and participation schedules
        drift_schedule = scenario.get_drift_schedule()
        participation_schedule = scenario.get_participation_schedule()

        # Initialize drift injector
        drift_injector = DriftInjector(
            drift_type=DriftType.SUDDEN,
            drift_magnitude=0.3
        )

        # Tracking metrics
        round_metrics = []
        detection_events = []
        true_drift_rounds = {client_id: info['start_round']
                            for client_id, info in drift_schedule.items()}

        # Run federated learning rounds
        for round_num in tqdm(range(self.num_rounds), desc="Training"):
            # Get participating clients for this round
            participating_clients = participation_schedule.get(
                round_num, list(range(self.num_clients))
            )

            # Prepare data for this round
            client_data = {}
            for client_id in participating_clients:
                # Check if drift should be applied
                drift_info = drift_schedule.get(client_id, {})
                drift_applied = (drift_info.get('start_round', 999) <= round_num <
                               drift_info.get('end_round', 999))

                # Generate data for this round
                X, y = stream_gen.generate_round_data(
                    round_num=round_num,
                    client_id=client_id,
                    samples_per_round=self.samples_per_round,
                    drift_applied=drift_applied,
                    drift_magnitude=drift_info.get('magnitude', 0.0) if drift_applied else 0.0
                )

                if len(X) > 0:
                    client_data[client_id] = (X, y)

            # Train round
            if client_data:
                round_result = server.train_round(
                    client_data=client_data,
                    participation_rate=1.0,
                    local_epochs=self.local_epochs
                )

                # Update drift detectors
                for client_id, metrics in round_result['client_metrics'].items():
                    error_rate = 1.0 - metrics['accuracy']

                    # Update detector based on type
                    if detector_name == 'ADWIN':
                        drift_detected = detector.update_client(
                            client_id, error_rate, round_num
                        )
                    elif detector_name == 'DDM':
                        # DDM expects binary values
                        drift_detected = detector.update_client(
                            client_id, int(error_rate > 0.5), round_num
                        )
                    elif detector_name == 'KSWIN':
                        drift_detected = detector.update_client(
                            client_id, error_rate, round_num
                        )
                    else:
                        drift_detected = False

                    if drift_detected:
                        detection_events.append({
                            'round': round_num,
                            'client_id': client_id,
                            'detector': detector_name
                        })

                # Update global detector
                global_error = 1.0 - round_result['avg_accuracy']
                if detector_name == 'ADWIN':
                    detector.update_global(global_error, round_num)
                elif detector_name == 'DDM':
                    detector.update_global(int(global_error > 0.5), round_num)
                elif detector_name == 'KSWIN':
                    detector.update_global(global_error, round_num)

                # Store metrics
                round_metrics.append({
                    'round': round_num,
                    'avg_accuracy': round_result['avg_accuracy'],
                    'avg_loss': round_result['avg_loss'],
                    'participating_clients': participating_clients,
                    'drifting_clients': [cid for cid in participating_clients
                                        if drift_schedule.get(cid, {}).get('start_round', 999) <= round_num]
                })

        # Evaluate final model
        test_X, test_y = dataset.get_test_data()
        final_eval = server.evaluate_global_model((test_X, test_y))

        # Compute detection metrics
        detection_metrics = self._compute_detection_metrics(
            detection_events=detection_events,
            true_drift_rounds=true_drift_rounds,
            num_rounds=self.num_rounds
        )

        # Get detector info
        detector_info = detector.get_all_drift_info()

        result = {
            'scenario': scenario.get_scenario_description(),
            'detector': detector_name,
            'final_accuracy': final_eval['accuracy'],
            'final_loss': final_eval['loss'],
            'detection_metrics': detection_metrics,
            'detector_info': detector_info,
            'round_metrics': round_metrics,
            'detection_events': detection_events
        }

        return result

    def _compute_detection_metrics(self, detection_events: List[Dict],
                                   true_drift_rounds: Dict[int, int],
                                   num_rounds: int,
                                   tolerance: int = 5) -> Dict:
        """
        Compute detection performance metrics

        Args:
            detection_events: List of detection events
            true_drift_rounds: True drift start rounds for each client
            num_rounds: Total number of rounds
            tolerance: Tolerance window for detection

        Returns:
            Dict with precision, recall, F1, detection delay
        """
        # Create detection map
        detected_clients = set()
        detection_delays = []

        for event in detection_events:
            client_id = event['client_id']
            detected_round = event['round']
            true_round = true_drift_rounds.get(client_id, num_rounds + 100)

            # Check if detection is within tolerance
            if abs(detected_round - true_round) <= tolerance:
                detected_clients.add(client_id)
                delay = detected_round - true_round
                if delay >= 0:  # Only count positive delays
                    detection_delays.append(delay)

        # Compute metrics
        true_positives = len(detected_clients)
        false_positives = len(detection_events) - true_positives
        false_negatives = len(true_drift_rounds) - true_positives

        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, len(true_drift_rounds))
        f1 = 2 * (precision * recall) / max(0.0001, precision + recall)

        avg_delay = np.mean(detection_delays) if detection_delays else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'avg_detection_delay': avg_delay,
            'total_detections': len(detection_events)
        }

    def run_full_benchmark(self):
        """Run full benchmark across all scenarios and detectors"""
        print("\n" + "="*70)
        print("FEDDRIFTBENCH - Automated Benchmark Platform")
        print("="*70)

        # Get all scenarios
        scenarios = get_all_scenarios(
            num_clients=self.num_clients,
            num_rounds=self.num_rounds
        )

        # Define detectors
        detectors = {
            'ADWIN': FederatedADWIN(
                num_clients=self.num_clients,
                delta=0.002,
                global_detector=True
            ),
            'DDM': FederatedDDM(
                num_clients=self.num_clients,
                warning_level=2.0,
                drift_level=3.0,
                global_detector=True
            ),
            'KSWIN': FederatedKSWIN(
                num_clients=self.num_clients,
                alpha=0.005,
                window_size=100,
                stat_size=30,
                global_detector=True
            )
        }

        # Run experiments
        for scenario in scenarios:
            scenario_name = scenario.scenario_type.value
            self.results['scenarios'][scenario_name] = {}

            for detector_name, detector in detectors.items():
                # Reset detector
                detector.reset_all()

                # Run experiment
                start_time = time.time()
                result = self.run_scenario_with_detector(
                    scenario, detector_name, detector
                )
                elapsed_time = time.time() - start_time

                result['elapsed_time'] = elapsed_time

                # Store result
                self.results['scenarios'][scenario_name][detector_name] = result

                print(f"\nCompleted in {elapsed_time:.2f}s")
                print(f"Detection Metrics: {result['detection_metrics']}")

        # Save results
        self.save_results()

        # Print summary
        self.print_summary()

    def save_results(self):
        """Save experiment results to JSON"""
        os.makedirs('results/benchmarks', exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/benchmarks/benchmark_{timestamp}.json'

        # Convert any non-serializable objects
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

        # Deep copy and convert
        import copy
        results_copy = copy.deepcopy(self.results)

        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2, default=convert)

        print(f"\nResults saved to: {filename}")

    def print_summary(self):
        """Print experiment summary"""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)

        for scenario_name, detectors in self.results['scenarios'].items():
            print(f"\n{scenario_name.upper()}")
            print("-" * 70)

            for detector_name, result in detectors.items():
                metrics = result['detection_metrics']
                print(f"{detector_name:10s} | "
                      f"F1: {metrics['f1_score']:.3f} | "
                      f"Precision: {metrics['precision']:.3f} | "
                      f"Recall: {metrics['recall']:.3f} | "
                      f"Delay: {metrics['avg_detection_delay']:.1f} rounds")


def main():
    """Main entry point"""
    # Configuration
    config = {
        'num_clients': 10,
        'num_rounds': 80,
        'samples_per_round': 100,
        'local_epochs': 1,
        'random_seed': 42
    }

    # Set random seeds
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    # Run benchmark
    experiment = BenchmarkExperiment(config)
    experiment.run_full_benchmark()


if __name__ == '__main__':
    main()
