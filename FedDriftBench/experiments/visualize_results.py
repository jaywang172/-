"""
Results Visualization Module
Generates plots and visualizations from benchmark results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List
import pandas as pd


class BenchmarkVisualizer:
    """Visualize benchmark results"""

    def __init__(self, results_file: str):
        """
        Initialize visualizer

        Args:
            results_file: Path to results JSON file
        """
        with open(results_file, 'r') as f:
            self.results = json.load(f)

        self.output_dir = 'results/visualizations'
        os.makedirs(self.output_dir, exist_ok=True)

        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)

    def plot_detection_metrics_comparison(self):
        """Plot comparison of detection metrics across detectors and scenarios"""
        scenarios = list(self.results['scenarios'].keys())
        detectors = list(self.results['scenarios'][scenarios[0]].keys())

        metrics = ['precision', 'recall', 'f1_score']

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, metric in enumerate(metrics):
            data = []
            for scenario in scenarios:
                for detector in detectors:
                    value = self.results['scenarios'][scenario][detector]['detection_metrics'][metric]
                    data.append({
                        'Scenario': scenario,
                        'Detector': detector,
                        'Value': value
                    })

            df = pd.DataFrame(data)

            # Create grouped bar plot
            pivot_df = df.pivot(index='Scenario', columns='Detector', values='Value')
            pivot_df.plot(kind='bar', ax=axes[idx], rot=45)
            axes[idx].set_title(f'{metric.upper()}', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('Score', fontsize=12)
            axes[idx].set_xlabel('')
            axes[idx].legend(title='Detector', fontsize=10)
            axes[idx].set_ylim([0, 1.0])
            axes[idx].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/detection_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir}/detection_metrics_comparison.png")
        plt.close()

    def plot_accuracy_over_time(self):
        """Plot accuracy over training rounds"""
        scenarios = list(self.results['scenarios'].keys())

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]

            for detector in self.results['scenarios'][scenario].keys():
                round_metrics = self.results['scenarios'][scenario][detector]['round_metrics']
                rounds = [m['round'] for m in round_metrics]
                accuracies = [m['avg_accuracy'] for m in round_metrics]

                ax.plot(rounds, accuracies, marker='o', markersize=3,
                       label=detector, linewidth=2, alpha=0.8)

            ax.set_title(f'{scenario.replace("_", " ").title()}',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.0])

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/accuracy_over_time.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir}/accuracy_over_time.png")
        plt.close()

    def plot_detection_delay_comparison(self):
        """Plot detection delay comparison"""
        scenarios = list(self.results['scenarios'].keys())
        detectors = list(self.results['scenarios'][scenarios[0]].keys())

        data = []
        for scenario in scenarios:
            for detector in detectors:
                delay = self.results['scenarios'][scenario][detector]['detection_metrics']['avg_detection_delay']
                data.append({
                    'Scenario': scenario,
                    'Detector': detector,
                    'Delay': delay
                })

        df = pd.DataFrame(data)

        plt.figure(figsize=(12, 6))
        pivot_df = df.pivot(index='Scenario', columns='Detector', values='Delay')
        pivot_df.plot(kind='bar', rot=45)

        plt.title('Average Detection Delay Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Detection Delay (rounds)', fontsize=12)
        plt.xlabel('')
        plt.legend(title='Detector', fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        plt.savefig(f'{self.output_dir}/detection_delay_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir}/detection_delay_comparison.png")
        plt.close()

    def plot_leaderboard(self):
        """Create leaderboard visualization"""
        scenarios = list(self.results['scenarios'].keys())
        detectors = list(self.results['scenarios'][scenarios[0]].keys())

        # Compute average F1 score across all scenarios
        leaderboard = []
        for detector in detectors:
            f1_scores = []
            for scenario in scenarios:
                f1 = self.results['scenarios'][scenario][detector]['detection_metrics']['f1_score']
                f1_scores.append(f1)

            avg_f1 = np.mean(f1_scores)
            leaderboard.append({
                'Detector': detector,
                'Avg F1 Score': avg_f1,
                'Scores': f1_scores
            })

        # Sort by average F1
        leaderboard.sort(key=lambda x: x['Avg F1 Score'], reverse=True)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        detectors_sorted = [item['Detector'] for item in leaderboard]
        avg_scores = [item['Avg F1 Score'] for item in leaderboard]

        bars = ax.barh(detectors_sorted, avg_scores, color=sns.color_palette("viridis", len(detectors_sorted)))

        ax.set_xlabel('Average F1 Score', fontsize=14, fontweight='bold')
        ax.set_title('Detector Leaderboard (Averaged Across All Scenarios)',
                    fontsize=16, fontweight='bold')
        ax.set_xlim([0, 1.0])

        # Add value labels
        for i, (detector, score) in enumerate(zip(detectors_sorted, avg_scores)):
            ax.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/leaderboard.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir}/leaderboard.png")
        plt.close()

        return leaderboard

    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60 + "\n")

        self.plot_detection_metrics_comparison()
        self.plot_accuracy_over_time()
        self.plot_detection_delay_comparison()
        leaderboard = self.plot_leaderboard()

        print("\n" + "="*60)
        print("LEADERBOARD")
        print("="*60)
        for rank, item in enumerate(leaderboard, 1):
            print(f"{rank}. {item['Detector']:10s} - Avg F1: {item['Avg F1 Score']:.3f}")

        print(f"\nAll visualizations saved to: {self.output_dir}/")


def main():
    """Main entry point"""
    import glob

    # Find most recent results file
    results_files = glob.glob('results/benchmarks/benchmark_*.json')
    if not results_files:
        print("No results files found. Please run the benchmark first.")
        return

    latest_file = max(results_files, key=os.path.getctime)
    print(f"Visualizing results from: {latest_file}\n")

    visualizer = BenchmarkVisualizer(latest_file)
    visualizer.generate_all_visualizations()


if __name__ == '__main__':
    main()
