# FedDriftBench: Automated Benchmark Platform for Concept Drift Detection in Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**FedDriftBench** is an automated benchmark platform for evaluating concept drift detection methods in federated learning environments with heterogeneous client participation.

## 🎯 Overview

Federated learning systems face the challenge of concept drift - changes in data distributions over time at different clients. This platform provides:

- ✅ **Standardized benchmark scenarios** covering diverse drift patterns
- ✅ **Automated experiment pipeline** with reproducible results
- ✅ **Multiple drift detection methods** implemented and compared
- ✅ **Comprehensive evaluation metrics** (precision, recall, F1, detection delay)
- ✅ **Automated visualization** of results
- ✅ **GitHub Actions integration** for continuous benchmarking

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FedDriftBench.git
cd FedDriftBench

# Install dependencies
pip install -r requirements.txt
```

### Run Quick Demo (5 minutes)

```bash
python quick_start.py
```

This runs a minimal benchmark with reduced parameters for quick testing.

### Run Full Benchmark

```bash
python experiments/run_benchmark.py
```

This runs the complete benchmark across all scenarios and detectors (~1-2 hours on CPU).

### Visualize Results

```bash
python experiments/visualize_results.py
```

Generates plots and leaderboard from the latest benchmark results.

## 📊 Benchmark Scenarios

FedDriftBench implements **4 standardized scenarios**:

### Scenario A: Homogeneous Drift
- All clients experience drift simultaneously
- Tests detector response to synchronized changes
- Baseline scenario for comparison

### Scenario B: Heterogeneous Drift
- Different clients drift at different times
- Simulates real-world asynchronous drift
- Tests detector's ability to identify individual client drift

### Scenario C: Partial Client Participation
- Only subset of clients participate each round
- Common in real federated learning deployments
- Tests robustness to missing data

### Scenario D: Staggered Drift with Dynamic Participation
- Clients drift in groups at staggered intervals
- Variable participation rates
- Most challenging and realistic scenario

## 🔬 Implemented Detectors

Currently implemented drift detection methods:

1. **ADWIN** (Adaptive Windowing)
   - Statistical change detection
   - Adaptive sliding window
   - Good for gradual drifts

2. **DDM** (Drift Detection Method)
   - Monitors error rate and standard deviation
   - Warning and drift levels
   - Fast detection of sudden drifts

3. **KSWIN** (Kolmogorov-Smirnov Windowing)
   - Statistical hypothesis testing
   - Compares recent vs. reference windows
   - Robust to noise

### Adding Your Own Detector

Create a new detector in `detectors/`:

```python
# detectors/your_detector.py
class YourDetector:
    def __init__(self, num_clients, **params):
        # Initialize your detector
        pass

    def update_client(self, client_id, metric, timestamp):
        # Update with client metric
        # Return True if drift detected
        pass

    def get_all_drift_info(self):
        # Return detection statistics
        pass
```

Add it to the benchmark in `experiments/run_benchmark.py`.

## 📈 Evaluation Metrics

FedDriftBench evaluates detectors using:

- **Precision**: Ratio of correct drift detections to total detections
- **Recall**: Ratio of detected drifts to total true drifts
- **F1 Score**: Harmonic mean of precision and recall
- **Detection Delay**: Average rounds between true drift and detection
- **False Positive Rate**: Incorrect drift detections
- **Final Model Accuracy**: End-to-end learning performance

## 📁 Project Structure

```
FedDriftBench/
├── core/
│   ├── federated_simulator.py    # FL simulation engine
│   └── drift_injector.py         # Drift injection logic
├── detectors/
│   └── baseline/
│       ├── adwin_detector.py     # ADWIN implementation
│       ├── ddm_detector.py       # DDM implementation
│       └── kswin_detector.py     # KSWIN implementation
├── scenarios/
│   └── standard_scenarios.py     # 4 benchmark scenarios
├── datasets/
│   └── data_generator.py         # Data loading and streaming
├── experiments/
│   ├── run_benchmark.py          # Main experiment runner
│   └── visualize_results.py      # Results visualization
├── results/
│   ├── benchmarks/               # JSON results
│   └── visualizations/           # Generated plots
├── .github/workflows/
│   └── weekly_benchmark.yml      # Automated CI/CD
├── requirements.txt
├── quick_start.py                # Quick demo
└── README.md
```

## 🤖 Automated Benchmarking

FedDriftBench includes GitHub Actions workflow for automated weekly benchmarking:

- Runs every Sunday at 00:00 UTC
- Generates fresh results and visualizations
- Commits results back to repository
- Can be triggered manually via GitHub UI

Enable by pushing to GitHub with Actions enabled.

## 📊 Example Results

After running the benchmark, you'll find:

**Results JSON** (`results/benchmarks/`):
```json
{
  "scenarios": {
    "homogeneous": {
      "ADWIN": {
        "detection_metrics": {
          "precision": 0.95,
          "recall": 0.88,
          "f1_score": 0.91,
          "avg_detection_delay": 3.2
        }
      }
    }
  }
}
```

**Visualizations** (`results/visualizations/`):
- `detection_metrics_comparison.png` - Bar charts comparing all metrics
- `accuracy_over_time.png` - Learning curves for each scenario
- `detection_delay_comparison.png` - Detection delay analysis
- `leaderboard.png` - Overall detector ranking

## 🔧 Configuration

Customize experiments by modifying config in `run_benchmark.py`:

```python
config = {
    'num_clients': 10,          # Number of federated clients
    'num_rounds': 100,          # Training rounds
    'samples_per_round': 100,   # Samples per client per round
    'local_epochs': 1,          # Local training epochs
    'random_seed': 42           # Reproducibility
}
```

## 📝 Citation

If you use FedDriftBench in your research, please cite:

```bibtex
@article{feddriftbench2024,
  title={FedDriftBench: An Automated Benchmark Platform for Evaluating Concept Drift Detection in Federated Learning},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for contribution:

- **New detectors**: Implement additional drift detection methods
- **New scenarios**: Add more realistic drift patterns
- **Real datasets**: Integrate real-world federated datasets
- **Performance optimization**: Improve computational efficiency
- **Documentation**: Enhance docs and tutorials

Please open an issue or pull request on GitHub.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Work

This benchmark addresses research gaps identified in:

- **Concept Drift in Federated Learning**: Limited standardized evaluation
- **Heterogeneous Client Participation**: Real-world complexity not well studied
- **Reproducibility**: Need for open benchmarks in FL + drift detection

Key references:
1. Lu et al. (2024) - "Concept Drift in Federated Learning: A Survey"
2. Zhang et al. (2023) - "Drift Detection in Distributed Learning"
3. Gama et al. (2014) - "A Survey on Concept Drift Adaptation"

## 💡 Research Applications

FedDriftBench enables research on:

- **IoT and Edge Computing**: Device networks with changing environments
- **Healthcare**: Medical device monitoring with patient diversity
- **Smart Cities**: Traffic systems with temporal patterns
- **Finance**: Fraud detection with evolving behaviors

## 🎓 Tutorial & Documentation

Comprehensive documentation available at:
- [Getting Started Guide](docs/getting_started.md)
- [API Reference](docs/api_reference.md)
- [Adding Custom Detectors](docs/custom_detectors.md)
- [Scenarios Design](docs/scenarios.md)

## 📧 Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Email: your.email@university.edu
- Twitter: @yourusername

## 🌟 Acknowledgments

Built with:
- **PyTorch** for deep learning
- **River** for online learning and drift detection
- **scikit-learn** for machine learning utilities
- **matplotlib/seaborn** for visualization

---

**Star ⭐ this repository if you find it useful!**

**Happy Benchmarking! 🚀**
