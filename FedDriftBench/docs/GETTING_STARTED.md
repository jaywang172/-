# Getting Started with FedDriftBench

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- At least 4GB RAM
- CUDA-capable GPU (optional, but recommended for faster training)

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/FedDriftBench.git
cd FedDriftBench
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; import river; print('Installation successful!')"
```

## Running Your First Experiment

### Quick Demo (5 minutes)

The quick demo runs a minimal benchmark to verify everything works:

```bash
python quick_start.py
```

This will:
- Create 5 federated clients
- Train for 30 rounds
- Inject drift at round 15
- Test ADWIN detector
- Display results

Expected output:
```
FEDDRIFTBENCH - QUICK START DEMO
================================
...
Detection Metrics:
  Precision: 0.800
  Recall: 0.800
  F1 Score: 0.800
  Avg Detection Delay: 2.0 rounds
```

### Full Benchmark (1-2 hours)

Run the complete benchmark across all scenarios and detectors:

```bash
python experiments/run_benchmark.py
```

This will:
- Test 4 benchmark scenarios
- Evaluate 3 drift detectors
- Generate comprehensive results
- Save to `results/benchmarks/`

### Visualize Results

After running experiments, generate visualizations:

```bash
python experiments/visualize_results.py
```

This creates:
- Detection metrics comparison plots
- Accuracy over time graphs
- Detection delay analysis
- Overall leaderboard

Find plots in `results/visualizations/`

## Understanding the Results

### Results JSON Structure

```json
{
  "config": { ... },
  "scenarios": {
    "homogeneous": {
      "ADWIN": {
        "detection_metrics": {
          "precision": 0.90,
          "recall": 0.85,
          "f1_score": 0.87,
          "avg_detection_delay": 3.2,
          "true_positives": 9,
          "false_positives": 1,
          "false_negatives": 1
        },
        "final_accuracy": 0.92,
        "final_loss": 0.25
      }
    }
  }
}
```

### Key Metrics Explained

- **Precision**: What fraction of drift detections were correct?
  - High precision = few false alarms

- **Recall**: What fraction of true drifts were detected?
  - High recall = catches most drifts

- **F1 Score**: Harmonic mean of precision and recall
  - Balanced metric (0-1, higher is better)

- **Detection Delay**: How many rounds after drift before detection?
  - Lower is better (faster detection)

## Customizing Experiments

### Modify Configuration

Edit `experiments/run_benchmark.py`:

```python
config = {
    'num_clients': 20,          # Increase clients
    'num_rounds': 150,          # More rounds
    'samples_per_round': 200,   # More data per round
    'local_epochs': 2,          # More local training
    'random_seed': 42           # Reproducibility
}
```

### Test Single Scenario

```python
from scenarios.standard_scenarios import HomogeneousDriftScenario
from detectors.baseline.adwin_detector import FederatedADWIN
from experiments.run_benchmark import BenchmarkExperiment

scenario = HomogeneousDriftScenario(
    num_clients=10,
    num_rounds=100,
    drift_start_round=50
)

detector = FederatedADWIN(num_clients=10, delta=0.002)

experiment = BenchmarkExperiment(config)
result = experiment.run_scenario_with_detector(scenario, 'ADWIN', detector)
```

### Adjust Drift Parameters

```python
scenario = HeterogeneousDriftScenario(
    num_clients=10,
    num_rounds=100,
    drift_start_round=50,
    drift_spread=30,           # Wider time spread
    drift_magnitude=0.7        # Stronger drift
)
```

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
- Reduce `num_clients` or `samples_per_round`
- Use CPU: `torch.device('cpu')`

**2. River module not found**
```bash
pip install river --upgrade
```

**3. MNIST download fails**
- Check internet connection
- Manual download: Place in `./data/MNIST/raw/`

**4. Slow performance**
- Reduce experiment size for testing
- Use GPU if available
- Close other applications

### Getting Help

- Check [FAQ](FAQ.md)
- Open GitHub issue
- Email: your.email@university.edu

## Next Steps

- Read [API Reference](api_reference.md)
- Learn about [Custom Detectors](custom_detectors.md)
- Explore [Scenarios Design](scenarios.md)
- Check example notebooks in `examples/`

## Tips for Research

1. **Start small**: Use quick_start.py to test ideas
2. **Version control**: Save configs and results
3. **Multiple runs**: Average over 3-5 random seeds
4. **Baseline comparison**: Always compare to existing methods
5. **Document changes**: Keep notes on modifications

Happy benchmarking! 🚀
