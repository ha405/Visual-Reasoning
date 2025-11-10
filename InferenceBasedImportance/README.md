# PACS Domain Generalization with Neuron Pruning

This repository implements a complete Leave-One-Domain-Out (LODO) experiment for domain generalization on the PACS dataset using Integrated Gradients-based neuron pruning.

## Project Overview

The implementation follows a three-phase approach:
1. **Phase 1**: Train an ERM (Empirical Risk Minimization) baseline model
2. **Phase 2A**: Random pruning baseline - prune neurons randomly and fine-tune
3. **Phase 2B**: IG-based pruning - use Integrated Gradients to identify important neurons, prune strategically, and fine-tune

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download and prepare the PACS dataset:
   - Download PACS from [official source](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017.html)
   - Extract to a directory with structure:
     ```
     pacs/
     ├── art_painting/
     │   ├── dog/
     │   ├── elephant/
     │   └── ...
     ├── cartoon/
     ├── photo/
     └── sketch/
     ```

3. Update the data path in `configs/pacs_config.py`:
```python
DATA_ROOT = "/path/to/your/pacs/dataset"
```

## Usage

### Quick Test
To verify the implementation is working correctly:
```bash
# Run unit tests
python testing/test_pruning.py

# Run a single experiment (sketch as target, seed 42)
python run_experiment.py sketch 42
```

### Full Experiment
To run the complete LODO experiment across all domains and seeds:
```bash
python main.py
```

This will:
- Train and evaluate models for each domain as target
- Run experiments with multiple random seeds (default: [42, 123, 456])
- Compare ERM baseline, random pruning, and IG-based pruning
- Save results and statistics to `artifacts/`

### Configuration

Edit `configs/pacs_config.py` to modify:
- **Training parameters**: epochs, learning rates, batch sizes
- **Pruning parameters**: target proportion, selection method, minimum neurons per layer
- **IG parameters**: number of samples, integration steps
- **System parameters**: device, workers, mixed precision

Available selection methods:
- `"greedy_iterative"`: Maximize coverage across domains iteratively
- `"union_topk"`: Union of top-k neurons from each domain
- `"weighted_avg"`: Select based on average importance
- `"random"`: Random selection baseline

## Project Structure

```
pacs_pruning/
├── artifacts/              # Experiment outputs
│   ├── baselines/         # Cached IG baselines
│   ├── class_map.json    # Class name mappings
│   └── results/          # Per-experiment results
├── configs/              # Configuration files
├── data_handling/        # Dataset and dataloader code
├── models/               # Network architecture and pruning
├── analysis/             # IG computation and neuron selection
├── testing/              # Unit tests
├── utils/                # Helper utilities
├── engine.py             # Training/evaluation loops
├── run_experiment.py     # Single LODO experiment
└── main.py              # Full experiment runner
```

## Expected Results

The experiment will produce:
1. **Per-domain results**: Accuracy for each method when domain is held out
2. **Overall statistics**: Mean and std across all domains
3. **Method comparison**: IG vs ERM, IG vs Random pruning

Results are saved to:
- `artifacts/final_results.csv`: Detailed results for all experiments
- `artifacts/final_statistics.csv`: Aggregated statistics
- `artifacts/results/{domain}/seed_{seed}/`: Individual run artifacts

## Implementation Details

### Integrated Gradients
- Computes importance scores by integrating gradients along path from baseline to input
- Uses Riemann sum approximation with configurable integration steps
- Normalizes scores per layer using L2 norm

### Neuron Selection
- **Greedy Iterative**: Maximizes coverage across source domains
- Weights importance by inverse coverage to prioritize underrepresented domains
- Enforces minimum neurons per layer to maintain model capacity

### Model Pruning
- Rebuilds model with reduced architecture
- Carefully copies weights maintaining alignment
- Validates correctness through masking equivalence test

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size or disable mixed precision
2. **Dataset not found**: Check DATA_ROOT path in config
3. **Poor performance**: Increase training epochs or adjust learning rates
4. **Pruning too aggressive**: Increase MIN_KEPT_PER_LAYER

## Citation

If you use this code, please cite the original PACS dataset paper:
```
@inproceedings{li2017deeper,
  title={Deeper, broader and artier domain generalization},
  author={Li, Da and Yang, Yongxin and Song, Yi-Zhe and Hospedales, Timothy M},
  booktitle={ICCV},
  year={2017}
}
```
