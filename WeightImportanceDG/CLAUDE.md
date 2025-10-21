# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**WeightImportanceDG** is a domain generalization research project implementing a two-phase training approach for visual reasoning on the PACS dataset (Photo, Art Painting, Cartoon, Sketch domains). The core idea is to identify "important" weights in a feed-forward network by training on individual classes from multiple domains, then use these importance masks to guide training on Leave-One-Domain-Out (LODO) generalization tasks.

The architecture consists of:
- **Frozen ViT Feature Extractor**: Pre-trained `google/vit-base-patch16-224-in21k` that generates 768-dim embeddings
- **Trainable FFN Head**: 4-layer feed-forward network (768 → 10 → 10 → 10 → 10 → 7 classes)

## Two-Phase Training Pipeline

### Phase 1: Weight Importance Discovery
1. Initialize a shared FFN head with random weights (saved as "initial")
2. For each domain in `PH1_DOMAINS` (art_painting, cartoon, photo):
   - Randomly select ONE class from that domain
   - Train a copy of the FFN on that single class for `PH1_EPOCHS`
   - Save final weights for that domain
3. Compute importance masks per domain: `|w_final - w_initial|`
4. Create consensus masks across domains using element-wise minimum
5. Binarize masks using `IMPORTANCE_THRESHOLD` (0.5)

**Key insight**: Weights that change significantly when training on single classes across multiple domains are considered "important" for domain-general features.

### Phase 2: LODO Training with Gradient Masking
1. For each domain as test domain (4 LODO splits):
   - Train on remaining 3 domains
   - Apply gradient masking during training:
     - Important weights (mask=1): normal learning rate
     - Non-important weights (mask=0): scaled gradients that decrease linearly to 0 by epoch `PH2_FREEZE_EPOCH`
2. Evaluate on held-out test domain
3. Report average LODO accuracy

The gradient masking formula in `train_phase2.py:135`:
```python
param.grad *= (imp_mask + nonimp_mask * nonimp_scale)
```
where `nonimp_scale = max(0.0, 1.0 - (epoch / PH2_FREEZE_EPOCH))`

## Key Configuration

All hyperparameters are centralized in `src/config.py`:
- Dataset: PACS at `DATA_ROOT` (needs manual path adjustment)
- Phase 1: 10 epochs, batch size 32, LR 1e-3
- Phase 2: 10 epochs, batch size 32, LR 1e-3 (important), gradual freeze by epoch 7
- Importance threshold: 0.5 (for binarization)

## Running Experiments

The primary workflow is through `notebooks/run_experiments.ipynb`:

```python
# 1. Phase 1: Discover importance
from src.train_phase1 import run_phase1
run_phase1()

# 2. Compute importance and consensus masks
from src.phase1_utils import compute_and_save_importance, create_consensus_mask, get_param_names
from src.models import FeedForwardHead
ffn_temp = FeedForwardHead(config.EMBEDDING_DIM)
param_names = get_param_names(ffn_temp)

for domain in config.PH1_DOMAINS:
    for name in param_names:
        compute_and_save_importance(domain, name)

for name in param_names:
    create_consensus_mask(name, domains=config.PH1_DOMAINS)

# 3. Phase 2: LODO training
from src.train_phase2 import run_phase2_lodo
lodo_results = {}
for test_domain in config.DOMAINS:
    train_domains = [d for d in config.DOMAINS if d != test_domain]
    test_accuracy = run_phase2_lodo(train_domains, test_domain)
    lodo_results[test_domain] = test_accuracy
```

## Architecture Details

### Module Organization
- `src/config.py`: All configuration and hyperparameters
- `src/models.py`: ViTFeatureExtractor (frozen) and FeedForwardHead (trainable)
- `src/dataset.py`: PACSDataset wrapper with methods for single-class and LODO dataloaders
- `src/train_phase1.py`: Phase 1 training loop (per-domain, single-class)
- `src/phase1_utils.py`: Weight saving, importance computation, consensus mask creation
- `src/train_phase2.py`: Phase 2 LODO training with gradient masking
- `src/visualization.py`: Plotting utilities for weight matrices and network graphs

### Data Flow
1. Images → ViT (frozen) → 768-dim embeddings → FFN head → logits → CrossEntropyLoss
2. Phase 1: embeddings extracted once, FFN trained on single class
3. Phase 2: embeddings extracted, FFN trained with masked gradients on all classes

### Results Structure
```
results/
├── phase1_weights/
│   ├── initial/           # Shared initial FFN weights (.npy)
│   └── final/
│       ├── domain_art_painting/
│       ├── domain_cartoon/
│       └── domain_photo/
├── phase1_importances/
│   ├── domain_art_painting/  # Per-domain importance (_norm.npy, _binary.npy)
│   ├── domain_cartoon/
│   ├── domain_photo/
│   └── consensus/            # Final masks used in Phase 2
├── phase2_results/
│   ├── lodo_art_painting/   # model_final.pt, metrics.json
│   ├── lodo_cartoon/
│   ├── lodo_photo/
│   └── lodo_sketch/
└── plots/                   # Visualizations
```

## Important Implementation Notes

1. **Parameter names**: Only weights and biases from `layers.{i}.weight`, `layers.{i}.bias`, `classifier.weight`, `classifier.bias` are tracked. BatchNorm parameters are intentionally excluded from masking.

2. **Consensus strategy**: Uses `np.min()` across domains (strict consensus) - a weight is only important if it's important in ALL Phase 1 domains.

3. **Gradient masking fallback**: If a mask doesn't exist for a parameter (e.g., BatchNorm), gradients are left unmasked (see `train_phase2.py:125`).

4. **Random seed**: Set to 42 for reproducibility across all random operations (class selection, data splits, initialization).

5. **Data root**: Hardcoded path in `config.py:13` needs adjustment per environment.

## Common Workflows

**To modify the importance threshold:**
Edit `config.py:34` and re-run importance computation cells in the notebook.

**To change Phase 2 freezing schedule:**
Adjust `PH2_FREEZE_EPOCH` in `config.py:41` or modify the `nonimp_scale` calculation in `train_phase2.py:104`.

**To train on different Phase 1 domains:**
Modify `PH1_DOMAINS` in `config.py:28` and re-run Phase 1.

**To visualize importance masks:**
Use `src/visualization.py` functions `plot_matrix()` or `create_graph_from_matrices()` as shown in the notebook.
