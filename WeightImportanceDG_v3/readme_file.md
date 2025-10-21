# Domain Generalization with Importance-Based Weight Masking

A PyTorch implementation of domain generalization using importance-based weight masking on the PACS dataset. This approach identifies important weights through domain-specific training and applies selective learning rates during cross-domain generalization.

## 📋 Overview

This project implements a two-phase training procedure:

**Phase 1: Domain-Specific Training**
- Train separate networks on individual domains
- Track weight changes to identify important connections
- Create importance masks (top 60% of weights marked as important)

**Phase 2: Cross-Domain Generalization**
- Train in Leave-One-Domain-Out (LODO) configuration
- Apply differential learning rates based on importance masks
- Freeze unimportant weights after epoch 7

## 🏗️ Architecture

- **Feature Extractor**: Frozen Vision Transformer (ViT-Base) for image embeddings (768-dim)
- **Classifier**: 4-layer feedforward network (768 → 10 → 10 → 10 → 10 → 7)
- **Importance Tracking**: Only the 10×10 hidden layer weights are tracked

## 📁 Project Structure

```
.
├── config.py              # Configuration and hyperparameters
├── dataset.py             # PACS dataset loading utilities
├── models.py              # ViT feature extractor and classifier
├── importance.py          # Importance calculation and masking
├── trainer_phase1.py      # Phase 1 training logic
├── trainer_phase2.py      # Phase 2 training logic
├── visualization.py       # Visualization utilities
├── train.ipynb            # Main training notebook
├── README.md              # This file
└── results/               # Output directory (created automatically)
    ├── weights/           # Saved model weights
    ├── plots/             # Visualization outputs
    ├── importance/        # Importance matrices and masks
    └── summary_report.txt # Final results summary
```

## 🔧 Installation

### Requirements

```bash
pip install torch torchvision transformers
pip install numpy matplotlib seaborn scikit-learn
pip install tqdm jupyter
```

### Dataset Setup

1. Download the PACS dataset
2. Organize it in the following structure:
```
pacs_data/
└── pacs_data/
    ├── art_painting/
    │   ├── dog/
    │   ├── elephant/
    │   └── ...
    ├── cartoon/
    ├── photo/
    └── sketch/
```
3. Update `DATA_ROOT` in `config.py` to point to your dataset location

## 🚀 Usage

### Quick Start

Simply run the Jupyter notebook:

```bash
jupyter notebook train.ipynb
```

Then execute all cells sequentially. The notebook will:
1. Run Phase 1 training on all domains
2. Compute importance masks
3. Run Phase 2 LODO training for all 4 configurations
4. Generate visualizations and reports

### Configuration

Edit `config.py` to modify:

```python
# Training parameters
PH1_NUM_EPOCHS = 10              # Phase 1 epochs
PH2_NUM_EPOCHS = 10              # Phase 2 epochs
PH1_LEARNING_RATE = 1e-3         # Phase 1 LR
PH2_LR_IMPORTANT = 1e-3          # Phase 2 LR for important weights
PH2_LR_UNIMPORTANT_START = 1e-4  # Phase 2 initial LR for unimportant weights
PH2_FREEZE_EPOCH = 7             # Epoch to freeze unimportant weights

# Importance threshold
IMPORTANCE_THRESHOLD_PERCENTILE = 60  # Top 60% marked as important

# Dataset
DATA_ROOT = "../../../pacs_data/pacs_data"
```

## 📊 Outputs

### Visualizations

For each domain in Phase 1, the following visualizations are generated:
1. **Initial Weights**: 4 grids showing 10×10 weight matrices before training
2. **Final Weights**: Weight matrices after training
3. **Continuous Importance**: Normalized importance values (0-1) for each domain
4. **Binary Masks**: Binary importance masks (0 or 1) for each domain

**Aggregated Importance Visualizations** (after Phase 1, before Phase 2):
1. **Summed Importance**: Element-wise sum of |weight changes| across all training domains
2. **Normalized Aggregated Importance**: Summed importance normalized to [0, 1]
3. **Binary Masks**: Final masks used in Phase 2 (top 60% of summed importance)

Each visualization shows 4 layers side-by-side as grids of colored circles, where:
- Circle color intensity represents the weight value or importance
- Numerical values are displayed inside each circle

### Training Curves

- Phase 1: Loss and accuracy curves for each domain
- Phase 2: Loss, training accuracy, and validation accuracy curves for each LODO configuration

### Results Summary

- Bar chart comparing test accuracies across all LODO configurations
- Text report with detailed statistics
- Saved numpy archives for further analysis

## 🧪 Experimental Results

The model is evaluated using Leave-One-Domain-Out cross-validation:

| Test Domain    | Train Domains              | Test Accuracy |
|---------------|----------------------------|---------------|
| art_painting  | cartoon, photo, sketch     | TBD           |
| cartoon       | art_painting, photo, sketch| TBD           |
| photo         | art_painting, cartoon, sketch | TBD        |
| sketch        | art_painting, cartoon, photo | TBD         |

**Average Test Accuracy**: TBD ± TBD

*Run the notebook to populate these results!*

## 🔬 Method Details

### Phase 1: Importance Discovery

1. Initialize 3 networks with identical random weights
2. Train each on a different domain for 10 epochs (using all data)
3. For each domain, compute importance: `|final_weights - initial_weights|`
4. **Element-wise sum** the importance matrices across all 3 domains for each layer
5. Normalize the summed importance globally to [0, 1] across all layers
6. Create binary masks: top 60% → important (1), rest → unimportant (0)

**Key insight**: Weights that change significantly across **multiple domains** receive higher importance scores after summation.

### Phase 2: Importance-Based Learning

1. Use aggregated masks from Phase 1 (created from summed importance)
2. Initialize a new random network
3. Apply differential learning rates:
   - Important weights (mask=1): LR = 1e-3
   - Unimportant weights (mask=0): LR = 1e-4 with exponential decay
4. After epoch 7: freeze unimportant weights completely
5. Evaluate on held-out domain

### Importance Aggregation Strategy

**Current Approach** (Element-wise Sum):
- Sum the absolute weight changes across all training domains
- Formula: `summed_importance[i,j] = Σ(|domain_k_final[i,j] - domain_k_initial[i,j]|)` for all domains k
- Weights that change consistently across multiple domains get higher scores
- Then normalize globally and threshold to get top 60%

This differs from the union approach where each domain votes independently. Here, the magnitude of change across domains matters.

### Key Features

- **Frozen ViT**: Feature extractor remains frozen, only classifier is trained
- **Selective Learning**: Important weights learn at full rate, unimportant weights are restricted
- **Progressive Freezing**: Unimportant weights are gradually frozen
- **Global Normalization**: Importance is normalized across all layers for consistency

## 📝 Notes

- The ViT model downloads automatically on first run (~300MB)
- Training Phase 1 on 3 domains takes approximately 30-60 minutes on GPU
- Phase 2 training takes approximately 20-30 minutes per LODO configuration
- Total experiment time: 2-3 hours for all 4 LODO configurations

## 🐛 Troubleshooting

**Out of Memory Error**:
- Reduce `BATCH_SIZE` in `config.py`
- Use a smaller ViT model (modify `VIT_MODEL` in config.py)

**Dataset Not Found**:
- Verify `DATA_ROOT` path in `config.py`
- Ensure PACS dataset is properly organized

**Slow Training**:
- Ensure CUDA is available: check `torch.cuda.is_available()`
- Reduce image size or batch size

## 📚 References

- **PACS Dataset**: [Domain Generalization Benchmark](https://arxiv.org/abs/1710.03077)
- **Vision Transformer**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **Importance-Based Learning**: Custom approach inspired by network pruning and transfer learning literature

## 📄 License

This project is for research and educational purposes.

## 👤 Author

Created for domain generalization research on the PACS dataset.

---

**Happy Experimenting! 🚀**
