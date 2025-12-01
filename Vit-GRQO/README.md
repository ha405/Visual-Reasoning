# GRQO Visual Reasoning

Domain generalization with Gradient-based Query Optimization (GRQO) for visual reasoning tasks.

## Quick Start

### GRQO Training
```bash
python train_grqo.py --model vit-tiny --dataset PACS --epochs 5
python train_grqo.py --model resnet18 --dataset VLCS --epochs 10
```

### Baseline Training
```bash
python train_baseline.py --model vit-tiny --dataset PACS --epochs 5
python train_baseline.py --model resnet50 --dataset OfficeHome --epochs 10
```

## Supported Models

- **ViT**: `vit-tiny`, `vit-small`
- **ResNet**: `resnet18`, `resnet34`, `resnet50`

## Supported Datasets

- **PACS**: 4 domains (art_painting, cartoon, photo, sketch), 7 classes
- **VLCS**: 4 domains (VOC2007, LabelMe, Caltech101, SUN09), 5 classes
- **OfficeHome**: 4 domains (Art, Clipart, Product, Real World), 65 classes
- **RMNIST**: 6 rotation angles, 10 classes
- **CMNIST**: 3 color domains (red, green, blue), 10 classes
- **TerraIncognita**: Multiple location domains, 10 classes

## Configuration

Edit `config.yaml` to modify:
- Training hyperparameters (batch size, learning rate, epochs)
- GRQO-specific parameters (alpha, beta, temperature, etc.)
- Dataset paths
- Model configurations

## Command Line Arguments

```bash
--model          Model architecture (required)
--dataset        Dataset name (required)
--epochs         Number of training epochs (optional, default from config)
--batch-size     Batch size (optional, default from config)
--lr             Learning rate (optional, default from config)
--config         Path to config file (default: config.yaml)
--output-dir     Output directory for results (auto-generated if not specified)
```

## Output Structure

Results are saved to `./results/MODEL_DATASET_TYPE_TIMESTAMP/`:
```
results/
└── vit-tiny_PACS_grqo_20251201_214617/
    ├── config.yaml                  # Training configuration
    ├── lodo_summary_TIMESTAMP.json  # LODO results
    └── best_DOMAIN.ckpt             # Best checkpoint per domain
```

## Examples

### Train ViT-Tiny with GRQO on PACS
```bash
python train_grqo.py --model vit-tiny --dataset PACS --epochs 5
```

### Train ResNet50 Baseline on OfficeHome
```bash
python train_baseline.py --model resnet50 --dataset OfficeHome --epochs 10 --batch-size 64
```

### Custom Learning Rate
```bash
python train_grqo.py --model vit-small --dataset VLCS --epochs 15 --lr 5e-5
```

## Dataset Setup

Update dataset paths in `config.yaml`:

```yaml
datasets:
  PACS:
    root: "path/to/pacs_data"
  VLCS:
    root: "path/to/vlcs"
```

## Architecture

- **models.py**: Unified model registry for all architectures
- **grqo.py**: Core GRQO implementation
- **decoder.py**: Visual query decoder
- **dataset.py**: Dataset loaders
- **train.py**: Training and evaluation functions
- **runner.py**: LODO training orchestration (legacy)  
- **transform.py**: Data augmentation and preprocessing
- **config.yaml**: Centralized configuration

## Results Format

JSON output includes:
```json
{
  "model": "vit-tiny",
  "dataset": "PACS",
  "lodo_results": {
    "sketch": 0.6501,
    "photo": 0.9671,
    "art_painting": 0.8756,
    "cartoon": 0.9318
  },
  "timestamp": "20251201_214617"
}
```

## Requirements

- PyTorch >= 1.12
- torchvision
- transformers
- PyYAML
- numpy
