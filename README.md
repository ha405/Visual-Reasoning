# Neural Network Pruning for Domain Generalization

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python main.py --data_dir /path/to/pacs_data
```

## Usage

### Basic Examples

```bash
# Default (activation + DI + ResNet18)
python main.py --data_dir /path/to/pacs

# Taylor pruning
python main.py --data_dir /path/to/pacs --pruning_method taylor --importance_type taylor_meanvar

# Different model
python main.py --data_dir /path/to/pacs --model resnet50 --batch_size 128

# Different target domain
python main.py --data_dir /path/to/pacs --target_domain photo --source_domains art_painting cartoon sketch

# Adaptive pruning
python main.py --data_dir /path/to/pacs --pruning_method adaptive --pruning_strategy target_error
```

## Available Options

### Pruning Methods
- `fixed_rate` (default)
- `taylor`
- `adaptive`

### Importance Types
- `activation` (default)
- `meanvar`
- `taylor`
- `taylor_domain`
- `taylor_meanvar`
- `magnitude`
- `error_correlated`

### Models
ResNet: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
VGG: `vgg11`, `vgg13`, `vgg16`, `vgg19`
DenseNet: `densenet121`, `densenet161`, `densenet169`, `densenet201`
MobileNet: `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`

### Training Strategies
- `DI` (default): Domain Invariance
- `SFT`: Supervised Fine-Tuning
- `vanilla`: Standard training

## Key Parameters

```bash
--data_dir PATH              # Required: dataset location
--model MODEL                # Model architecture
--importance_type TYPE       # Importance metric
--pruning_method METHOD      # Pruning strategy
--training_strategy STRAT    # Training method
--target_domain DOMAIN       # Target domain
--prune_rates R1 R2 R3      # Pruning rates per iteration
--lr FLOAT                   # Learning rate (default: 0.001)
--alpha FLOAT                # DI variance weight (default: 1.0)
--batch_size INT             # Batch size (default: 256)
```

## Output Organization

Results automatically organized by experiment:

```
outputs/
└── {dataset}_{target}_{importance}_{training}_{model}/
    ├── config.yaml
    ├── final_model.pth
    └── final_mask.pth
```

Example: `outputs/pacs_sketch_taylor_meanvar_DI_resnet50/`

## Project Structure

```
Pruning/
├── pruning/              # Main package
├── configs/
│   └── default.yaml     # Default configuration
├── legacy/              # Original implementation
├── main.py              # CLI entry point
└── README.md
```
