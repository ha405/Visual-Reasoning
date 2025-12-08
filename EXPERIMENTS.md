# Reproducing Experiments

## Original Notebook Experiments

### Activation + DI
```bash
python main.py --data_dir /path/to/pacs
```
Output: `outputs/pacs_sketch_activation_DI_resnet18/`

### Mean-Variance
```bash
python main.py --data_dir /path/to/pacs --importance_type meanvar
```
Output: `outputs/pacs_sketch_meanvar_DI_resnet18/`

### Taylor
```bash
python main.py --data_dir /path/to/pacs --pruning_method taylor --importance_type taylor
```
Output: `outputs/pacs_sketch_taylor_DI_resnet18/`

### Taylor Mean-Variance
```bash
python main.py --data_dir /path/to/pacs --pruning_method taylor --importance_type taylor_meanvar
```
Output: `outputs/pacs_sketch_taylor_meanvar_DI_resnet18/`

### Adaptive Pruning
```bash
python main.py --data_dir /path/to/pacs --pruning_method adaptive --pruning_strategy target_error
```
Output: `outputs/pacs_sketch_error_correlated_DI_resnet18/`

## Different Target Domains

### Photo as Target
```bash
python main.py --data_dir /path/to/pacs --target_domain photo --source_domains art_painting cartoon sketch
```

### Cartoon as Target
```bash
python main.py --data_dir /path/to/pacs --target_domain cartoon --source_domains art_painting photo sketch
```

## Common Parameters

```bash
--importance_type activation | meanvar | taylor | taylor_domain | taylor_meanvar
--pruning_method fixed_rate | taylor | adaptive
--training_strategy DI | SFT | vanilla
--model resnet18 | resnet50 | vgg16 | densenet121
--prune_rates 0.1 0.1 0.1          # Default
--alpha 1.0                         # DI variance weight
--batch_size 256                    # Default (128 for ResNet50)
```

## Full Example Suite

```bash
# 1. Activation
python main.py --data_dir /path/to/pacs

# 2. Mean-Variance
python main.py --data_dir /path/to/pacs --importance_type meanvar

# 3. Taylor
python main.py --data_dir /path/to/pacs --pruning_method taylor --importance_type taylor

# 4. Taylor Mean-Variance
python main.py --data_dir /path/to/pacs --pruning_method taylor --importance_type taylor_meanvar

# 5. Adaptive
python main.py --data_dir /path/to/pacs --pruning_method adaptive --pruning_strategy target_error

# 6. ResNet50 + Taylor
python main.py --data_dir /path/to/pacs --model resnet50 --batch_size 128 --pruning_method taylor --importance_type taylor_meanvar
```
