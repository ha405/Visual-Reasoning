import yaml
import os
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional, Dict, Any

DATASET_DOMAINS = {
    'pacs': ['art_painting', 'cartoon', 'photo', 'sketch'],
    'vlcs': ['caltech101', 'labelme', 'sun09', 'voc2007'],
    'office_home': ['Art', 'Clipart', 'Product', 'Real_World'],
    'terra_incognita': ['location_38', 'location_43', 'location_46', 'location_100'],
    'domain_net': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
}

@dataclass
class PruningConfig:
    # Dataset
    dataset: str = 'pacs'
    data_dir: Optional[str] = None
    domains: Optional[List[str]] = None  # Full list of domains for the dataset
    source_domains: List[str] = field(default_factory=list)
    target_domain: str = 'sketch'
    batch_size: int = 256
    num_workers: int = 2
    num_classes: Optional[int] = 7
    
    custom_dataset_config: Dict[str, Any] = field(default_factory=lambda: {
        'domain_folders': None,
        'class_to_idx': None,
        'transform': 'imagenet'
    })
    
    # Model
    model: str = 'resnet18'
    pretrained: bool = True
    checkpoint_path: Optional[str] = None
    
    # Training
    lr: float = 0.001
    alpha: float = 1.0
    training_strategy: str = 'DI'
    warmup_epochs: int = 5
    retrain_epochs: int = 5
    
    # Pruning
    pruning_method: str = 'fixed_rate'
    importance_type: str = 'activation'
    prune_rates: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    keep_overall_best: bool = True
    
    # Adaptive Pruning
    pruning_strategy: str = 'target_error'
    candidate_rates: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    iterations: int = 3
    calibration_samples: int = 1000
    relative_acc_drop_threshold: float = 0.1
    
    # Output
    output_dir: str = './outputs'
    checkpoint_dir: str = './checkpoints'
    experiment_name: Optional[str] = None
    
    # System
    device: str = 'cuda'
    seed: int = 42
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        # Filter out keys that are not in the dataclass
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save_yaml(self, yaml_path):
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def update_from_dict(self, updates):
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}' will be ignored.")
    
    def validate(self):
        errors = []
        
        # if self.data_dir is None:
        #     errors.append("data_dir is required")
        
        if self.pruning_method not in ['fixed_rate', 'adaptive', 'taylor']:
            errors.append(f"Invalid pruning_method: {self.pruning_method}")
        
        if self.training_strategy not in ['DI', 'SFT', 'vanilla']:
            errors.append(f"Invalid training_strategy: {self.training_strategy}")
        
        if self.importance_type not in ['activation', 'meanvar', 'taylor', 'taylor_domain', 'taylor_meanvar', 'magnitude', 'error_correlated']:
            errors.append(f"Invalid importance_type: {self.importance_type}")
        
        if self.pruning_method == 'adaptive' and self.pruning_strategy not in ['target_error', 'source_magnitude']:
            errors.append(f"Invalid pruning_strategy for adaptive method: {self.pruning_strategy}")
        
        if errors:
            raise ValueError("Config validation errors:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True


def load_config(config_path=None, cli_overrides=None):
    if config_path:
        config = PruningConfig.from_yaml(config_path)
        print(f"Loaded config from: {config_path}")
    else:
        # Try to load default config.yaml if it exists
        default_path = 'config.yaml'
        if os.path.exists(default_path):
            config = PruningConfig.from_yaml(default_path)
            print(f"Loaded default config from: {default_path}")
        else:
            config = PruningConfig()
            print("Using default internal configuration")
    
    if cli_overrides:
        config.update_from_dict(cli_overrides)
        print(f"Applied {len(cli_overrides)} CLI overrides")
    
    config.validate()
    return config
