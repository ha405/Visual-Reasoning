from .importance import (
    compute_activation_importance,
    compute_meanvar_importance,
    compute_taylor_importance,
    compute_taylor_domain_stats,
    compute_taylor_meanvar_importance,
    compute_magnitude_importance,
    compute_error_correlated_importance,
)

from .masking import (
    generate_mask_from_importance,
    generate_mask_from_taylor_importance,
    select_prune_mask_by_source_heuristics,
    select_prune_mask_by_target_error,
)

from .training import (
    train_DI,
    train_SFT,
    train_vanilla,
    evaluate,
)

from .strategies import (
    fixed_rate_pruning,
    adaptive_layerwise_pruning,
    taylor_pruning,
)

from .dataset import (
    DomainDataset,
    get_pacs_dataloaders,
    get_dataloaders,
)

from .models import get_model

from .utils import (
    get_layer,
    combine_source_loaders,
    create_calibration_loaders,
    apply_mask,
)

__version__ = "1.0.0"
__all__ = [
    "compute_activation_importance",
    "compute_meanvar_importance",
    "compute_taylor_importance",
    "compute_taylor_domain_stats",
    "compute_taylor_meanvar_importance",
    "compute_magnitude_importance",
    "compute_error_correlated_importance",
    "generate_mask_from_importance",
    "generate_mask_from_taylor_importance",
    "apply_mask",
    "select_prune_mask_by_source_heuristics",
    "select_prune_mask_by_target_error",
    "train_DI",
    "train_SFT",
    "train_vanilla",
    "evaluate",
    "fixed_rate_pruning",
    "adaptive_layerwise_pruning",
    "taylor_pruning",
    "DomainDataset",
    "get_pacs_dataloaders",
    "get_dataloaders",
    "get_model",
    "get_layer",
    "combine_source_loaders",
    "create_calibration_loaders",
]
