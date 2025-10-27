"""
Importance Calculation and Masking Utilities
Computes weight importance based on training-induced changes
Creates binary masks for selective learning rate application
"""

import torch
import numpy as np
from config_file import *


class ImportanceCalculator:
    """
    Calculates importance of weights based on change during training
    Importance = |final_weights - initial_weights|
    """
    
    def __init__(self):
        """Initialize importance calculator"""
        pass
    
    def compute_importance(self, initial_weights, final_weights):
        """
        Compute continuous importance values for each weight
        
        Args:
            initial_weights (list): List of 4 initial weight tensors [10, 10]
            final_weights (list): List of 4 final weight tensors [10, 10]
        
        Returns:
            list: List of 4 importance tensors [10, 10] with values >= 0
        """
        assert len(initial_weights) == len(final_weights), "Weight lists must have same length"
        
        importance_matrices = []
        for init_w, final_w in zip(initial_weights, final_weights):
            # Compute absolute difference
            importance = torch.abs(final_w - init_w)
            importance_matrices.append(importance)
        
        return importance_matrices
    
    def normalize_importance_global(self, importance_matrices):
        """
        Normalize importance values globally across all layers to [0, 1]
        Uses global min and max across all 4 layers
        
        Args:
            importance_matrices (list): List of 4 importance tensors [10, 10]
        
        Returns:
            list: List of 4 normalized importance tensors [10, 10] with values in [0, 1]
        """
        # Stack all importance matrices to find global min/max
        all_importance = torch.stack(importance_matrices)
        global_min = all_importance.min()
        global_max = all_importance.max()
        
        # Avoid division by zero
        if global_max - global_min < 1e-10:
            return [torch.ones_like(imp) * 0.5 for imp in importance_matrices]
        
        # Normalize each matrix using global min/max
        normalized = []
        for imp in importance_matrices:
            norm_imp = (imp - global_min) / (global_max - global_min)
            normalized.append(norm_imp)
        
        return normalized
    
    def create_binary_mask(self, normalized_importance, threshold_percentile=IMPORTANCE_THRESHOLD_PERCENTILE):
        """
        Create binary mask: top threshold_percentile% of weights = 1 (important), rest = 0
        
        Args:
            normalized_importance (list): List of 4 normalized importance tensors [10, 10]
            threshold_percentile (int): Percentile threshold (e.g., 60 = top 60% are important)
        
        Returns:
            list: List of 4 binary mask tensors [10, 10] with values {0, 1}
        """
        # Stack all normalized importance values
        all_importance = torch.stack(normalized_importance).flatten()
        
        # Compute threshold value at the given percentile
        threshold_value = torch.quantile(all_importance, q=(100 - threshold_percentile) / 100.0)
        
        # Create binary masks
        binary_masks = []
        for norm_imp in normalized_importance:
            mask = (norm_imp >= threshold_value).float()  # 1 if >= threshold, else 0
            binary_masks.append(mask)
        
        return binary_masks
    
    def save_importance_data(self, domain_name, initial_weights, final_weights, 
                            importance_matrices, normalized_importance, binary_masks, save_dir):
        """
        Save all importance-related data for a domain
        
        Args:
            domain_name (str): Name of the domain
            initial_weights (list): Initial weights
            final_weights (list): Final trained weights
            importance_matrices (list): Raw importance matrices
            normalized_importance (list): Normalized importance [0, 1]
            binary_masks (list): Binary importance masks {0, 1}
            save_dir (str): Directory to save data
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        data = {
            'domain': domain_name,
            'initial_weights': [w.cpu().numpy() for w in initial_weights],
            'final_weights': [w.cpu().numpy() for w in final_weights],
            'importance_matrices': [imp.cpu().numpy() for imp in importance_matrices],
            'normalized_importance': [norm.cpu().numpy() for norm in normalized_importance],
            'binary_masks': [mask.cpu().numpy() for mask in binary_masks]
        }
        
        save_path = os.path.join(save_dir, f'{domain_name}_importance.npz')
        np.savez(save_path, **data)
        print(f"Saved importance data for {domain_name} to {save_path}")
    
    def load_importance_data(self, domain_name, save_dir):
        """
        Load importance data for a domain
        
        Args:
            domain_name (str): Name of the domain
            save_dir (str): Directory containing saved data
        
        Returns:
            dict: Dictionary containing all importance data
        """
        import os
        load_path = os.path.join(save_dir, f'{domain_name}_importance.npz')
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No importance data found at {load_path}")
        
        data = np.load(load_path, allow_pickle=True)
        
        return {
            'domain': str(data['domain']),
            'initial_weights': [torch.tensor(w) for w in data['initial_weights']],
            'final_weights': [torch.tensor(w) for w in data['final_weights']],
            'importance_matrices': [torch.tensor(imp) for imp in data['importance_matrices']],
            'normalized_importance': [torch.tensor(norm) for norm in data['normalized_importance']],
            'binary_masks': [torch.tensor(mask) for mask in data['binary_masks']]
        }


def aggregate_importance_from_raw_changes(importance_matrices_dict):
    """
    Aggregate importance by summing absolute weight changes across all domains
    This is the NEW approach: element-wise sum of |weight_changes| across domains
    
    Args:
        importance_matrices_dict (dict): Dictionary mapping domain_name -> list of 4 importance tensors [10, 10]
                                        Each importance tensor contains |final - initial| for that domain
    
    Returns:
        tuple: (summed_importance, normalized_importance, binary_masks)
            - summed_importance: List of 4 tensors [10, 10] with raw summed values
            - normalized_importance: List of 4 tensors [10, 10] normalized to [0, 1]
            - binary_masks: List of 4 tensors [10, 10] with top 60% marked as 1
    """
    domains = list(importance_matrices_dict.keys())
    num_layers = len(importance_matrices_dict[domains[0]])
    
    print(f"\nAggregating importance from {len(domains)} domains: {domains}")
    
    # Element-wise sum across all domains for each layer
    summed_importance = []
    for layer_idx in range(num_layers):
        # Stack all domains' importance matrices for this layer
        layer_importance_stack = torch.stack([
            importance_matrices_dict[domain][layer_idx] 
            for domain in domains
        ])  # Shape: [num_domains, 10, 10]
        
        # Sum across domains (element-wise)
        summed = torch.sum(layer_importance_stack, dim=0)  # Shape: [10, 10]
        summed_importance.append(summed)
    
    print(f"Computed element-wise sum of importance across {len(domains)} domains")
    
    # Normalize globally to [0, 1]
    all_summed = torch.stack(summed_importance)
    global_min = all_summed.min()
    global_max = all_summed.max()
    
    if global_max - global_min < 1e-10:
        normalized_importance = [torch.ones_like(imp) * 0.5 for imp in summed_importance]
    else:
        normalized_importance = [
            (imp - global_min) / (global_max - global_min)
            for imp in summed_importance
        ]
    
    print(f"Normalized summed importance to [0, 1] range")
    
    # Create binary masks: top 60% = important
    all_normalized = torch.stack(normalized_importance).flatten()
    threshold_value = torch.quantile(all_normalized, q=(100 - IMPORTANCE_THRESHOLD_PERCENTILE) / 100.0)
    
    binary_masks = []
    for norm_imp in normalized_importance:
        mask = (norm_imp >= threshold_value).float()
        binary_masks.append(mask)
    
    print(f"Created binary masks (top {IMPORTANCE_THRESHOLD_PERCENTILE}% marked as important)")
    
    return summed_importance, normalized_importance, binary_masks


def aggregate_importance_masks(mask_list):
    """
    OLD APPROACH: Aggregate binary masks from multiple domains using union strategy
    This function is kept for backward compatibility but is NOT used in the new approach
    
    Args:
        mask_list (list): List of mask lists, where each mask list contains 4 tensors [10, 10]
                         e.g., [domain1_masks, domain2_masks, domain3_masks]
    
    Returns:
        list: Aggregated masks (4 tensors [10, 10])
    """
    # Default strategy: Union (if any domain finds a weight important, mark it as important)
    aggregated_masks = []
    
    num_layers = len(mask_list[0])
    for layer_idx in range(num_layers):
        # Stack masks for this layer from all domains
        layer_masks = torch.stack([masks[layer_idx] for masks in mask_list])
        
        # Union: max across domains (1 if any domain has 1)
        aggregated = torch.max(layer_masks, dim=0)[0]
        aggregated_masks.append(aggregated)
    
    return aggregated_masks


def compute_mask_statistics(binary_masks):
    """
    Compute statistics about binary masks
    
    Args:
        binary_masks (list): List of 4 binary mask tensors [10, 10]
    
    Returns:
        dict: Statistics including percentage of important weights per layer
    """
    stats = {}
    
    for i, mask in enumerate(binary_masks):
        total_weights = mask.numel()
        important_weights = mask.sum().item()
        percentage_important = (important_weights / total_weights) * 100
        
        stats[f'layer_{i+1}'] = {
            'total_weights': total_weights,
            'important_weights': int(important_weights),
            'percentage_important': percentage_important
        }
    
    # Overall statistics
    total_all = sum([mask.numel() for mask in binary_masks])
    important_all = sum([mask.sum().item() for mask in binary_masks])
    stats['overall'] = {
        'total_weights': total_all,
        'important_weights': int(important_all),
        'percentage_important': (important_all / total_all) * 100
    }
    
    return stats
