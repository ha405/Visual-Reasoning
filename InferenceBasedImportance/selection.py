# analysis/selection.py
import numpy as np
import logging


logger = logging.getLogger(__name__)


def select_neurons(domain_importances: dict, total_neurons: int, config):
    """
    Select neurons to keep based on importance scores.
    
    Args:
        domain_importances: Dictionary mapping domain names to importance score arrays
        total_neurons: Total number of neurons in the MLP
        config: Configuration object
        
    Returns:
        selected_indices: Sorted list of neuron indices to keep
    """
    target_keep = int(total_neurons * config.PRUNING_TARGET_PROPORTION)
    logger.info(f"Selecting {target_keep}/{total_neurons} neurons using {config.SELECTION_METHOD}")
    
    # Get neurons per layer from the importance score shape
    n_domains = len(domain_importances)
    importance_array = np.stack(list(domain_importances.values()))
    
    # Select based on method
    if config.SELECTION_METHOD == "greedy_iterative":
        selected_indices = _select_greedy_iterative(importance_array, target_keep)
    elif config.SELECTION_METHOD == "union_topk":
        selected_indices = _select_union_topk(importance_array, target_keep)
    elif config.SELECTION_METHOD == "weighted_avg":
        selected_indices = _select_weighted_avg(importance_array, target_keep)
    elif config.SELECTION_METHOD == "random":
        selected_indices = _select_random(total_neurons, target_keep)
    else:
        raise ValueError(f"Unknown selection method: {config.SELECTION_METHOD}")
    
    # Enforce minimum kept per layer
    selected_indices = _enforce_min_kept(selected_indices, total_neurons, config)
    
    logger.info(f"Selected {len(selected_indices)} neurons")
    return sorted(selected_indices)


def _select_greedy_iterative(importance_array, target_keep):
    """
    Greedy iterative selection: maximize coverage across domains.
    
    Args:
        importance_array: Shape (n_domains, n_neurons)
        target_keep: Number of neurons to keep
        
    Returns:
        List of selected neuron indices
    """
    n_domains, n_neurons = importance_array.shape
    selected = set()
    
    # Track coverage per domain
    domain_coverage = np.zeros(n_domains)
    
    while len(selected) < target_keep:
        best_neuron = -1
        best_gain = -np.inf
        
        # Find neuron that maximizes total coverage gain
        for neuron_idx in range(n_neurons):
            if neuron_idx in selected:
                continue
            
            # Calculate coverage gain if we add this neuron
            gain = 0
            for domain_idx in range(n_domains):
                # Gain is the importance score weighted by how much we need it
                current_coverage = domain_coverage[domain_idx]
                neuron_importance = importance_array[domain_idx, neuron_idx]
                
                # Weight by inverse of current coverage (prioritize less covered domains)
                weight = 1.0 / (1.0 + current_coverage)
                gain += weight * neuron_importance
            
            if gain > best_gain:
                best_gain = gain
                best_neuron = neuron_idx
        
        # Add best neuron
        selected.add(best_neuron)
        
        # Update coverage
        for domain_idx in range(n_domains):
            domain_coverage[domain_idx] += importance_array[domain_idx, best_neuron]
    
    return list(selected)


def _select_union_topk(importance_array, target_keep):
    """
    Union of top-k: take top neurons from each domain.
    
    Args:
        importance_array: Shape (n_domains, n_neurons)
        target_keep: Number of neurons to keep
        
    Returns:
        List of selected neuron indices
    """
    n_domains, n_neurons = importance_array.shape
    k_per_domain = target_keep // n_domains + 1
    
    selected = set()
    
    for domain_idx in range(n_domains):
        # Get top-k neurons for this domain
        domain_scores = importance_array[domain_idx]
        top_k_indices = np.argsort(domain_scores)[-k_per_domain:]
        selected.update(top_k_indices)
        
        if len(selected) >= target_keep:
            break
    
    # If we have too many, trim based on average importance
    if len(selected) > target_keep:
        selected_list = list(selected)
        avg_importance = importance_array.mean(axis=0)
        selected_importance = [(idx, avg_importance[idx]) for idx in selected_list]
        selected_importance.sort(key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in selected_importance[:target_keep]]
    else:
        selected = list(selected)
    
    return selected


def _select_weighted_avg(importance_array, target_keep):
    """
    Select based on weighted average importance across domains.
    
    Args:
        importance_array: Shape (n_domains, n_neurons)
        target_keep: Number of neurons to keep
        
    Returns:
        List of selected neuron indices
    """
    # Simple average across domains
    avg_importance = importance_array.mean(axis=0)
    
    # Select top-k based on average importance
    top_indices = np.argsort(avg_importance)[-target_keep:]
    
    return list(top_indices)


def _select_random(total_neurons, target_keep):
    """
    Random selection baseline.
    
    Args:
        total_neurons: Total number of neurons
        target_keep: Number of neurons to keep
        
    Returns:
        List of selected neuron indices
    """
    indices = np.random.choice(total_neurons, size=target_keep, replace=False)
    return list(indices)


def _enforce_min_kept(selected_indices, total_neurons, config):
    """
    Ensure minimum neurons are kept per layer.
    
    Args:
        selected_indices: List of selected neuron indices
        total_neurons: Total number of neurons
        config: Configuration object
        
    Returns:
        Updated list of selected indices
    """
    # This is a simplified version - in practice, you'd need the layer structure
    # For now, just ensure we don't prune too aggressively
    min_total = len(config.HIDDEN_SIZES) * config.MIN_KEPT_PER_LAYER
    
    if len(selected_indices) < min_total:
        # Add random neurons to meet minimum
        all_indices = set(range(total_neurons))
        available = list(all_indices - set(selected_indices))
        
        n_to_add = min_total - len(selected_indices)
        additional = np.random.choice(available, size=n_to_add, replace=False)
        
        selected_indices = list(selected_indices) + list(additional)
        logger.warning(f"Added {n_to_add} neurons to meet minimum requirements")
    
    return selected_indices


def save_selected_neurons(selected_indices, save_path):
    """Save selected neuron indices to JSON."""
    import json
    with open(save_path, 'w') as f:
        json.dump({'selected_indices': selected_indices}, f, indent=2)
    logger.info(f"Saved selected neurons to {save_path}")


def load_selected_neurons(load_path):
    """Load selected neuron indices from JSON."""
    import json
    with open(load_path, 'r') as f:
        data = json.load(f)
    return data['selected_indices']
