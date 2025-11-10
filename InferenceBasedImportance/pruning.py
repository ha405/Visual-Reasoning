# models/pruning.py
import torch
import logging
from .pacs_net import PACSNet


logger = logging.getLogger(__name__)


def rebuild_pruned_model(original_model, selected_indices: list, config):
    """
    Rebuild a pruned model with only selected neurons.
    
    Args:
        original_model: Original full model
        selected_indices: List of neuron indices to keep (flat across all layers)
        config: Configuration object
        
    Returns:
        pruned_model: New model with pruned architecture
    """
    # Get original layer structure
    original_neurons_per_layer = original_model.get_num_neurons_per_layer()
    n_layers = len(original_neurons_per_layer)
    
    # Map flat indices to per-layer structure
    layer_map = {i: [] for i in range(n_layers)}
    
    offset = 0
    for layer_idx, n_neurons in enumerate(original_neurons_per_layer):
        layer_start = offset
        layer_end = offset + n_neurons
        
        # Find which selected indices belong to this layer
        for idx in selected_indices:
            if layer_start <= idx < layer_end:
                # Convert to layer-local index
                local_idx = idx - layer_start
                layer_map[layer_idx].append(local_idx)
        
        offset += n_neurons
    
    # Calculate new hidden sizes
    pruned_hidden_sizes = [len(layer_map[i]) for i in range(n_layers)]
    
    logger.info(f"Original layer sizes: {original_neurons_per_layer}")
    logger.info(f"Pruned layer sizes: {pruned_hidden_sizes}")
    
    # Create new model with pruned architecture
    pruned_model = PACSNet(num_classes=config.NUM_CLASSES, hidden_sizes=pruned_hidden_sizes)
    
    # Copy backbone weights (unchanged)
    pruned_model.backbone.load_state_dict(original_model.backbone.state_dict())
    
    # Copy MLP weights (pruned)
    _copy_pruned_mlp_weights(original_model, pruned_model, layer_map)
    
    # Copy classifier weights (with pruned input)
    _copy_classifier_weights(original_model, pruned_model, layer_map[n_layers-1])
    
    return pruned_model


def _copy_pruned_mlp_weights(original_model, pruned_model, layer_map):
    """
    Copy weights for pruned MLP layers.
    
    Args:
        original_model: Original model
        pruned_model: Pruned model
        layer_map: Dictionary mapping layer index to kept neuron indices
    """
    # Extract modules
    original_modules = list(original_model.fc_stack.children())
    pruned_modules = list(pruned_model.fc_stack.children())
    
    layer_idx = 0
    prev_kept_indices = None  # For input dimension slicing
    
    for i in range(0, len(original_modules), 3):  # Each block has Linear, BatchNorm, ReLU
        if i + 2 >= len(original_modules):
            break
            
        # Get modules for this layer
        original_linear = original_modules[i]
        original_bn = original_modules[i + 1]
        pruned_linear = pruned_modules[i]
        pruned_bn = pruned_modules[i + 1]
        
        # Get kept indices for this layer
        kept_out_indices = layer_map[layer_idx]
        
        # Copy Linear weights
        with torch.no_grad():
            # Slice weights: [out_features, in_features]
            if prev_kept_indices is None:
                # First layer - keep all input features from backbone
                pruned_linear.weight.data = original_linear.weight[kept_out_indices, :].clone()
            else:
                # Subsequent layers - slice both dimensions
                pruned_linear.weight.data = original_linear.weight[kept_out_indices, :][:, prev_kept_indices].clone()
            
            # Slice bias: [out_features]
            pruned_linear.bias.data = original_linear.bias[kept_out_indices].clone()
        
        # Copy BatchNorm parameters
        with torch.no_grad():
            # All BN parameters are indexed by output features
            pruned_bn.weight.data = original_bn.weight[kept_out_indices].clone()
            pruned_bn.bias.data = original_bn.bias[kept_out_indices].clone()
            pruned_bn.running_mean.data = original_bn.running_mean[kept_out_indices].clone()
            pruned_bn.running_var.data = original_bn.running_var[kept_out_indices].clone()
        
        # Update for next iteration
        prev_kept_indices = kept_out_indices
        layer_idx += 1


def _copy_classifier_weights(original_model, pruned_model, last_layer_indices):
    """
    Copy classifier weights with pruned input dimension.
    
    Args:
        original_model: Original model
        pruned_model: Pruned model
        last_layer_indices: Kept indices from the last MLP layer
    """
    with torch.no_grad():
        # Classifier weights: [num_classes, in_features]
        # Only slice the input dimension
        pruned_model.classifier.weight.data = original_model.classifier.weight[:, last_layer_indices].clone()
        pruned_model.classifier.bias.data = original_model.classifier.bias.clone()
    
    logger.info(f"Copied classifier weights with input dim {len(last_layer_indices)}")


def create_masked_model(original_model, selected_indices: list):
    """
    Create a masked version of the model (for testing).
    Instead of rebuilding, this zeros out pruned neurons.
    
    Args:
        original_model: Original model
        selected_indices: List of neuron indices to keep
        
    Returns:
        masked_model: Model with pruned neurons masked
    """
    import copy
    masked_model = copy.deepcopy(original_model)
    
    # Get layer structure
    neurons_per_layer = original_model.get_num_neurons_per_layer()
    
    # Create masks for each layer
    layer_masks = []
    offset = 0
    
    for n_neurons in neurons_per_layer:
        mask = torch.zeros(n_neurons)
        for idx in selected_indices:
            if offset <= idx < offset + n_neurons:
                mask[idx - offset] = 1.0
        layer_masks.append(mask)
        offset += n_neurons
    
    # Apply masks to weights
    with torch.no_grad():
        layer_idx = 0
        modules = list(masked_model.fc_stack.children())
        
        for i in range(0, len(modules), 3):  # Linear, BatchNorm, ReLU
            if i >= len(modules):
                break
                
            linear = modules[i]
            mask = layer_masks[layer_idx].to(linear.weight.device)
            
            # Mask output neurons
            linear.weight.data *= mask.unsqueeze(1)
            linear.bias.data *= mask
            
            # Mask BatchNorm if present
            if i + 1 < len(modules):
                bn = modules[i + 1]
                bn.weight.data *= mask
                bn.bias.data *= mask
            
            layer_idx += 1
    
    return masked_model
