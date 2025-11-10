# analysis/importance.py
import torch
import numpy as np
from tqdm import tqdm
import logging
import os


logger = logging.getLogger(__name__)


def compute_domain_ig(model, dataloader, domain_name: str, device, config):
    """
    Compute Integrated Gradients importance scores for a specific domain.
    
    Args:
        model: Neural network model
        dataloader: DataLoader for the domain
        domain_name: Name of the domain
        device: Device to run on
        config: Configuration object
        
    Returns:
        importance_scores: NumPy array of importance scores for each neuron
    """
    model.eval()
    model = model.to(device)
    
    # Disable mixed precision for IG computation
    with torch.cuda.amp.autocast(enabled=False):
        # Load pre-computed baseline tensor
        baseline_path = os.path.join(config.ARTIFACTS_DIR, "baselines", f"{domain_name}_mean.pt")
        baseline = torch.load(baseline_path).to(device).unsqueeze(0)
        
        # Setup for tracking activations
        activations = {}
        hook_handles = []
        
        def get_forward_hook(name):
            """Create a forward hook to save activations and enable gradient computation."""
            def hook(module, input, output):
                activations[name] = output
                output.retain_grad()
            return hook
        
        # Register hooks on all ReLU modules in fc_stack
        layer_idx = 0
        for i, module in enumerate(model.fc_stack):
            if isinstance(module, torch.nn.ReLU):
                handle = module.register_forward_hook(get_forward_hook(f'relu_{layer_idx}'))
                hook_handles.append(handle)
                layer_idx += 1
        
        # Get neuron counts per layer
        neurons_per_layer = model.get_num_neurons_per_layer()
        total_neurons = sum(neurons_per_layer)
        
        # Initialize importance accumulator
        importance_accumulator = torch.zeros(total_neurons).to(device)
        sample_count = 0
        
        try:
            logger.info(f"Computing IG for domain {domain_name} with {config.NUM_IG_SAMPLES} samples")
            
            # Process samples
            progress_bar = tqdm(dataloader, total=min(config.NUM_IG_SAMPLES, len(dataloader)))
            
            for sample_idx, (image, label) in enumerate(progress_bar):
                if sample_idx >= config.NUM_IG_SAMPLES:
                    break
                
                image = image.to(device)
                label = label.to(device)
                
                # Initialize gradient accumulator for this sample
                sample_grads = torch.zeros(total_neurons).to(device)
                
                # Riemann sum approximation
                for m_idx in range(1, config.IG_INTEGRATION_STEPS + 1):
                    # Clear previous gradients
                    model.zero_grad()
                    activations.clear()
                    
                    # Interpolate between baseline and input
                    alpha = m_idx / config.IG_INTEGRATION_STEPS
                    x_t = baseline + alpha * (image - baseline)
                    x_t.requires_grad = True
                    
                    # Forward pass
                    outputs = model(x_t)
                    
                    # Get predicted class probability
                    probs = torch.softmax(outputs, dim=1)
                    target_prob = probs[0, label]
                    
                    # Backward pass
                    target_prob.backward(retain_graph=True)
                    
                    # Accumulate gradients from activations
                    offset = 0
                    for layer_idx, n_neurons in enumerate(neurons_per_layer):
                        if f'relu_{layer_idx}' in activations:
                            act = activations[f'relu_{layer_idx}']
                            if act.grad is not None:
                                # Sum gradients across batch dimension
                                layer_grads = act.grad.squeeze(0)
                                sample_grads[offset:offset+n_neurons] += layer_grads
                        offset += n_neurons
                
                # Apply IG formula: (x - baseline) * (accumulated_gradients / m)
                model.zero_grad()
                activations.clear()
                
                # Get activations at input
                with torch.no_grad():
                    _ = model(image)
                    act_x = []
                    for layer_idx in range(len(neurons_per_layer)):
                        if f'relu_{layer_idx}' in activations:
                            act_x.append(activations[f'relu_{layer_idx}'].squeeze(0))
                    act_x = torch.cat(act_x) if act_x else torch.zeros(total_neurons).to(device)
                
                # Get activations at baseline
                activations.clear()
                with torch.no_grad():
                    _ = model(baseline)
                    act_baseline = []
                    for layer_idx in range(len(neurons_per_layer)):
                        if f'relu_{layer_idx}' in activations:
                            act_baseline.append(activations[f'relu_{layer_idx}'].squeeze(0))
                    act_baseline = torch.cat(act_baseline) if act_baseline else torch.zeros(total_neurons).to(device)
                
                # Compute IG scores for this sample
                sample_ig = (act_x - act_baseline) * (sample_grads / config.IG_INTEGRATION_STEPS)
                
                # Accumulate absolute values
                importance_accumulator += torch.abs(sample_ig)
                sample_count += 1
                
                progress_bar.set_postfix({'samples': sample_count})
        
        finally:
            # Remove all hooks
            for handle in hook_handles:
                handle.remove()
        
        # Average over samples
        if sample_count > 0:
            importance_scores = importance_accumulator / sample_count
        else:
            importance_scores = importance_accumulator
        
        # Post-process: normalize per layer
        importance_scores_np = importance_scores.cpu().numpy()
        normalized_scores = []
        
        offset = 0
        for n_neurons in neurons_per_layer:
            layer_scores = importance_scores_np[offset:offset+n_neurons]
            
            # L2 normalize per layer
            layer_norm = np.linalg.norm(layer_scores)
            if layer_norm > 0:
                layer_scores = layer_scores / layer_norm
            
            normalized_scores.append(layer_scores)
            offset += n_neurons
        
        # Flatten back
        final_scores = np.concatenate(normalized_scores)
        
        logger.info(f"Computed IG scores for {sample_count} samples from {domain_name}")
        logger.info(f"Score statistics - Min: {final_scores.min():.6f}, "
                   f"Max: {final_scores.max():.6f}, Mean: {final_scores.mean():.6f}")
        
        return final_scores


def save_importance_scores(scores_dict, save_path):
    """Save importance scores to a numpy file."""
    np.savez(save_path, **scores_dict)
    logger.info(f"Saved importance scores to {save_path}")


def load_importance_scores(load_path):
    """Load importance scores from a numpy file."""
    data = np.load(load_path)
    return {key: data[key] for key in data.files}
