# src/phase1_utils.py

import numpy as np
import os
from . import config

def get_param_names(model):
    """Returns a list of parameter names for weights and biases."""
    names = []
    for i in range(len(model.layers)):
        names.append(f"layers.{i}.weight")
        names.append(f"layers.{i}.bias")
    names.append("classifier.weight")
    names.append("classifier.bias")
    return names

def save_weights(model, directory):
    """Saves model weights and biases as individual .npy files."""
    os.makedirs(directory, exist_ok=True)
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        filename = name.replace('.', '_') + '.npy'
        np.save(os.path.join(directory, filename), param.cpu().numpy())

def compute_and_save_importance(domain, param_name):
    """Computes and saves importance masks for a given parameter."""
    initial_dir = os.path.join(config.PHASE1_WEIGHTS_DIR, "initial")
    final_dir = os.path.join(config.PHASE1_WEIGHTS_DIR, "final", f"domain_{domain}")
    
    filename = param_name.replace('.', '_') + '.npy'
    
    w_init = np.load(os.path.join(initial_dir, filename))
    w_final = np.load(os.path.join(final_dir, filename))
    
    importance = np.abs(w_final - w_init)
    
    # Per-matrix max absolute normalization
    norm_factor = np.max(importance) + 1e-12
    importance_norm = importance / norm_factor
    
    importance_binary = (importance_norm > config.IMPORTANCE_THRESHOLD).astype(np.uint8)
    
    # Save masks
    out_dir = os.path.join(config.PHASE1_IMPORTANCE_DIR, f"domain_{domain}")
    os.makedirs(out_dir, exist_ok=True)
    
    norm_path = os.path.join(out_dir, f"{filename.replace('.npy', '')}_norm.npy")
    binary_path = os.path.join(out_dir, f"{filename.replace('.npy', '')}_binary.npy")
    
    np.save(norm_path, importance_norm)
    np.save(binary_path, importance_binary)
    
    return importance_norm, importance_binary

def create_consensus_mask(param_name, domains=config.PH1_DOMAINS):
    """Creates and saves a consensus mask from multiple domain importances."""
    filename = param_name.replace('.', '_') + '.npy'
    norm_filename = filename.replace('.npy', '_norm.npy')
    
    importance_stack = []
    for domain in domains:
        imp_path = os.path.join(config.PHASE1_IMPORTANCE_DIR, f"domain_{domain}", norm_filename)
        importance_stack.append(np.load(imp_path))
        
    stacked_imps = np.stack(importance_stack, axis=0)
    
    # Strict consensus: use the minimum importance value across domains
    consensus_norm = np.min(stacked_imps, axis=0)
    consensus_binary = (consensus_norm > config.IMPORTANCE_THRESHOLD).astype(np.uint8)
    
    # Save consensus masks
    out_dir = os.path.join(config.PHASE1_IMPORTANCE_DIR, "consensus")
    os.makedirs(out_dir, exist_ok=True)
    
    norm_path = os.path.join(out_dir, f"{filename.replace('.npy', '')}_norm.npy")
    binary_path = os.path.join(out_dir, f"{filename.replace('.npy', '')}_binary.npy")
    
    np.save(norm_path, consensus_norm)
    np.save(binary_path, consensus_binary)
    
    print(f"Created consensus mask for {param_name}")
    return consensus_norm, consensus_binary