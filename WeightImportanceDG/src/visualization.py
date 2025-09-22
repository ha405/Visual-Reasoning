# src/visualization.py

import numpy as np
import matplotlib.pyplot as plt
import os
from . import config
from graphviz import Digraph

def plot_matrix(mat, save_path, title, cmap='viridis', vmin=None, vmax=None):
    """
    Plots a matrix as a heatmap and saves it to a file.
    """
    plt.figure(figsize=(8, 6), dpi=config.PLOT_DPI)
    plt.imshow(mat, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Neurons")
    plt.ylabel("Inputs/Features")
    plt.tight_layout()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


# In src/visualization.py

# ... (keep the existing plot_matrix function) ...


# ... (plot_matrix function is here) ...

def create_graph_from_matrices(param_matrices, title, save_path, highlight_binary=False):
    """
    Visualizes a set of parameter matrices as a network graph.

    Args:
        param_matrices (dict): A dictionary where keys are param names ('layers.0.weight')
                               and values are the corresponding numpy arrays.
        title (str): The title for the graph.
        save_path (str): Path to save the output graph file (e.g., 'network.png').
        highlight_binary (bool): If True, styles edges for binary 0/1 values.
                                 Edges with value 1 are bold/colored, edges with 0 are omitted.
    """
    layer_sizes = [config.EMBEDDING_DIM] + [config.FFN_HIDDEN] * config.FFN_LAYERS + [config.NUM_CLASSES]
    
    dot = Digraph(comment=title)
    dot.attr('graph', label=title, fontsize='24', labelloc='t', rankdir='LR') # LR for left-to-right
    dot.attr('node', shape='circle', style='filled', color='skyblue', fontsize='10')
    dot.attr('edge', fontsize='8')

    # --- Create Nodes ---
    for i in range(len(layer_sizes)):
        with dot.subgraph(name=f'cluster_{i}') as c:
            c.attr(style='invis') # Invisible subgraph for alignment
            
            # Truncate large layers for readability
            nodes_to_show = layer_sizes[i]
            is_truncated = nodes_to_show > 10
            if is_truncated:
                nodes_to_show = 5
            
            for n in range(nodes_to_show):
                node_id = f'L{i}_N{n}'
                label_text = ''
                
                # Add bias value to node if available
                if i > 0: # Input layer has no biases
                    param_prefix = "classifier" if i == len(layer_sizes) - 1 else f"layers.{i-1}"
                    bias_name = f"{param_prefix}.bias"
                    if bias_name in param_matrices and n < len(param_matrices[bias_name]):
                        bias_val = param_matrices[bias_name][n]
                        label_text = f'b={bias_val:.2f}' if not highlight_binary else str(int(bias_val))
                
                c.node(node_id, label=label_text)
            
            if is_truncated:
                c.node(f'L{i}_ellipsis', '...', shape='none', width='0', height='0')

    # --- Create Edges ---
    for i in range(len(layer_sizes) - 1):
        param_prefix = "classifier" if i == len(layer_sizes) - 2 else f"layers.{i}"
        weight_name = f"{param_prefix}.weight"
        
        if weight_name not in param_matrices:
            continue
            
        weights = param_matrices[weight_name] # Shape: (out_features, in_features)
        
        prev_nodes = layer_sizes[i]
        if prev_nodes > 10: prev_nodes = 5
        curr_nodes = layer_sizes[i+1]
        if curr_nodes > 10: curr_nodes = 5
        
        for n_prev in range(prev_nodes):
            for n_curr in range(curr_nodes):
                weight_val = weights[n_curr, n_prev]
                
                if highlight_binary:
                    # For binary masks, only draw the important connections
                    if weight_val == 1:
                        dot.edge(f'L{i}_N{n_prev}', f'L{i+1}_N{n_curr}', 
                                 color='crimson', penwidth='1.5')
                else:
                    # For regular weights/importance, label the edge with the value
                    dot.edge(f'L{i}_N{n_prev}', f'L{i+1}_N{n_curr}', label=f'{weight_val:.2f}')

    # Save and render the graph
    output_format = os.path.splitext(save_path)[1][1:] or 'png'
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
        
    dot.render(os.path.splitext(save_path)[0], format=output_format, view=False, cleanup=True)
    print(f"Saved network graph to {save_path}")