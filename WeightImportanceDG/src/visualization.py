# src/visualization.py

import numpy as np
import matplotlib.pyplot as plt
import os
from . import config

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