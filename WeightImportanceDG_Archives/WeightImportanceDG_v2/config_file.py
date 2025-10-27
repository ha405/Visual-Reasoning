"""
Configuration file for Domain Generalization with Importance-Based Weight Masking
Contains all hyperparameters, paths, and constants used across the project
"""

import torch
import numpy as np
import random

# ============================================================================
# DEVICE AND SEED CONFIGURATION
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATA_ROOT = r"E:\Fatim\LUMS\SPROJ\Codes\Visual Reasoning\Data\PACS\kfold"
DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
NUM_CLASSES = 7

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Vision Transformer model
VIT_MODEL = "google/vit-base-patch16-224-in21k"
VIT_EMBEDDING_DIM = 768

# Feedforward Network Architecture (4 hidden layers, each with 10 neurons)
HIDDEN_LAYER_SIZES = [10, 10, 10, 10]  # 4 layers of 10 neurons each
NUM_HIDDEN_LAYERS = len(HIDDEN_LAYER_SIZES)

# ============================================================================
# PHASE 1: DOMAIN-SPECIFIC TRAINING CONFIGURATION
# ============================================================================
PH1_NUM_EPOCHS = 10
PH1_BATCH_SIZE = 32
PH1_LEARNING_RATE = 1e-3

# ============================================================================
# IMPORTANCE CALCULATION CONFIGURATION
# ============================================================================
# Threshold for binary mask: top 60% of weights are important
IMPORTANCE_THRESHOLD_PERCENTILE = 60  # Top 60% marked as important (1), rest as 0

# ============================================================================
# PHASE 2: CROSS-DOMAIN GENERALIZATION CONFIGURATION
# ============================================================================
PH2_NUM_EPOCHS = 10
PH2_BATCH_SIZE = 32

# Learning rates for Phase 2
PH2_LR_IMPORTANT = 1e-3  # Normal learning rate for important weights
PH2_LR_UNIMPORTANT_START = 1e-4  # Starting LR for unimportant weights
PH2_LR_DECAY_RATE = 0.5  # Exponential decay rate for unimportant weights
PH2_FREEZE_EPOCH = 7  # Freeze unimportant weights after this epoch (1-indexed)

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
IMAGE_SIZE = 224
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

# ============================================================================
# PATHS FOR SAVING RESULTS
# ============================================================================
RESULTS_DIR = "./results"
WEIGHTS_DIR = "./results/weights"
PLOTS_DIR = "./results/plots"
IMPORTANCE_DIR = "./results/importance"

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
VIZ_CIRCLE_SIZE = 300  # Size of circles in weight visualization
VIZ_FIGSIZE_PER_LAYER = (4, 4)  # Figure size for each 10x10 layer
VIZ_COLORMAP = "RdYlGn"  # Colormap for continuous values
VIZ_DPI = 100  # DPI for saved figures
