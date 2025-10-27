# src/config.py

import torch
import os

# --- General Settings ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Dataset Settings ---
# DATA_ROOT = "../pacs_data/pacs_data" # Adjust this path to your dataset location
DATA_ROOT = 'D:\Haseeb\SPROJ\PACS ViT\pacs_data\pacs_data'
DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
NUM_CLASSES = 7

# --- ViT Model Settings ---
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
EMBEDDING_DIM = 768  # From ViT-Base config

# --- FFN Head Architecture ---
FFN_HIDDEN = 10
FFN_LAYERS = 4
DROPOUT = 0.2
BATCH_NORM = True

# --- Phase 1 Training ---
PH1_DOMAINS = ["art_painting", "cartoon", "photo"] # Domains to train on for importance
PH1_EPOCHS = 10
PH1_BATCH_SIZE = 32
PH1_LR = 1e-3

# --- Importance Computation ---
IMPORTANCE_THRESHOLD = 0.5

# --- Phase 2 Training (LODO) ---
PH2_EPOCHS = 10
PH2_BATCH_SIZE = 32
PH2_LR_IMPORTANT = 1e-3
PH2_LR_NONIMPORTANT_START = 1e-4
PH2_FREEZE_EPOCH = 7

# --- Visualization ---
PLOT_DPI = 150

# --- Directory Paths ---
RESULTS_DIR = "../results"
PHASE1_WEIGHTS_DIR = os.path.join(RESULTS_DIR, "phase1_weights")
PHASE1_IMPORTANCE_DIR = os.path.join(RESULTS_DIR, "phase1_importances")
PHASE2_RESULTS_DIR = os.path.join(RESULTS_DIR, "phase2_results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Create directories if they don't exist
os.makedirs(os.path.join(PHASE1_WEIGHTS_DIR, "initial"), exist_ok=True)
os.makedirs(os.path.join(PHASE1_WEIGHTS_DIR, "final"), exist_ok=True)
os.makedirs(PHASE1_IMPORTANCE_DIR, exist_ok=True)
os.makedirs(PHASE2_RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)