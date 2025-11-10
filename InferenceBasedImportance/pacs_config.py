# configs/pacs_config.py
import torch

# Dataset configuration
DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
NUM_CLASSES = 7
DATA_ROOT = "/path/to/pacs/dataset"  # Update this path

# Model architecture
HIDDEN_SIZES = [512, 256]

# Training configuration - Phase 1 (ERM)
PHASE1_EPOCHS = 30
PHASE1_LR = 0.001
PHASE1_BATCH_SIZE = 32
PHASE1_WEIGHT_DECAY = 1e-4
PHASE1_LR_SCHEDULE = "cosine"

# Training configuration - Phase 2 (Fine-tuning)
PHASE2_EPOCHS = 20
PHASE2_LR = 0.0001
PHASE2_BATCH_SIZE = 32
PHASE2_WEIGHT_DECAY = 1e-4
PHASE2_LR_SCHEDULE = "cosine"

# Data augmentation
IMAGE_SIZE = 224
CROP_SIZE = 224
COLOR_JITTER = 0.4
AFFINE_DEGREES = 30

# Integrated Gradients configuration
NUM_IG_SAMPLES = 100
IG_INTEGRATION_STEPS = 50

# Neuron selection configuration
SELECTION_METHOD = "greedy_iterative"  # Options: "greedy_iterative", "union_topk", "weighted_avg", "random"
PRUNING_TARGET_PROPORTION = 0.5  # Keep 50% of neurons
MIN_KEPT_PER_LAYER = 20  # Minimum neurons to keep per layer

# System configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
USE_AMP = True
PIN_MEMORY = True

# Experiment configuration
RANDOM_SEEDS = [42, 123, 456]
ARTIFACTS_DIR = "artifacts"

# Normalization values (ImageNet)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
