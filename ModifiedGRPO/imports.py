import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
# Corrected import: Use an alias for the imported ViTModel to avoid name collision
from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTModel as ViTModelBase
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random