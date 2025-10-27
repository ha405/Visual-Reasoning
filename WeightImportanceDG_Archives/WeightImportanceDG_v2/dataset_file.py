"""
Dataset handling for PACS domain generalization
Provides utilities to load PACS data with different domain configurations
"""

import os
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from config_file import *


class PACSDataset:
    """
    Wrapper class for PACS dataset handling
    Supports loading single domains or multiple domains with train/test splits
    """
    
    def __init__(self, data_root, domains, transform=None):
        """
        Initialize PACS dataset loader
        
        Args:
            data_root (str): Root directory containing PACS domain folders
            domains (list): List of domain names (e.g., ['art_painting', 'cartoon', ...])
            transform: Torchvision transforms to apply to images
        """
        self.data_root = data_root
        self.domains = domains
        self.transform = transform or self._default_transform()
    
    def _default_transform(self):
        """Default image transformations for ViT input"""
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
    
    def get_single_domain_loader(self, domain, batch_size=32, shuffle=True, train_split=None):
        """
        Get dataloader for a single domain
        
        Args:
            domain (str): Domain name (e.g., 'art_painting')
            batch_size (int): Batch size for dataloader
            shuffle (bool): Whether to shuffle data
            train_split (float or None): If None, use all data. If float (e.g., 0.8), 
                                        split into train/val and return train portion
        
        Returns:
            DataLoader: PyTorch dataloader for the domain
        """
        domain_path = os.path.join(self.data_root, domain)
        dataset = datasets.ImageFolder(domain_path, transform=self.transform)
        
        if train_split is None:
            # Use all data (for Phase 1 training)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            # Split data for train/val
            indices = list(range(len(dataset)))
            train_idx, _ = train_test_split(
                indices, 
                train_size=train_split, 
                stratify=[dataset.targets[i] for i in indices],
                random_state=SEED
            )
            subset = Subset(dataset, train_idx)
            loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
        
        return loader
    
    def get_multi_domain_loader(self, domains, batch_size=32, shuffle=True, train_split=0.8):
        """
        Get concatenated dataloader from multiple domains
        
        Args:
            domains (list): List of domain names to combine
            batch_size (int): Batch size for dataloader
            shuffle (bool): Whether to shuffle data
            train_split (float): Train/val split ratio (0.8 = 80% train, 20% val)
        
        Returns:
            tuple: (train_loader, val_loader) - DataLoaders for training and validation
        """
        train_datasets = []
        val_datasets = []
        
        for domain in domains:
            domain_path = os.path.join(self.data_root, domain)
            dataset = datasets.ImageFolder(domain_path, transform=self.transform)
            
            # Split each domain into train/val
            indices = list(range(len(dataset)))
            train_idx, val_idx = train_test_split(
                indices,
                train_size=train_split,
                stratify=[dataset.targets[i] for i in indices],
                random_state=SEED
            )
            
            train_datasets.append(Subset(dataset, train_idx))
            val_datasets.append(Subset(dataset, val_idx))
        
        # Concatenate all domains
        combined_train = ConcatDataset(train_datasets)
        combined_val = ConcatDataset(val_datasets)
        
        train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(combined_val, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def get_lodo_loaders(self, test_domain, batch_size=32):
        """
        Get Leave-One-Domain-Out (LODO) configuration loaders
        Train on all domains except test_domain, test on test_domain
        
        Args:
            test_domain (str): Domain to hold out for testing
            batch_size (int): Batch size for dataloaders
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        train_domains = [d for d in self.domains if d != test_domain]
        
        # Get train/val loaders from training domains
        train_loader, val_loader = self.get_multi_domain_loader(
            train_domains, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # Get test loader from held-out domain (use validation split as test)
        test_domain_path = os.path.join(self.data_root, test_domain)
        test_dataset = datasets.ImageFolder(test_domain_path, transform=self.transform)
        
        indices = list(range(len(test_dataset)))
        _, test_idx = train_test_split(
            indices,
            train_size=0.8,
            stratify=[test_dataset.targets[i] for i in indices],
            random_state=SEED
        )
        
        test_subset = Subset(test_dataset, test_idx)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader


def get_transform():
    """Returns the standard transform for PACS images"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
