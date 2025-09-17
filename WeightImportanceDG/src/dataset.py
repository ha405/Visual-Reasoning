# src/dataset.py

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
from . import config

# Standard ViT transformation
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class PACSDataset:
    def __init__(self, data_root, domains, transform):
        self.data_root = data_root
        self.domains = domains
        self.transform = transform

    def get_full_domain_dataset(self, domain):
        """Returns the full ImageFolder dataset for a domain."""
        return datasets.ImageFolder(os.path.join(self.data_root, domain), transform=self.transform)

    def get_split_dataloader(self, domain, batch_size, train=True, shuffle=True):
        """Returns a dataloader for the train or validation split of a domain."""
        dataset = self.get_full_domain_dataset(domain)
        
        indices = list(range(len(dataset)))
        # Ensure consistent split for all runs
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, stratify=dataset.targets, random_state=config.SEED
        )
        
        selected_idx = train_idx if train else val_idx
        subset = Subset(dataset, selected_idx)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
        return loader

    def get_single_class_dataloader(self, domain, class_idx, batch_size, shuffle=True):
        """Returns a dataloader for a single class from a single domain."""
        dataset = self.get_full_domain_dataset(domain)
        
        indices = [i for i, target in enumerate(dataset.targets) if target == class_idx]
        if not indices:
            raise ValueError(f"Class index {class_idx} not found in domain {domain}")
            
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
        return loader

def get_lodo_dataloaders(pacs_dataset, train_domains, test_domain, batch_size):
    """Prepares dataloaders for Leave-One-Domain-Out training."""
    # Training loaders for the specified domains
    train_datasets = [pacs_dataset.get_split_dataloader(d, batch_size, train=True, shuffle=False).dataset for d in train_domains]
    train_ds = ConcatDataset(train_datasets)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Validation loaders for the training domains
    val_datasets = [pacs_dataset.get_split_dataloader(d, batch_size, train=False, shuffle=False).dataset for d in train_domains]
    val_ds = ConcatDataset(val_datasets)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Test loader for the held-out domain
    test_loader = pacs_dataset.get_split_dataloader(test_domain, batch_size, train=False, shuffle=False)
    
    return train_loader, val_loader, test_loader