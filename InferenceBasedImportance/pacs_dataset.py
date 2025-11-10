# data_handling/pacs_dataset.py
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging


logger = logging.getLogger(__name__)


class PACSDataset(Dataset):
    """PACS dataset for domain generalization."""
    
    def __init__(self, root_dir: str, domains: list, class_map: dict, transform=None):
        """
        Args:
            root_dir: Root directory of PACS dataset
            domains: List of domains to include
            class_map: Mapping from class names to integer labels
            transform: Optional transform to be applied on images
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        # Iterate through each domain
        for domain in domains:
            domain_dir = os.path.join(root_dir, domain)
            if not os.path.exists(domain_dir):
                logger.warning(f"Domain directory not found: {domain_dir}")
                continue
                
            # Walk through the directory structure
            for class_name in os.listdir(domain_dir):
                if class_name not in class_map:
                    continue
                    
                class_dir = os.path.join(domain_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                    
                # Get all image files in the class directory
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(class_map[class_name])
        
        logger.info(f"Loaded {len(self.image_paths)} images from domains: {domains}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Load image and apply transform."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_pacs_dataloaders(config, source_domains: list, target_domain: str, class_map: dict):
    """
    Create dataloaders for training and evaluation.
    
    Args:
        config: Configuration object
        source_domains: List of source domains for training
        target_domain: Target domain for evaluation
        class_map: Class name to label mapping
    
    Returns:
        train_loader: DataLoader for training (combined source domains)
        target_loader: DataLoader for target domain evaluation
        ig_loaders: Dictionary of DataLoaders for IG computation per source domain
    """
    
    # Define training transform with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.CROP_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(
            brightness=config.COLOR_JITTER,
            contrast=config.COLOR_JITTER,
            saturation=config.COLOR_JITTER,
            hue=config.COLOR_JITTER * 0.25
        ),
        transforms.RandomAffine(
            degrees=config.AFFINE_DEGREES,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    # Define evaluation transform (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.CenterCrop(config.CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    # Create combined source dataset for training
    train_dataset = PACSDataset(
        root_dir=config.DATA_ROOT,
        domains=source_domains,
        class_map=class_map,
        transform=train_transform
    )
    
    # Create target dataset for evaluation
    target_dataset = PACSDataset(
        root_dir=config.DATA_ROOT,
        domains=[target_domain],
        class_map=class_map,
        transform=eval_transform
    )
    
    # Create IG datasets for each source domain (with eval transform for consistency)
    ig_datasets = {}
    for domain in source_domains:
        ig_datasets[domain] = PACSDataset(
            root_dir=config.DATA_ROOT,
            domains=[domain],
            class_map=class_map,
            transform=eval_transform
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.PHASE1_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )
    
    target_loader = DataLoader(
        target_dataset,
        batch_size=config.PHASE1_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    ig_loaders = {}
    for domain, dataset in ig_datasets.items():
        ig_loaders[domain] = DataLoader(
            dataset,
            batch_size=1,  # IG computation is done sample by sample
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
    
    return train_loader, target_loader, ig_loaders


def compute_and_cache_baselines(config, class_map: dict):
    """
    Compute and cache baseline tensors for Integrated Gradients.
    
    Args:
        config: Configuration object
        class_map: Class name to label mapping
    """
    baseline_dir = os.path.join(config.ARTIFACTS_DIR, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)
    
    # Define evaluation transform for baseline computation
    eval_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.CenterCrop(config.CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    for domain in config.DOMAINS:
        baseline_path = os.path.join(baseline_dir, f"{domain}_mean.pt")
        
        # Skip if baseline already exists
        if os.path.exists(baseline_path):
            logger.info(f"Baseline for {domain} already exists at {baseline_path}")
            continue
        
        logger.info(f"Computing baseline for {domain}...")
        
        # Create dataset for the domain
        dataset = PACSDataset(
            root_dir=config.DATA_ROOT,
            domains=[domain],
            class_map=class_map,
            transform=eval_transform
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.NUM_WORKERS
        )
        
        # Compute mean over NUM_IG_SAMPLES samples
        sum_tensor = None
        count = 0
        
        for i, (image, _) in enumerate(dataloader):
            if i >= config.NUM_IG_SAMPLES:
                break
                
            if sum_tensor is None:
                sum_tensor = image.squeeze(0).clone()
            else:
                sum_tensor += image.squeeze(0)
            count += 1
        
        # Calculate mean
        mean_tensor = sum_tensor / count
        
        # Save baseline
        torch.save(mean_tensor, baseline_path)
        logger.info(f"Saved baseline for {domain} to {baseline_path}")
