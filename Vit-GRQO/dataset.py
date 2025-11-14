import os
import torch
import numpy as np
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True
SEED = 42

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class SafeImageFolder(datasets.ImageFolder):
    """
    Extends ImageFolder to be robust to corrupted images and to return the domain index.
    """
    def __init__(self, root, transform=None, domain_idx=0):
        super().__init__(root, transform=transform)
        self.domain_idx = domain_idx

    def __getitem__(self, index):
        original_index = index
        for _ in range(len(self)):
            try:
                # Original __getitem__ returns (image, class_label)
                img, target = super().__getitem__(index)
                # Return (image, class_label, domain_label)
                return img, target, self.domain_idx
            except (OSError, UnidentifiedImageError):
                # If an image is corrupt, try the next one
                index = (index + 1) % len(self)
        raise RuntimeError(f"No valid images found starting from index {original_index}")

class BaseDomainDataset:
    def __init__(self, data_root, domains, transform, batch_size):
        self.data_root = data_root
        self.domains = domains
        self.transform = transform
        self.batch_size = batch_size
    
    def get_dataset(self, domain, train=True):
        if domain not in self.domains:
            raise ValueError(f"Unknown domain '{domain}'. Available: {self.domains}")
        domain_idx = self.domains.index(domain)
        dataset = SafeImageFolder(
            os.path.join(self.data_root, domain),
            transform=self.transform,
            domain_idx=domain_idx
        )
        return dataset

    def get_dataloader(self, domain, train=True):
        dataset = self.get_dataset(domain, train)
        g = torch.Generator()
        g.manual_seed(SEED)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=0,
            # pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        return loader

class PACSDataset(BaseDomainDataset):
    def __init__(self, data_root, transform, batch_size):
        super().__init__(data_root, ["art_painting", "cartoon", "photo", "sketch"], transform, batch_size)

class VLCSDataset(BaseDomainDataset):
    def __init__(self, data_root, transform, batch_size):
        super().__init__(data_root, ["VOC2007", "LabelMe", "Caltech101", "SUN09"], transform, batch_size)

class OfficeHomeDataset(BaseDomainDataset):
    def __init__(self, data_root, transform, batch_size):
        super().__init__(data_root, ["Art", "Clipart", "Product", "Real World"], transform, batch_size)

class RMNISTFolderSafe(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None, domain_idx=0):
        self.samples = samples
        self.transform = transform
        self.domain_idx = domain_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        original_idx = idx
        for _ in range(len(self.samples)):
            img_path, label = self.samples[idx]
            try:
                img = Image.open(img_path).convert("L")
                if self.transform:
                    img = self.transform(img)
                # Return (image, class_label, domain_label)
                return img, label, self.domain_idx
            except (OSError, UnidentifiedImageError):
                idx = (idx + 1) % len(self.samples)
        raise RuntimeError(f"No valid images found starting from index {original_idx}")

class RMNISTDataset(BaseDomainDataset):
    def __init__(self, data_root, transform, batch_size):
        super().__init__(data_root, ["0", "15", "30", "46", "60", "75"], transform, batch_size)

    def get_dataloader(self, domain, train=True):
        path = os.path.join(self.data_root, domain)
        if not os.path.exists(path):
            raise RuntimeError(f"Domain folder not found: {path}")
        samples = []
        for fname in os.listdir(path):
            full_path = os.path.join(path, fname)
            if os.path.isfile(full_path) and fname.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    label = int(fname.split("_")[-1].split(".")[0])
                    samples.append((full_path, label))
                except:
                    continue
        if not samples:
            raise RuntimeError(f"No images found in domain {domain} at {path}")
        
        domain_idx = self.domains.index(domain)
        dataset = RMNISTFolderSafe(samples, transform=self.transform, domain_idx=domain_idx)
        
        g = torch.Generator()
        g.manual_seed(SEED)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        return loader

class CMNISTFolderSafe(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None, domain_idx=0):
        self.samples = samples
        self.transform = transform
        self.domain_idx = domain_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        original_idx = idx
        for _ in range(len(self.samples)):
            img_path, label = self.samples[idx]
            try:
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                # Return (image, class_label, domain_label)
                return img, label, self.domain_idx
            except (OSError, UnidentifiedImageError):
                idx = (idx + 1) % len(self.samples)
        raise RuntimeError(f"No valid images found starting from index {original_idx}")

class CMNISTDataset(BaseDomainDataset):
    def __init__(self, data_root, transform, batch_size):
        super().__init__(data_root, ["red", "green", "blue"], transform, batch_size)

    def get_dataloader(self, domain, train=True):
        splits = ["training", "testing"]
        samples = []
        for split in splits:
            split_path = os.path.join(self.data_root, split)
            if not os.path.exists(split_path):
                continue
            for label_str in map(str, range(10)):
                label_path = os.path.join(split_path, label_str)
                if not os.path.exists(label_path):
                    continue
                for fname in os.listdir(label_path):
                    full_path = os.path.join(label_path, fname)
                    if not os.path.isfile(full_path):
                        continue
                    if fname.lower().startswith(domain.lower()) and fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        samples.append((full_path, int(label_str)))
        if not samples:
            raise RuntimeError(f"No images found for color domain {domain} in {self.data_root}")
        
        domain_idx = self.domains.index(domain)
        dataset = CMNISTFolderSafe(samples, transform=self.transform, domain_idx=domain_idx)
        
        g = torch.Generator()
        g.manual_seed(SEED)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        return loader

class TerraIncognitaDataset(BaseDomainDataset):
    def __init__(self, data_root, transform, batch_size):
        domains = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        super().__init__(data_root, domains, transform, batch_size)

    def get_dataloader(self, domain, train=True):
        if domain not in self.domains:
            raise ValueError(f"Unknown domain '{domain}'. Available: {self.domains}")
        domain_idx = self.domains.index(domain)
        domain_path = os.path.join(self.data_root, domain)
        
        dataset = SafeImageFolder(domain_path, transform=self.transform, domain_idx=domain_idx)
        
        g = torch.Generator()
        g.manual_seed(SEED)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        return loader

DATASET_REGISTRY = {
    "PACS": PACSDataset,
    "VLCS": VLCSDataset,
    "OfficeHome": OfficeHomeDataset,
    "RMNIST": RMNISTDataset,
    "CMNIST": CMNISTDataset,
    "TerraIncognita": TerraIncognitaDataset,
}

def get_dataset_class(name: str):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset name '{name}'. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]