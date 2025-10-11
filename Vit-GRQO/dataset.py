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
    def __getitem__(self, index):
        original_index = index
        for _ in range(len(self)):
            try:
                return super().__getitem__(index)
            except (OSError, UnidentifiedImageError):
                index = (index + 1) % len(self)
        raise RuntimeError(f"No valid images found starting from index {original_index}")

class BaseDomainDataset:
    def __init__(self, data_root, domains, transform, batch_size):
        self.data_root = data_root
        self.domains = domains
        self.transform = transform
        self.batch_size = batch_size
    
    def get_dataset(self, domain, train=True):
        dataset = SafeImageFolder(os.path.join(self.data_root, domain), transform=self.transform)
        return dataset

    def get_dataloader(self, domain, train=True):
        dataset = SafeImageFolder(os.path.join(self.data_root, domain), transform=self.transform)
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

class PACSDataset(BaseDomainDataset):
    def __init__(self, data_root, transform, batch_size):
        super().__init__(data_root, ["art_painting", "cartoon", "photo", "sketch"], transform, batch_size)

class VLCSDataset(BaseDomainDataset):
    def __init__(self, data_root, transform, batch_size):
        super().__init__(data_root, ["VOC2007", "LabelMe", "Caltech101", "SUN09"], transform, batch_size)

class OfficeHomeDataset(BaseDomainDataset):
    def __init__(self, data_root, transform, batch_size):
        super().__init__(data_root, ["Art", "Clipart", "Product", "Real_World"], transform, batch_size)

class RMNISTFolderSafe(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

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
                return img, label
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
        dataset = RMNISTFolderSafe(samples, transform=self.transform)
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
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

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
                return img, label
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
        dataset = CMNISTFolderSafe(samples, transform=self.transform)
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
        domain_path = os.path.join(self.data_root, domain)
        if not os.path.exists(domain_path):
            raise RuntimeError(f"Domain folder not found: {domain_path}")
        dataset = SafeImageFolder(domain_path, transform=self.transform)
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
