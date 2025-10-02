import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from PIL import ImageFile, UnidentifiedImageError

# Allow loading of truncated images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42  # reproducibility


class SafeImageFolder(datasets.ImageFolder):
    """ImageFolder that skips unreadable/corrupted images."""

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (OSError, UnidentifiedImageError):
            # Skip corrupted file -> fallback to another sample
            new_index = (index + 1) % len(self)
            return self.__getitem__(new_index)


class BaseDomainDataset:
    """Base class for multi-domain datasets with train/val split."""

    def __init__(self, data_root, domains, transform, batch_size):
        self.data_root = data_root
        self.domains = domains
        self.transform = transform
        self.batch_size = batch_size

    def get_dataloader(self, domain, train=True):
        # Use SafeImageFolder to avoid crashes
        dataset = SafeImageFolder(
            os.path.join(self.data_root, domain),
            transform=self.transform
        )

        # Stratified split into train and validation
        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            stratify=[dataset.targets[i] for i in indices],
            random_state=SEED
        )

        selected_idx = train_idx if train else val_idx
        subset = Subset(dataset, selected_idx)

        # DataLoader
        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=4,
            pin_memory=True
        )
        return loader


class PACSDataset(BaseDomainDataset):
    """PACS: art_painting, cartoon, photo, sketch."""

    def __init__(self, data_root, transform, batch_size):
        DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
        super().__init__(data_root, DOMAINS, transform, batch_size)


class VLCSDataset(BaseDomainDataset):
    """VLCS: VOC2007, LabelMe, Caltech101, SUN09."""

    def __init__(self, data_root, transform, batch_size):
        DOMAINS = ["VOC2007", "LabelMe", "Caltech101", "SUN09"]
        super().__init__(data_root, DOMAINS, transform, batch_size)


class OfficeHomeDataset(BaseDomainDataset):
    """Office-Home: Art, Clipart, Product, Real_World."""

    def __init__(self, data_root, transform, batch_size):
        DOMAINS = ["Art", "Clipart", "Product", "Real_World"]
        super().__init__(data_root, DOMAINS, transform, batch_size)
