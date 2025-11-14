import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

class DomainDataset(Dataset):
    def __init__(self, base_dataset, domain_id):
        self.base_dataset = base_dataset
        self.domain_id = domain_id

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        return img, label, self.domain_id

def get_pacs_dataloaders(data_dir, source_domains, target_domain, batch_size, num_workers=2, combine_sources=True):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    source_datasets = []
    print(f"Creating source datasets for: {source_domains}")
    for i, domain in enumerate(source_domains):
        domain_path = os.path.join(data_dir, domain)
        if not os.path.isdir(domain_path):
            raise FileNotFoundError(f"Source domain directory not found: {domain_path}")
        
        base_dataset = datasets.ImageFolder(domain_path, transform=train_transform)
        domain_dataset = DomainDataset(base_dataset, domain_id=i)
        source_datasets.append(domain_dataset)
        print(f"  - Domain '{domain}' (ID {i}) loaded with {len(base_dataset)} images.")

    if combine_sources:
        concat_source_dataset = ConcatDataset(source_datasets)
        source_loader = DataLoader(
            concat_source_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        print(f"Combined source dataloader created with {len(concat_source_dataset)} total images.")
        final_source_loaders = source_loader
    else:
        source_loaders = []
        for ds in source_datasets:
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                pin_memory=True, drop_last=True
            )
            source_loaders.append(loader)
        print(f"Created {len(source_loaders)} separate source dataloaders.")
        final_source_loaders = source_loaders

    print(f"Creating target dataloader for: {target_domain}")
    target_path = os.path.join(data_dir, target_domain)
    target_dataset = datasets.ImageFolder(target_path, transform=val_transform)
    target_loader = DataLoader(
        target_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    print(f"  - Domain '{target_domain}' loaded with {len(target_dataset)} images.")

    class_to_idx = target_dataset.class_to_idx
    return final_source_loaders, target_loader, class_to_idx
