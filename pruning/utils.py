import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset


def get_layer(model, name):
    return dict(model.named_modules())[name]


def combine_source_loaders(source_loaders, batch_size, num_workers):
    datasets = [loader.dataset for loader in source_loaders]
    combined_dataset = ConcatDataset(datasets)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def create_calibration_loaders(loaders, num_samples, is_target=False):
    if num_samples is None:
        return loaders
    
    calib_loaders = []
    if not isinstance(loaders, list):
        loaders = [loaders]

    print(f"Creating calibration loaders with {num_samples} samples each...")
    for loader in loaders:
        dataset = loader.dataset
        num_dataset_samples = len(dataset)
        samples_to_take = min(num_samples, num_dataset_samples)
        
        random_indices = torch.randperm(num_dataset_samples)[:samples_to_take].tolist()
        subset = Subset(dataset, random_indices)
        calib_loader = DataLoader(subset, batch_size=loader.batch_size, shuffle=False, num_workers=loader.num_workers)
        calib_loaders.append(calib_loader)
        
    return calib_loaders[0] if is_target else calib_loaders


def apply_mask(model, mask):
    if not mask:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.data.mul_(mask[name])
