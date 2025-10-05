from torchvision import transforms
import torch

def get_transforms(img_size=224, augment=True, use_imagenet_norm=True):
    if use_imagenet_norm:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, test_transform


def prepare_batch(batch, device):
    images, labels = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
