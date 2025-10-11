from torchvision import transforms
import torch

def to_3_channels(t):
    if t.shape[0] == 1:
        return t.repeat(3, 1, 1)
    return t

def get_transforms(img_size=224, augment=True, use_imagenet_norm=True, keep_channels=False):
    if use_imagenet_norm:
        mean, std = [0.485,0.456,0.406],[0.229,0.224,0.225]
    else:
        mean, std = [0.5,0.5,0.5],[0.5,0.5,0.5]

    if keep_channels and len(mean)==3:
        mean,std=[mean[0]],[std[0]]

    normalize = transforms.Normalize(mean=mean,std=std)

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.RandomResizedCrop(img_size, scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4,0.4,0.4),
            transforms.ToTensor(),
            transforms.Lambda(to_3_channels),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Lambda(to_3_channels),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Lambda(to_3_channels),
        normalize,
    ])

    return train_transform,test_transform




def prepare_batch(batch, device):
    images, labels = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
