from torchvision import transforms


def to_3_channels(t):
    if t.shape[0] == 1:
        return t.repeat(3, 1, 1)
    return t


def get_transforms(dataset_cfg):
    img_size = dataset_cfg.get("image_size", 224)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    normalize = transforms.Normalize(mean=mean, std=std)
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Lambda(to_3_channels),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(to_3_channels),
        normalize,
    ])
    
    return {"train": train_transform, "val": val_transform}
