import torch
from tqdm import tqdm

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_grqo_loss = 0.0
    correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # Forward pass
        output = model(images, labels)

        # Extract losses
        loss = output['loss']
        cls_loss = output['cls_loss']
        grqo_loss = output['grqo_loss']
        preds = output['preds']

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item() * images.size(0)
        total_cls_loss += cls_loss.item() * images.size(0)
        total_grqo_loss += grqo_loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # Logging
        # if batch_idx % 100 == 0:
        #     print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, '
        #           f'Cls: {cls_loss.item():.4f}, GRQO: {grqo_loss.item():.4f}')

    avg_loss = total_loss / total_samples
    avg_cls_loss = total_cls_loss / total_samples
    avg_grqo_loss = total_grqo_loss / total_samples
    accuracy = correct / total_samples

    return avg_loss, avg_cls_loss, avg_grqo_loss, accuracy


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_grqo_loss = 0.0
    correct = 0
    total_samples = 0

    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(True):
                
                output = model(images, labels)
                loss = output['loss']
                cls_loss = output['cls_loss']
                grqo_loss = output['grqo_loss']
                preds = output['preds']

        total_loss += loss.item() * images.size(0)
        total_cls_loss += cls_loss.item() * images.size(0)
        total_grqo_loss += grqo_loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_cls_loss = total_cls_loss / total_samples
    avg_grqo_loss = total_grqo_loss / total_samples
    accuracy = correct / total_samples

    return avg_loss, avg_cls_loss, avg_grqo_loss, accuracy
