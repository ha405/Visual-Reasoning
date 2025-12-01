import torch
from tqdm import tqdm

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_grqo_loss = 0.0
    correct = 0
    total_samples = 0

    for batch_idx, (images, labels, domain_labels) in enumerate(train_loader):
        images, labels, domain_labels = images.to(device), labels.to(device), domain_labels.to(device)
        
        optimizer.zero_grad()

        # Forward pass
        output = model(images, labels, domain_labels)

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


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = total_cls_loss = total_grqo_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():  # Fixed: use no_grad instead of enable_grad
        for batch_idx, (images, labels, domain_labels) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            domain_labels = domain_labels.to(device, non_blocking=True)

            output = model(images, labels, domain_labels)
            loss = output['loss']
            cls_loss = output.get('cls_loss', torch.tensor(0.0, device=device))
            grqo_loss = output.get('grqo_loss', torch.tensor(0.0, device=device))
            preds = output['preds']

            total_loss += float(loss.item()) * images.size(0)
            total_cls_loss += float(cls_loss.item()) * images.size(0)
            total_grqo_loss += float(grqo_loss.item()) * images.size(0)
            correct += int((preds == labels).sum().item())
            total_samples += labels.size(0)

            # Periodic CUDA synchronization and cleanup
            if (batch_idx + 1) % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

    avg_loss = total_loss / total_samples if total_samples else 0.0
    avg_cls_loss = total_cls_loss / total_samples if total_samples else 0.0
    avg_grqo_loss = total_grqo_loss / total_samples if total_samples else 0.0
    accuracy = correct / total_samples if total_samples else 0.0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return avg_loss, avg_cls_loss, avg_grqo_loss, accuracy


