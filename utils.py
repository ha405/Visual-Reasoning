import torch
import torch.nn as nn
from tqdm.notebook import tqdm

def apply_mask(model, mask):
    if not mask:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.data.mul_(mask[name])

def evaluate(model, loader, device, mask=None):
    model.to(device)
    apply_mask(model, mask)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            inputs, targets = batch[0], batch[1]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_DI(model, source_loader, optimizer, device, epoch, alpha=1.0, mask=None):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='none')
    pbar = tqdm(source_loader, desc=f"Epoch {epoch+1} Training", leave=False)
    for inputs, targets, domain_ids in pbar:
        optimizer.zero_grad()
        inputs, targets, domain_ids = inputs.to(device), targets.to(device), domain_ids.to(device)
        outputs = model(inputs)
        per_sample_losses = criterion(outputs, targets)
        unique_domains = torch.unique(domain_ids)
        domain_losses = []
        for d_id in unique_domains:
            domain_mask = (domain_ids == d_id)
            if torch.any(domain_mask):
                domain_loss = per_sample_losses[domain_mask].mean()
                domain_losses.append(domain_loss)
        if len(domain_losses) > 0:
            loss_tensor = torch.stack(domain_losses)
            mean_loss = loss_tensor.mean()
            variance_loss = loss_tensor.var() if len(loss_tensor) > 1 else torch.tensor(0.0).to(device)
            total_loss = mean_loss + alpha * variance_loss
            total_loss.backward()
            optimizer.step()
            apply_mask(model, mask)
            pbar.set_postfix({"Mean Loss": f"{mean_loss.item():.4f}", "Var Loss": f"{variance_loss.item():.4f}"})

def train_vanilla(model, train_loader, optimizer, device, epoch, mask=None):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Vanilla Training", leave=False)
    for batch in pbar:
        inputs, targets = batch[0], batch[1]
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        apply_mask(model, mask)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

def train_SFT(model, source_loader, optimizer, device, epoch, alpha=1.0, mask=None):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(source_loader, desc=f"Epoch {epoch+1} Train2 (Normal)", leave=False)

    for inputs, targets, domain_ids in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        apply_mask(model, mask)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
