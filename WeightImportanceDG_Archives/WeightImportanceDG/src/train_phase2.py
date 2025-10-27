# src/train_phase2.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import json

from . import config
from .models import ViTFeatureExtractor, FeedForwardHead
from .dataset import PACSDataset, TRANSFORM, get_lodo_dataloaders
from .phase1_utils import get_param_names

def evaluate(vit_extractor, ffn_head, dataloader):
    """Evaluates the model on a given dataloader."""
    vit_extractor.eval()
    ffn_head.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            embeddings = vit_extractor(images)
            logits = ffn_head(embeddings)
            _, preds = torch.max(logits, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples

def run_phase2_lodo(train_domains, test_domain):
    """
    Executes Phase 2 Leave-One-Domain-Out training for a given split.
    """
    print(f"\n--- Starting Phase 2 LODO ---")
    print(f"Training on: {train_domains}, Testing on: {test_domain}")
    
    # 1. Load consensus masks
    consensus_masks = {}
    ffn_temp = FeedForwardHead(config.EMBEDDING_DIM)  # Temp model to get param names and shapes
    param_names = get_param_names(ffn_temp)
    
    # Build a dict of temp params to inspect shapes if needed
    temp_params = dict(ffn_temp.named_parameters())

    for name in param_names:
        mask_path = os.path.join(
            config.PHASE1_IMPORTANCE_DIR, "consensus", 
            f"{name.replace('.', '_')}_binary.npy"
        )

        if os.path.exists(mask_path):
            try:
                mask_np = np.load(mask_path)
                mask = torch.from_numpy(mask_np).to(config.DEVICE)
            except Exception as e:
                # If file exists but failed to load for some reason, fallback to ones and warn
                print(f"Warning: failed to load mask for '{name}' ({e}). Using all-ones mask.")
                if name in temp_params:
                    mask = torch.ones_like(temp_params[name].data).to(config.DEVICE)
                else:
                    # safety fallback: scalar 1
                    mask = torch.tensor(1.0, device=config.DEVICE)
        else:
            # Mask file not found -> create an all-ones mask of the same shape as the parameter
            if name in temp_params:
                mask = torch.ones_like(temp_params[name].data).to(config.DEVICE)
                print(f"Warning: mask not found for '{name}'. Using all-ones mask (trainable).")
            else:
                # shouldn't normally happen; fallback to scalar 1
                mask = torch.tensor(1.0, device=config.DEVICE)
                print(f"Warning: '{name}' not present in temp model; using scalar-1 fallback mask.")

        consensus_masks[name] = mask

    # 2. Initialize models
    vit_extractor = ViTFeatureExtractor().to(config.DEVICE)
    ffn_head = FeedForwardHead(
        embedding_dim=config.EMBEDDING_DIM,
        hidden=config.FFN_HIDDEN,
        layers=config.FFN_LAYERS,
        dropout=config.DROPOUT,
        batch_norm=config.BATCH_NORM,
        num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)
    
    # 3. Prepare dataloaders
    pacs_dataset = PACSDataset(config.DATA_ROOT, config.DOMAINS, TRANSFORM)
    train_loader, val_loader, test_loader = get_lodo_dataloaders(
        pacs_dataset, train_domains, test_domain, config.PH2_BATCH_SIZE
    )
    
    # 4. Set up optimizer and loss
    optimizer = optim.AdamW(ffn_head.parameters(), lr=config.PH2_LR_IMPORTANT)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training loop with gradient masking
    history = {'train_loss': [], 'val_acc': []}
    for epoch in range(config.PH2_EPOCHS):
        ffn_head.train()
        
        # Calculate scaling factor for non-important weights
        nonimp_scale = max(0.0, 1.0 - (epoch / config.PH2_FREEZE_EPOCH))
        
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.PH2_EPOCHS}"):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            with torch.no_grad():
                embeddings = vit_extractor(images)
            
            optimizer.zero_grad()
            logits = ffn_head(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            
            # --- Apply gradient masking ---
            with torch.no_grad():
                for name, param in ffn_head.named_parameters():
                    if param.grad is None:
                        continue
                    
                    # If we don't have a mask for this parameter, skip masking (leave gradients intact)
                    if name not in consensus_masks:
                        # Option A: skip masking (recommended for BN, biases, etc.)
                        continue

                    mask = consensus_masks[name]
                    # ensure mask is float and same device; broadcasting is OK if scalar fallback used
                    imp_mask = mask.float().to(param.grad.device)
                    nonimp_mask = 1.0 - imp_mask
                    
                    # Scale gradients: keep important gradients, scale non-important by nonimp_scale
                    param.grad *= (imp_mask + nonimp_mask * nonimp_scale)

            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(vit_extractor, ffn_head, val_loader)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Non-Imp Scale: {nonimp_scale:.2f}")

    # 6. Final evaluation on the held-out test domain
    test_acc = evaluate(vit_extractor, ffn_head, test_loader)
    print(f"--- Final Test Accuracy on {test_domain}: {test_acc:.4f} ---")
    
    # 7. Save results
    result_dir = os.path.join(config.PHASE2_RESULTS_DIR, f"lodo_{test_domain}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Save model
    torch.save(ffn_head.state_dict(), os.path.join(result_dir, "model_final.pt"))
    
    # Save metrics
    metrics = {
        'test_domain': test_domain,
        'final_test_accuracy': test_acc,
        'history': history
    }
    with open(os.path.join(result_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return test_acc
