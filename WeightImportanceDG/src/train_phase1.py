# src/train_phase1.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import os
import copy

from . import config
from .models import ViTFeatureExtractor, FeedForwardHead
from .dataset import PACSDataset, TRANSFORM
from .phase1_utils import save_weights, get_param_names

def run_phase1():
    """
    Executes the full Phase 1 training pipeline.
    1. Initializes a shared FFN and saves its initial weights.
    2. For each specified domain, trains a copy of the FFN on a single random class.
    3. Saves the final weights for each domain-specific FFN.
    """
    print("--- Starting Phase 1 Training ---")
    
    # Set seed for reproducibility
    torch.manual_seed(config.SEED)
    random.seed(config.SEED)

    # 1. Initialize models
    vit_extractor = ViTFeatureExtractor().to(config.DEVICE)
    vit_extractor.eval() # Ensure it's in eval mode

    ffn_head = FeedForwardHead(
        embedding_dim=config.EMBEDDING_DIM,
        hidden=config.FFN_HIDDEN,
        layers=config.FFN_LAYERS,
        dropout=config.DROPOUT,
        batch_norm=config.BATCH_NORM,
        num_classes=config.NUM_CLASSES
    )
    
    # Save initial shared weights
    initial_weights_dir = os.path.join(config.PHASE1_WEIGHTS_DIR, "initial")
    save_weights(ffn_head, initial_weights_dir)
    print(f"Saved initial shared weights to {initial_weights_dir}")

    pacs_dataset = PACSDataset(config.DATA_ROOT, config.DOMAINS, TRANSFORM)
    
    # 2. Loop through domains and train
    for domain in config.PH1_DOMAINS:
        print(f"\n--- Training on domain: {domain} ---")
        
        # Randomly select one class
        chosen_class = random.randint(0, config.NUM_CLASSES - 1)
        print(f"Randomly selected class: {chosen_class}")
        with open(os.path.join(config.RESULTS_DIR, f"phase1_{domain}_chosen_class.txt"), "w") as f:
            f.write(str(chosen_class))

        # Get dataloader for the single class
        dataloader = pacs_dataset.get_single_class_dataloader(
            domain, chosen_class, batch_size=config.PH1_BATCH_SIZE
        )
        
        # Create a fresh copy of the FFN for this domain and load initial weights
        domain_ffn = copy.deepcopy(ffn_head).to(config.DEVICE)
        
        optimizer = optim.AdamW(domain_ffn.parameters(), lr=config.PH1_LR)
        criterion = nn.CrossEntropyLoss()
        
        domain_ffn.train()
        for epoch in range(config.PH1_EPOCHS):
            total_loss = 0
            for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.PH1_EPOCHS}"):
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                
                with torch.no_grad():
                    embeddings = vit_extractor(images)
                
                optimizer.zero_grad()
                logits = domain_ffn(embeddings)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Domain {domain} | Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        # 3. Save final weights for this domain
        final_weights_dir = os.path.join(config.PHASE1_WEIGHTS_DIR, "final", f"domain_{domain}")
        save_weights(domain_ffn, final_weights_dir)
        print(f"Saved final weights for domain {domain} to {final_weights_dir}")

    print("\n--- Phase 1 Training Complete ---")