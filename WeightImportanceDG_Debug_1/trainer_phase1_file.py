"""
Phase 1 Training: Domain-Specific Training with OBD Saliency Computation
Trains separate networks on each domain and computes OBD saliencies
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
from config_file import *
from models_file import DomainGeneralizationModel
from importance_file import HessianComputer, compute_obd_saliency


class Phase1Trainer:
    """
    Trainer for Phase 1: Domain-Specific Training
    Trains a network on a single domain and computes OBD saliencies
    """
    
    def __init__(self, model, device=DEVICE):
        """
        Initialize Phase 1 trainer
        
        Args:
            model (DomainGeneralizationModel): Model to train
            device: Device to train on (CPU/GPU)
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # Only optimize classifier parameters (ViT is frozen)
        self.optimizer = optim.Adam(
            self.model.classifier.parameters(),
            lr=PH1_LEARNING_RATE
        )
        
        # Storage for training history
        self.train_losses = []
        self.train_accuracies = []
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        # Keep feature extractor in eval mode (it's frozen)
        self.model.feature_extractor.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, dataloader, num_epochs=PH1_NUM_EPOCHS, domain_name="unknown"):
        """
        Train the model for multiple epochs on a single domain
        
        Args:
            dataloader: Training data loader
            num_epochs (int): Number of epochs to train
            domain_name (str): Name of domain being trained on
        
        Returns:
            dict: Training results including final weights and model
        """
        print(f"\n{'='*60}")
        print(f"Phase 1 Training on domain: {domain_name}")
        print(f"{'='*60}")
        
        # Training loop
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(dataloader)
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        # Record final weights after training
        final_weights = self.model.get_hidden_layer_weights()
        print(f"Recorded final weights after {num_epochs} epochs")
        
        results = {
            'domain': domain_name,
            'final_weights': final_weights,
            'train_losses': self.train_losses.copy(),
            'train_accuracies': self.train_accuracies.copy(),
            'model': self.model  # Keep reference to model for Hessian computation
        }
        
        print(f"Phase 1 training completed for {domain_name}")
        print(f"Final Loss: {train_loss:.4f}, Final Accuracy: {train_acc:.4f}")
        
        return results


def run_phase1_all_domains(dataset_handler, train_domains, save_results=True):
    """
    Run Phase 1 training on all specified domains with OBD saliency computation
    Each domain gets its own network initialized with the same random seed
    
    Args:
        dataset_handler (PACSDataset): Dataset handler for loading data
        train_domains (list): List of domains to train on (e.g., ['cartoon', 'photo', 'sketch'])
        save_results (bool): Whether to save saliency data
    
    Returns:
        dict: Results for all domains including OBD saliencies
    """
    print("\n" + "="*80)
    print("PHASE 1: DOMAIN-SPECIFIC TRAINING WITH OBD SALIENCY COMPUTATION")
    print(f"Training on {len(train_domains)} domains: {train_domains}")
    print("="*80 + "\n")
    
    all_results = {}
    all_saliencies = {}
    hessian_computer = HessianComputer(use_lm_approximation=True)
    
    for domain in train_domains:
        print(f"\n>>> Starting training on domain: {domain}")
        
        # Create fresh model for this domain
        model = DomainGeneralizationModel()
        
        # Get dataloader for this domain (use ALL data, no train/test split)
        dataloader = dataset_handler.get_single_domain_loader(
            domain=domain,
            batch_size=PH1_BATCH_SIZE,
            shuffle=True,
            train_split=None  # Use all data
        )
        
        # Train on this domain
        trainer = Phase1Trainer(model)
        results = trainer.train(
            dataloader=dataloader,
            num_epochs=PH1_NUM_EPOCHS,
            domain_name=domain
        )
        
        # ========================================================================
        # Compute OBD Saliencies
        # ========================================================================
        print(f"\n>>> Computing OBD saliencies for {domain}...")
        
        # Compute diagonal Hessian at convergence
        hessian_diagonal = hessian_computer.compute_diagonal_hessian(
            model=results['model'],
            dataloader=dataloader,
            device=DEVICE
        )
        
        # Get final trained weights
        final_weights = results['final_weights']
        
        # Compute OBD saliencies: s_k = 0.5 * h_kk * u_k²
        saliencies = compute_obd_saliency(final_weights, hessian_diagonal)
        
        print(f"Computed OBD saliencies for {domain}")
        
        # DEBUG: Check if Hessian has meaningful values
        print(f"\n  [DEBUG] Checking Hessian and Saliency values:")
        for layer_idx in range(4):
            h = hessian_diagonal[layer_idx]
            s = saliencies[layer_idx]
            print(f"    Layer {layer_idx}:")
            print(f"      Hessian - min: {h.min().item():.8f}, max: {h.max().item():.8f}, mean: {h.mean().item():.8f}")
            print(f"      Hessian zeros: {(h == 0).sum().item()}/100")
            print(f"      Hessian std: {h.std().item():.8f}")
            print(f"      Saliency - min: {s.min().item():.8f}, max: {s.max().item():.8f}, mean: {s.mean().item():.8f}")
        print()  # Extra newline for readability
        
        # Add saliency data to results
        results['hessian_diagonal'] = hessian_diagonal
        results['obd_saliencies'] = saliencies
        
        # Store for aggregation
        all_saliencies[domain] = saliencies
        
        # Save results for this domain
        all_results[domain] = results
        
        # Optionally save to disk
        if save_results:
            os.makedirs(IMPORTANCE_DIR, exist_ok=True)
            
            # Save saliencies and Hessian
            save_data = {
                'domain': domain,
                'final_weights': [w.cpu().numpy() for w in final_weights],
                'hessian_diagonal': [h.cpu().numpy() for h in hessian_diagonal],
                'obd_saliencies': [s.cpu().numpy() for s in saliencies]
            }
            
            save_path = os.path.join(IMPORTANCE_DIR, f'{domain}_obd_saliency.npz')
            np.savez(save_path, **save_data)
            print(f"Saved OBD saliency data for {domain} to {save_path}")
        
        print(f"<<< Completed training and saliency computation for {domain}\n")
    
    print("\n" + "="*80)
    print("PHASE 1 COMPLETE")
    print(f"Trained on {len(train_domains)} domains and computed OBD saliencies")
    print("="*80 + "\n")
    
    # Store aggregated saliencies in results
    all_results['aggregated_saliencies'] = all_saliencies
    
    return all_results