"""
Phase 1 Training: Domain-Specific Training
Trains separate networks on each domain to discover weight importance patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
from config import *
from models import DomainGeneralizationModel
from importance import ImportanceCalculator


class Phase1Trainer:
    """
    Trainer for Phase 1: Domain-Specific Training
    Trains a network on a single domain and tracks weight changes
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
            dict: Training results including initial and final weights
        """
        print(f"\n{'='*60}")
        print(f"Phase 1 Training on domain: {domain_name}")
        print(f"{'='*60}")
        
        # Record initial weights before training
        initial_weights = self.model.get_hidden_layer_weights()
        print(f"Recorded initial weights (4 layers of 10x10 matrices)")
        
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
            'initial_weights': initial_weights,
            'final_weights': final_weights,
            'train_losses': self.train_losses.copy(),
            'train_accuracies': self.train_accuracies.copy()
        }
        
        print(f"Phase 1 training completed for {domain_name}")
        print(f"Final Loss: {train_loss:.4f}, Final Accuracy: {train_acc:.4f}")
        
        return results


def run_phase1_all_domains(dataset_handler, train_domains, save_results=True):
    """
    Run Phase 1 training on all specified domains
    Each domain gets its own network initialized with the same random seed
    
    Args:
        dataset_handler (PACSDataset): Dataset handler for loading data
        train_domains (list): List of domains to train on (e.g., ['art_painting', 'cartoon', 'photo'])
        save_results (bool): Whether to save importance data
    
    Returns:
        dict: Results for all domains including importance matrices
    """
    print("\n" + "="*80)
    print("PHASE 1: DOMAIN-SPECIFIC TRAINING")
    print(f"Training on {len(train_domains)} domains: {train_domains}")
    print("="*80 + "\n")
    
    all_results = {}
    importance_calc = ImportanceCalculator()
    
    # Store the shared initial weights (same random initialization for all domains)
    shared_initial_weights = None
    
    for domain in train_domains:
        print(f"\n>>> Starting training on domain: {domain}")
        
        # Create fresh model for this domain
        model = DomainGeneralizationModel()
        
        # Store shared initial weights from first model
        if shared_initial_weights is None:
            shared_initial_weights = model.get_hidden_layer_weights()
            print("Stored shared initial weights (same for all domains)")
        else:
            # Ensure all models start with same weights
            model.set_hidden_layer_weights(shared_initial_weights)
            print("Set model to use shared initial weights")
        
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
        
        # Compute importance matrices
        importance_matrices = importance_calc.compute_importance(
            results['initial_weights'],
            results['final_weights']
        )
        
        # Normalize importance globally
        normalized_importance = importance_calc.normalize_importance_global(importance_matrices)
        
        # Create binary masks (top 60% = important)
        binary_masks = importance_calc.create_binary_mask(normalized_importance)
        
        # Add importance data to results
        results['importance_matrices'] = importance_matrices
        results['normalized_importance'] = normalized_importance
        results['binary_masks'] = binary_masks
        
        # Save results for this domain
        all_results[domain] = results
        
        # Optionally save to disk
        if save_results:
            import os
            os.makedirs(IMPORTANCE_DIR, exist_ok=True)
            importance_calc.save_importance_data(
                domain_name=domain,
                initial_weights=results['initial_weights'],
                final_weights=results['final_weights'],
                importance_matrices=importance_matrices,
                normalized_importance=normalized_importance,
                binary_masks=binary_masks,
                save_dir=IMPORTANCE_DIR
            )
        
        print(f"<<< Completed training on domain: {domain}\n")
    
    print("\n" + "="*80)
    print("PHASE 1 COMPLETE")
    print(f"Trained on {len(train_domains)} domains")
    print("="*80 + "\n")
    
    return all_results