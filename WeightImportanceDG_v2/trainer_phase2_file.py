"""
Phase 2 Training: Cross-Domain Generalization with Importance-Based Learning
Uses importance masks from Phase 1 to apply differential learning rates
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from config import *
from models import DomainGeneralizationModel
from importance import aggregate_importance_masks


class Phase2Trainer:
    """
    Trainer for Phase 2: Cross-Domain Generalization
    Applies importance masks to use different learning rates for important vs unimportant weights
    """
    
    def __init__(self, model, importance_masks, device=DEVICE):
        """
        Initialize Phase 2 trainer with importance-based learning
        
        Args:
            model (DomainGeneralizationModel): Model to train
            importance_masks (list): List of 4 binary mask tensors [10, 10]
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # Store importance masks
        self.importance_masks = [mask.to(device) for mask in importance_masks]
        
        # Setup optimizers with different learning rates
        self._setup_optimizers()
        
        # Storage for training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Track current epoch for freezing unimportant weights
        self.current_epoch = 0
        self.unimportant_frozen = False
    
    def _setup_optimizers(self):
        """
        Setup separate optimizers for important and unimportant weights
        Important weights: normal learning rate
        Unimportant weights: smaller learning rate with decay
        """
        # Only optimize classifier parameters (ViT is frozen)
        important_params = []
        unimportant_params = []
        
        # Separate parameters based on importance masks
        for layer_idx, layer in enumerate(self.model.classifier.hidden_layers):
            mask = self.importance_masks[layer_idx]
            
            # For each weight in the layer, determine if it's important
            # We need to manually track which weights are important
            important_params.append({
                'params': [layer.weight],
                'lr': PH2_LR_IMPORTANT,
                'layer_idx': layer_idx,
                'is_important': True
            })
        
        # Create optimizer for important weights
        self.optimizer_important = optim.Adam(
            [p for p in important_params if p['is_important']],
            lr=PH2_LR_IMPORTANT
        )
        
        # Create optimizer for unimportant weights with decay
        self.optimizer_unimportant = optim.Adam(
            self.model.classifier.hidden_layers.parameters(),
            lr=PH2_LR_UNIMPORTANT_START
        )
        
        # Learning rate scheduler for unimportant weights (exponential decay)
        self.scheduler_unimportant = optim.lr_scheduler.ExponentialLR(
            self.optimizer_unimportant,
            gamma=PH2_LR_DECAY_RATE
        )
    
    def _apply_masked_gradients(self):
        """
        Apply importance masks to gradients
        Important weights (mask=1): use normal gradients
        Unimportant weights (mask=0): use reduced gradients or freeze
        """
        if self.unimportant_frozen:
            # After freeze epoch, zero out gradients for unimportant weights
            for layer_idx, layer in enumerate(self.model.classifier.hidden_layers):
                if layer.weight.grad is not None:
                    mask = self.importance_masks[layer_idx]
                    # Zero gradients for unimportant weights (mask=0)
                    layer.weight.grad *= mask
        else:
            # Before freeze epoch, scale gradients based on importance
            for layer_idx, layer in enumerate(self.model.classifier.hidden_layers):
                if layer.weight.grad is not None:
                    mask = self.importance_masks[layer_idx]
                    # Important weights (mask=1): keep full gradient
                    # Unimportant weights (mask=0): scale gradient by smaller LR ratio
                    scale_factor = mask + (1 - mask) * (PH2_LR_UNIMPORTANT_START / PH2_LR_IMPORTANT)
                    layer.weight.grad *= scale_factor
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch with importance-based learning
        
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
        
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {self.current_epoch+1}", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer_important.zero_grad()
            if not self.unimportant_frozen:
                self.optimizer_unimportant.zero_grad()
            
            loss.backward()
            
            # Apply importance masks to gradients
            self._apply_masked_gradients()
            
            # Update weights
            self.optimizer_important.step()
            if not self.unimportant_frozen:
                self.optimizer_unimportant.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        """
        Evaluate model on validation/test set
        
        Args:
            dataloader: Validation/test data loader
        
        Returns:
            float: Accuracy
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy
    
    def train(self, train_loader, val_loader, num_epochs=PH2_NUM_EPOCHS, test_domain="unknown"):
        """
        Train the model for multiple epochs with importance-based learning
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs (int): Number of epochs to train
            test_domain (str): Name of held-out test domain
        
        Returns:
            dict: Training results
        """
        print(f"\n{'='*60}")
        print(f"Phase 2 Training (LODO - Test Domain: {test_domain})")
        print(f"{'='*60}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Freeze unimportant weights after specified epoch
            if epoch >= PH2_FREEZE_EPOCH and not self.unimportant_frozen:
                self.unimportant_frozen = True
                print(f"\n>>> Freezing unimportant weights (mask=0) after epoch {epoch}")
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_acc = self.evaluate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Decay learning rate for unimportant weights (if not frozen)
            if not self.unimportant_frozen:
                self.scheduler_unimportant.step()
                current_lr = self.scheduler_unimportant.get_last_lr()[0]
                print(f"Epoch [{epoch+1}/{num_epochs}] - "
                      f"Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, "
                      f"Val Acc: {val_acc:.4f}, "
                      f"Unimportant LR: {current_lr:.6f}")
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}] - "
                      f"Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, "
                      f"Val Acc: {val_acc:.4f} "
                      f"[Unimportant weights FROZEN]")
        
        results = {
            'test_domain': test_domain,
            'train_losses': self.train_losses.copy(),
            'train_accuracies': self.train_accuracies.copy(),
            'val_accuracies': self.val_accuracies.copy()
        }
        
        print(f"\nPhase 2 training completed")
        print(f"Final Train Acc: {train_acc:.4f}, Final Val Acc: {val_acc:.4f}")
        
        return results


def run_phase2_lodo(dataset_handler, test_domain, importance_masks):
    """
    Run Phase 2 training in LODO configuration
    Train on all domains except test_domain, using importance masks from Phase 1
    
    Args:
        dataset_handler (PACSDataset): Dataset handler for loading data
        test_domain (str): Domain to hold out for testing
        importance_masks (list): Aggregated importance masks from Phase 1
    
    Returns:
        dict: Results including test accuracy
    """
    print(f"\n>>> Phase 2: Leave-One-Domain-Out (Test Domain: {test_domain})")
    
    # Get LODO dataloaders
    train_loader, val_loader, test_loader = dataset_handler.get_lodo_loaders(
        test_domain=test_domain,
        batch_size=PH2_BATCH_SIZE
    )
    
    # Create fresh model for Phase 2
    model = DomainGeneralizationModel()
    
    # Initialize trainer with importance masks
    trainer = Phase2Trainer(model, importance_masks)
    
    # Train
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=PH2_NUM_EPOCHS,
        test_domain=test_domain
    )
    
    # Test on held-out domain
    test_acc = trainer.evaluate(test_loader)
    results['test_accuracy'] = test_acc
    
    print(f"\n{'='*60}")
    print(f"Test Accuracy on {test_domain}: {test_acc:.4f}")
    print(f"{'='*60}\n")
    
    return results