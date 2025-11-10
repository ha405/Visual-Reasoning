"""
Phase 2 Training: Cross-Domain Generalization with Weight Pruning
Prunes unimportant weights (sets to 0) and trains only important weights
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from config_file import *
from models_file import DomainGeneralizationModel


class Phase2Trainer:
    """
    Trainer for Phase 2: Cross-Domain Generalization with Pruning
    Prunes unimportant weights and trains only the important (non-pruned) weights
    """
    
    def __init__(self, model, importance_masks, device=DEVICE):
        """
        Initialize Phase 2 trainer with weight pruning
        
        Args:
            model (DomainGeneralizationModel): Model to train
            importance_masks (list): List of 4 binary mask tensors [10, 10]
                                     1 = important (keep), 0 = unimportant (prune)
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # Store importance masks
        self.importance_masks = [mask.to(device) for mask in importance_masks]
        
        # Apply pruning: set unimportant weights to 0
        self._prune_weights()
        
        # Setup optimizer (only important weights will update due to gradient masking)
        self.optimizer = optim.Adam(
            self.model.classifier.hidden_layers.parameters(),
            lr=PH2_LEARNING_RATE
        )
        
        # Storage for training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Track pruning statistics
        self._print_pruning_stats()
    
    def _prune_weights(self):
        """
        Prune (set to 0) all weights where mask = 0
        This is done once at initialization
        """
        print("\n>>> Applying weight pruning...")
        for layer_idx, layer in enumerate(self.model.classifier.hidden_layers):
            mask = self.importance_masks[layer_idx]
            # Set pruned weights to 0
            with torch.no_grad():
                layer.weight.data *= mask
        print("Pruned all unimportant weights (set to 0)")
        
        # DEBUG: Check pruning statistics
        print("\n  [DEBUG] Pruning verification:")
        for layer_idx, layer in enumerate(self.model.classifier.hidden_layers):
            total = layer.weight.numel()
            zeros = (layer.weight.data == 0).sum().item()
            nonzeros = total - zeros
            print(f"    Layer {layer_idx}: {zeros}/{total} pruned ({zeros/total*100:.1f}%), {nonzeros} active")
            # Check if any active weights
            active_weights = layer.weight.data[layer.weight.data != 0]
            if len(active_weights) > 0:
                print(f"      Active weight range: [{active_weights.min().item():.4f}, {active_weights.max().item():.4f}]")
    
    def _print_pruning_stats(self):
        """Print statistics about pruned vs non-pruned weights"""
        total_weights = 0
        pruned_weights = 0
        
        for mask in self.importance_masks:
            total_weights += mask.numel()
            pruned_weights += (mask == 0).sum().item()
        
        kept_weights = total_weights - pruned_weights
        pruning_ratio = (pruned_weights / total_weights) * 100
        
        print(f"\nPruning Statistics:")
        print(f"  Total weights: {total_weights}")
        print(f"  Pruned weights (set to 0): {pruned_weights} ({pruning_ratio:.1f}%)")
        print(f"  Kept weights (trainable): {kept_weights} ({100-pruning_ratio:.1f}%)")
    
    def _mask_gradients(self):
        """
        Mask gradients so pruned weights (mask=0) never update
        This ensures pruned weights stay at exactly 0 throughout training
        """
        for layer_idx, layer in enumerate(self.model.classifier.hidden_layers):
            if layer.weight.grad is not None:
                mask = self.importance_masks[layer_idx]
                # Zero out gradients for pruned weights
                layer.weight.grad *= mask
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch with gradient masking for pruned weights
        
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
        
        batch_count = 0
        for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # DEBUG: Print first batch details
            if batch_count == 0:
                print(f"\n    [DEBUG] First batch - input shape: {inputs.shape}, labels shape: {labels.shape}")
            
            batch_count += 1
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Mask gradients to ensure pruned weights don't update
            self._mask_gradients()
            
            # Update only non-pruned weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        # DEBUG: Check if weights changed during epoch
        print(f"\n    [DEBUG] After epoch - checking weight updates:")
        for layer_idx, layer in enumerate(self.model.classifier.hidden_layers):
            weight_norm = layer.weight.data.norm().item()
            num_zeros = (layer.weight.data == 0).sum().item()
            print(f"      Layer {layer_idx}: Weight norm: {weight_norm:.4f}, Still {num_zeros}/100 pruned")
        
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
        Train the pruned model for multiple epochs
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs (int): Number of epochs to train
            test_domain (str): Name of held-out test domain
        
        Returns:
            dict: Training results
        """
        print(f"\n{'='*60}")
        print(f"Phase 2 Training with Pruning (LODO - Test Domain: {test_domain})")
        print(f"{'='*60}")
        print(f"Training only the top {IMPORTANCE_THRESHOLD_PERCENTILE}% important weights")
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_acc = self.evaluate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
        
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
    Run Phase 2 training in LODO configuration with weight pruning
    Train on all domains except test_domain, using importance masks from Phase 1
    
    Args:
        dataset_handler (PACSDataset): Dataset handler for loading data
        test_domain (str): Domain to hold out for testing
        importance_masks (list): Binary importance masks from Phase 1 (top 60%)
    
    Returns:
        dict: Results including test accuracy
    """
    print(f"\n>>> Phase 2: Leave-One-Domain-Out with Pruning (Test Domain: {test_domain})")
    
    # Get LODO dataloaders
    train_loader, val_loader, test_loader = dataset_handler.get_lodo_loaders(
        test_domain=test_domain,
        batch_size=PH2_BATCH_SIZE
    )
    
    # Create fresh model for Phase 2
    model = DomainGeneralizationModel()
    
    # Initialize trainer with importance masks (pruning happens in __init__)
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