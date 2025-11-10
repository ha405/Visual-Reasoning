# models/pacs_net.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class PACSNet(nn.Module):
    """ResNet18-based network for PACS domain generalization."""
    
    def __init__(self, num_classes: int, hidden_sizes: list):
        """
        Args:
            num_classes: Number of output classes
            hidden_sizes: List of hidden layer sizes for the MLP head
        """
        super().__init__()
        
        # Load pretrained ResNet18 backbone
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the original classifier
        backbone_output_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Build MLP head
        self.fc_stack = self._make_mlp(backbone_output_dim, hidden_sizes)
        
        # Final classifier
        final_hidden_size = hidden_sizes[-1] if hidden_sizes else backbone_output_dim
        self.classifier = nn.Linear(final_hidden_size, num_classes)
    
    def _make_mlp(self, input_dim: int, hidden_sizes: list):
        """
        Create MLP with BatchNorm and ReLU activations.
        
        Args:
            input_dim: Input dimension
            hidden_sizes: List of hidden layer sizes
            
        Returns:
            nn.Sequential module containing the MLP
        """
        layers = []
        
        for h_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, h_size))
            layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.ReLU(inplace=True))
            input_dim = h_size
        
        return nn.Sequential(*layers)
    
    def forward(self, x, return_activations=False):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            return_activations: If True, return intermediate activations
            
        Returns:
            If return_activations is False: output logits
            If return_activations is True: (logits, activations_dict)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        if return_activations:
            activations = {}
            
            # Track activations through MLP layers
            h = features
            for i, module in enumerate(self.fc_stack):
                h = module(h)
                if isinstance(module, nn.ReLU):
                    # Store ReLU activations
                    layer_idx = i // 3  # Each block has Linear, BatchNorm, ReLU
                    activations[f'fc_{layer_idx}'] = h.clone()
            
            # Final classifier
            output = self.classifier(h)
            
            return output, activations
        else:
            # Standard forward pass
            h = self.fc_stack(features)
            output = self.classifier(h)
            return output
    
    def get_num_neurons_per_layer(self):
        """
        Get the number of neurons in each MLP layer.
        
        Returns:
            List of neuron counts per layer
        """
        neuron_counts = []
        
        for module in self.fc_stack:
            if isinstance(module, nn.Linear):
                neuron_counts.append(module.out_features)
        
        return neuron_counts
    
    def get_total_mlp_neurons(self):
        """
        Get the total number of neurons in the MLP head.
        
        Returns:
            Total neuron count
        """
        return sum(self.get_num_neurons_per_layer())
