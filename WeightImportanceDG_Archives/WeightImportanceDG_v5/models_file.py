"""
Neural Network Models for Domain Generalization
- ViTFeatureExtractor: Frozen ViT for extracting image embeddings
- FeedforwardClassifier: 4-layer feedforward network with 10 neurons per layer
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor
from config_file import *


class ViTFeatureExtractor(nn.Module):
    """
    Vision Transformer (ViT) feature extractor
    This module is frozen and only used to extract embeddings from images
    """

    def __init__(self, model_name=VIT_MODEL):
        """
        Initialize ViT feature extractor

        Args:
            model_name (str): HuggingFace model identifier for ViT
        """
        super(ViTFeatureExtractor, self).__init__()

        # Load pretrained ViT model
        self.vit = ViTModel.from_pretrained(model_name)

        # Freeze all ViT parameters (we don't train the feature extractor)
        for param in self.vit.parameters():
            param.requires_grad = False

        self.vit.eval()  # Set to evaluation mode

    def forward(self, x):
        """
        Extract features from images

        Args:
            x (torch.Tensor): Input images [batch_size, 3, 224, 224]

        Returns:
            torch.Tensor: Image embeddings [batch_size, 768]
        """
        with torch.no_grad():  # No gradient computation for frozen ViT
            outputs = self.vit(pixel_values=x)
            # Use [CLS] token embedding as image representation
            embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        return embeddings


class FeedforwardClassifier(nn.Module):
    """
    Feedforward classifier network with 4 hidden layers (10 neurons each)
    Architecture: 768 -> 10 -> 10 -> 10 -> 10 -> 7
    Only the 10x10 weight matrices between hidden layers are tracked for importance
    """

    def __init__(self, input_dim=VIT_EMBEDDING_DIM, num_classes=NUM_CLASSES):
        """
        Initialize feedforward classifier

        Args:
            input_dim (int): Input dimension (ViT embedding size = 768)
            num_classes (int): Number of output classes (7 for PACS)
        """
        super(FeedforwardClassifier, self).__init__()

        # Layer 0: Input projection (768 -> 10) - NOT tracked for importance
        self.input_layer = nn.Linear(input_dim, HIDDEN_LAYER_SIZES[0])

        # Layers 1-4: Hidden layers (10 -> 10) - TRACKED for importance
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(HIDDEN_LAYER_SIZES[i], HIDDEN_LAYER_SIZES[i])
                for i in range(NUM_HIDDEN_LAYERS)
            ]
        )

        # Output layer: (10 -> 7) - NOT tracked for importance
        self.output_layer = nn.Linear(HIDDEN_LAYER_SIZES[-1], num_classes)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x (torch.Tensor): Input embeddings [batch_size, 768]

        Returns:
            torch.Tensor: Logits [batch_size, 7]
        """
        # Input projection
        x = self.relu(self.input_layer(x))

        # Hidden layers with ReLU activation
        for layer in self.hidden_layers:
            x = self.relu(layer(x))

        # Output layer (no activation, return logits)
        x = self.output_layer(x)

        return x

    def get_hidden_layer_weights(self):
        """
        Get weights from the 4 hidden layers (10x10 matrices)
        These are the weights we track for importance computation

        Returns:
            list: List of 4 weight tensors, each of shape [10, 10]
        """
        weights = []
        for layer in self.hidden_layers:
            weights.append(
                layer.weight.data.clone()
            )  # [out_features, in_features] = [10, 10]
        return weights

    def set_hidden_layer_weights(self, weights):
        """
        Set weights for the 4 hidden layers

        Args:
            weights (list): List of 4 weight tensors, each of shape [10, 10]
        """
        assert (
            len(weights) == NUM_HIDDEN_LAYERS
        ), f"Expected {NUM_HIDDEN_LAYERS} weight matrices"

        for i, layer in enumerate(self.hidden_layers):
            layer.weight.data = weights[i].clone()


class DomainGeneralizationModel(nn.Module):
    """
    Complete model combining frozen ViT feature extractor and trainable classifier
    """

    def __init__(self):
        """Initialize the complete model"""
        super(DomainGeneralizationModel, self).__init__()

        self.feature_extractor = ViTFeatureExtractor()
        self.classifier = FeedforwardClassifier()

    def forward(self, x):
        """
        Forward pass through the complete model

        Args:
            x (torch.Tensor): Input images [batch_size, 3, 224, 224]

        Returns:
            torch.Tensor: Class logits [batch_size, 7]
        """
        # Extract features with frozen ViT
        features = self.feature_extractor(x)

        # Classify with trainable feedforward network
        logits = self.classifier(features)

        return logits

    def get_hidden_layer_weights(self):
        """Get weights from classifier's hidden layers"""
        return self.classifier.get_hidden_layer_weights()

    def set_hidden_layer_weights(self, weights):
        """Set weights for classifier's hidden layers"""
        self.classifier.set_hidden_layer_weights(weights)
