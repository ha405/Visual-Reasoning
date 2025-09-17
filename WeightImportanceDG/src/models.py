# src/models.py

import torch
import torch.nn as nn
from transformers import ViTModel
from . import config

class ViTFeatureExtractor(nn.Module):
    """
    A wrapper for the ViT model from HuggingFace to act as a frozen feature extractor.
    """
    def __init__(self, model_name=config.VIT_MODEL_NAME):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        # Freeze all parameters of the ViT model
        for param in self.vit.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # We only need the features of the [CLS] token
        outputs = self.vit(x)
        return outputs.last_hidden_state[:, 0, :]

class FeedForwardHead(nn.Module):
    """
    Custom FFN head as specified.
    embedding_dim -> 10 -> 10 -> 10 -> 10 -> num_classes
    """
    def __init__(self, embedding_dim, hidden=10, layers=4, dropout=0.2, batch_norm=True, num_classes=7):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if batch_norm else None
        
        # Build hidden layers
        for i in range(layers):
            in_dim = embedding_dim if i == 0 else hidden
            self.layers.append(nn.Linear(in_dim, hidden))
            if batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden))
        
        # Final classifier layer
        self.classifier = nn.Linear(hidden, num_classes)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, linear in enumerate(self.layers):
            x = linear(x)
            if self.bns:
                # BatchNorm expects (N, C, L) or (N, C), handle (N, L) case
                if x.dim() == 2:
                    x = self.bns[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        logits = self.classifier(x)
        return logits