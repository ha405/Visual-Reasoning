import torch
import torch.nn as nn
from torchvision.models import vgg11, VGG11_Weights

class VGGWrapper(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = VGG11_Weights.IMAGENET1K_V1 if pretrained else None
        self.vgg = vgg11(weights=weights)

        # Keep the original classifier up to the penultimate layer
        in_features = self.vgg.classifier[-1].in_features
        self.vgg.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)

    def get_intermediate_features(self, x):
        features = []
        for layer in self.vgg.features:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                features.append(x)

        x = torch.flatten(x, 1)

        # Pass through classifier, capturing activations
        for layer in self.vgg.classifier[:-1]:
            x = layer(x)
            features.append(x)

        logits = self.vgg.classifier[-1](x)
        return logits, features
