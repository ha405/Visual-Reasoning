import torch
import torch.nn as nn
from torchvision import models


MODEL_REGISTRY = {
    'resnet18': lambda pretrained, num_classes: models.resnet18(pretrained=pretrained),
    'resnet34': lambda pretrained, num_classes: models.resnet34(pretrained=pretrained),
    'resnet50': lambda pretrained, num_classes: models.resnet50(pretrained=pretrained),
    'resnet101': lambda pretrained, num_classes: models.resnet101(pretrained=pretrained),
    'resnet152': lambda pretrained, num_classes: models.resnet152(pretrained=pretrained),
    'vgg11': lambda pretrained, num_classes: models.vgg11(pretrained=pretrained),
    'vgg13': lambda pretrained, num_classes: models.vgg13(pretrained=pretrained),
    'vgg16': lambda pretrained, num_classes: models.vgg16(pretrained=pretrained),
    'vgg19': lambda pretrained, num_classes: models.vgg19(pretrained=pretrained),
    'densenet121': lambda pretrained, num_classes: models.densenet121(pretrained=pretrained),
    'densenet161': lambda pretrained, num_classes: models.densenet161(pretrained=pretrained),
    'densenet169': lambda pretrained, num_classes: models.densenet169(pretrained=pretrained),
    'densenet201': lambda pretrained, num_classes: models.densenet201(pretrained=pretrained),
    'mobilenet_v2': lambda pretrained, num_classes: models.mobilenet_v2(pretrained=pretrained),
    'mobilenet_v3_small': lambda pretrained, num_classes: models.mobilenet_v3_small(pretrained=pretrained),
    'mobilenet_v3_large': lambda pretrained, num_classes: models.mobilenet_v3_large(pretrained=pretrained),
    'efficientnet_b0': lambda pretrained, num_classes: models.efficientnet_b0(pretrained=pretrained),
    'efficientnet_b1': lambda pretrained, num_classes: models.efficientnet_b1(pretrained=pretrained),
    'efficientnet_b2': lambda pretrained, num_classes: models.efficientnet_b2(pretrained=pretrained),
}


def _modify_final_layer(model, model_name, num_classes):
    if num_classes is None:
        return model
    
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    elif hasattr(model, 'heads') and hasattr(model.heads, 'head'):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    
    else:
        print(f"Warning: Could not automatically modify final layer for {model_name}. "
              f"You may need to do this manually.")
    
    return model


def get_model(model_name, pretrained=True, num_classes=None, checkpoint_path=None):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    model = MODEL_REGISTRY[model_name](pretrained=pretrained, num_classes=num_classes)
    
    if num_classes is not None:
        model = _modify_final_layer(model, model_name, num_classes)
    
    if checkpoint_path is not None:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
    
    return model


def register_model(name, model_func):
    MODEL_REGISTRY[name] = model_func
    print(f"Registered model: {name}")


def list_available_models():
    return sorted(MODEL_REGISTRY.keys())
