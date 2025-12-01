import torch
import torch.nn as nn
from transformers import ViTModel, ViTForImageClassification
from torchvision import models
from grqo import GRQO


class ViTGRQO(nn.Module):
    def __init__(self, vit_encoder, grqo_model):
        super().__init__()
        self.vit = vit_encoder
        self.grqo = grqo_model

    def forward(self, x, labels, domains=None):
        outputs = self.vit(pixel_values=x, output_hidden_states=True)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        return self.grqo(patch_tokens, labels, domains)


class ResNetGRQO(nn.Module):
    def __init__(self, backbone, grqo_model, backbone_dim, grqo_dim):
        super().__init__()
        self.backbone = backbone
        self.grqo = grqo_model
        self.projection = nn.Linear(backbone_dim, grqo_dim)

    def forward(self, x, labels, domains=None):
        features = self.backbone(x)
        B, D, H, W = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        projected_tokens = self.projection(tokens)
        return self.grqo(projected_tokens, labels, domains)


def _get_vit_grqo_model(model_name, dataset, cfg):
    grqo_cfg = cfg["grqo"]
    data_cfg = cfg["datasets"][dataset]
    model_cfg = cfg["models"][model_name]

    vit_encoder = ViTModel.from_pretrained(model_cfg["pretrained"])
    hidden_dim = model_cfg["hidden_dim"]

    grqo_model = GRQO(
        Hidden_dim=hidden_dim,
        num_heads=grqo_cfg["num_heads"],
        dropout=grqo_cfg["dropout"],
        num_tokens=grqo_cfg["num_tokens"],
        ddropout=grqo_cfg["ddropout"],
        num_layers=grqo_cfg["num_layers"],
        num_classes=data_cfg["num_classes"],
        temperature=grqo_cfg["temperature"],
        alpha=grqo_cfg["alpha"],
        beta=grqo_cfg["beta"],
        tau=grqo_cfg["tau"],
        lambda_grqo=grqo_cfg["lambda_grqo"],
        teacher_ema=grqo_cfg["teacher_ema"],
        reward_proxy=grqo_cfg["reward_proxy"],
        random_k=grqo_cfg.get("random_k"),
        alpha_invar=grqo_cfg["alpha_invar"],
        gamma_var=grqo_cfg["gamma_var"]
    )

    model = ViTGRQO(vit_encoder, grqo_model)
    return model


def _get_resnet_grqo_model(model_name, dataset, cfg):
    grqo_cfg = cfg["grqo"]
    data_cfg = cfg["datasets"][dataset]
    model_cfg = cfg["models"][model_name]

    if model_name == "resnet18":
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnet34":
        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown ResNet model: {model_name}")

    backbone = nn.Sequential(*list(backbone.children())[:-2])
    backbone_dim = model_cfg["backbone_dim"]
    grqo_dim = model_cfg["grqo_dim"]

    grqo_model = GRQO(
        Hidden_dim=grqo_dim,
        num_heads=grqo_cfg["num_heads"],
        dropout=grqo_cfg["dropout"],
        num_tokens=grqo_cfg["num_tokens"],
        ddropout=grqo_cfg["ddropout"],
        num_layers=grqo_cfg["num_layers"],
        num_classes=data_cfg["num_classes"],
        temperature=grqo_cfg["temperature"],
        alpha=grqo_cfg["alpha"],
        beta=grqo_cfg["beta"],
        tau=grqo_cfg["tau"],
        lambda_grqo=grqo_cfg["lambda_grqo"],
        teacher_ema=grqo_cfg["teacher_ema"],
        reward_proxy=grqo_cfg["reward_proxy"],
        random_k=grqo_cfg.get("random_k"),
        alpha_invar=grqo_cfg["alpha_invar"],
        gamma_var=grqo_cfg["gamma_var"],
        resnet=True
    )

    model = ResNetGRQO(backbone, grqo_model, backbone_dim, grqo_dim)
    return model


def _get_vit_baseline_model(model_name, dataset, cfg):
    data_cfg = cfg["datasets"][dataset]
    model_cfg = cfg["models"][model_name]

    model = ViTForImageClassification.from_pretrained(
        model_cfg["pretrained"],
        num_labels=data_cfg["num_classes"],
        ignore_mismatched_sizes=True
    )
    return model


def _get_resnet_baseline_model(model_name, dataset, cfg):
    data_cfg = cfg["datasets"][dataset]

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown ResNet model: {model_name}")

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, data_cfg["num_classes"])
    return model


def get_grqo_model(model_name, dataset, cfg):
    if "vit" in model_name.lower():
        return _get_vit_grqo_model(model_name, dataset, cfg)
    elif "resnet" in model_name.lower():
        return _get_resnet_grqo_model(model_name, dataset, cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_baseline_model(model_name, dataset, cfg):
    if "vit" in model_name.lower():
        return _get_vit_baseline_model(model_name, dataset, cfg)
    elif "resnet" in model_name.lower():
        return _get_resnet_baseline_model(model_name, dataset, cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")
