import torch
import torch.nn as nn
from torchvision import models
import sys, os

sys.path.append(os.path.abspath(".."))
from grqo import GRQO


class resnetGRQO(nn.Module):
    def __init__(self, backbone, grqo_model, HD):
        super().__init__()
        self.backbone = backbone
        self.grqo = grqo_model
        self.projection_head = nn.Linear(HD, 384)

    def forward(self, x, labels=None, domains=None):
        outputs = self.backbone(x)
        B, D, H, W = outputs.shape
        outputs = outputs.flatten(2).transpose(1, 2)
        tokens = self.projection_head(outputs)
        return self.grqo(tokens, labels, domains)


def get_model(cfg, dataset="VLCS"):
    grqo_cfg = cfg["grqo"]
    data_cfg = cfg["datasets"][dataset]
    device = cfg["system"]["device"]

    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    HD = 512

    grqo_model = GRQO(
        Hidden_dim=384,
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
    )

    model = resnetGRQO(backbone, grqo_model, HD).to(device)
    return model
