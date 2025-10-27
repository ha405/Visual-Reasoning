import torch
import torch.nn as nn
from transformers import ViTModel
import sys, os

sys.path.append(os.path.abspath(".."))  
from grqo import GRQO

class ViTGRQO(nn.Module):
    def __init__(self, vit_encoder, grqo_model):
        super().__init__()
        self.vit = vit_encoder
        self.grqo = grqo_model

    def forward(self, x, labels=None):
        outputs = self.vit(pixel_values=x, output_hidden_states=True)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        return self.grqo(patch_tokens, labels)

def get_model(cfg, dataset="VLCS"):
    grqo_cfg = cfg["grqo"]
    data_cfg = cfg["datasets"][dataset]

    vit_encoder = ViTModel.from_pretrained("WinKawaks/vit-small-patch16-224")
    hidden_dim = vit_encoder.config.hidden_size

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
    )

    model = ViTGRQO(vit_encoder, grqo_model).to(cfg["system"]["device"])
    return model
