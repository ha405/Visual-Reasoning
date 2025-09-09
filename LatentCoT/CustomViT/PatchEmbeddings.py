import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patches_size=14, hidden_dim=768, in_channels=3):
        super().__init__()

        self.image_size = image_size
        self.patches_size = patches_size
        self.num_patches = image_size/patches_size
        self.hidden_dim = hidden_dim

        self.proj = nn.Conv2d(in_channels=in_channels,out_channels=hidden_dim, kernel_size=self.patches_size,stride=self.patches_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))

    def forward(self,x):
        B = x.shape[0]

        x = self.proj(x)  # [B, Dim, H/Patches, W/Patches]
        x = x.flatten()  # [B, Dim, Num_Patches]
        x = x.transpose(1,2)

        cls_token = self.cls_token.expand(B,-1,-1) # [B, 1, Dim]
        x = torch.cat((cls_token,x), dim=1)  # [B, num_patches + cls, Dim]
        x = x + self.pos_embed 

        return x