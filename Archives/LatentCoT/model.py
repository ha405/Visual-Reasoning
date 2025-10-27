import torchvision
import torch.nn as nn
import timm
from LatentWrapper import LatentRepeat, LatentRepeatTiny
import torch

# [DEBUG]
# nrepeat = 3
# model = timm.create_model('tiny_vit_21m_224.dist_in22k_ft_in1k', pretrained=True)
# print(model.pos_embed)
# blocks = model.blocks
# head = model.head
# print(head)
# cls = model.cls_token
# print(cls)
# weight = blocks[0].attn.qkv.weight
# print(weight)
# unpooled = model.forward_features(torch.randn(2,3,224,224))
# pooled = model.forward_head(unpooled)
# print(unpooled.shape)
# print(pooled.shape)
# [DEBUG] 

class LatenViTSmall(nn.Module):
    def __init__(self, model, nrepeat, start,end, num_layers):
        super().__init__()
        self.model = model
        self.nrepeat = nrepeat
        self.start = start
        self.end = end
        self.num_layers = num_layers
        self.cotformers = self.model.blocks[start:end]
        self.latent_module = LatentRepeat(self.cotformers, self.nrepeat)
    
    def forward(self, x):
        # x is a batch of images (B, H, E) 
        x = self.model.patch_embed(x)
        # cls token is [1,1,D] -> after expansion its [B, 1, D]
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1) # expand cls token across all batch of images
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        for i in range(self.start):
            x = self.model.blocks[i](x)
        x = self.latent_module(x)
        for i in range(self.end,self.num_layers + 1):
            x = self.model.blocks[i](x)
        
        x = self.model.norm(x)
        x = x[:, 0]
        x = self.model.fc_norm(x)
        x = self.model.head_drop(x)
        x = self.model.head(x)

        return x 
  
class LatenViTtiny(nn.Module):
    def __init__(self, model, nrepeat, stage):
        super().__init__()
        self.model = model
        self.nrepeat = nrepeat
        self.stage = stage
        self.cotformers = self.model.stages[stage].blocks
        self.latent_module = LatentRepeatTiny(self.cotformers, self.nrepeat)
    
    def forward(self, x):
        # x is a batch of images (B, H, E) 
        x = self.model.patch_embed(x)
        x = self.model.stages[0](x)
        x = self.model.stages[1](x)
        x = self.model.stages[2].downsample(x)
        x = self.latent_module(x)
        x = self.model.stages[3](x)
        x = self.model.head(x)
        return x 
