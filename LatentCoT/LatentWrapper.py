import torch
import torch.nn as nn
import torch.nn.functional as F

# Logic:
# Q = [B,H,L,D], K = [B, H, L, D], V = [B, H, L , D]
# Q @ K.tranpose(-2,-1) => attn_scores = [B, H, L, L]
# attn_scores @ V => [B, H, L, D] (Restored)
# Now lets consider with history:
# q = [B, H, L, D], K = [B, H, L~, D], V = [B, H, L~, D]
# attn_scores = [B, H, L, L~]
# attn_scores @ V => [B, H, L, D] (Restored but now attention happened over historical sequence) 

# FLow =>
# Iter1: x -> block1 -> block2 -> block3 -> get cls_token from sequences (cls_prim) -> cls_prim to history
# Iter2: Restore x to original but update cls_token to cls_prim  

class LatentRepeat(nn.Module):
    def __init__(self, blocks, nrepeat):
        super().__init__()
        self.blocks = blocks
        self.nrepeat = nrepeat
        self.history = []
        self.D = 384

    def forward(self, x):
        x_prime = x
        for _ in range(self.nrepeat):
            mem = torch.cat([x] + [m for m in self.history], dim=1)
            x_prime[:, 0] = x[:, 0]
            x = x_prime
            for block in self.blocks:
                weight = block.attn.qkv.weight
                bias = block.attn.qkv.bias

                q_projs = weight[:self.D]
                k_projs = weight[self.D:2*self.D]
                v_projs = weight[2*self.D:]

                q_bias = bias[:self.D]
                k_bias = bias[self.D:2*self.D]
                v_bias = bias[2*self.D:]

                q = F.linear(x,q_projs,q_bias)
                k = F.linear(mem,k_projs,k_bias)
                v = F.linear(mem,v_projs,v_bias)
                x = block(x)

            self.history.append(x[:,0])
            
