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
        self.history = []
        x_prime = x.clone()
        for _ in range(self.nrepeat):
            # if (len(self.history > 0)):
            new_cls = x[:,0:1].clone()
            patches = x_prime[:,1:]
            x = torch.cat((new_cls,patches),dim=1)   
            mem = torch.cat([x] + [m for m in self.history], dim=1)
            for block in self.blocks:
                res1 = x
                x = block.norm1(x)
                weight = block.attn.qkv.weight
                bias   = block.attn.qkv.bias

                q_projs = weight[:self.D]
                k_projs = weight[self.D:2*self.D]
                v_projs = weight[2*self.D:]

                q_bias = bias[:self.D]
                k_bias = bias[self.D:2*self.D]
                v_bias = bias[2*self.D:]

                q = F.linear(x,        q_projs, q_bias)
                k = F.linear(mem,      k_projs, k_bias)
                v = F.linear(mem,      v_projs, v_bias)
                
                q = block.attn.q_norm(q)
                k = block.attn.k_norm(k)

                product = F.scaled_dot_product_attention(q, k, v, dropout_p=block.attn.attn_drop.p if self.training else 0.0)

                product = block.attn.norm(product)
                proj      = block.attn.proj(product)
                proj_drop = block.attn.proj_drop(proj)

                proj_drop = block.ls1(proj_drop)
                x = block.drop_path1(proj_drop) + res1

                res2 = x
                x2   = block.norm2(x)
                x1   = block.mlp(x2)
                x1 = block.ls2(x1)
                x    = block.drop_path2(x1) + res2


            # new_cls2 = x[:,0:1].clone()
            self.history.append(new_cls)

        # self.history = []
        return x





# Block(
#       (norm1): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
#       (attn): Attention(
#         (qkv): Linear(in_features=384, out_features=1152, bias=True)
#         (q_norm): Identity()
#         (k_norm): Identity()
#         (attn_drop): Dropout(p=0.0, inplace=False)
#         (norm): Identity()
#         (proj): Linear(in_features=384, out_features=384, bias=True)
#         (proj_drop): Dropout(p=0.0, inplace=False)
#       )
#       (ls1): Identity()
#       (drop_path1): Identity()
#       (norm2): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
#       (mlp): Mlp(
#         (fc1): Linear(in_features=384, out_features=1536, bias=True)
#         (act): GELU(approximate='none')
#         (drop1): Dropout(p=0.0, inplace=False)
#         (norm): Identity()
#         (fc2): Linear(in_features=1536, out_features=384, bias=True)
#         (drop2): Dropout(p=0.0, inplace=False)
#       )
#       (ls2): Identity()
#       (drop_path2): Identity()
#     )


class LatentRepeatTiny(nn.Module):
    def __init__(self, blocks, nrepeat):
        super().__init__()
        self.blocks = blocks
        self.nrepeat = nrepeat
        self.history = []
        self.D = 384

    def forward(self, x):
        self.history = []
        B, E, H, W = x.shape
        x = x.view(B,E,H*W,).transpose(-2,-1)
        for _ in range(self.nrepeat):
            mem = torch.cat([x] + [m for m in self.history], dim=1)
            for block in self.blocks:
                res1 = x
                x = block.attn.norm(x)
                weight = block.attn.qkv.weight
                bias   = block.attn.qkv.bias

                q_projs = weight[:self.D]
                k_projs = weight[self.D:2*self.D]
                v_projs = weight[2*self.D:]

                q_bias = bias[:self.D]
                k_bias = bias[self.D:2*self.D]
                v_bias = bias[2*self.D:]

                q = F.linear(x,        q_projs, q_bias)
                k = F.linear(mem,      k_projs, k_bias)
                v = F.linear(mem,      v_projs, v_bias)
                
                product = F.scaled_dot_product_attention(q, k, v)

                projection = block.attn.proj(product)
                x = block.drop_path1(projection) + res1
                res2 = x
                x = block.mlp(x)
                x    = block.drop_path2(x) + res2
                # print(x.shape)
                B,N,D = x.shape
                gs = 14
                x = x.transpose(1,2).view(B,D,gs,gs)
                x = block.local_conv(x)
                x = x.view(B, D, gs * gs).transpose(1, 2)
            self.history.append(x)
        # print(x.shape)
        x = x.view(B,D,gs,gs)
        return x




    #  (blocks): Sequential(
#         (0): TinyVitBlock(
#           dim=384, num_heads=12, window_size=14, mlp_ratio=4.0
#           (attn): Attention(
#             (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
#             (qkv): Linear(in_features=384, out_features=1152, bias=True)
#             (proj): Linear(in_features=384, out_features=384, bias=True)
#           )
#           (drop_path1): DropPath(drop_prob=0.073)
#           (mlp): NormMlp(
#             (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
#             (fc1): Linear(in_features=384, out_features=1536, bias=True)
#             (act): GELU(approximate='none')
#             (drop1): Dropout(p=0.0, inplace=False)
#             (fc2): Linear(in_features=1536, out_features=384, bias=True)
#             (drop2): Dropout(p=0.0, inplace=False)
#           )
#           (drop_path2): DropPath(drop_prob=0.073)
#           (local_conv): ConvNorm(
#             (conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
#             (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           )
#         )