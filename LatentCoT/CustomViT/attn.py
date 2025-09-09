import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.hd = hidden_dim

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x):
        residual = x

        q = self.q(x)   # [B, L, d]
        k = self.k(x)   # [B, L, d]
        v = self.v(x)   # [B, L, d]

        attn_scores = q @ k.transpose(-2, -1)  # [B, L, L]
        attn_scores = attn_scores / math.sqrt(self.hd)

        attn = F.softmax(attn_scores, dim=-1)  # [B, L, L]

        out = attn @ v  # [B, L, d]
        out = self.out(out)  # [B, L, d]

        x = self.norm1(residual + out)

        residual = x
        out = self.ff(x)
        x = self.norm2(residual + out)

        return x



class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.hd = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.heads = nn.ModuleList([AttentionBlock(self.head_dim) for _ in range(num_heads)])
        self.out = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x):
        B, L, _ = x.shape
        residual = x

        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(x).view(B, L, self.head_dim))

        out = torch.cat(head_outputs, dim=-1)  # [B, L, hidden_dim]

        # final linear projection
        out = self.out(out)

        # first residual + norm
        x = self.norm1(residual + out)

        residual = x
        out = self.ff(x)
        x = self.norm2(residual + out)

        return x
