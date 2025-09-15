import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class DecoderAttn(nn.Module):
    def __init__(self, input_dim: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        # projections into head subspace
        self.q_proj = nn.Linear(input_dim, head_dim)
        self.k_proj = nn.Linear(input_dim, head_dim)
        self.v_proj = nn.Linear(input_dim, head_dim)

        self.out_proj = nn.Linear(head_dim, head_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        queries: torch.Tensor,  # [B, M, D]  decoder queries
        keys: torch.Tensor,     # [B, N, D]  encoder patch tokens
        values: torch.Tensor    # [B, N, D]  encoder patch tokens
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        q = self.q_proj(queries)  # [B, M, head_dim]
        k = self.k_proj(keys)     # [B, N, head_dim]
        v = self.v_proj(values)   # [B, N, head_dim]

        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B, M, N]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)  # [B, M, head_dim]
        out = self.out_proj(out)
        return out, attn


class MultiheadAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_queries: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_queries = num_queries

        # learnable content queries
        self.query_content = nn.Parameter(torch.randn(1, num_queries, dim))

        # per-head attention
        self.heads = nn.ModuleList([
            DecoderAttn(dim, self.head_dim, dropout=dropout)
            for _ in range(num_heads)
        ])

        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        pos_queries: torch.Tensor,  # [B, M, D] from encoder (pos embeddings for queries)
        enc_tokens: torch.Tensor    # [B, N, D] encoder patch tokens (keys/values)
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, M, D = pos_queries.shape
        _, N, _ = enc_tokens.shape

        # add content + positional
        queries = self.query_content.expand(B, -1, -1) + pos_queries  # [B, M, D]

        # collect outputs per head
        head_outputs, head_attns = [], []
        for head in self.heads:
            out, attn = head(queries, enc_tokens, enc_tokens)  # cross-attn
            head_outputs.append(out)
            head_attns.append(attn)

        # concat along feature dim
        out = torch.cat(head_outputs, dim=-1)  # [B, M, D]
        out = self.out_proj(out)

        # stack attention maps
        attn = torch.stack(head_attns, dim=1).mean(dim=1)  # [B, M, N]

        return out, attn



class VisualDecoder(nn.Module):
    def __init__(self, latent_tokens, Hidden_dim, patch_embeddings):
        super().__init__()
        self.Z = latent_tokens,
        self.D = Hidden_dim
    
    def forward(self):
        pass
        
        