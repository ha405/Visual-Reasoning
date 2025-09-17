import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class VisualDecoder(nn.Module):
    def __init__(self, Hidden_dim, num_heads, dropout, num_tokens, ddropout, num_layers, temperature):
        super().__init__()
        self.p = nn.Linear(Hidden_dim,1)
        self.cross_attention = nn.ModuleList(MultiheadAttn(Hidden_dim, num_heads,num_tokens,ddropout) for _ in range(num_layers))
        self.selector = nn.Parameter(torch.randn(num_tokens, Hidden_dim))
        self.temperature = temperature
        self.num_queries = num_tokens
        
    def forward(self, latent_tokens):
        B, N, D = latent_tokens.shape
        logits = torch.einsum('bnd,md->bmn', latent_tokens, self.selector)
        weights = F.softmax(logits / self.temperature, dim=-1)  # [B, M, N]
        pos_queries = torch.einsum('bmn,bnd->bmd', weights, latent_tokens)  # [B, M, D]
        decoder_out,_ = self.cross_attention[0](pos_queries,latent_tokens)
        for layer in self.cross_attention[1:]:
            decoder_out, _ = layer(decoder_out, latent_tokens)
        return decoder_out

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
        self.dropout = nn.Dropout(dropout)
        # per-head attention
        self.heads = nn.ModuleList([
            DecoderAttn(dim, self.head_dim, dropout=dropout)
            for _ in range(num_heads)
        ])
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
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
        out = self.dropout(out)
        # stack attention maps
        attn = torch.stack(head_attns, dim=1).mean(dim=1)  # [B, M, N]
        out = self.norm(out + queries)
        mlp_out = self.mlp(out)
        out = self.norm2(out + mlp_out)
        
        return out, attn

        
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
        