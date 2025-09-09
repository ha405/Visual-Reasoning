import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  

class ViTGRQO(nn.Module):
    def __init__(self, num_classes, vit_model='vit_small_patch16_224', token_dim=384, topk=16):
        super().__init__()
        self.vit = timm.create_model(vit_model, pretrained=True)
        self.num_classes = num_classes
        self.token_dim = token_dim
        self.topk = topk
        self.selection_head = nn.Linear(token_dim, 1)
        self.register_buffer('teacher_probs', torch.zeros(1))  

    def forward(self, x):
        # x: [B, 3, H, W]
        B = x.size(0)
        features = self.vit.forward_features(x)  # [B, N+1, token_dim], includes CLS
        cls_token = features[:, 0]               # CLS token
        patch_tokens = features[:, 1:, :]        # [B, N, token_dim]
        
        # Classification logits
        logits = self.vit.head(cls_token)
        
        # Token importance scores
        token_scores = self.selection_head(patch_tokens).squeeze(-1)  # [B, N]
        token_probs = torch.softmax(token_scores, dim=-1)
        
        return logits, patch_tokens, token_probs

def grqo_loss_from_gradients(
    logits,
    patch_tokens,
    token_probs,
    labels,
    teacher_probs=None,
    alpha=1.0,
    beta=0.5,
    topk=None,
    tau=1e-3,
    eps=1e-6
):
    device = logits.device
    B, N, D = patch_tokens.shape
    per_sample_loss = F.cross_entropy(logits, labels, reduction='none')
    patch_tokens.requires_grad_(True)
    token_rewards = torch.zeros(B, N, device=device)
    
    for b in range(B):
        loss_b = per_sample_loss[b]
        grads = torch.autograd.grad(loss_b, patch_tokens, retain_graph=True, create_graph=False)[0]
        token_rewards[b] = grads[b].norm(dim=-1)
    
    token_rewards = token_rewards.detach()
    
    if topk is None or topk >= N:
        topk_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        K = N
    else:
        K = topk
        topk_idx = token_probs.topk(K, dim=1).indices
    
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, K)
    rewards_topk = token_rewards.gather(1, topk_idx)
    probs_topk = token_probs.gather(1, topk_idx)
    
    mu = rewards_topk.mean(dim=1, keepdim=True)
    sigma = rewards_topk.std(dim=1, unbiased=False, keepdim=True) + eps
    advantage = (rewards_topk - mu) / sigma
    
    if teacher_probs is None:
        O_ref_topk = torch.full_like(probs_topk, 1.0 / float(K))
    else:
        O_ref_topk = teacher_probs.gather(1, topk_idx).clamp(min=eps)
    
    O_theta = probs_topk.clamp(min=eps)
    kl_per_image = (O_theta * (O_theta.log() - O_ref_topk.log())).sum(dim=1)
    
    alpha_mask = (O_theta > tau).float()
    alpha_effective = alpha * alpha_mask
    
    adv_term = (alpha_effective * advantage).mean(dim=1)
    grqo_per_image = - (adv_term - beta * kl_per_image)
    L_grqo = grqo_per_image.mean()
    
    return L_grqo