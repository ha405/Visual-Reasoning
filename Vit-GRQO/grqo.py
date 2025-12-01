from decoder import VisualDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class QueryLosses(nn.Module):
    def __init__(self, Hidden_dim, num_heads, dropout,
                 num_tokens, ddropout, num_layers, num_classes, temperature, randomk: Optional[int]=None):
        super().__init__()
        self.decoder = VisualDecoder(Hidden_dim, num_heads, dropout,
                                     num_tokens, ddropout, num_layers, temperature, random_k=randomk)
        self.selection_head = nn.Linear(Hidden_dim, 1)
        self.cls_head = nn.Linear(Hidden_dim, num_classes)

    def forward(self, tokens, labels):
        decoder_out = self.decoder(tokens)           # [B, M, D]
        per_query_logits = self.cls_head(decoder_out)  # [B, M, C]
        sel_scores = self.selection_head(decoder_out).squeeze(-1)  # [B, M]
        prob_scores = F.softmax(sel_scores, dim=1)    # [B, M]
        img_logits = torch.einsum("bq,bqc->bc", prob_scores, per_query_logits)  # [B, C]
        cls_loss = F.cross_entropy(img_logits, labels)  # scalar
        preds = img_logits.argmax(dim=1)    # [B]
        return cls_loss, prob_scores, img_logits, preds, per_query_logits, decoder_out

class GRQO(nn.Module):
    def __init__(self, Hidden_dim, num_heads, dropout,
                 num_tokens, ddropout, num_layers, num_classes,
                 temperature,
                 alpha=1.0, beta=1.0, tau=1e-3,
                 lambda_grqo=1.0, teacher_ema=0.99,
                 reward_proxy="taylor",  
                 resnet = False,
                 random_k: Optional[int]=None,
                 alpha_invar=0.5, 
                 gamma_var=1.0    
                 ):
        super().__init__()
        self.ql = QueryLosses(Hidden_dim, num_heads, dropout,
                              num_tokens, ddropout, num_layers, num_classes, temperature, randomk=random_k)

        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.lambda_grqo = lambda_grqo
        self.teacher_ema = teacher_ema
        assert reward_proxy in ("taylor", "gradnorm"), "reward_proxy must be 'taylor' or 'gradnorm'"
        self.reward_proxy = reward_proxy
        self.resnet = resnet
        
        self.alpha_invar = alpha_invar
        self.gamma_var = gamma_var

        self.register_buffer("teacher_ref", None)

    def _init_teacher(self, M, device):
        if self.teacher_ref is None:
            self.teacher_ref = torch.full((M,), 1.0 / M, device=device)

    def forward(self, x, y, domains=None):
        device = x.device
        if self.resnet:
            B,D,H,W = x.shape
            x = x.flatten(2).transpose(1,2)
            
        # --- 1) Base forward ---
        cls_loss, prob_scores, img_logits, preds, per_query_logits, decoder_out = self.ql(x, y)
        
        B, M, D = decoder_out.shape
        self._init_teacher(M, device)

        # --- 2) Reward proxy computation ---
        
        # [FIX 1] Compute Global Gradients ONCE here.
        # This fixes the "Tensor not used in graph" error.
        # retain_graph=True is REQUIRED because we need the graph for total_loss.backward() later.
        # create_graph=False ensures these gradients are treated as constants (targets).
        grads = torch.autograd.grad(cls_loss, decoder_out, retain_graph=True, create_graph=False)[0] # [B, M, D]

        raw_rewards = torch.zeros(B, M, device=device)
        
        if domains is not None:
            unique_domains = torch.unique(domains)
            domain_means_list = []
            valid_domains_found = False

            for d in unique_domains:
                mask = (domains == d)
                if mask.sum() == 0:
                    continue
                
                # [FIX 2] Instead of recalculating loss/autograd, SLICE the pre-computed global grads.
                grads_d = grads[mask]              # [N_d, M, D]
                decoder_out_d = decoder_out[mask]  # [N_d, M, D]

                if self.reward_proxy == "taylor":
                    # Normalize gradients and features to make dot product stable
                    grads_d_norm = F.normalize(grads_d, dim=-1)
                    decoder_out_d_norm = F.normalize(decoder_out_d, dim=-1)
                    raw_r_d = - (grads_d_norm * decoder_out_d_norm).sum(dim=-1)
                else:
                    raw_r_d = torch.norm(grads_d, p=2, dim=-1)
                
                # Store values
                raw_rewards[mask] = raw_r_d.detach()
                
                domain_means_list.append(raw_r_d.detach().mean(dim=0))
                valid_domains_found = True

            if valid_domains_found and len(domain_means_list) > 1:
                # Invariance Calculation
                stacked_means = torch.stack(domain_means_list, dim=0) # [Num_Domains, M]
                var_across_domains = stacked_means.var(dim=0, unbiased=False) # [M]
                
                # Penalty on high variance
                invariance_penalty = (self.gamma_var * var_across_domains).unsqueeze(0)
                
                # Reward blending
                raw_rewards = raw_rewards - invariance_penalty
        else:
            # Fallback
            if self.reward_proxy == "taylor":
                raw_rewards = - (grads * decoder_out).sum(dim=-1)
            else:
                raw_rewards = torch.norm(grads, p=2, dim=-1)

        # Detach rewards
        rewards = raw_rewards.detach()

        # --- 3) Advantage Normalization ---
        eps = 1e-6
        mu = rewards.mean(dim=1, keepdim=True)
        sigma = rewards.std(dim=1, keepdim=True) + eps
        
        adv = (rewards - mu) / sigma
        # Clamp advantage to stabilize training
        adv = torch.clamp(adv, -5.0, 5.0)
        adv = adv.detach()

        # --- 4) Mask and RL-like Gradient Injection ---
        mask = (prob_scores > self.tau).float().detach()

        # Policy Gradient injection: prob_scores * advantage
        masked_adv_rl = prob_scores * adv * mask 

        counts = mask.sum(dim=1, keepdim=True)
        denom = torch.where(counts > 0, counts, torch.ones_like(counts))
        
        mean_masked_adv = (masked_adv_rl.sum(dim=1, keepdim=True) / denom).squeeze(1)

        # --- 5) Teacher KL anchor term ---
        teacher = self.teacher_ref.unsqueeze(0).expand(B, M)
        kl_per_image = (prob_scores * (torch.log(prob_scores + 1e-12) - torch.log(teacher + 1e-12))).sum(dim=1)

        # --- 6) GRQO Loss ---
        grqo_per_image = - (self.alpha * mean_masked_adv - self.beta * kl_per_image)
        grqo_loss = grqo_per_image.mean()

        # --- 7) Total loss ---
        total_loss = cls_loss + self.lambda_grqo * grqo_loss

        # --- 8) EMA update ---
        if self.training:
            with torch.no_grad():
                batch_mean_w = prob_scores.mean(dim=0)
                self.teacher_ref = self.teacher_ema * self.teacher_ref + (1.0 - self.teacher_ema) * batch_mean_w.detach()

        out = {
            "loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "grqo_loss": grqo_loss.detach(),
            "grqo_per_image": grqo_per_image.detach(),
            "mean_masked_adv": mean_masked_adv.detach(),
            "rewards": rewards.detach(),
            "advantage": adv.detach(),
            "mask": mask.detach(),
            "kl_per_image": kl_per_image.detach(),
            "ent": -(prob_scores * torch.log(prob_scores + 1e-12)).sum(dim=1).mean().detach(),
            "prob_scores": prob_scores.detach(),
            "img_logits": img_logits.detach(),
            "preds": preds.detach()
        }
        return out