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

# AI code (needs to be reviewed and fixed)
class GRQO(nn.Module):
    """
    Wraps QueryLosses and adds GRQO loss terms:
      - reward proxy (Taylor or grad-norm)
      - group-normalized advantage (per-image)
      - mask by selection prob > tau
      - KL anchor to teacher_ref (EMA)
      - final GRQO loss per the math:
            L_GRQO_b = - ( alpha * mean_j ( m_{b,j} * Ahat_{b,j} ) - beta * D_KL(W_b || W_ref_b) )
      - total loss = cls_loss + lambda_grqo * mean_b L_GRQO_b
    """
    def __init__(self, Hidden_dim, num_heads, dropout,
                 num_tokens, ddropout, num_layers, num_classes,
                 temperature,
                 # GRQO hyperparams:
                 alpha=1.0, beta=1.0, tau=1e-3,
                 lambda_grqo=1.0, teacher_ema=0.99,
                 reward_proxy="taylor",  # either "taylor" or "gradnorm"
                 resnet = False,
                 random_k: Optional[int]=None
                 ):
        super().__init__()
        # QueryLosses does decoding + heads
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
        # teacher reference distribution for queries (vector of length M). We'll
        # lazily initialize once we know M (after first forward) and register as buffer.
        self.register_buffer("teacher_ref", None)

    def _init_teacher(self, M, device):
        # initialize uniform teacher if not set
        if self.teacher_ref is None:
            self.teacher_ref = torch.full((M,), 1.0 / M, device=device)

    def forward(self, x, y):
        """
        x: backbone tokens [B, N, D] - ViT
        x: backbone tokens [B,D,H,W] - Resnet
        y: labels [B]
        Returns dict with 'loss' and diagnostics
        """
        device = x.device
        if self.resnet:
            B,D,H,W = x.shape
            x = x.flatten(2).transpose(1,2)
        # --- 1) Base forward: classification and selection ---
        cls_loss, prob_scores, img_logits, preds, per_query_logits, decoder_out = self.ql(x, y)
        # cls_loss: scalar tensor
        # prob_scores: [B, M]
        # per_query_logits: [B, M, C]
        # decoder_out: [B, M, D]

        B, M, D = decoder_out.shape
        _, _, C = per_query_logits.shape

        # ensure teacher ref exists
        self._init_teacher(M, device)

        # --- 2) Reward proxy computation ---
        # We use either Taylor first-order proxy:
        #   r_{b,j} = - < G_{b,j}, decoder_out_{b,j} >
        # where G = d(sum_b cls_loss_b)/d decoder_out (shape [B,M,D])
        # or gradient-norm proxy:
        #   r_{b,j} = || G_{b,j} ||_2
        #
        # compute grads of the scalar classification loss w.r.t decoder_out
        # Note: cls_loss is scalar; ensure computation graph is present (it is).
        grads = torch.autograd.grad(cls_loss, decoder_out, retain_graph=True, create_graph=False)[0]  # [B,M,D]

        if self.reward_proxy == "taylor":
            # Taylor proxy: negative dot product between grad and feature
            # r = - sum_d G * q
            raw_rewards = - (grads * decoder_out).sum(dim=-1)  # [B, M]
        else:
            # grad-norm proxy
            raw_rewards = torch.norm(grads, p=2, dim=-1)  # [B, M]

        # detach rewards -- they are targets for GRQO, not a path for grads
        rewards = raw_rewards.detach()

        # --- 3) Group-normalized advantage (per image) ---
        # compute mean & std per image (over queries)
        eps = 1e-6
        mu = rewards.mean(dim=1, keepdim=True)           # [B,1]
        sigma = rewards.std(dim=1, keepdim=True) + eps   # [B,1]
        adv = (rewards - mu) / sigma                     # [B, M]
        # optional clamp to avoid very large values (you can tune or remove)
        adv = torch.clamp(adv, -10.0, 10.0)

        # --- 4) Mask tiny-prob queries by threshold tau ---
        mask = (prob_scores > self.tau).float()  # [B, M]
        masked_adv = adv * mask                   # [B, M]

        # compute mean advantage per image over queries (only masked ones contribute)
        # to follow your math: (1/|J|) sum_{j in J} m_{b,j} * Ahat_{b,j}
        # Note: use count per image to normalize correctly
        counts = mask.sum(dim=1, keepdim=True)   # [B,1]
        # avoid divide by zero: if counts==0, fallback to uniform denom 1
        denom = torch.where(counts > 0, counts, torch.ones_like(counts))
        mean_masked_adv = (masked_adv.sum(dim=1, keepdim=True) / denom).squeeze(1)  # [B]

        # --- 5) Teacher KL anchor term (per image) ---
        # teacher_ref is shape [M]; expand to [B,M] (global teacher)
        teacher = self.teacher_ref.unsqueeze(0).expand(B, M)  # [B,M]
        # compute KL per image: sum_j w_{b,j} * log(w_{b,j} / teacher_{b,j})
        # stable compute: use log probs
        kl_per_image = (prob_scores * (torch.log(prob_scores + 1e-12) - torch.log(teacher + 1e-12))).sum(dim=1)  # [B]

        # --- 6) GRQO per-image term (minimize) as in eq:
        # L_GRQO_b = - ( alpha * mean_masked_adv_b  -  beta * KL_b )
        grqo_per_image = - (self.alpha * mean_masked_adv - self.beta * kl_per_image)  # [B]

        # mean over batch
        grqo_loss = grqo_per_image.mean()

        # --- 7) Total loss ---
        total_loss = cls_loss + self.lambda_grqo * grqo_loss

        # --- 8) EMA update of teacher_ref (global) using batch mean W ---
        with torch.no_grad():
            batch_mean_w = prob_scores.mean(dim=0)  # [M]
            self.teacher_ref = self.teacher_ema * self.teacher_ref + (1.0 - self.teacher_ema) * batch_mean_w.detach()

        # --- 9) Package outputs / diagnostics ---
        out = {
            "loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "grqo_loss": grqo_loss.detach(),
            "grqo_per_image": grqo_per_image.detach(),
            "mean_masked_adv": mean_masked_adv.detach(),
            "rewards": rewards.detach(),              # [B,M]
            "advantage": adv.detach(),                # [B,M]
            "mask": mask.detach(),                    # [B,M]
            "kl_per_image": kl_per_image.detach(),    # [B]
            "ent": -(prob_scores * torch.log(prob_scores + 1e-12)).sum(dim=1).mean().detach(),
            "prob_scores": prob_scores.detach(),
            "img_logits": img_logits.detach(),
            "preds": preds.detach()
        }
        return out
