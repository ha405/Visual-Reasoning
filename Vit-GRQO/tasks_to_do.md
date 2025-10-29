# GRQO DG Ablations

## Core Ablations
- **Ablate GRQO** — set `lambda_grqo = 0`  
  Goal: confirm whether GRQO is the cause of OOD improvement.

- **Ablate KL Anchor** — set `beta = 0`  
  Goal: test if EMA smoothing helps or hurts generalization.

- **Compare Reward Proxies** — run with  
  - `reward_proxy = "taylor"`  
  - `reward_proxy = "gradnorm"`  
  Goal: identify which signal better supports OOD robustness.

---

## Domain-Specific Diagnostics
- **Domain Leakage Test (Qdec)**
  - Train a small MLP: `Q_pool = Qdec.mean(dim=1)` → `domain_labels`
  - Evaluate domain classification accuracy.  
  High accuracy ⇒ domain info leaked.

- **Per-Query Selection Distribution**
  - Compute per-domain mean selection probabilities:  
    \[
    \bar{p}_j(d) = \mathbb{E}_{b \in d}[p_{b,j}]
    \]
  - Plot variance of \(\bar{p}_j(d)\) across domains.  
  High variance ⇒ query specialization to domains.

- **Reward Variance Across Domains**
  - Log per-domain average reward \(r_{d,j}\) and variance across domains.  
  High variance ⇒ domain-dependent reward signal.

- **Selection Entropy**
  - Compute entropy of \(p_j\) per image and per domain.  
  Low entropy ⇒ collapsed selection; high variance ⇒ unstable across domains.

---