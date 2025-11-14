# %%writefile /kaggle/working/pruning.py
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import ConcatDataset, DataLoader
import torch.optim as optim
from utils import *

def get_layer(model, name):
    return dict(model.named_modules())[name]

def compute_magnitude_importance(model, cumulative_mask):
    """Computes importance as the L1-norm of each neuron's weights."""
    importance = OrderedDict()
    for layer_name, module in model.named_modules():
        if not (hasattr(module, "weight") and isinstance(module, (nn.Conv2d, nn.Linear))):
            continue
        
        weights = module.weight.data
        if isinstance(module, nn.Conv2d):
            # For Conv layers, sum over all dims except the output channel dim (dim 0)
            l1_norm = torch.sum(torch.abs(weights), dim=(1, 2, 3))
        else: # Linear
            # For Linear layers, sum over the input feature dim (dim 1)
            l1_norm = torch.sum(torch.abs(weights), dim=1)
        importance[layer_name] = l1_norm
    return importance

def _create_layer_mask(module, indices_to_prune):
    device = module.weight.device
    num_units = module.weight.shape[0]
    
    mask = torch.ones(num_units, device=device)
    if len(indices_to_prune) > 0:
        mask[torch.tensor(indices_to_prune, device=device)] = 0.0

    if isinstance(module, nn.Conv2d):
        return mask.view(-1, 1, 1, 1).expand_as(module.weight)
    else: # Linear
        return mask.view(-1, 1).expand_as(module.weight)

def select_prune_mask_by_source_heuristics(model, source_loaders, device, cumulative_mask,
                                           candidate_rates=(0.20, 0.30, 0.40, 0.60), acc_drop_threshold=5.0):
    model.to(device).eval()
    iter_mask = {}
    importance = compute_magnitude_importance(model, cumulative_mask)
    print("\nSelecting per-layer prune rates using source domain heuristics...")

    baseline_accs = torch.tensor([evaluate(model, loader, device, mask=cumulative_mask)[1] for loader in source_loaders])
    
    for layer_name, scores in importance.items():
        module = get_layer(model, layer_name)
        weight_name = f"{layer_name}.weight"
        
        if cumulative_mask and weight_name in cumulative_mask:
            prev_mask_flat = cumulative_mask[weight_name][:, 0, 0, 0] if module.weight.dim() == 4 else cumulative_mask[weight_name][:, 0]
            active_indices = torch.where(prev_mask_flat == 1)[0]
        else:
            active_indices = torch.arange(scores.shape[0], device=scores.device)
        
        if len(active_indices) == 0: continue

        # Sort by magnitude ASCENDING (smallest first, as they are least important)
        active_scores = scores[active_indices]
        sorted_indices_in_active = torch.argsort(active_scores, descending=False) 
        
        best_rate = 0.0
        for r in sorted(candidate_rates):
            k = int(r * len(active_indices))
            if k == 0: continue
            
            prune_indices_in_active = sorted_indices_in_active[:k]
            prune_indices = active_indices[prune_indices_in_active]
            
            temp_mask = {k_c: v_c.clone() for k_c, v_c in cumulative_mask.items()} if cumulative_mask else {}
            current_layer_mask = _create_layer_mask(module, prune_indices)
            temp_mask[weight_name] = cumulative_mask.get(weight_name, torch.ones_like(current_layer_mask)) * current_layer_mask

            current_accs = torch.tensor([evaluate(model, loader, device, mask=temp_mask)[1] for loader in source_loaders])
            if torch.all(current_accs >= (baseline_accs - acc_drop_threshold)):
                best_rate = r

        if best_rate > 0:
            k = int(best_rate * len(active_indices))
            prune_indices_in_active = sorted_indices_in_active[:k]
            prune_indices = active_indices[prune_indices_in_active]
            iter_mask[weight_name] = _create_layer_mask(module, prune_indices)
            print(f"  - Layer {layer_name}: selected rate {best_rate*100:.0f}%. All source accs within threshold.")
        else:
            print(f"  - Layer {layer_name}: No rate met the threshold. Layer not pruned.")

    return iter_mask

def select_prune_mask_by_target_testing(model, target_loader, device, cumulative_mask,
                                        candidate_rates=(0.20, 0.30, 0.40, 0.60)):
    model.to(device).eval()
    iter_mask = {}
    importance = compute_magnitude_importance(model, cumulative_mask)
    print("\nSelecting per-layer prune rates by testing on the target loader...")
    _, base_acc = evaluate(model, target_loader, device, mask=cumulative_mask)

    for layer_name, scores in importance.items():
        module = get_layer(model, layer_name)
        weight_name = f"{layer_name}.weight"

        if cumulative_mask and weight_name in cumulative_mask:
            prev_mask_flat = cumulative_mask[weight_name][:, 0, 0, 0] if module.weight.dim() == 4 else cumulative_mask[weight_name][:, 0]
            active_indices = torch.where(prev_mask_flat == 1)[0]
        else:
            active_indices = torch.arange(scores.shape[0], device=scores.device)

        if len(active_indices) == 0: continue

        # Sort by magnitude ASCENDING (smallest first)
        active_scores = scores[active_indices]
        sorted_indices_in_active = torch.argsort(active_scores, descending=False)
        
        best_acc = base_acc
        best_rate = 0.0
        for r in candidate_rates:
            k = int(r * len(active_indices))
            if k == 0: continue

            prune_indices_in_active = sorted_indices_in_active[:k]
            prune_indices = active_indices[prune_indices_in_active]

            temp_mask = {k_c: v_c.clone() for k_c, v_c in cumulative_mask.items()} if cumulative_mask else {}
            current_layer_mask = _create_layer_mask(module, prune_indices)
            temp_mask[weight_name] = cumulative_mask.get(weight_name, torch.ones_like(current_layer_mask)) * current_layer_mask
            
            _, acc = evaluate(model, target_loader, device, mask=temp_mask)
            if acc > best_acc:
                best_acc = acc
                best_rate = r
        
        if best_rate > 0:
            k = int(best_rate * len(active_indices))
            prune_indices_in_active = sorted_indices_in_active[:k]
            prune_indices = active_indices[prune_indices_in_active]
            iter_mask[weight_name] = _create_layer_mask(module, prune_indices)
            print(f"  - Layer {layer_name}: selected rate {best_rate*100:.0f}% -> target acc {best_acc:.2f}%")
        else:
            print(f"  - Layer {layer_name}: No rate improved target accuracy over baseline {base_acc:.2f}%.")
            
    return iter_mask

def combine_source_loaders(source_loaders, batch_size, num_workers):
    combined_dataset = ConcatDataset([loader.dataset for loader in source_loaders])
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def iterative_pruning(model, source_loaders_list, target_loader, device,
                      retrain_epochs, lr, alpha, batch_size, num_workers, SFT=False,
                      mask_selection_strategy="by_target",
                      candidate_rates=(0.20, 0.30, 0.40, 0.60),
                      iterations=3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    cumulative_mask = {}

    _, best_overall_acc = evaluate(model, target_loader, device)
    print(f"Initial Baseline Target Accuracy: {best_overall_acc:.2f}%")
    torch.save(model.state_dict(), "best_pruned_model.pth")
    combined_source_loader = combine_source_loaders(source_loaders_list, batch_size, num_workers)

    for i in range(iterations):
        print(f"\n--- Pruning Iteration {i+1}/{iterations} ---")
        model.load_state_dict(torch.load("best_pruned_model.pth"))

        if mask_selection_strategy == "by_source":
            iter_mask = select_prune_mask_by_source_heuristics(model, source_loaders_list, device, cumulative_mask, candidate_rates)
        elif mask_selection_strategy == "by_target":
            iter_mask = select_prune_mask_by_target_testing(model, target_loader, device, cumulative_mask, candidate_rates)
        else:
            raise ValueError(f"Unknown mask_selection_strategy: {mask_selection_strategy}")

        if not iter_mask:
            print("No further pruning was selected in this iteration. Stopping.")
            break

        for k, v in iter_mask.items():
            cumulative_mask[k] = cumulative_mask.get(k, torch.ones_like(v)) * v

        best_iter_acc = 0.0
        for epoch in range(retrain_epochs):
            print(f"\nRetraining Epoch {epoch+1}/{retrain_epochs}")
            train_func = train_SFT if SFT else train_DI
            train_func(model, combined_source_loader, optimizer, device, epoch, alpha, cumulative_mask)

            _, target_acc = evaluate(model, target_loader, device, mask=cumulative_mask)
            print(f"  Epoch {epoch+1} Target Accuracy: {target_acc:.2f}%")
            if target_acc > best_iter_acc:
                best_iter_acc = target_acc
                torch.save(model.state_dict(), "best_pruned_model.pth")
        print(f"Iteration {i+1} | Best Accuracy in this round: {best_iter_acc:.2f}%")

    model.load_state_dict(torch.load("best_pruned_model.pth"))
    apply_mask(model, cumulative_mask)
    return model, cumulative_mask