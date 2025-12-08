import torch
import torch.nn as nn
from collections import OrderedDict

from .utils import get_layer, apply_mask
from .training import evaluate


def generate_mask_from_importance(model, importance, prune_rate=0.1, cumulative_mask=None):
    if cumulative_mask is None: 
        cumulative_mask = {}
    new_mask = {}
    print(f"\nGenerating mask with base iterative prune rate: {prune_rate}")
    
    total_pruned = 0
    total_params = 0
    
    for layer_name, scores in importance.items():
        weight_name = f"{layer_name}.weight"

        if scores.dim() == 1:
            var_across_domains = scores
        else:
            var_across_domains = scores.var(dim=1)

        num_total_filters = scores.shape[0]
        total_params += num_total_filters

        if weight_name in cumulative_mask:
            prev_mask_flat = cumulative_mask[weight_name][:, 0, 0, 0]
            active_indices = torch.where(prev_mask_flat == 1)[0]
        else:
            active_indices = torch.arange(num_total_filters, device=scores.device)
        
        num_active_filters = len(active_indices)
        if layer_name.startswith('conv1') or layer_name.startswith('layer1'): 
            current_prune_rate = prune_rate / 4.0
        elif layer_name.startswith('layer2'): 
            current_prune_rate = prune_rate / 2.0
        else: 
            current_prune_rate = prune_rate

        k = int(current_prune_rate * num_active_filters)

        if k > 0 and num_active_filters > 0:
            active_scores_var = var_across_domains[active_indices]
            largest = False if scores.dim() == 1 else True
            _, prune_indices_in_subset = torch.topk(active_scores_var, k, largest=largest)
            prune_indices_original = active_indices[prune_indices_in_subset]
            
            layer_mask = torch.ones(num_total_filters, device=scores.device)
            layer_mask[prune_indices_original] = 0.0
            total_pruned += k
            
            module = get_layer(model, layer_name)
            full_mask = layer_mask.view(-1, 1, 1, 1).expand_as(module.weight)
            new_mask[weight_name] = full_mask.clone()
    
    print(f"  - Pruned {total_pruned} filters across all layers.")
    return new_mask


def generate_mask_from_taylor_importance(model, importance, prune_rate=0.1, cumulative_mask=None):
    if cumulative_mask is None:
        cumulative_mask = {}
    new_mask = {}
    print(f"\nGenerating Taylor mask with base prune rate: {prune_rate}")

    total_pruned = 0
    for layer_name, scores in importance.items():
        weight_name = f"{layer_name}.weight"
        num_total_filters = scores.shape[0]

        if weight_name in cumulative_mask:
            prev_mask_flat = cumulative_mask[weight_name][:, 0, 0, 0]
            active_indices = torch.where(prev_mask_flat == 1)[0]
        else:
            active_indices = torch.arange(num_total_filters, device=scores.device)

        num_active_filters = len(active_indices)
        
        if layer_name.startswith('conv1') or layer_name.startswith('layer1'):
            current_prune_rate = prune_rate / 4.0
        elif layer_name.startswith('layer2'):
            current_prune_rate = prune_rate / 2.0
        else:
            current_prune_rate = prune_rate

        k = int(current_prune_rate * num_active_filters)

        if k > 0 and num_active_filters > 0:
            active_scores = scores[active_indices]
            _, prune_indices_in_subset = torch.topk(active_scores, k, largest=False)
            prune_indices_original = active_indices[prune_indices_in_subset]

            layer_mask = torch.ones(num_total_filters, device=scores.device)
            layer_mask[prune_indices_original] = 0.0
            total_pruned += k

            module = get_layer(model, layer_name)
            full_mask = layer_mask.view(-1, 1, 1, 1).expand_as(module.weight)
            new_mask[weight_name] = full_mask.clone()

    print(f"  - Pruned {total_pruned} filters across all layers.")
    return new_mask


def _create_layer_mask(module, indices_to_prune):
    device = module.weight.device
    num_units = module.weight.shape[0]
    mask = torch.ones(num_units, device=device)
    if indices_to_prune.numel() > 0:
        mask[indices_to_prune] = 0.0
    if isinstance(module, nn.Conv2d):
        return mask.view(-1, 1, 1, 1).expand_as(module.weight)
    else:
        return mask.view(-1, 1).expand_as(module.weight)


def select_prune_mask_by_source_heuristics(model, source_calib_loaders, device, cumulative_mask,
                                           candidate_rates, relative_acc_drop_threshold):
    model.to(device).eval()
    iter_mask = {}
    from .importance import compute_magnitude_importance
    importance = compute_magnitude_importance(model)
    print("\nSelecting per-layer prune rates using source domain heuristics (magnitude)...")
    baseline_accs = torch.tensor([evaluate(model, loader, device, mask=cumulative_mask)[1] for loader in source_calib_loaders])
    
    for layer_name, scores in importance.items():
        module = get_layer(model, layer_name)
        weight_name = f"{layer_name}.weight"
        
        active_indices = torch.where(cumulative_mask.get(weight_name, torch.ones_like(scores))[:, 0, 0, 0] == 1)[0] if scores.dim() > 1 else torch.where(cumulative_mask.get(weight_name, torch.ones_like(scores)) == 1)[0]
        if len(active_indices) == 0: 
            continue
        
        active_scores = scores[active_indices]
        sorted_indices_in_active = torch.argsort(active_scores, descending=False)
        best_rate = 0.0
        for r in sorted(candidate_rates):
            k = int(r * len(active_indices))
            if k == 0: 
                continue
            prune_indices = active_indices[sorted_indices_in_active[:k]]
            
            temp_mask = {k_c: v_c.clone() for k_c, v_c in cumulative_mask.items()} if cumulative_mask else {}
            current_layer_mask = _create_layer_mask(module, prune_indices)
            temp_mask[weight_name] = cumulative_mask.get(weight_name, torch.ones_like(current_layer_mask)) * current_layer_mask

            current_accs = torch.tensor([evaluate(model, loader, device, mask=temp_mask)[1] for loader in source_calib_loaders])
            if torch.all(current_accs >= baseline_accs * (1.0 - relative_acc_drop_threshold)):
                best_rate = r

        if best_rate > 0:
            k = int(best_rate * len(active_indices))
            prune_indices = active_indices[sorted_indices_in_active[:k]]
            iter_mask[weight_name] = _create_layer_mask(module, prune_indices)
            print(f"  - Layer {layer_name}: selected rate {best_rate*100:.0f}%.")
        else:
            print(f"  - Layer {layer_name}: No rate met the threshold. Not pruned.")
    return iter_mask


def select_prune_mask_by_target_error(model, target_calib_loader, device, cumulative_mask, candidate_rates):
    model.to(device).eval()
    iter_mask = {}
    from .importance import compute_error_correlated_importance
    importance = compute_error_correlated_importance(model, target_calib_loader, device, cumulative_mask)
    
    if importance is None:
        print("Could not compute error-based importance. Skipping target pruning.")
        return {}

    print("\nSelecting per-layer prune rates by removing neurons correlated with target error...")
    _, base_acc = evaluate(model, target_calib_loader, device, mask=cumulative_mask)

    for layer_name, scores in importance.items():
        module = get_layer(model, layer_name)
        weight_name = f"{layer_name}.weight"

        active_indices = torch.where(cumulative_mask.get(weight_name, torch.ones_like(scores))[:, 0, 0, 0] == 1)[0] if scores.dim() > 1 else torch.where(cumulative_mask.get(weight_name, torch.ones_like(scores)) == 1)[0]
        if len(active_indices) == 0: 
            continue
        
        active_scores = scores[active_indices]
        sorted_indices_in_active = torch.argsort(active_scores, descending=True)
        
        best_post_prune_acc = -1.0
        best_rate = 0.0
        for r in candidate_rates:
            k = int(r * len(active_indices))
            if k == 0: 
                continue
            
            prune_indices = active_indices[sorted_indices_in_active[:k]]
            temp_mask = {k_c: v_c.clone() for k_c, v_c in cumulative_mask.items()} if cumulative_mask else {}
            current_layer_mask = _create_layer_mask(module, prune_indices)
            temp_mask[weight_name] = cumulative_mask.get(weight_name, torch.ones_like(current_layer_mask)) * current_layer_mask
            
            _, acc = evaluate(model, target_calib_loader, device, mask=temp_mask)
            if acc > best_post_prune_acc:
                best_post_prune_acc = acc
                best_rate = r
        
        if best_rate > 0:
            k = int(best_rate * len(active_indices))
            prune_indices = active_indices[sorted_indices_in_active[:k]]
            iter_mask[weight_name] = _create_layer_mask(module, prune_indices)
            print(f"  - Layer {layer_name}: selected rate {best_rate*100:.0f}%. Best post-prune acc: {best_post_prune_acc:.2f}% (from {base_acc:.2f}%)")
        else:
            print(f"  - Layer {layer_name}: No pruning of error-correlated neurons improved accuracy.")

    return iter_mask
