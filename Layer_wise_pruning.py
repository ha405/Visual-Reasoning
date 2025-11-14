import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torch.optim as optim
from utils import *

def get_layer(model, name):
    return dict(model.named_modules())[name]

def create_calibration_loaders(loaders, num_samples, is_target=False):
    if num_samples is None:
        return loaders
    
    calib_loaders = []
    if not isinstance(loaders, list):
        loaders = [loaders]

    print(f"Creating calibration loaders with {num_samples} samples each...")
    for loader in loaders:
        dataset = loader.dataset
        num_dataset_samples = len(dataset)
        samples_to_take = min(num_samples, num_dataset_samples)
        
        random_indices = torch.randperm(num_dataset_samples)[:samples_to_take].tolist()
        subset = Subset(dataset, random_indices)
        calib_loader = DataLoader(subset, batch_size=loader.batch_size, shuffle=False, num_workers=loader.num_workers)
        calib_loaders.append(calib_loader)
        
    return calib_loaders[0] if is_target else calib_loaders

def compute_magnitude_importance(model):
    importance = OrderedDict()
    for name, module in model.named_modules():
        if not (hasattr(module, "weight") and isinstance(module, (nn.Conv2d, nn.Linear))):
            continue
        weights = module.weight.data
        if isinstance(module, nn.Conv2d):
            l1_norm = torch.sum(torch.abs(weights), dim=(1, 2, 3))
        else:
            l1_norm = torch.sum(torch.abs(weights), dim=1)
        importance[name] = l1_norm
    return importance

def compute_error_correlated_importance(model, target_calib_loader, device, mask=None):
    model.to(device).eval()
    apply_mask(model, mask)
    
    hooks, activations = [], defaultdict(list)
    layers_to_hook = [name for name, module in model.named_modules() if isinstance(module, (nn.Conv2d, nn.Linear))]

    def get_activation(name):
        def hook(model, input, output):
            activations[name].append(output)
        return hook

    for name in layers_to_hook:
        hooks.append(get_layer(model, name).register_forward_hook(get_activation(name)))

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in target_calib_loader:
            outputs = model(inputs.to(device))
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    for hook in hooks: hook.remove()

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    incorrect_indices = torch.where(all_preds != all_labels)[0]

    if len(incorrect_indices) == 0:
        print("Warning: No incorrect predictions on calibration set. Cannot compute error-correlated importance.")
        return None

    importance = OrderedDict()
    for layer_name, layer_activations_list in activations.items():
        layer_activations = torch.cat(layer_activations_list, dim=0)
        error_activations = layer_activations[incorrect_indices]
        
        if error_activations.dim() == 4:
            mean_error_activations = error_activations.mean(dim=[0, 2, 3])
        else:
            mean_error_activations = error_activations.mean(dim=0)
        importance[layer_name] = mean_error_activations.detach()

    print("Computed importance based on correlation with target errors.")
    return importance

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
    importance = compute_magnitude_importance(model)
    print("\nSelecting per-layer prune rates using source domain heuristics (magnitude)...")
    baseline_accs = torch.tensor([evaluate(model, loader, device, mask=cumulative_mask)[1] for loader in source_calib_loaders])
    
    for layer_name, scores in importance.items():
        module = get_layer(model, layer_name)
        weight_name = f"{layer_name}.weight"
        
        active_indices = torch.where(cumulative_mask.get(weight_name, torch.ones_like(scores))[:, 0, 0, 0] == 1)[0] if scores.dim() > 1 else torch.where(cumulative_mask.get(weight_name, torch.ones_like(scores)) == 1)[0]
        if len(active_indices) == 0: continue
        
        active_scores = scores[active_indices]
        sorted_indices_in_active = torch.argsort(active_scores, descending=False)
        best_rate = 0.0
        for r in sorted(candidate_rates):
            k = int(r * len(active_indices))
            if k == 0: continue
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
        if len(active_indices) == 0: continue
        
        active_scores = scores[active_indices]
        sorted_indices_in_active = torch.argsort(active_scores, descending=True)
        
        best_post_prune_acc = -1.0
        best_rate = 0.0
        for r in candidate_rates:
            k = int(r * len(active_indices))
            if k == 0: continue
            
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

def combine_source_loaders(source_loaders, batch_size, num_workers):
    combined_dataset = ConcatDataset([loader.dataset for loader in source_loaders])
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def iterative_pruning(model, source_loaders_list, target_loader, device,
                      retrain_epochs, lr, alpha, batch_size, num_workers, SFT=False,
                      pruning_strategy="target_error",
                      candidate_rates=(0.20, 0.30, 0.40, 0.60),
                      iterations=3, calibration_samples=1000,
                      relative_acc_drop_threshold=0.05):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    cumulative_mask = {}

    _, best_overall_acc = evaluate(model, target_loader, device)
    print(f"Initial Baseline Target Accuracy: {best_overall_acc:.2f}%")
    torch.save(model.state_dict(), "best_pruned_model.pth")
    
    combined_source_loader = combine_source_loaders(source_loaders_list, batch_size, num_workers)
    source_calib_loaders = create_calibration_loaders(source_loaders_list, calibration_samples)
    target_calib_loader = create_calibration_loaders(target_loader, calibration_samples, is_target=True)

    for i in range(iterations):
        print(f"\n--- Pruning Iteration {i+1}/{iterations} ---")
        model.load_state_dict(torch.load("best_pruned_model.pth"))

        if pruning_strategy == "source_magnitude":
            iter_mask = select_prune_mask_by_source_heuristics(model, source_calib_loaders, device, cumulative_mask, candidate_rates, relative_acc_drop_threshold)
        elif pruning_strategy == "target_error":
            iter_mask = select_prune_mask_by_target_error(model, target_calib_loader, device, cumulative_mask, candidate_rates)
        else:
            raise ValueError(f"Unknown pruning_strategy: {pruning_strategy}")

        if not iter_mask:
            print("No further pruning was selected. Stopping.")
            break

        for k, v in iter_mask.items():
            cumulative_mask[k] = cumulative_mask.get(k, torch.ones_like(v)) * v

        best_iter_acc = 0.0
        for epoch in range(retrain_epochs):
            print(f"\nRetraining Epoch {epoch+1}/{retrain_epochs} on full source data")
            train_func = train_SFT if SFT else train_DI
            train_func(model, combined_source_loader, optimizer, device, epoch, alpha, cumulative_mask)

            _, target_acc = evaluate(model, target_loader, device, mask=cumulative_mask)
            print(f"  Epoch {epoch+1} Full Target Accuracy: {target_acc:.2f}%")
            if target_acc > best_iter_acc:
                best_iter_acc = target_acc
                torch.save(model.state_dict(), "best_pruned_model.pth")
        print(f"Iteration {i+1} | Best Accuracy in this round: {best_iter_acc:.2f}%")

    model.load_state_dict(torch.load("best_pruned_model.pth"))
    apply_mask(model, cumulative_mask)
    return model, cumulative_mask