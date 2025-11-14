import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict
from tqdm.notebook import tqdm
from torch.utils.data import ConcatDataset, DataLoader
import torch.optim as optim
from utils import *

def get_layer(model, name):
    return dict(model.named_modules())[name]

def compute_activation_importance(model, source_loaders, device, mask=None):
    model.to(device)
    apply_mask(model, mask)
    model.eval()
    conv_layer_names = [name for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    activations = defaultdict(list)

    def hook_fn(module, input, output):
        activations[module.layer_name].append(output.mean(dim=[0, 2, 3]).detach())

    hooks = []
    for name in conv_layer_names:
        layer = get_layer(model, name)
        layer.layer_name = name 
        hooks.append(layer.register_forward_hook(hook_fn))
    
    importance = OrderedDict()
    for name in conv_layer_names:
        num_filters = get_layer(model, name).weight.shape[0]
        importance[name] = torch.zeros((num_filters, len(source_loaders)), device=device)

    print("Computing filter activations per domain...")
    for d_idx, loader in enumerate(source_loaders):
        print(f"  - Domain {d_idx+1}/{len(source_loaders)}")
        activations.clear()
        with torch.no_grad():
            for inputs, _, _ in tqdm(loader, leave=False):
                model(inputs.to(device))
        for name in conv_layer_names:
            domain_mean_activations = torch.mean(torch.stack(activations[name], dim=0), dim=0)
            importance[name][:, d_idx] = domain_mean_activations
    for h in hooks:
        h.remove()
    return importance

def compute_meanvar_importance(model, source_loaders, device, mask=None, eps=1e-6):
    model.to(device)
    apply_mask(model, mask)
    model.eval()

    conv_layer_names = [name for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    activations = defaultdict(list)

    def hook_fn(module, input, output):
        activations[module.layer_name].append(output.mean(dim=[0, 2, 3]).detach())

    hooks = []
    for name in conv_layer_names:
        layer = get_layer(model, name)
        layer.layer_name = name
        hooks.append(layer.register_forward_hook(hook_fn))

    domain_means = OrderedDict()
    for name in conv_layer_names:
        nf = get_layer(model, name).weight.shape[0]
        domain_means[name] = torch.zeros((nf, len(source_loaders)), device=device)

    print("Computing mean/variance importance...")
    for d_idx, loader in enumerate(source_loaders):
        activations.clear()
        with torch.no_grad():
            for x, _, _ in loader:
                model(x.to(device))
        for name in conv_layer_names:
            domain_means[name][:, d_idx] = torch.stack(activations[name]).mean(0)

    # Convert domain_means -> final importance score
    importance = OrderedDict()
    for name, scores in domain_means.items():
        mean = scores.mean(dim=1)
        var = scores.var(dim=1)
        importance[name] = (mean / (var + eps)).detach()

    for h in hooks: h.remove()
    return importance


def compute_taylor_importance(model, loader, device, mask=None, num_batches=20):
    model.to(device)
    apply_mask(model, mask)
    model.train()  # we need gradients

    # Zero accumulator
    taylor = {}
    conv_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight
            conv_names.append(name)
            taylor[name] = torch.zeros(w.shape[0], device=device)

    # Accumulate |w * grad|
    count = 0
    for xb, yb, _ in loader:
        xb, yb = xb.to(device), yb.to(device)

        model.zero_grad()
        out = model(xb)
        loss = nn.CrossEntropyLoss()(out, yb)
        loss.backward()

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                w = module.weight
                g = module.weight.grad
                if g is not None:
                    score = (w * g).abs().mean(dim=[1, 2, 3])
                    taylor[name] += score.detach()

        count += 1
        if count >= num_batches:
            break

    # Normalize and format into your importance structure
    importance = OrderedDict()
    for name in conv_names:
        importance[name] = taylor[name] / count

    return importance


def generate_mask_from_importance(model, importance, prune_rate=0.1, cumulative_mask=None):
    if cumulative_mask is None: cumulative_mask = {}
    new_mask = {}
    print(f"\nGenerating mask with base iterative prune rate: {prune_rate}")
    
    for layer_name, scores in importance.items():
        weight_name = f"{layer_name}.weight"

        # handle both 1D and 2D importance
        if scores.dim() == 1:
            var_across_domains = scores
        else:
            var_across_domains = scores.var(dim=1)

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
            active_scores_var = var_across_domains[active_indices]
            # For 1D importance, prune smallest scores
            largest = False if scores.dim() == 1 else True
            _, prune_indices_in_subset = torch.topk(active_scores_var, k, largest=largest)
            prune_indices_original = active_indices[prune_indices_in_subset]
            
            layer_mask = torch.ones(num_total_filters, device=scores.device)
            layer_mask[prune_indices_original] = 0.0
            print(f"  - Layer '{layer_name}': Pruning {k}/{num_active_filters} active filters (rate {current_prune_rate:.3f}).")
            
            module = get_layer(model, layer_name)
            full_mask = layer_mask.view(-1, 1, 1, 1).expand_as(module.weight)
            new_mask[weight_name] = full_mask.clone()
        else:
            print(f"  - Layer '{layer_name}': No filters pruned.")

    return new_mask


def combine_source_loaders(source_loaders, batch_size, num_workers):
    datasets = [loader.dataset for loader in source_loaders]
    combined_dataset = ConcatDataset(datasets)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def iterative_pruning(model, source_loaders_list, target_loader, device,
                      prune_rates, retrain_epochs, lr, alpha, batch_size, num_workers, SFT=False, importance_type="activation"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    cumulative_mask = {} 
    
    _, best_overall_acc = evaluate(model, target_loader, device)
    print(f"Initial Baseline Target Accuracy: {best_overall_acc:.2f}%")
    torch.save(model.state_dict(), "best_pruned_model.pth")

    combined_source_loader = combine_source_loaders(source_loaders_list, batch_size, num_workers)

    for i, p_rate in enumerate(prune_rates):
        it = i + 1
        print(f"\n--- Pruning Iteration {it}/{len(prune_rates)} with base rate {p_rate} ---")
        
        model.load_state_dict(torch.load("best_pruned_model.pth"))
        
        if importance_type == "activation":
            importance = compute_activation_importance(
                model, source_loaders_list, device, mask=cumulative_mask
            )
        elif importance_type == "meanvar":
            importance = compute_meanvar_importance(
                model, source_loaders_list, device, mask=cumulative_mask
            )
        else:
            importance = compute_taylor_importance(
                model, combined_source_loader, device, mask=cumulative_mask, num_batches=20
            )
        iter_mask = generate_mask_from_importance(model, importance, prune_rate=p_rate, cumulative_mask=cumulative_mask)
        
        for k, v in iter_mask.items():
            if k in cumulative_mask: cumulative_mask[k] = cumulative_mask[k] * v
            else: cumulative_mask[k] = v
        
        best_iter_acc = 0.0
        for epoch in range(retrain_epochs):
            print(f"\nRetraining Epoch {epoch+1}/{retrain_epochs}")
            if not SFT:
                train_DI(model, combined_source_loader, optimizer, device, epoch, alpha, cumulative_mask)
            else:
                train_SFT(model, combined_source_loader, optimizer, device, epoch, alpha, cumulative_mask)
            
            _, target_acc = evaluate(model, target_loader, device, mask=cumulative_mask)
            print(f"  Epoch {epoch+1} Target Accuracy: {target_acc:.2f}%")
            if target_acc > best_iter_acc:
                best_iter_acc = target_acc
                torch.save(model.state_dict(), "best_pruned_model.pth")
        print(f"Iteration {it} | Best Accuracy in this round: {best_iter_acc:.2f}%")

    model.load_state_dict(torch.load("best_pruned_model.pth"))
    return model, cumulative_mask