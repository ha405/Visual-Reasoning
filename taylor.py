import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.notebook import tqdm
from utils import *


def generate_mask_from_importance(model, importance, prune_rate=0.1, cumulative_mask=None):
    if cumulative_mask is None: cumulative_mask = {}
    new_mask = {}
    print(f"\nGenerating mask with base iterative prune rate: {prune_rate}")
    
    for layer_name, scores in importance.items():
        weight_name = f"{layer_name}.weight"
        if scores.dim() == 1:
            importance_metric = scores
        else:
            importance_metric = scores.var(dim=1)

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
            active_scores = importance_metric[active_indices]
            largest = False if scores.dim() == 1 else True
            _, prune_indices_in_subset = torch.topk(active_scores, k, largest=largest)
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

def compute_taylor_importance(model, loader, device, mask=None, num_batches=20):
    model.to(device)
    apply_mask(model, mask)
    model.train()
    
    taylor = {}
    conv_names = [name for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    
    for name in conv_names:
        nf = get_layer(model, name).weight.shape[0]
        taylor[name] = torch.zeros(nf, device=device)
    
    loss_fn = nn.CrossEntropyLoss()
    count = 0
    for xb, yb, _ in loader:
        xb, yb = xb.to(device), yb.to(device)
        model.zero_grad()
        out = model(xb)
        loss_fn(out, yb).backward()
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                g = module.weight.grad
                if g is not None:
                    score = (module.weight * g).abs().mean(dim=[1,2,3])
                    taylor[name] += score.detach()
        
        count += 1
        if count >= num_batches:
            break

    for name in conv_names:
        taylor[name] /= float(count)
    return taylor

def compute_taylor_domain_stats(model, source_loaders, device, mask=None, num_batches_per_domain=20):
    model.to(device)
    apply_mask(model, mask)
    model.train()
    
    conv_names = [name for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    per_domain_scores = OrderedDict()
    for name in conv_names:
        nf = get_layer(model, name).weight.shape[0]
        per_domain_scores[name] = torch.zeros((nf, len(source_loaders)), device=device)
    
    loss_fn = nn.CrossEntropyLoss()
    for d_idx, loader in enumerate(source_loaders):
        count = 0
        for xb, yb, _ in loader:
            xb, yb = xb.to(device), yb.to(device)
            model.zero_grad()
            out = model(xb)
            loss_fn(out, yb).backward()
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    g = module.weight.grad
                    if g is not None:
                        score = (module.weight * g).abs().mean(dim=[1,2,3])
                        per_domain_scores[name][:, d_idx] += score.detach()
            
            count += 1
            if count >= num_batches_per_domain:
                break
        
        if count > 0:
            for name in conv_names:
                per_domain_scores[name][:, d_idx] /= float(count)
    
    return per_domain_scores

def compute_taylor_meanvar_importance(model, source_loaders, device, mask=None, eps=1e-6, num_batches_per_domain=20):
    per_domain_scores = compute_taylor_domain_stats(model, source_loaders, device, mask, num_batches_per_domain)
    
    importance = OrderedDict()
    for name, scores in per_domain_scores.items():
        mean = scores.mean(dim=1)
        var = scores.var(dim=1)
        importance[name] = (mean / (var + eps)).detach()
    return importance

def generate_mask_from_taylor_importance(model, importance, prune_rate=0.1, cumulative_mask=None):
    if cumulative_mask is None: cumulative_mask = {}
    new_mask = {}
    
    for layer_name, scores in importance.items():
        weight_name = f"{layer_name}.weight"

        if scores.dim() == 1:
            importance_metric = scores
        else:
            importance_metric = scores.var(dim=1)

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
            active_scores = importance_metric[active_indices]
            largest = False if scores.dim() == 1 else True
            _, prune_indices_in_subset = torch.topk(active_scores, k, largest=largest)
            prune_indices_original = active_indices[prune_indices_in_subset]
            
            layer_mask = torch.ones(num_total_filters, device=scores.device)
            layer_mask[prune_indices_original] = 0.0
            
            module = get_layer(model, layer_name)
            full_mask = layer_mask.view(-1, 1, 1, 1).expand_as(module.weight)
            new_mask[weight_name] = full_mask.clone()
        else:
            layer_mask = torch.ones(num_total_filters, device=scores.device)
            module = get_layer(model, layer_name)
            full_mask = layer_mask.view(-1, 1, 1, 1).expand_as(module.weight)
            new_mask[weight_name] = full_mask.clone()
    return new_mask

def iterative_pruning_taylor(model, source_loaders_list, target_loader, device,
                             prune_rates, retrain_epochs, lr, alpha, batch_size, num_workers,
                             SFT=False, importance_type="taylor", keep_overall_best=True):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cumulative_mask = {} 
    
    _, best_overall_acc = evaluate(model, target_loader, device)
    torch.save(model.state_dict(), "best_pruned_model.pth")

    if keep_overall_best:
        overall_best_acc = best_overall_acc
        overall_best_ckpt = "best_overall_model.pth"
        torch.save(model.state_dict(), overall_best_ckpt)

    combined_source_loader = combine_source_loaders(source_loaders_list, batch_size, num_workers)

    for i, p_rate in enumerate(prune_rates):
        model.load_state_dict(torch.load("best_pruned_model.pth"))
        
        if importance_type == "taylor":
            importance = compute_taylor_importance(
                model, combined_source_loader, device, mask=cumulative_mask, num_batches=20
            )
        elif importance_type == "taylor_domain":
            importance = compute_taylor_domain_stats(
                model, source_loaders_list, device, mask=cumulative_mask, num_batches_per_domain=20
            )
        elif importance_type == "taylor_meanvar":
            importance = compute_taylor_meanvar_importance(
                model, source_loaders_list, device, mask=cumulative_mask, num_batches_per_domain=20
            )
        iter_mask = generate_mask_from_taylor_importance(model, importance, prune_rate=p_rate, cumulative_mask=cumulative_mask)
        
        for k, v in iter_mask.items():
            if k in cumulative_mask: cumulative_mask[k] = cumulative_mask[k] * v
            else: cumulative_mask[k] = v
        
        best_iter_acc = 0.0
        for epoch in range(retrain_epochs):
            if not SFT:
                train_DI(model, combined_source_loader, optimizer, device, epoch, alpha, cumulative_mask)
            else:
                train_SFT(model, combined_source_loader, optimizer, device, epoch, alpha, cumulative_mask)
            
            _, target_acc = evaluate(model, target_loader, device, mask=cumulative_mask)
            
            if target_acc > best_iter_acc:
                best_iter_acc = target_acc
                torch.save(model.state_dict(), "best_pruned_model.pth")
            
            if keep_overall_best and target_acc > overall_best_acc:
                overall_best_acc = target_acc
                torch.save(model.state_dict(), overall_best_ckpt)
    
    if keep_overall_best:
        model.load_state_dict(torch.load(overall_best_ckpt))
    else:
        model.load_state_dict(torch.load("best_pruned_model.pth"))
    
    return model, cumulative_mask

