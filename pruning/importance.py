import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict
from tqdm.notebook import tqdm

from .utils import get_layer


def compute_activation_importance(model, source_loaders, device, mask=None):
    from .masking import apply_mask
    
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
    from .masking import apply_mask
    
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

    importance = OrderedDict()
    for name, scores in domain_means.items():
        mean = scores.mean(dim=1)
        var = scores.var(dim=1)
        importance[name] = (mean / (var + eps)).detach()

    for h in hooks: 
        h.remove()
    return importance


def compute_taylor_importance(model, loader, device, mask=None, num_batches=20):
    from .masking import apply_mask
    
    model.to(device)
    apply_mask(model, mask)
    model.train()

    taylor = {}
    conv_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight
            conv_names.append(name)
            taylor[name] = torch.zeros(w.shape[0], device=device)

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

    importance = OrderedDict()
    for name in conv_names:
        importance[name] = taylor[name] / count
    return importance


def compute_taylor_domain_stats(model, source_loaders, device, mask=None, num_batches_per_domain=20):
    from .masking import apply_mask
    
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
    from .masking import apply_mask
    
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
    
    for hook in hooks: 
        hook.remove()

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
