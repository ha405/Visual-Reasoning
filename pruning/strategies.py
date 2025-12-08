import torch
import torch.optim as optim

from .importance import (
    compute_activation_importance,
    compute_meanvar_importance,
    compute_taylor_importance,
    compute_taylor_domain_stats,
    compute_taylor_meanvar_importance,
)
from .masking import (
    generate_mask_from_importance,
    generate_mask_from_taylor_importance,
    select_prune_mask_by_source_heuristics,
    select_prune_mask_by_target_error,
    apply_mask,
)
from .training import train_DI, train_SFT, evaluate
from .utils import combine_source_loaders, create_calibration_loaders


def fixed_rate_pruning(model, source_loaders_list, target_loader, device,
                       prune_rates, retrain_epochs, lr, alpha, batch_size, num_workers,
                       SFT=False, importance_type="activation", keep_overall_best=True):
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    cumulative_mask = {} 
    
    _, best_overall_acc = evaluate(model, target_loader, device)
    print(f"Initial Baseline Target Accuracy: {best_overall_acc:.2f}%")
    torch.save({'model': model.state_dict(), 'mask': cumulative_mask}, "best_pruned_model.pth")

    if keep_overall_best:
        overall_best_acc = best_overall_acc
        overall_best_ckpt = "best_overall_model.pth"
        torch.save({'model': model.state_dict(), 'mask': cumulative_mask}, overall_best_ckpt)

    combined_source_loader = combine_source_loaders(source_loaders_list, batch_size, num_workers)

    for i, p_rate in enumerate(prune_rates):
        it = i + 1
        print(f"\n--- Pruning Iteration {it}/{len(prune_rates)} with base rate {p_rate} ---")
        
        checkpoint = torch.load("best_pruned_model.pth")
        model.load_state_dict(checkpoint['model'])
        cumulative_mask = checkpoint['mask']
        
        if importance_type == "activation":
            importance = compute_activation_importance(
                model, source_loaders_list, device, mask=cumulative_mask
            )
        elif importance_type == "meanvar":
            importance = compute_meanvar_importance(
                model, source_loaders_list, device, mask=cumulative_mask
            )
        elif importance_type == "taylor_meanvar":
            importance = compute_taylor_meanvar_importance(
                model, source_loaders_list, device, mask=cumulative_mask
            )
        else:
            importance = compute_taylor_importance(
                model, combined_source_loader, device, mask=cumulative_mask, num_batches=20
            )
        iter_mask = generate_mask_from_importance(model, importance, prune_rate=p_rate, cumulative_mask=cumulative_mask)
        
        for k, v in iter_mask.items():
            if k in cumulative_mask: 
                cumulative_mask[k] = cumulative_mask[k] * v
            else: 
                cumulative_mask[k] = v
        
        best_iter_acc = 0.0
        best_iter_ckpt = "best_iter_model.pth"
        
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
                torch.save({'model': model.state_dict(), 'mask': cumulative_mask}, best_iter_ckpt)
            
            if keep_overall_best and target_acc > overall_best_acc:
                overall_best_acc = target_acc
                torch.save({'model': model.state_dict(), 'mask': cumulative_mask}, overall_best_ckpt)
                print(f"  [New Overall Best: {overall_best_acc:.2f}%]")
        
        print(f"Iteration {it} | Best Accuracy in this round: {best_iter_acc:.2f}%")
        if keep_overall_best:
            print(f"Overall Best Accuracy so far: {overall_best_acc:.2f}%")
        
        # Always load the best model from THIS iteration to proceed
        checkpoint = torch.load(best_iter_ckpt)
        model.load_state_dict(checkpoint['model'])
        cumulative_mask = checkpoint['mask']
        torch.save(checkpoint, "best_pruned_model.pth")
    
    if keep_overall_best:
        print(f"\nLoading overall best model (Acc: {overall_best_acc:.2f}%)")
        checkpoint = torch.load(overall_best_ckpt)
        model.load_state_dict(checkpoint['model'])
        cumulative_mask = checkpoint['mask']
    else:
        print(f"\nLoading final pruned model (Acc: {best_iter_acc:.2f}%)")
        checkpoint = torch.load("best_pruned_model.pth")
        model.load_state_dict(checkpoint['model'])
        cumulative_mask = checkpoint['mask']
    
    return model, cumulative_mask


def taylor_pruning(model, source_loaders_list, target_loader, device,
                   prune_rates, retrain_epochs, lr, alpha, batch_size, num_workers,
                   SFT=False, importance_type="taylor", keep_overall_best=True):
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    cumulative_mask = {}
    
    _, best_overall_acc = evaluate(model, target_loader, device)
    torch.save({'model': model.state_dict(), 'mask': cumulative_mask}, "best_pruned_model.pth")

    if keep_overall_best:
        overall_best_acc = best_overall_acc
        overall_best_ckpt = "best_overall_model.pth"
        torch.save({'model': model.state_dict(), 'mask': cumulative_mask}, overall_best_ckpt)

    combined_source_loader = combine_source_loaders(source_loaders_list, batch_size, num_workers)

    for i, p_rate in enumerate(prune_rates):
        checkpoint = torch.load("best_pruned_model.pth")
        model.load_state_dict(checkpoint['model'])
        cumulative_mask = checkpoint['mask']
        
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
            if k in cumulative_mask:
                cumulative_mask[k] = cumulative_mask[k] * v
            else:
                cumulative_mask[k] = v
        
        best_iter_acc = 0.0
        best_iter_ckpt = "best_iter_model.pth"

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
                torch.save({'model': model.state_dict(), 'mask': cumulative_mask}, best_iter_ckpt)
            
            if keep_overall_best and target_acc > overall_best_acc:
                overall_best_acc = target_acc
                torch.save({'model': model.state_dict(), 'mask': cumulative_mask}, overall_best_ckpt)
                print(f"  [New Overall Best: {overall_best_acc:.2f}%]")
        
        print(f"Iteration {i+1} | Best Accuracy in this round: {best_iter_acc:.2f}%")
        if keep_overall_best:
            print(f"Overall Best Accuracy so far: {overall_best_acc:.2f}%")

        # Always load the best model from THIS iteration to proceed
        checkpoint = torch.load(best_iter_ckpt)
        model.load_state_dict(checkpoint['model'])
        cumulative_mask = checkpoint['mask']
        torch.save(checkpoint, "best_pruned_model.pth")
    
    if keep_overall_best:
        print(f"\nLoading overall best model (Acc: {overall_best_acc:.2f}%)")
        checkpoint = torch.load(overall_best_ckpt)
        model.load_state_dict(checkpoint['model'])
        cumulative_mask = checkpoint['mask']
    else:
        print(f"\nLoading final pruned model (Acc: {best_iter_acc:.2f}%)")
        checkpoint = torch.load("best_pruned_model.pth")
        model.load_state_dict(checkpoint['model'])
        cumulative_mask = checkpoint['mask']
    
    return model, cumulative_mask


def adaptive_layerwise_pruning(model, source_loaders_list, target_loader, device,
                                retrain_epochs, lr, alpha, batch_size, num_workers, SFT=False,
                                pruning_strategy="target_error",
                                candidate_rates=(0.10, 0.20, 0.30, 0.40),
                                iterations=3, calibration_samples=1000,
                                relative_acc_drop_threshold=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    cumulative_mask = {}

    _, best_overall_acc = evaluate(model, target_loader, device)
    print(f"Initial Baseline Target Accuracy: {best_overall_acc:.2f}%")
    torch.save({'model': model.state_dict(), 'mask': cumulative_mask}, "best_pruned_model.pth")
    
    combined_source_loader = combine_source_loaders(source_loaders_list, batch_size, num_workers)
    source_calib_loaders = create_calibration_loaders(source_loaders_list, calibration_samples)
    target_calib_loader = create_calibration_loaders(target_loader, calibration_samples, is_target=True)

    for i in range(iterations):
        print(f"\n--- Pruning Iteration {i+1}/{iterations} ---")
        checkpoint = torch.load("best_pruned_model.pth")
        model.load_state_dict(checkpoint['model'])
        cumulative_mask = checkpoint['mask']

        if pruning_strategy == "source_magnitude":
            iter_mask = select_prune_mask_by_source_heuristics(
                model, source_calib_loaders, device, cumulative_mask, 
                candidate_rates, relative_acc_drop_threshold
            )
        elif pruning_strategy == "target_error":
            iter_mask = select_prune_mask_by_target_error(
                model, target_calib_loader, device, cumulative_mask, candidate_rates
            )
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
                torch.save({'model': model.state_dict(), 'mask': cumulative_mask}, "best_pruned_model.pth")
        print(f"Iteration {i+1} | Best Accuracy in this round: {best_iter_acc:.2f}%")

    checkpoint = torch.load("best_pruned_model.pth")
    model.load_state_dict(checkpoint['model'])
    cumulative_mask = checkpoint['mask']
    apply_mask(model, cumulative_mask)
    return model, cumulative_mask
