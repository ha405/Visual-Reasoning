#!/usr/bin/env python
import os
import argparse
import random
import numpy as np
import torch

from pruning.config import load_config
from pruning import (
    get_dataloaders,
    get_model,
    fixed_rate_pruning,
    adaptive_layerwise_pruning,
    taylor_pruning,
    train_vanilla,
    evaluate,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Neural Network Pruning for Domain Generalization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--data_dir', type=str, default=None, help='Root directory of dataset')
    parser.add_argument('--target_domain', type=str, default=None, help='Target domain name')
    parser.add_argument('--source_domains', nargs='+', default=None, help='Source domain names')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of dataloader workers')
    
    parser.add_argument('--model', type=str, default=None, help='Model architecture')
    parser.add_argument('--pretrained', type=lambda x: str(x).lower() == 'true', default=None, help='Use pretrained weights')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to model checkpoint')
    
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=None, help='DI variance weight')
    parser.add_argument('--training_strategy', type=str, default=None, choices=['DI', 'SFT', 'vanilla'], help='Training strategy')
    parser.add_argument('--warmup_epochs', type=int, default=None, help='Number of warmup epochs')
    parser.add_argument('--retrain_epochs', type=int, default=None, help='Number of retraining epochs per pruning iteration')
    
    parser.add_argument('--pruning_method', type=str, default=None, choices=['fixed_rate', 'adaptive', 'taylor'], help='Pruning method')
    parser.add_argument('--importance_type', type=str, default=None,
                       choices=['activation', 'meanvar', 'taylor', 'taylor_domain', 'taylor_meanvar', 'magnitude', 'error_correlated'],
                       help='Importance metric type')
    parser.add_argument('--prune_rates', nargs='+', type=float, default=None, help='Pruning rates for each iteration')
    parser.add_argument('--pruning_strategy', type=str, default=None, choices=['target_error', 'source_magnitude'], help='Adaptive pruning strategy')
    parser.add_argument('--iterations', type=int, default=None, help='Number of pruning iterations (for adaptive)')
    
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    
    parser.add_argument('--device', type=str, default=None, help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    cli_overrides = {}
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name != 'config':
            cli_overrides[arg_name] = arg_value
    
    config = load_config(config_path=args.config, cli_overrides=cli_overrides)
    
    set_seed(config.seed)
    
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if config.experiment_name is None:
        config.experiment_name = f"{config.dataset}_{config.target_domain}_{config.importance_type}_{config.training_strategy}_{config.model}"
    
    if config.output_dir == './outputs':
        config.output_dir = f"./outputs/{config.experiment_name}"
    
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    config_save_path = os.path.join(config.output_dir, 'config.yaml')
    config.save_yaml(config_save_path)
    print(f"Saved configuration to: {config_save_path}")
    
    print("\n" + "="*60)
    print("EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"Dataset: {config.dataset}")
    print(f"Model: {config.model}")
    print(f"Pruning Method: {config.pruning_method}")
    print(f"Importance Type: {config.importance_type}")
    print(f"Training Strategy: {config.training_strategy}")
    print(f"Target Domain: {config.target_domain}")
    print(f"Source Domains: {config.source_domains}")
    print("="*60 + "\n")
    
    print("Loading dataset...")
    source_loaders, target_loader, class_to_idx = get_dataloaders(
        dataset_name=config.dataset,
        data_dir=config.data_dir,
        source_domains=config.source_domains,
        target_domain=config.target_domain,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        combine_sources=False
    )
    
    num_classes = config.num_classes if config.num_classes else len(class_to_idx)
    print(f"Number of classes: {num_classes}\n")
    
    print(f"Loading model: {config.model}")
    model = get_model(
        model_name=config.model,
        pretrained=config.pretrained,
        num_classes=num_classes,
        checkpoint_path=config.checkpoint_path
    )
    model = model.to(device)
    print(f"Model loaded successfully\n")
    
    if config.warmup_epochs > 0 and config.checkpoint_path is None:
        print("="*60)
        print("WARMUP PHASE")
        print("="*60)
        from pruning import combine_source_loaders
        import torch.optim as optim
        
        combined_loader = combine_source_loaders(source_loaders, config.batch_size, config.num_workers)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        
        best_acc = 0.0
        for epoch in range(config.warmup_epochs):
            print(f"\nWarmup Epoch {epoch+1}/{config.warmup_epochs}")
            train_vanilla(model, combined_loader, optimizer, device, epoch)
            
            _, target_acc = evaluate(model, target_loader, device)
            print(f"Target Accuracy: {target_acc:.2f}%")
            
            if target_acc > best_acc:
                best_acc = target_acc
                warmup_ckpt = os.path.join(config.checkpoint_dir, 'warmup_best.pth')
                torch.save(model.state_dict(), warmup_ckpt)
                print(f"Saved warmup checkpoint: {warmup_ckpt}")
        
        model.load_state_dict(torch.load(warmup_ckpt))
        print(f"\nWarmup complete. Best accuracy: {best_acc:.2f}%\n")
    
    print("="*60)
    print("PRUNING PHASE")
    print("="*60)
    
    SFT = (config.training_strategy == 'SFT')
    
    if config.pruning_method == 'fixed_rate':
        pruned_model, mask = fixed_rate_pruning(
            model=model,
            source_loaders_list=source_loaders,
            target_loader=target_loader,
            device=device,
            prune_rates=config.prune_rates,
            retrain_epochs=config.retrain_epochs,
            lr=config.lr,
            alpha=config.alpha,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            SFT=SFT,
            importance_type=config.importance_type,
            keep_overall_best=config.keep_overall_best
        )
    
    elif config.pruning_method == 'taylor':
        pruned_model, mask = taylor_pruning(
            model=model,
            source_loaders_list=source_loaders,
            target_loader=target_loader,
            device=device,
            prune_rates=config.prune_rates,
            retrain_epochs=config.retrain_epochs,
            lr=config.lr,
            alpha=config.alpha,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            SFT=SFT,
            importance_type=config.importance_type,
            keep_overall_best=config.keep_overall_best
        )
    
    elif config.pruning_method == 'adaptive':
        pruned_model, mask = adaptive_layerwise_pruning(
            model=model,
            source_loaders_list=source_loaders,
            target_loader=target_loader,
            device=device,
            retrain_epochs=config.retrain_epochs,
            lr=config.lr,
            alpha=config.alpha,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            SFT=SFT,
            pruning_strategy=config.pruning_strategy,
            candidate_rates=config.candidate_rates,
            iterations=config.iterations,
            calibration_samples=config.calibration_samples,
            relative_acc_drop_threshold=config.relative_acc_drop_threshold
        )
    
    else:
        raise ValueError(f"Unknown pruning method: {config.pruning_method}")
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    _, final_acc = evaluate(pruned_model, target_loader, device, mask=mask)
    print(f"Final Target Accuracy: {final_acc:.2f}%")
    
    final_model_path = os.path.join(config.output_dir, 'final_model.pth')
    final_mask_path = os.path.join(config.output_dir, 'final_mask.pth')
    torch.save(pruned_model.state_dict(), final_model_path)
    torch.save(mask, final_mask_path)
    print(f"Saved final model to: {final_model_path}")
    print(f"Saved final mask to: {final_mask_path}")
    
    total_params = sum(p.numel() for p in pruned_model.parameters())
    if mask:
        pruned_params = sum((m == 0).sum().item() for m in mask.values())
        pruning_ratio = pruned_params / total_params * 100
        print(f"\nPruning Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Pruned parameters: {pruned_params:,}")
        print(f"  Pruning ratio: {pruning_ratio:.2f}%")
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Results saved to: {config.output_dir}\n")


if __name__ == '__main__':
    main()
